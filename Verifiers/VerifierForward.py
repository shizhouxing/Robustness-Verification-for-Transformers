# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import math, time, random, copy
from Verifiers import Verifier
from Verifiers.Bounds import Bounds
from Verifiers.utils import check

# can only accept one example in each batch
class VerifierForward(Verifier):
    def __init__(self, args, target, logger):
        super(VerifierForward, self).__init__(args, target, logger)
        self.ibp = args.method == "ibp"

    def verify_safety(self, example, embeddings, index, eps):
        errorType = OSError if self.debug else AssertionError

        try:
            with torch.no_grad():
                bounds = self._bound_input(embeddings, index=index, eps=eps) # hard-coded yet

                check("embedding", bounds=bounds, std=self.std["embedding_output"], verbose=self.verbose)
            
                if self.verbose:
                    bounds.print("embedding")

                for i, layer in enumerate(self.encoding_layers):
                    attention_scores, attention_probs, bounds = self._bound_layer(bounds, layer)

                    check("layer %d attention_scores" % i, 
                        bounds=attention_scores, std=self.std["attention_scores"][i][0], verbose=self.verbose)
                    check("layer %d attention_probs" % i, 
                        bounds=attention_probs, std=self.std["attention_probs"][i][0], verbose=self.verbose)
                    check("layer %d" % i, bounds=bounds, std=self.std["encoded_layers"][i], verbose=self.verbose)
                    
                bounds = self._bound_pooling(bounds, self.pooler)
                check("pooled output", bounds=bounds, std=self.std["pooled_output"], verbose=self.verbose)

                safety = self._bound_classifier(bounds, self.classifier, example["label"])

                return safety
        except errorType as err: # for debug
            if self.verbose:
                print("Warning: failed assertion")
                print(err)
            print("Warning: failed assertion", eps)
            return False

    def _bound_input(self, embeddings, index, eps):
        length, dim = embeddings.shape[1], embeddings.shape[2]

        w = torch.zeros((length, dim * self.perturbed_words, dim)).to(self.device)
        b = embeddings[0]   
        lb, ub = b, b.clone()     
        
        if self.perturbed_words == 1:
            if self.ibp:
                lb[index], ub[index] = lb[index] - eps, ub[index] + eps
            else:
                w[index] = torch.eye(dim).to(self.device)
        else:
            if self.ibp:
                for i in range(self.perturbed_words):
                    lb[index[i]], ub[index[i]] = lb[index[i]] - eps, ub[index[i]] + eps
            else:
                for i in range(self.perturbed_words):
                    w[index[i], (dim * i):(dim * (i + 1)), :] = torch.eye(dim).to(self.device)
            
        lw = w.unsqueeze(0)
        uw = lw.clone()
        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)

        bounds = Bounds(self.args, self.p, eps, lw=lw, lb=lb, uw=uw, ub=ub)

        bounds = bounds.layer_norm(self.embeddings.LayerNorm, self.layer_norm)

        return bounds

    def _bound_layer(self, bounds_input, layer):
        start_time = time.time()

        # main self-attention
        attention_scores, attention_probs, attention = \
            self._bound_attention(bounds_input, layer.attention)

        attention = attention.dense(layer.attention.output.dense)
        attention = attention.add(bounds_input).layer_norm(layer.attention.output.LayerNorm, self.layer_norm)
        del(bounds_input)

        if self.verbose:
            attention.print("after attention layernorm")
            attention.dense(layer.intermediate.dense).print("intermediate pre-activation")
            print("dense norm", torch.norm(layer.intermediate.dense.weight, p=self.p))
            start_time = time.time()

        intermediate = attention.dense(layer.intermediate.dense).act(self.hidden_act)

        if self.verbose:
            intermediate.print("intermediate")
            start_time = time.time()            

        dense = intermediate.dense(layer.output.dense).add(attention)
        del(intermediate)
        del(attention)

        if self.verbose:
            print("dense norm", torch.norm(layer.output.dense.weight, p=self.p))
            dense.print("output pre layer norm")

        output = dense.layer_norm(layer.output.LayerNorm, self.layer_norm)

        if self.verbose:
            output.print("output")
            # print(" time", time.time() - start_time)
            start_time = time.time()            

        return attention_scores, attention_probs, output

    def _bound_attention(self, bounds_input, attention):
        num_attention_heads = attention.self.num_attention_heads
        attention_head_size = attention.self.attention_head_size

        query = bounds_input.dense(attention.self.query)
        key = bounds_input.dense(attention.self.key)
        value = bounds_input.dense(attention.self.value)

        del(bounds_input)

        def transpose_for_scores(x):
            def transpose_w(x):
                return x\
                    .reshape(
                        x.shape[0], x.shape[1], x.shape[2], 
                        num_attention_heads, attention_head_size)\
                    .permute(0, 3, 1, 2, 4)\
                    .reshape(-1, x.shape[1], x.shape[2], attention_head_size)
            def transpose_b(x):
                return x\
                    .reshape(
                        x.shape[0], x.shape[1], num_attention_heads, attention_head_size)\
                    .permute(0, 2, 1, 3)\
                    .reshape(-1, x.shape[1], attention_head_size)
            x.lw = transpose_w(x.lw)
            x.uw = transpose_w(x.uw)
            x.lb = transpose_b(x.lb)
            x.ub = transpose_b(x.ub)
            x.update_shape()

        transpose_for_scores(query)
        transpose_for_scores(key)

        # TODO: no attention mask for now (doesn't matter for batch_size=1)
        attention_scores = query.dot_product(key, verbose=self.verbose)\
            .multiply(1. / math.sqrt(attention_head_size))        

        if self.verbose:
            attention_scores.print("attention score")

        del(query)
        del(key)
        attention_probs = attention_scores.softmax(verbose=self.verbose)

        if self.verbose:
            attention_probs.print("attention probs")

        transpose_for_scores(value)  

        context = attention_probs.context(value)

        if self.verbose:
            value.print("value")        
            context.print("context")

        def transpose_back(x):
            def transpose_w(x):
                return x.permute(1, 2, 0, 3).reshape(1, x.shape[1], x.shape[2], -1)
            def transpose_b(x):
                return x.permute(1, 0, 2).reshape(1, x.shape[1], -1)

            x.lw = transpose_w(x.lw)
            x.uw = transpose_w(x.uw)
            x.lb = transpose_b(x.lb)
            x.ub = transpose_b(x.ub)
            x.update_shape()
        
        transpose_back(context)

        return attention_scores, attention_probs, context
        
    def _bound_pooling(self, bounds, pooler):
        bounds = Bounds(
            self.args, bounds.p, bounds.eps,
            lw = bounds.lw[:, :1, :, :], lb = bounds.lb[:, :1, :],
            uw = bounds.uw[:, :1, :, :], ub = bounds.ub[:, :1, :]
        )
        if self.verbose:
            bounds.print("pooling before dense")

        bounds = bounds.dense(pooler.dense)

        if self.verbose:
            bounds.print("pooling pre-activation")

        bounds = bounds.tanh()

        if self.verbose:
            bounds.print("pooling after activation")
        return bounds

    def _bound_classifier(self, bounds, classifier, label):
        classifier = copy.deepcopy(classifier)
        classifier.weight[0, :] -= classifier.weight[1, :]
        classifier.bias[0] -= classifier.bias[1]

        if self.verbose:
            bounds.print("before dense")
            print(torch.norm(classifier.weight[0, :]))
            print(torch.mean(torch.norm(bounds.lw, dim=-2)))
            print(torch.mean(torch.norm(bounds.dense(classifier).lw, dim=-2)))

        bounds = bounds.dense(classifier)
        
        if self.verbose:
            bounds.print("after dense")

        l, u = bounds.concretize()

        if self.verbose:
            print(l[0][0][0])
            print(u[0][0][0])

        if label == 0:
            safe = l[0][0][0] > 0
        else:
            safe = u[0][0][0] < 0

        if self.verbose:
            print("Safe" if safe else "Unsafe")

        return safe
