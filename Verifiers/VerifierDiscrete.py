# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import math, time, random, copy
from Verifiers.Verifier import Verifier
from data_utils import get_batches

class VerifierDiscrete(Verifier):
    def __init__(self, args, target, logger):
        super(VerifierDiscrete, self).__init__(args, target, logger)
        self.word_embeddings = target.model.bert.embeddings.word_embeddings.weight
        self.vocab = target.vocab
        self.vocab_size = len(self.vocab)
        self.words = []
        self.batch_size = args.batch_size
        for w in target.vocab:
            if not w in ["#", "["]:
                self.words.append(w)

    def verify(self, example):
        start_time = time.time()      

        embeddings, tokens = self.target.get_embeddings([example])
        length = embeddings.shape[1]
        tokens = tokens[0]
        
        self.logger.write("tokens:", " ".join(tokens))
        self.logger.write("length:", length)        
        self.logger.write("label:", example["label"])

        self.std = self.target.step([example], infer_grad=True)[-1] 

        result = {
            "tokens": tokens,
            "label": float(example["label"]),
            "bounds": []
        }        

        cnt = 0
        sum_eps, min_eps = 0, 1e30        

        assert(self.perturbed_words == 1)
        # [CLS] and [SEP] cannot be perturbed
        for i in range(1, length - 1):
            # skip OOV
            if tokens[i][0] == "#" or tokens[i + 1][0] == "#":
                continue
            
            candidates = []
            for w in self.words:
                _tokens = copy.deepcopy(tokens)
                _tokens[i] = w
                sent = ""
                for _w in _tokens[1:-1]:
                    if _w[0] == "#":
                        sent += _w[2:] + " "
                    else:
                        sent += _w + " "
                candidates.append({
                    "sent_a": sent.split(),
                    "label": example["label"]
                })
            
            epsilon = 1e10
            epsilon_max = 0
            for batch in get_batches(candidates, self.batch_size):
                r = self.target.step(batch)[-1]
                dist = torch.norm(
                    r["embedding_output"][:, i] - embeddings[0][i].unsqueeze(0), p=self.p, dim=-1)
                for j in range(len(batch)):
                    if r["pred_labels"][j] != example["label"]:
                        epsilon = min(epsilon, float(dist[j]))
                    epsilon_max = max(epsilon_max, float(dist[j]))
            epsilon = min(epsilon, epsilon_max)
                    
            epsilon_normalized = epsilon / torch.norm(embeddings[0, i], p=self.p)
                        
            self.logger.write("Position %d: %s %.5f %.5f" % (
                i, tokens[i], epsilon, epsilon_normalized))

            result["bounds"].append({
                "position": i,
                "eps": float(epsilon),
                "eps_normalized": float(epsilon_normalized)
            })       

            cnt += 1
            sum_eps += epsilon
            min_eps = min(min_eps, epsilon)         

        result["time"] = time.time() - start_time            

        self.logger.write("Time elapsed", result["time"])

        return result, sum_eps / cnt, min_eps