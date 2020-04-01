# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights rved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse, csv, logging, os, random, sys, shutil, scipy, pdb
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from Models.modeling import BertForSequenceClassification
from Models.utils import truncate_seq_pair, acc_and_f1, InputExample, \
    InputFeatures, convert_examples_to_features

class BERT:
    def __init__(self, args, data_train):
        self.general_init(args, data_train)
        self.load_pretrained()
        self._build_trainer()

    def general_init(self, args, data_train):
        self.args = args
        self.max_seq_length = args.max_sent_length
        self.do_lower_case = True
        self.learning_rate = args.lr
        self.gradient_accumulation_steps = 1
        self.seed = args.seed
        self.num_labels = args.num_labels
        self.label_list = range(args.num_labels) 
        self.num_train_optimization_steps = \
            args.num_epoches * (len(data_train) + args.batch_size - 1) // args.batch_size
        self.warmup_proportion = args.warmup
        self.weight_decay = args.weight_decay
        self.device = args.device

        self.dir = self.bert_model = args.dir
        self.checkpoint = False
        if not os.path.exists(self.dir):
            os.system("cp -r %s %s" % (args.base_dir, self.dir))
        if os.path.exists(os.path.join(self.bert_model, "checkpoint")):
            with open(os.path.join(self.bert_model, "checkpoint")) as file:
                self.bert_model = os.path.join(self.bert_model, "ckpt-%d" % (int(file.readline())))
                self.checkpoint = True
                print("BERT checkpoint:", self.bert_model)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case)
        self.vocab = self.tokenizer.vocab
        self.softmax = torch.nn.Softmax(dim=-1)        

    def load_pretrained(self):
        cache_dir = "cache/bert"
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model,
                cache_dir=cache_dir,
                num_labels=self.num_labels)      
        self.model.to(self.device)

    def _build_trainer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer\
                if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer\
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = BertAdam(optimizer_grouped_parameters,
            lr=self.learning_rate,
            warmup=self.warmup_proportion,
            t_total=self.num_train_optimization_steps
        )

    def save(self, epoch):
        # Save a trained model, configuration and tokenizer
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_dir = os.path.join(self.dir, "ckpt-%d" % epoch)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        self.tokenizer.save_vocabulary(output_dir)    

        with open(os.path.join(self.dir, "checkpoint"), "w") as file:
            file.write("%d" % epoch)   

        print("BERT saved: %s" % output_dir) 

    def get_input(self, batch):
        features = convert_examples_to_features(
            batch, self.label_list, self.max_seq_length, self.tokenizer)
        
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)        

        input_ids = input_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        segment_ids = segment_ids.to(self.device)

        return input_ids, input_mask, segment_ids, features

    def get_embeddings(self, batch):
        input_ids, input_mask, token_type_ids, features = self.get_input(batch)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.model.bert.embeddings.word_embeddings(input_ids)
        position_embeddings = self.model.bert.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(token_type_ids)
        embeddings = (word_embeddings + position_embeddings + token_type_embeddings)
        tokens = [feature.tokens for feature in features]

        return embeddings, tokens

    def step(self, batch, is_train=False, infer_grad=False):
        input_ids, input_mask, segment_ids, features = self.get_input(batch)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        label_ids = label_ids.to(self.device)  
        
        if is_train:
            self.model.train()
            logits, embedding_output, encoded_layers, attention_scores, attention_probs, \
                self_output, pooled_output = self.model(
                    input_ids, segment_ids, input_mask, labels=None)
        else:
            self.model.eval()
            grad = torch.enable_grad() if infer_grad else torch.no_grad()
            with grad:
                logits, embedding_output, encoded_layers, attention_scores, attention_probs, \
                    self_output, pooled_output = self.model(
                        input_ids, segment_ids, input_mask, labels=None)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))

        preds = self.softmax(logits).detach().cpu().numpy()
        pred_labels = np.argmax(preds, axis=1)
        acc = acc_and_f1(pred_labels, label_ids.cpu().numpy())["acc"]

        if infer_grad:
            gradients = torch.autograd.grad(loss, embedding_output)[0]
        else:
            gradients = None

        ret = [
            loss, acc,
            {
                "pred_scores": preds, 
                "pred_labels": pred_labels,
                "embedding_output": embedding_output,
                "encoded_layers": encoded_layers,
                "attention_scores": attention_scores,
                "attention_probs": attention_probs,
                "self_output": self_output,
                "pooled_output": pooled_output,
                "features": features,
                "gradients": gradients
            }
        ]

        if is_train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()    
    
        return ret

    def predict(self, X):
        return self.step([
            { "sent_a": x, "label": 1}
            for x in X
        ])[-1]["pred_scores"]

    def get_gradients(self, batch, embeddings):
        input_ids, input_mask, segment_ids, features = self.get_input(batch)
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        label_ids = label_ids.to(self.device)        

        self.model.eval()
        logits, embedding_output, encoded_layers, attention_scores, attention_probs, \
            self_output, pooled_output = self.model(
                input_ids, segment_ids, input_mask, labels=None)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))

        grad = torch.zeros(preds.shape[0], preds.shape[1], embeddings.shape[-1]).to(self.device)
        grad[:, 0, :] = torch.autograd.grad(torch.sum(preds, dim=0)[0], embeddings,
            retain_graph=True, only_inputs=True)[0][:, pos, :]
        grad[:, 1, :] = -grad[:, 0, :]

        return grad