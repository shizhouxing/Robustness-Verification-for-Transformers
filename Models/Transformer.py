# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import os
from Models.BERT import BERT
from Models.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer

class Transformer(BERT):
    def __init__(self, args, data_train):
        self.general_init(args, data_train)

        self.min_word_freq = args.min_word_freq
        self.update_vocabulary(data_train)

        config = BertConfig(self.vocab_size)
        config.num_hidden_layers = args.num_layers
        config.hidden_size = args.hidden_size
        config.intermediate_size = args.intermediate_size
        config.hidden_act = args.hidden_act
        config.num_attention_heads = args.num_attention_heads
        config.layer_norm = args.layer_norm

        self.load_pretrained()
        if not self.checkpoint:
            bert = self.model.bert

            self.model = BertForSequenceClassification(config, self.num_labels)
            self.model.to(self.device)

        self._build_trainer()

    """
    Build a vocabulary from the training data instead of using BERT's vocabulary.
    Because we are now training the Transformer from scratch.
    """
    def update_vocabulary(self, data_train):        
        vocab_base = os.path.join(self.bert_model, "vocab_base.txt")
        if not os.path.exists(vocab_base):
            with open(os.path.join(self.bert_model, "vocab.txt")) as file:
                self.vocab_size = len(file.readlines())
            return
        cnt = {}
        in_bert = {}
        with open(vocab_base) as file:
            for line in file.readlines():
                cnt[line[:-1]] = 0
                in_bert[line[:-1]] = True
        for example in data_train:
            for token in example["sent_a"]:
                if not token in cnt:
                    cnt[token] = 0
                cnt[token] += 1
        cnt["[PAD]"] = 1e8
        words = []
        for w in cnt:
            if w[0] == "#" or w[0] == "[" or w in in_bert or cnt[w] >= self.min_word_freq:
                words.append(w)
        words = sorted(words, key=lambda w:cnt[w], reverse=True)          
        with open(os.path.join(self.bert_model, "vocab.txt"), "w") as file:
            for w in words:
                file.write("%s\n" % w)

        self.vocab_size = len(words)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case)
        self.vocab = self.tokenizer.vocab        
