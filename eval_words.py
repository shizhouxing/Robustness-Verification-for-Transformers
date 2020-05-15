# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

"""Experiments for comparing whether each method can identify important words"""
import torch
import copy, argparse, random, pdb
import numpy as np
from dump_bounds import load_result

random.seed(0)

methods = ["Grad", "Upper", "Ours"]

def eval_words(args, target, data_test):
    if args.data == "yelp":
        bounds = {
            "Upper": load_result("results/res_model_yelp_1_discrete_2_1.json"),
            "Ours": load_result("results/res_model_yelp_1_baf_2_1.json")
        } 
    else:
        bounds = {
            "Upper": load_result("res_model_sst_1_discrete_2_1_100s.json"),
            "Ours": load_result("res_model_sst_1_baf_2_1_100s.json")
        } 

    label_dict = {}
    for example in data_test:
        for i in range(len(example["sent_a"])):
            if "word_labels" in example:
                label_dict[example["sent_a"][i].lower()] = example["word_labels"][i]

    words_sum_top = {}
    words_sum_bottom = {}
    words_top = {}
    words_bottom = {}
    for method in methods:
        words_sum_top[method] = np.zeros(100)
        words_sum_bottom[method] = np.zeros(100)
        words_top[method] = []
        words_bottom[method] = []

    def add(method, w):
        w = sorted(w, key=lambda x:x[0], reverse=True)
        if method in ["Upper", "Ours"]:
            w = w[::-1]
        for i in range(len(w)):
            if w[i][1] in label_dict:
                words_sum_top[method][i] += abs(label_dict[w[i][1]] - 2)
        words_top[method].append(w[0][1])                
        w = w[::-1]
        for i in range(len(w)):
            if w[i][1] in label_dict:
                words_sum_bottom[method][i] += abs(label_dict[w[i][1]] - 2)               
        words_bottom[method].append(w[0][1])

    for t, example in enumerate(bounds["Ours"]["examples"]):
        tokens = copy.deepcopy(example["tokens"])  

        sent = ""
        for j in range(1, len(tokens) - 1):
            cur = tokens[j]
            if cur[0] == "#":
                cur = cur[2:]
            sent += cur + " "
        std = target.step([{
            "sent_a": sent.split(), 
            "label": int(example["label"])
        }], infer_grad=True)[-1]

        valid = [0] * std["embedding_output"][0].shape[0]
        for p in example["bounds"]:
            valid[p["position"]] = True

        # assume we only consider one word now

        for method in ["Upper", "Ours"]:
            if method in bounds:
                w = []
                for p in bounds[method]["examples"][t]["bounds"]:
                    w.append((p["eps_normalized"], tokens[p["position"]]))
                add(method, w)

        grad = torch.norm(std["gradients"][0], p=2, dim=-1) \
            # / torch.norm(std["embedding_output"][0], p=2, dim=-1) # this results in worse results
        w = []
        for i in range(1, len(tokens) - 1):
            if valid[i]:
                w.append((float(grad[i]), tokens[i]))
        add("Grad", w)

    if args.data == "yelp":
        important_words = ["terrible", "great", "best", "good", "slow", 
                        "perfect", "typical", "decadent"]
        for word_list in [words_top, words_bottom]:
            if word_list == words_top:
                type = "Most"
            else:
                type = "Least"
            k = 10
            for t, method in enumerate(methods):
                print("%s & 0.00 & " % method, end="")
                used = {}
                for i, w in enumerate(word_list[method][:k]):
                    used[w] = True
                    if w == "&": _w = "\\&"
                    else: _w = w
                    if w in important_words:
                        print("\\textbf{\\texttt{%s}}" % _w, end="")
                    else:
                        print("\\texttt{%s}" % _w, end="")
                    if i + 1 < len(word_list[method][:k]):
                        print(" /", end=" ")
                print("\\\\")
            print()
    else:
        cnt = len(bounds[method]["examples"])
        for method in methods:
            ours = method == "Ours"
            print(method)
            print("{:.2f}".format(np.sum(words_sum_top[method][0]) * 1. / cnt))
            print("{:.2f}".format(np.sum(words_sum_bottom[method][0]) * 1. / cnt))
