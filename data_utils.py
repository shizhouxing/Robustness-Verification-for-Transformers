# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import numpy as np
import json, re, os, nltk, pickle, gzip, random, csv
import torch
from tqdm import tqdm
from multiprocessing import Pool

if not os.path.exists("tmp"): os.mkdir("tmp")    

def tokenize(sent):
    return nltk.word_tokenize(sent)

def tokenize_example(example):
    for key in ["sent_a", "sent_b"]:
        if key in example:
            example[key] = tokenize(example[key])
    return example    

def load_data_yelp(args, set):
    path = "data/%s/%s.csv" % (args.data, set)    
    print("Loading yelp data from " + path)    
    data = []
    with open(path) as file:
        raw = csv.reader(file)
        for row in raw:
            if row[0] == "label": 
                continue
            text = row[1]
            text = text.replace("\\n", " ")
            text = text.replace('\\"', '"')
            data.append({
                "label": int(row[0]),
                "sent_a": text
            })
    with Pool(processes=args.cpus) as pool:
        data = pool.map(tokenize_example, data)   
    return data

def load_data_sst(args, set):
    if set != "train":
        path = "data/%s/%s.txt" % (args.data, set)    
        print("Loading sst data from " + path) 
        data = []  
        with open(path) as file:
            for line in file.readlines():
                segs = line[:-1].split(" ")
                tokens = []
                word_labels = []
                label = int(segs[0][1])
                if label < 2:
                    label = 0
                elif label >= 3:
                    label = 1
                else: 
                    continue
                for i in range(len(segs) - 1):
                    if segs[i][0] == "(" and segs[i][1] in ["0", "1", "2", "3", "4"]\
                            and segs[i + 1][0] != "(":
                        tokens.append(segs[i + 1][:segs[i + 1].find(")")])
                        word_labels.append(int(segs[i][1]))
                data.append({
                    "label": label,
                    "sent_a": tokens,
                    "word_labels": word_labels
                })
        for example in data:
            for i, token in enumerate(example["sent_a"]):
                if token == "-LRB-":
                    example["sent_a"][i] = "("
                if token == "-RRB-":
                    example["sent_a"][i] = ")"
    else:
        path = "data/sst/train-nodes.tsv"
        print("Loading sst data from " + path) 
        data = []  
        with open(path) as file:
            for line in file.readlines()[1:]:
                data.append({
                    "sent_a": line.split("\t")[0],
                    "label": int(line.split("\t")[1])
                })
        with Pool(processes=args.cpus) as pool:
            data = pool.map(tokenize_example, data)   
    return data
    
def load_data_raw(args, set):
    if args.data == "yelp":
        data = load_data_yelp(args, set)
    elif args.data == "sst":
        data = load_data_sst(args, set)
    else:
        raise NotImplementedError
    return data

def load_data(args):
    if args.small:
        path = "tmp/data_%s_small.pkl.gz" % (args.data)
        path_no_train = "tmp/data_%s_no_train_small.pkl.gz" % (args.data)
    else:
        path = "tmp/data_%s.pkl.gz" % (args.data)
        path_no_train = "tmp/data_%s_no_train.pkl.gz" % (args.data)
    path_load = path if args.train else path_no_train
    if os.path.exists(path_load):
        print("Loading cached data...")
        with gzip.open(path_load, "rb") as file:
            data_train, data_valid, data_test, vocab_char, vocab_word = pickle.load(file)
    else:
        data_train = load_data_raw(args, "train")
        if args.small: 
            random.shuffle(data_train)
            data_train = data_train[:len(data_train)//10]
        vocab_char, vocab_word = None, None
        data_test = load_data_raw(args, "test")
        if args.small: 
            random.shuffle(data_test)
            data_test = data_test[:len(data_test)//10]
        try:
            data_valid = load_data_raw(args, "dev")
            if args.small:
                random.shuffle(data_valid)
                data_valid = data_valid[:len(data_valid)//10]
        except FileNotFoundError:
            data_valid = []
        with gzip.open(path, "wb") as file:
            pickle.dump((data_train, data_valid, data_test, vocab_char, vocab_word), file)
        with gzip.open(path_no_train, "wb") as file:
            pickle.dump(([], data_valid, data_test, vocab_char, vocab_word), file)

    # in the yelp dataset labels are among {1, 2}
    if args.data == "yelp":
        for example in data_train + data_valid + data_test:
            example["label"] -= 1

    return data_train, data_valid, data_test, vocab_char, vocab_word

def get_batches(data, batch_size):
    batches = []
    for i in range((len(data) + batch_size - 1) // batch_size):
        batches.append(data[i * batch_size : (i + 1) * batch_size])
    return batches

def sample(args, data, target):
    examples = []
    for i in range(args.samples):
        while True:
            example = data[random.randint(0, len(data) - 1)]
            std = target.step([example])[-1]
            # too long
            if std["embedding_output"][0].shape[0] > args.max_verify_length:
                continue
            # incorrectly classified            
            if std["pred_labels"][0] != example["label"]:
                continue
            examples.append(example)
            break
    return examples

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
