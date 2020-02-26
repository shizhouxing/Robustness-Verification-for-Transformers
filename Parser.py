# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import argparse, os

class Parser(object):
    def getParser(self):
        parser = argparse.ArgumentParser()

        # modes
        parser.add_argument("--train", action="store_true")
        parser.add_argument("--infer", action="store_true")
        parser.add_argument("--verify", action="store_true")
        parser.add_argument("--word_label", action="store_true")

        # data
        parser.add_argument("--dir", type=str, default="dev")
        parser.add_argument("--base_dir", type=str, default="model_base")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--data", type=str, default="yelp",
                            choices=["yelp", "sst", "cifar", "mnist"])
        parser.add_argument("--use_tsv", action="store_true")    
        parser.add_argument("--vocab_size", type=int, default=50000)
        parser.add_argument("--small", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--use_dev", action="store_true")  
        parser.add_argument("--num_classes", type=int, default=2) 
        parser.add_argument("--task", type=str, default="text_classification", 
                            choices=["text_classification", "image"])
        
        # runtime
        parser.add_argument("--cpu", action="store_true")
        parser.add_argument("--cpus", type=int, default=32)
        parser.add_argument("--display_interval", type=int, default=50)

        # model
        parser.add_argument("--num_epoches", type=int, default=3)        
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--max_sent_length", type=int, default=128)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--num_labels", type=int, default=2) 
        parser.add_argument("--num_layers", type=int, default=12)
        parser.add_argument("--num_attention_heads", type=int, default=4)
        parser.add_argument("--hidden_size", type=int, default=256)
        parser.add_argument("--intermediate_size", type=int, default=512)
        parser.add_argument("--warmup", type=float, default=-1)
        parser.add_argument("--hidden_act", type=str, default="relu")
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--min_word_freq", type=int, default=50)
        parser.add_argument("--layer_norm", type=str, default="no_var",
                            choices=["standard", "no", "no_var"])

        # verification
        parser.add_argument("--samples", type=int, default=10)
        parser.add_argument("--p", type=int, default=2)
        parser.add_argument("--eps", type=float, default=1e-5)
        parser.add_argument("--max_eps", type=float, default=0.01)
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--debug_pos", type=int, default=1)
        parser.add_argument("--log", type=str, default="log.txt")
        parser.add_argument("--res", type=str, default="res.json")
        parser.add_argument("--max_verify_length", type=int, default=32)
        parser.add_argument("--method", type=str, default="forward",
                            choices=["baf", "backward", "forward", "ibp", "discrete"])
        parser.add_argument("--num_verify_iters", type=int, default=10)
        parser.add_argument("--view_embed_dist", action="store_true")
        parser.add_argument("--empty_cache", action="store_true")
        parser.add_argument("--perturbed_words", type=int, default=1, choices=[1, 2])

        return parser

def update_arguments(args):
    if args.infer or args.verify or args.word_label:
        args.small = True

    if not args.train:
        args.batch_size *= 30

    if args.cpu:
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
