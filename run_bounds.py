# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

"""
Computer certified bounds for main results
"""

import os, argparse, json
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=None, required=True)
parser.add_argument("--model", type=str, nargs="*")
parser.add_argument("--p", type=str, nargs="*")
parser.add_argument("--method", type=str, nargs="*")
parser.add_argument("--perturbed_words", type=int, default=1)
parser.add_argument("--samples", type=int, default=10)
parser.add_argument("--suffix", type=str, default="")
args = parser.parse_args()

if len(args.suffix) > 0:
    args.suffix = "_" + args.suffix
if args.perturbed_words == 2:
    max_verify_length = 16
else:
    max_verify_length = 32    

res_all = []

def verify(model, method, p):
    log = "log_{}_{}_{}_{}{}.txt".format(model, method, p, args.perturbed_words, args.suffix)
    res = "res_{}_{}_{}_{}{}.json".format(model, method, p, args.perturbed_words, args.suffix)
    res_all.append(res)
    cmd = "python main.py --verify --data={} --dir={} --method={} --p={}\
            --max_verify_length={} --perturbed_words={} --samples={}\
            --log={} --res={}".format(
                args.data, model, method, p,
                max_verify_length, args.perturbed_words, args.samples,
                log, res
            )
    print(cmd)
    os.system(cmd)

for model in args.model:
    for method in args.method:
        for p in args.p:
            verify(model, method, p)

for res in res_all:
    print(res)
