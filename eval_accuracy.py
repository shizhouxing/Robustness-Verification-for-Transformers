# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import os, json

res = {}

for dataset in ["yelp", "sst"]:
    for num_layers in range(1, 4):
        for ln in ["", "_no", "_standard"]:
            dir = "model_{}_{}{}".format(dataset, num_layers, ln)
            command = "python main.py --dir={} --data={} --log=log.txt".format(dir, dataset)
            print(command)
            os.system(command)
            with open("log.txt") as file:
                acc = float(file.readline())
            res[dir] = acc

with open("res_acc.json", "w") as file:
    file.write(json.dumps(res, indent=4))
