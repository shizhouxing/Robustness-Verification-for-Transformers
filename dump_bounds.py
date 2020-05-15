# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

"""
Main results for 1-word perturbation
"""
import sys, os, json, pdb, argparse

def load_result(filename):
    with open(filename) as file:
        return json.loads(file.read())

def print_small_float(x):
    str = "%.1E" % x
    idx = str.find("E") 
    if str[idx + 2] == "0":
        str = str[:(idx+2)] + str[(idx+3):]
    return str

if __name__ == "__main__":
    acc = load_result("results/res_accuracy.json")
p_list = ["1", "2", "100"]

def dump_one_word():
    for i, dataset in enumerate(["Yelp", "SST"]):
        for j, model in enumerate(["1", "2", "3"]):
            if j == 0:
                print("\multirow{9}{*}{%s} " % dataset, end="")
            else:
                print("\cline{2-12}")
            print("& ", end="")

            for k, p in enumerate(["1", "2", "\infty"]):
                if k == 0:
                    print("\multirow{3}{*}{%s} & \multirow{3}{*}{%.2f} " % (
                        model, acc["model_{}_{}".format(dataset.lower(), model)] * 100), end="")
                else:
                    print("& & ", end="")
                print("& ", end="")

                print("$\ell_%s$ & " % p, end="")

                res_upper = load_result("results/res_model_{}_{}_discrete_{}_1.json".format(
                    dataset.lower(), model, p_list[k]))
                print("{:.3f} & {:.3f} & ".format(res_upper["minimum"], res_upper["average"]))

                res = load_result("results/res_model_{}_{}_ibp_{}_1.json".format(
                    dataset.lower(), model, p_list[k]))
                print("{} & {} & ".format(
                    print_small_float(res["minimum"]), print_small_float(res["average"])))

                res = load_result("results/res_model_{}_{}_baf_{}_1.json".format(
                    dataset.lower(), model, p_list[k]))
                print("{:.3f} & {:.3f} & ".format(res["minimum"], res["average"]))                

                print("{:.0f}\\% & {:.0f}\\% ".format(
                    res["minimum"] / res_upper["minimum"] * 100, 
                    res["average"] / res_upper["average"] * 100), end="")

                print("\\\\")

        print("\hline")

def dump_two_word():
    for j, model in enumerate(["1", "2", "3"]):
        print("{} & ".format(model))
        for i, dataset in enumerate(["Yelp", "SST"]):
            res = load_result("results/res_model_{}_{}_ibp_2_2.json".format(
                dataset.lower(), model))
            print("{} & {} & ".format(
                print_small_float(res["minimum"]), print_small_float(res["average"])))
            res = load_result("results/res_model_{}_{}_baf_2_2.json".format(
                dataset.lower(), model))
            print("{:.3f} & {:.3f} ".format(
                res["minimum"], res["average"]))
        print("\\\\")

def dump_framework():
    acc = [91.3, 83.3]
    for i, dataset in enumerate(["Yelp", "SST"]):
        for j, p in enumerate(["1", "2", "\infty"]):
            if j == 0:
                print("\multirow{3}{*}{%s} & \multirow{3}{*}{%.2f}" % (dataset, acc[i]), end="")
            else:
                print("& ", end="")
            print(" & $\ell_%s$ & " % p, end="")

            for k, method in enumerate(["forward", "backward", "baf"]):
                if k > 0:
                    print(" & ", end="")
                _p = p if p != "\infty" else "100"
                res = load_result("results/res_model_{}_small_1_{}_{}_1.json".format(
                    dataset.lower(), method, p_list[j]))
                avg_time = 0
                for example in res["examples"]:
                    avg_time += example["time"]
                avg_time /= len(res["examples"])
                print("%.3f & %.3f & %.1f " % (
                    res["minimum"], res["average"], avg_time), end="")

            print("\\\\")

        print("\hline")

def dump_layer_normalization():
    ln_list = ["_standard", "_no", ""]
    for i, dataset in enumerate(["Yelp", "SST"]):
        for j, model in enumerate(["1", "2"]):
            for jj, ln in enumerate(["Standard", "None", "Ours"]):
                for k, p in enumerate(["1", "2", "\infty"]):
                    if j == 0 and jj == 0 and k == 0:
                        print("\multirow{%d}{*}{%s} " % (2 * 3 * 3, dataset), end="")
                    if jj == 0 and k == 0:
                        print(" & \multirow{%d}{*}{%s} " % (3 * 3, model), end="")
                    else:
                        print(" & ", end="")
                    if k == 0:
                        print(" & \multirow{3}{*}{%s} " % (ln), end="")
                        print(" & \multirow{3}{*}{%.1f} " % (
                            acc["model_{}_{}{}".format(dataset.lower(), model, ln_list[jj])] * 100), end="")
                    else:
                        print(" & ", end="")
                        print(" & ", end="")
                    print(" & $\ell_%s$ " % p, end="")

                    res_upper = load_result("results/res_model_{}_{}{}_discrete_{}_1{}.json".format(
                        dataset.lower(), model, ln_list[jj], p_list[k], ln_list[jj]))
                    print("& {:.3f} & {:.3f} ".format(res_upper["minimum"], res_upper["average"]))

                    res = load_result("results/res_model_{}_{}{}_baf_{}_1{}.json".format(
                        dataset.lower(), model, ln_list[jj], p_list[k], ln_list[jj]))
                    print("& {:.3f} & {:.3f} ".format(res["minimum"], res["average"]))

                    if jj == 0:
                        print("& %s\\ & %s\\ " % (
                            print_small_float(res["minimum"] / res_upper["minimum"]), 
                            print_small_float(res["average"] / res_upper["average"])), end="")    
                    else:
                        print("& %.0f\\%% & %.0f\\%% " % (
                            res["minimum"] / res_upper["minimum"] * 100, 
                            res["average"] / res_upper["average"] * 100), end="")

                    print("\\\\")

                if jj < 2:
                    print("\cline{3-11}")

            if j == 0:
                print("\cline{2-11}")

        print("\hline")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str, default=None, 
                        choices=["one", "two", "frameworks", "ln"])
    args = parser.parse_args()

    if args.exp == "one":
        dump_one_word()
    elif args.exp == "two":
        dump_two_word()
    elif args.exp == "frameworks":
        dump_framework()
    elif args.exp == "ln":
        dump_layer_normalization()
    else:
        raise NotImplementedError