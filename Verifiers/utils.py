# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch

def check(name, bounds=None, l=None, u=None, std=None, verbose=False):
    if verbose:
        print("Check ", name)
    eps = 1e-4
    if bounds is not None:
        l, u = bounds.concretize()
    if len(l.shape) == 3:
        l, u, std = l[0], u[0], std[0]
    c = torch.gt(l - eps, std).to(torch.float) + torch.lt(u + eps, std).to(torch.float)
    if bounds is not None:
        c += torch.gt(bounds.lb[0] - eps, std).to(torch.float) + torch.lt(bounds.ub[0] + eps, std).to(torch.float)
    errors = torch.sum(c)
    score = float(torch.mean(u - l))
    if verbose:
        print("%d errors, %.5f average range" % (errors, score))
        if errors > 0:
            cnt = 0
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    if c[i][j] > 0:
                        print(i, j)
                        print(l[i][j], u[i][j], std[i][j])
                        cnt += 1
                        if cnt >= 10: 
                            assert(0)
    assert(errors == 0)