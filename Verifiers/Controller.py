# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch

class Controller:
    def __init__(self, args, eps):
        self.args = args
        self.layers = []
        self.p = args.p
        self.q = 1. / (1. - 1. / args.p) if args.p != 1 else float("inf") # dual norm
        self.eps = eps
        self.perturbed_words = args.perturbed_words

    def append(self, layer):
        self.layers.append(layer)

    def compute(self, length, dim):
        self.lb = torch.zeros(length, dim).cuda()
        self.ub = self.lb.clone()
        self.final_lw = self.final_uw = None
        self.final_lb = self.final_ub = None
        for layer in self.layers[::-1]:
            if layer.lw is not None:
                layer.backward()

    def concretize_l(self, lw=None):
        return -self.eps * torch.norm(lw, p=self.q, dim=-1)

    def concretize_u(self, uw=None):      
        return self.eps * torch.norm(uw, p=self.q, dim=-1)

    def concretize(self, lw, uw):
        if self.perturbed_words == 2:
            assert(len(lw.shape) == 3) 
            half = lw.shape[-1] // 2
            return \
                self.concretize_l(lw[:, :, :half]) + self.concretize_l(lw[:, :, half:]),\
                self.concretize_u(uw[:, :, :half]) + self.concretize_u(uw[:, :, half:])
        elif self.perturbed_words == 1:
            return self.concretize_l(lw), self.concretize_u(uw)      
        else:
            raise NotImplementedError