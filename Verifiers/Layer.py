# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import torch.nn as nn
import math, time
from Verifiers.Controller import Controller
from Verifiers.Bounds import Bounds

epsilon = 1e-12

class Layer:
    def __init__(self, args, controller, length, dim, bounds=None):
        self.args = args
        self.controller = controller
        self.length = length
        self.dim = dim
        self.use_forward = args.method == "baf"
        self.parents = []
        self.l = self.u = None
        # for back propagation
        self.lw = self.uw = None         
        # bounds of the layer 
        self.final_lw = self.final_uw = None 
        self.final_lb = self.final_ub = None
        self.empty_cache = args.empty_cache
        self.controller.append(self)

        # bounds obtained from the forward framework
        if bounds is not None:
            self.back = False
            self.bounds = bounds

            self.l, self.u = bounds.concretize()
            self.final_lw, self.final_lb = bounds.lw.transpose(-1, -2), bounds.lb
            self.final_uw, self.final_ub = bounds.uw.transpose(-1, -2), bounds.ub

            # incompatible format (batch)
            self.l = self.l[0]
            self.u = self.u[0]
            self.final_lw = self.final_lw[0]
            self.final_lb = self.final_lb[0]
            self.final_uw = self.final_uw[0]
            self.final_ub = self.final_ub[0]
        else:
            self.back = True
            
    def print(self, message):
        print(message)
        print("shape (%d, %d)" % (self.length, self.dim))
        print("mean abs %.5f %.5f" % (torch.mean(torch.abs(self.l)), torch.mean(torch.abs(self.u))))
        print("diff %.5f %.5f %.5f" % (torch.min(self.u - self.l), torch.max(self.u - self.l), torch.mean(self.u - self.l)))
        print("min", torch.min(self.l))
        print("max", torch.max(self.u))
        print()

    def add_edge(self, edge):
        self.parents.append(edge)

    def next(self, edge, length=None, dim=None):
        if length is None:
            length = self.length
        if dim is None:
            dim = self.dim
        layer = Layer(self.args, self.controller, length, dim)
        layer.add_edge(edge)
        layer.compute()
        return layer

    def compute(self):
        if self.use_forward:    
            self.lw = torch.eye(self.dim).cuda()\
                .reshape(1, self.dim, self.dim).repeat(self.length, 1, 1)
        else:
            self.lw = torch.eye(self.length * self.dim).cuda()\
                .reshape(self.length, self.dim, self.length, self.dim) 
        self.uw = self.lw.clone()
        self.controller.compute(self.length, self.dim)
        self.l, self.u = self.controller.lb, self.controller.ub
        self.final_lw, self.final_uw = self.controller.final_lw, self.controller.final_uw
        self.final_lb, self.final_ub = self.controller.final_lb, self.controller.final_ub

    def backward_buffer(self, lw, uw):    
        if self.lw is None:
            self.lw, self.uw = lw, uw
        else:
            self.lw += lw
            self.uw += uw

    def backward(self):
        if self.back:
            for edge in self.parents:
                edge.backward(self.lw, self.uw)
        else:
            bounds_l = self.bounds.matmul(self.lw)\
                .add(self.controller.lb.unsqueeze(0))
            bounds_u = self.bounds.matmul(self.uw)\
                .add(self.controller.ub.unsqueeze(0))
            bounds = Bounds(
                bounds_l.args, bounds_l.p, bounds_l.eps,
                lw = bounds_l.lw, lb = bounds_l.lb,
                uw = bounds_u.uw, ub = bounds_u.ub   
            )
            self.controller.final_lw = bounds.lw[0].transpose(1, 2)
            self.controller.final_uw = bounds.uw[0].transpose(1, 2)
            self.controller.final_lb = bounds.lb[0]
            self.controller.final_ub = bounds.ub[0]
            self.controller.lb, self.controller.ub = bounds.concretize()
            self.controller.lb = self.controller.lb[0]
            self.controller.ub = self.controller.ub[0]

        del(self.lw)
        del(self.uw)
        if self.empty_cache:
            torch.cuda.empty_cache()
        self.lw, self.uw = None, None