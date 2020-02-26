# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import numpy as np
from Verifiers.Layer import Layer
from Verifiers.Controller import Controller
from Verifiers.Bounds import Bounds
import pdb

epsilon = 1e-12

def get_bounds_xy(l_x, u_x, l_y, u_y):
    alpha_l = l_y
    beta_l = l_x
    gamma_l = -alpha_l * beta_l        

    alpha_u = u_y
    beta_u = l_x
    gamma_u = -alpha_u * beta_u 

    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u    

class Edge:
    def __init__(self, args, controller):
        self.args = args
        self.controller = controller
        self.use_forward = args.method == "baf"
        self.empty_cache = args.empty_cache

    def backward(self, lw, uw):
        raise NotImplementedError     

class EdgeComplex(Edge):
    def __init__(self, args, controller):
        super(EdgeComplex, self).__init__(args, controller)

    def backward(self, lw, uw):
        self.res.backward_buffer(lw, uw)

class EdgeDirect(Edge):
    def __init__(self, args, controller, par):
        super(EdgeDirect, self).__init__(args, controller)
        self.par = par

    def backward(self, lw, uw):  
        self.par.backward_buffer(lw, uw)

class EdgeInput(Edge):
    def __init__(self, args, controller, embeddings, index):
        super(EdgeInput, self).__init__(args, controller)
        self.embeddings = embeddings
        self.index = index
        self.perturbed_words = args.perturbed_words

    def backward(self, lw, uw):
        if self.use_forward:
            if self.perturbed_words == 2:
                assert(type(self.index) == list)
                dim = lw.shape[2]
                self.controller.final_lw = torch.zeros(lw.shape[0], lw.shape[1], dim * 2).cuda() 
                self.controller.final_uw = torch.zeros(lw.shape[0], lw.shape[1], dim * 2).cuda() 
                self.controller.final_lw[self.index[0], :, :dim] = lw[self.index[0], :, :]
                self.controller.final_uw[self.index[0], :, :dim] = lw[self.index[0], :, :]
                self.controller.final_lw[self.index[1], :, dim:] = lw[self.index[1], :, :]
                self.controller.final_uw[self.index[1], :, dim:] = lw[self.index[1], :, :]
                _lb = torch.sum(self.embeddings.unsqueeze(1) * lw, dim=-1)
                _ub = torch.sum(self.embeddings.unsqueeze(1) * uw, dim=-1)
            elif self.perturbed_words == 1:
                assert(type(self.index) == int)
                self.controller.final_lw = torch.zeros(lw.shape[0], lw.shape[1], lw.shape[2]).cuda() 
                self.controller.final_uw = torch.zeros(lw.shape[0], lw.shape[1], lw.shape[2]).cuda() 
                self.controller.final_lw[self.index, :, :] = lw[self.index, :, :]
                self.controller.final_uw[self.index, :, :] = uw[self.index, :, :]
                _lb = torch.sum(self.embeddings.unsqueeze(1) * lw, dim=-1)
                _ub = torch.sum(self.embeddings.unsqueeze(1) * uw, dim=-1)
            else:
                raise NotImplementedError
        else:
            assert(type(self.index) == int)
            self.controller.final_lw = lw[:, :, self.index, :]
            self.controller.final_uw = uw[:, :, self.index, :]
            _lb = torch.sum(lw * self.embeddings.unsqueeze(0).unsqueeze(0), dim=[-1, -2])
            _ub = torch.sum(uw * self.embeddings.unsqueeze(0).unsqueeze(0), dim=[-1, -2]) 

        self.controller.lb += _lb
        self.controller.ub += _ub

        self.controller.final_lb = self.controller.lb.reshape(_lb.shape).clone()
        self.controller.final_ub = self.controller.ub.reshape(_lb.shape).clone()

        l, u = self.controller.concretize(self.controller.final_lw, self.controller.final_uw)
        l = l.reshape(_lb.shape)
        u = u.reshape(_lb.shape)

        self.controller.lb += l
        self.controller.ub += u

        if self.empty_cache:
            torch.cuda.empty_cache()


class EdgeSoftmax(EdgeComplex):
    def __init__(self, args, controller, par, num_attention_heads):
        super(EdgeSoftmax, self).__init__(args, controller)

        self.length = par.length
        self.num_attention_heads = num_attention_heads

        self.par = par       

        self.exp = self.par.next(EdgeExp(self.args, self.controller, self.par))

        if self.use_forward:
            raise NotImplementedError
        ones = torch.ones(1, self.length, self.length).cuda()
        zeros = torch.zeros(num_attention_heads, self.length, self.length).cuda()
        w = torch.cat([
            ones, 
            torch.cat([zeros, ones], dim=0).repeat(num_attention_heads - 1, 1, 1)
        ], dim=0)\
        .reshape(num_attention_heads, num_attention_heads, self.length, self.length)\
        .permute(0, 2, 1, 3)\
        .reshape(num_attention_heads * self.length, num_attention_heads * self.length)
        self.sum = self.exp.next(EdgeDense(self.args, self.controller, self.exp, w=w, b=0.))
        self.res = self.exp.next(EdgeDivide(self.args, self.controller, self.exp, self.sum))

class EdgePooling(Edge):
    def __init__(self, args, controller, par):
        super(EdgePooling, self).__init__(args, controller)

        self.par = par
        self.length = par.length

    def backward(self, lw, uw):
        if self.use_forward:
            dim = 0
            zeros = torch.zeros(self.length - 1, lw.shape[1], lw.shape[2]).cuda()
        else:
            dim = 2
            zeros = torch.zeros(lw.shape[0], lw.shape[1], self.length - 1, lw.shape[3]).cuda()
        lw = torch.cat([lw, zeros], dim=dim)
        uw = torch.cat([uw, zeros], dim=dim)
        self.par.backward_buffer(lw, uw)            

class EdgeDense(Edge):
    def __init__(self, args, controller, par, w=0., b=0., dense=None):
        super(EdgeDense, self).__init__(args, controller)
        self.par = par
        if dense is not None:
            w = dense.weight
            b = dense.bias
        self.w = w
        if type(b) == float:
            self.b = torch.ones(w.shape[-1]).cuda() * b
        else:
            self.b = b

    def backward(self, lw, uw):
        _lw = torch.matmul(lw, self.w)
        _uw = torch.matmul(uw, self.w)     

        if self.use_forward:
            self.controller.lb += torch.sum(lw * self.b, dim=-1)
            self.controller.ub += torch.sum(uw * self.b, dim=-1)
        else:
            self.controller.lb += torch.sum(lw * self.b.reshape(1, 1, 1, -1), dim=[-1, -2]) 
            self.controller.ub += torch.sum(uw * self.b.reshape(1, 1, 1, -1), dim=[-1, -2]) 

        return self.par.backward_buffer(_lw, _uw)
    
class EdgeActivation(Edge):
    def __init__(self, args, controller, par, par2=None):
        super(EdgeActivation, self).__init__(args, controller)
        self.par = par
        self.par2 = par2
        self.init_linear()

    def init_linear(self):
        self.mask_pos = torch.gt(self.par.l, 0).to(torch.float)
        self.mask_neg = torch.lt(self.par.u, 0).to(torch.float)
        self.mask_both = 1 - self.mask_pos - self.mask_neg 

        # element-wise for now
        shape = (self.par.length, self.par.dim)
        self.lw = torch.zeros(shape).cuda()
        self.lb = torch.zeros(shape).cuda()
        self.uw = torch.zeros(shape).cuda()
        self.ub = torch.zeros(shape).cuda()

        if self.par2 is not None:
            shape = (self.par2.length, self.par2.dim)
            self.lw2 = torch.zeros(shape).cuda()
            self.lb2 = torch.zeros(shape).cuda()
            self.uw2 = torch.zeros(shape).cuda()
            self.ub2 = torch.zeros(shape).cuda()

    def add_linear(self, mask, type, k, x0, y0, second=False):
        if mask is None:
            mask = 1
        if type == "lower":
            if second:
                w_out, b_out = self.lw2, self.lb2
            else:
                w_out, b_out = self.lw, self.lb
        else:
            if second:
                w_out, b_out = self.uw2, self.ub2
            else:
                w_out, b_out = self.uw, self.ub  
        w_out += mask * k
        b_out += mask * (-x0 * k + y0)

    def backward_par(self, lw, uw, self_lw, self_lb, self_uw, self_ub, par):
        mask_l = torch.gt(lw, 0.).to(torch.float)
        mask_u = torch.gt(uw, 0.).to(torch.float)
        if self.use_forward:
            _lw = mask_l * lw * self_lw.unsqueeze(1) +\
                (1 - mask_l) * lw * self_uw.unsqueeze(1)
            _lb = torch.sum(mask_l * lw * self_lb.unsqueeze(1) +\
                (1 - mask_l) * lw * self_ub.unsqueeze(1), dim=-1)
            _uw = mask_u * uw * self_uw.unsqueeze(1) +\
                (1 - mask_u) * uw * self_lw.unsqueeze(1)
            _ub = torch.sum(mask_u * uw * self_ub.unsqueeze(1) +\
                (1 - mask_u) * uw * self_lb.unsqueeze(1), dim=-1)
        else:
            _lw = mask_l * lw * self_lw.unsqueeze(0).unsqueeze(0) +\
                (1 - mask_l) * lw * self_uw.unsqueeze(0).unsqueeze(0)
            _lb = torch.sum(mask_l * lw * self_lb.unsqueeze(0).unsqueeze(0) + \
                (1 - mask_l) * lw * self_ub.unsqueeze(0).unsqueeze(0), dim=[-1, -2])
            _uw = mask_u * uw * self_uw.unsqueeze(0).unsqueeze(0) +\
                (1 - mask_u) * uw * self_lw.unsqueeze(0).unsqueeze(0)
            _ub = torch.sum(mask_u * uw * self_ub.unsqueeze(0).unsqueeze(0) + \
                (1 - mask_u) * uw * self_lb.unsqueeze(0).unsqueeze(0), dim=[-1, -2])

        self.controller.lb += _lb
        self.controller.ub += _ub
        
        par.backward_buffer(_lw, _uw)    

    def backward(self, lw, uw):  
        self.backward_par(lw, uw, self.lw, self.lb, self.uw, self.ub, self.par)
        if self.par2 is not None:
            self.backward_par(lw, uw, self.lw2, self.lb2, self.uw2, self.ub2, self.par2)        

# cannot be combined with the forward framework

class EdgeDotProduct(Edge):
    def __init__(self, args, controller, a, b, num_attention_heads):
        super(EdgeDotProduct, self).__init__(args, controller)

        assert(args.method != "baf")

        self.a = a
        self.b = b
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = a.l.shape[-1] // num_attention_heads

        l_a = a.l.reshape(a.length, num_attention_heads, self.attention_head_size)\
            .repeat(1, 1, b.length).reshape(-1)
        u_a = a.u.reshape(a.length, num_attention_heads, self.attention_head_size)\
            .repeat(1, 1, b.length).reshape(-1)

        l_b = b.l.reshape(b.length, num_attention_heads, self.attention_head_size)\
            .transpose(0, 1).repeat(a.length, 1, 1).reshape(-1)
        u_b = b.u.reshape(b.length, num_attention_heads, self.attention_head_size)\
            .transpose(0, 1).repeat(a.length, 1, 1).reshape(-1)

        self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u = \
            get_bounds_xy(l_a, u_a, l_b, u_b)

        # batch_size, length, h, h_size*length
        self.alpha_l = self.alpha_l\
            .reshape(a.length, num_attention_heads, 
                b.length, self.attention_head_size)
        self.alpha_u = self.alpha_u\
            .reshape(a.length, num_attention_heads, 
                b.length, self.attention_head_size)

        self.beta_l = self.beta_l\
            .reshape(a.length, num_attention_heads,
                b.length, self.attention_head_size)#.transpose(0, 2)
        self.beta_u = self.beta_u\
            .reshape(a.length, num_attention_heads,
                b.length, self.attention_head_size)#.transpose(0, 2)

        # batch_size, length, h, length*h_size
        self.gamma_l = self.gamma_l\
            .reshape(a.length, num_attention_heads, b.length, self.attention_head_size)\
            .sum(dim=-1)
        self.gamma_u = self.gamma_u\
            .reshape(a.length, num_attention_heads, b.length, self.attention_head_size)\
            .sum(dim=-1)

    def backward(self, lw, uw):    
        # [length, 1, h, length, r]
        alpha_l = self.alpha_l.unsqueeze(0).unsqueeze(0)
        alpha_u = self.alpha_u.unsqueeze(0).unsqueeze(0)
        beta_l = self.beta_l.unsqueeze(0).unsqueeze(0)
        beta_u = self.beta_u.unsqueeze(0).unsqueeze(0)
        gamma_l = self.gamma_l.reshape(1, 1, self.a.length, -1)
        gamma_u = self.gamma_u.reshape(1, 1, self.a.length, -1)

        mask = torch.gt(lw, 0.).to(torch.float)
        _lb = torch.sum(lw * (
            mask * gamma_l + \
            (1 - mask) * gamma_u)
        , dim=[-1, -2])
        del(mask)

        mask = torch.gt(uw, 0.).to(torch.float)
        _ub = torch.sum(uw * (
            mask * gamma_u + \
            (1 - mask) * gamma_l)
        , dim=[-1, -2])      
        del(mask)
        del(gamma_l)        
        del(gamma_u)  

        if self.empty_cache:
            torch.cuda.empty_cache()              

        self.controller.lb += _lb
        self.controller.ub += _ub          

        # [length, h * length (o), h, length, 1]
        _lw = lw\
            .reshape(lw.shape[0], lw.shape[1], lw.shape[2], self.num_attention_heads, self.b.length, 1)
        mask = torch.gt(_lw, 0.).to(torch.float)
        _lw = torch.sum(mask * _lw * alpha_l + (1 - mask) * _lw * alpha_u, dim=-2)\
            .reshape(lw.shape[0], lw.shape[1], lw.shape[2], -1)

        # [length, h * length (o), h, length, 1]
        _uw = uw\
            .reshape(uw.shape[0], uw.shape[1], uw.shape[2], self.num_attention_heads, self.b.length, 1)
        mask = torch.gt(_uw, 0.).to(torch.float)
        _uw = torch.sum(mask * _uw * alpha_u + (1 - mask) * _uw * alpha_l, dim=-2)\
            .reshape(uw.shape[0], uw.shape[1], uw.shape[2], -1)            

        del(mask)
        if self.empty_cache:
            torch.cuda.empty_cache()

        self.a.backward_buffer(_lw, _uw)    

        del(_lw)
        del(_uw)
        if self.empty_cache:
            torch.cuda.empty_cache()

        _lw2 = lw\
            .reshape(lw.shape[0], lw.shape[1], lw.shape[2], self.num_attention_heads, self.b.length, 1)
        mask = torch.gt(_lw2, 0.).to(torch.float)
        _lw2 = torch.sum(mask * _lw2 * beta_l + (1 - mask) * _lw2 * beta_u, dim=-4)\
            .transpose(2, 3)
        _lw2 = _lw2.reshape(_lw2.shape[0], _lw2.shape[1], _lw2.shape[2], -1)

        _uw2 = uw\
            .reshape(uw.shape[0], uw.shape[1], uw.shape[2], self.num_attention_heads, self.b.length, 1)
        mask = torch.gt(_uw2, 0.).to(torch.float)
        _uw2 = torch.sum(mask * _uw2 * beta_u + (1 - mask) * _uw2 * beta_l, dim=-4)\
            .transpose(2, 3)
        _uw2 = _uw2.reshape(_uw2.shape[0], _uw2.shape[1], _uw2.shape[2], -1)            

        del(mask)
        if self.empty_cache:
            torch.cuda.empty_cache()

        self.b.backward_buffer(_lw2, _uw2)    

class EdgeTranspose(Edge):
    def __init__(self, args, controller, par, num_attention_heads):
        super(EdgeTranspose, self).__init__(args, controller)

        assert(args.method != "baf")

        self.par = par
        self.num_attention_heads = num_attention_heads

    def transpose(self, w):
        w = w.reshape(
            w.shape[0], w.shape[1], w.shape[2], 
            self.num_attention_heads, -1
        ).transpose(2, 4)
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], -1)    
        return w

    def backward(self, lw, uw): 
        lw = self.transpose(lw)
        uw = self.transpose(uw)

        self.par.backward_buffer(lw, uw)

class EdgeMultiply(EdgeActivation):
    def __init__(self, args, controller, a, b):
        super(EdgeMultiply, self).__init__(args, controller, par=a, par2=b)

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = get_bounds_xy(
            a.l.reshape(-1),
            a.u.reshape(-1),
            b.l.reshape(-1),
            b.u.reshape(-1)
        )
        alpha_l = alpha_l.reshape(a.l.shape)
        beta_l = beta_l.reshape(a.l.shape)
        gamma_l = gamma_l.reshape(a.l.shape)
        alpha_u = alpha_u.reshape(a.l.shape)
        beta_u = beta_u.reshape(a.l.shape)
        gamma_u = gamma_u.reshape(a.l.shape)

        self.add_linear(mask=None, type="lower", k=alpha_l, x0=0, y0=gamma_l)
        self.add_linear(mask=None, type="lower", k=beta_l, x0=0, y0=0, second=True)
        self.add_linear(mask=None, type="upper", k=alpha_u, x0=0, y0=gamma_u)
        self.add_linear(mask=None, type="upper", k=beta_u, x0=0, y0=0, second=True)

class EdgeSqr(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeSqr, self).__init__(args, controller, par)

        k = self.par.u + self.par.l
        self.add_linear(mask=None, type="upper", k=k, x0=self.par.l, y0=self.par.l.pow(2))
        m = torch.max((self.par.l + self.par.u) / 2, 2 * self.par.u)
        self.add_linear(mask=self.mask_neg, type="lower", k=2*m, x0=m, y0=m.pow(2))
        m = torch.min((self.par.l + self.par.u) / 2, 2 * self.par.l)
        self.add_linear(mask=self.mask_pos, type="lower", k=2*m, x0=m, y0=m.pow(2))

class EdgeSqrt(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeSqrt, self).__init__(args, controller, par)

        assert(torch.min(self.par.l) >= 0)
        k = (torch.sqrt(self.par.u) - torch.sqrt(self.par.l)) / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=None, type="lower", k=k, x0=self.par.l, y0=torch.sqrt(self.par.l) + epsilon)
        m = (self.par.l + self.par.u) / 2
        k = 0.5 / torch.sqrt(m)
        self.add_linear(mask=None, type="upper", k=k, x0=m, y0=torch.sqrt(m) + epsilon)

class EdgeReciprocal(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeReciprocal, self).__init__(args, controller, par)

        assert(torch.min(self.par.l))
        m = (self.par.l + self.par.u) / 2
        kl = -1 / m.pow(2)
        self.add_linear(mask=None, type="lower", k=kl, x0=m, y0=1. / m)
        ku = -1. / (self.par.l * self.par.u)
        self.add_linear(mask=None, type="upper", k=ku, x0=self.par.l, y0=1. / self.par.l)

class EdgeLinear(EdgeActivation):
    def __init__(self, args, controller, par, w, b): 
        super(EdgeLinear, self).__init__(args, controller, par)

        self.add_linear(mask=None, type="lower", k=w, x0=0., y0=b)
        self.add_linear(mask=None, type="upper", k=w, x0=0., y0=b)

class EdgeExp(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeExp, self).__init__(args, controller, par)

        m = torch.min((self.par.l + self.par.u) / 2, self.par.l + 0.99)
        k = torch.exp(m)
        self.add_linear(mask=None, type="lower", k=k, x0=m, y0=torch.exp(m))
        k = (torch.exp(self.par.u) - torch.exp(self.par.l)) / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=None, type="upper", k=k, x0=self.par.l, y0=torch.exp(self.par.l))

class EdgeDivide(EdgeComplex):
    def __init__(self, args, controller, a, b):
        super(EdgeDivide, self).__init__(args, controller)
        b_reciprocal = b.next(EdgeReciprocal(args, controller, b))
        self.res = a.next(EdgeMultiply(args, controller, a, b_reciprocal))

class EdgeRelu(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeRelu, self).__init__(args, controller, par)

        self.add_linear(mask=self.mask_neg, type="lower", k=0., x0=0, y0=0)
        self.add_linear(mask=self.mask_neg, type="upper", k=0., x0=0, y0=0)        
        self.add_linear(mask=self.mask_pos, type="lower", k=1., x0=0, y0=0)
        self.add_linear(mask=self.mask_pos, type="upper", k=1., x0=0, y0=0)

        k = self.par.u / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=self.mask_both, type="upper", k=k, x0=self.par.l, y0=0)

        k = torch.gt(torch.abs(self.par.u), torch.abs(self.par.l)).to(torch.float)
        self.add_linear(mask=self.mask_both, type="lower", k=k, x0=0, y0=0)

class EdgeTanh(EdgeActivation):
    def __init__(self, args, controller, par):
        super(EdgeTanh, self).__init__(args, controller, par)

        def dtanh(x):
            return 1. / torch.cosh(x).pow(2)
            
        # lower bound for negative
        m = (self.par.l + self.par.u) / 2
        k = dtanh(m)
        self.add_linear(mask=self.mask_neg, type="lower", k=k, x0=m, y0=torch.tanh(m))
        # upper bound for positive
        self.add_linear(mask=self.mask_pos, type="upper", k=k, x0=m, y0=torch.tanh(m))

        # upper bound for negative
        k = (torch.tanh(self.par.u) - torch.tanh(self.par.l)) / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=self.mask_neg, type="upper", k=k, x0=self.par.l, y0=torch.tanh(self.par.l))
        # lower bound for positive
        self.add_linear(mask=self.mask_pos, type="lower", k=k, x0=self.par.l, y0=torch.tanh(self.par.l))

        # bounds for both
        max_iter = 10

        # lower bound for both
        diff = lambda d: (torch.tanh(self.par.u) - torch.tanh(d)) / (self.par.u - d + epsilon) - dtanh(d)
        d = self.par.l / 2
        _l = self.par.l
        _u = torch.zeros(self.par.l.shape).cuda()
        for t in range(max_iter):
            v = diff(d)
            mask_p = torch.gt(v, 0).to(torch.float)
            _l = d * mask_p + _l * (1 - mask_p)
            _u = d * (1 - mask_p) + _u * mask_p
            d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
        k = (torch.tanh(d) - torch.tanh(self.par.u)) / (d - self.par.u + epsilon)
        self.add_linear(mask=self.mask_both, type="lower", k=k, x0=d, y0=torch.tanh(d))

        # upper bound for both
        diff = lambda d: (torch.tanh(d) - torch.tanh(self.par.l))/ (d - self.par.l + epsilon) - dtanh(d)
        d = self.par.u / 2
        _l = torch.zeros(self.par.l.shape).cuda()
        _u = self.par.u
        for t in range(max_iter):
            v = diff(d)
            mask_p = torch.gt(v, 0).to(torch.float)
            _l = d * (1 - mask_p) + _l * mask_p
            _u = d * mask_p + _u * (1 - mask_p)
            d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
        k = (torch.tanh(d) - torch.tanh(self.par.l)) / (d - self.par.l + epsilon)
        self.add_linear(mask=self.mask_both, type="upper", k=k, x0=d, y0=torch.tanh(d))        
