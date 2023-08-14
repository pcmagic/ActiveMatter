# coding=utf-8
"""
20210810
Zhang Ji

RNN class, training the network.
"""
# import abc
# import pickle
# import logging
# from tqdm import tqdm
import numpy as np
# from tqdm.notebook import tqdm as tqdm_notebook
# from petsc4py import PETSc
# from datetime import datetime
# import time
# import shutil
import os
# import h5py
# from matplotlib import pyplot as plt
# pytorch
import torch
from torch import nn
import random


# from act_src import baseClass
# from act_src import particleClass
# from act_src import interactionClass
# from act_src import relationClass
# from act_codeStore.support_class import *
# from act_codeStore import support_fun as spf


class rnnbase(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self._act_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._input_size = input_size
        self._hidden_size = hidden_size

    @property
    def act_device(self):
        return self._act_device

    @property
    def input_size(self):
        return self._input_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @staticmethod
    def set_seed(seed: int = 1) -> None:
        if seed is None:
            print("Random seed set as None")
            return
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")


class chimeraRNN(rnnbase):
    def __init__(self, input_size, hidden_size, output_size=None, num_layers=1, nonlinearity='tanh', bias=True,
                 batch_first=False,
                 *args, **kwargs):
        super().__init__(input_size, hidden_size)
        self._output_size = input_size if output_size is None else output_size
        self._num_layers = num_layers
        self._nonlinearity = nonlinearity
        self._bias = bias
        self._batch_first = batch_first
        self._rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, bias=self.bias,
                           batch_first=self.batch_first, nonlinearity=self.nonlinearity).to(self.act_device)
        self._fc = nn.Linear(self.hidden_size, self.output_size).to(self.act_device)

    @property
    def output_size(self):
        return self._output_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def nonlinearity(self):
        return self._nonlinearity

    @property
    def bias(self):
        return self._bias

    @property
    def batch_first(self):
        return self._batch_first

    @property
    def rnn(self):
        return self._rnn

    @property
    def fc(self):
        return self._fc

    def forward_step(self, x, h0):
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        return out, hn

    def forward(self, x):
        # self.rnn.flatten_parameters()
        h0 = self.init_hidden(x)
        return self.forward_step(x, h0)

    def init_hidden(self, x):
        if len(x.shape) == 2:
            hidden = torch.zeros(self.num_layers, self.hidden_size).to(self.act_device)
        else:
            batch_size = x.shape[0] if self.batch_first else x.shape[1]
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.act_device)
        return hidden

    # def sample_step(self, start, out_len, hn):
    #     self.eval()  # eval mode
    #     out = start
    #     out_list = [out, ]
    #     for _ in range(out_len):
    #         out, hn = self.forward(out, hn)
    #         out_list.append(out)
    #     return torch.vstack(out_list)

    def sample(self, start, out_len):
        self.eval()  # eval mode
        out = start
        out_list = [out, ]
        for _ in range(out_len):
            out, hn = self.forward(out)
            out_list.append(out)
        return torch.transpose(torch.vstack(out_list), 0, 1)


class chimeraLSTM(chimeraRNN):
    def __init__(self, input_size, hidden_size, output_size=None, num_layers=1, bias=True,
                 batch_first=False,
                 *args, **kwargs):
        rnnbase.__init__(self, input_size, hidden_size)
        self._output_size = input_size if output_size is None else output_size
        self._num_layers = num_layers
        self._bias = bias
        self._batch_first = batch_first
        self._rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bias=self.bias,
                             batch_first=self.batch_first).to(self.act_device)
        self._fc = nn.Linear(self.hidden_size, self.output_size).to(self.act_device)

    def init_c0(self, x):
        if len(x.shape) == 2:
            c0 = torch.zeros(self.num_layers, self.hidden_size).to(self.act_device)
        else:
            batch_size = x.shape[0] if self.batch_first else x.shape[1]
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.act_device)
        return c0

    def forward(self, x):
        # self.rnn.flatten_parameters()
        h0 = self.init_hidden(x)
        c0 = self.init_c0(x)
        out, hn = self.rnn(x, (h0, c0))
        out = self.fc(out)
        return out, hn


class chimeraFORCE(rnnbase):
    def __init__(self, input_size, hidden_size=1500, dens_w0=0.1, eval_dt=1,
                 rate_w0=1.5, rate_w1=1, lmd=1):
        super().__init__(input_size, hidden_size)
        self._eval_dt = eval_dt
        #
        tA = torch.normal(0, 1, (hidden_size, hidden_size), device=self.act_device)
        tB = torch.rand(hidden_size, hidden_size, device=self.act_device) < dens_w0
        self._omega = rate_w0 * torch.multiply(tA, tB) / np.sqrt(hidden_size * dens_w0
                                                                 )  # Initial weight matrix
        self._eta = (torch.rand(hidden_size, input_size, device=self.act_device) * 2 - 1
                     ) * rate_w1  # random eta variable
        self._h = torch.normal(0., 1., size=(hidden_size,), device=self.act_device)  # initial conditions
        self._out = torch.zeros(input_size, device=self.act_device)
        self._d = torch.zeros(hidden_size, input_size, device=self.act_device, requires_grad=False)
        self._Pinv = torch.eye(hidden_size, device=self.act_device) / lmd
        # self._Tanh = nn.Tanh().to(self.act_device)
        # self._linear = nn.Linear(hidden_size, input_size, bias=False, device=self.act_device).to(self.act_device)
        self._Tanh = None
        self._linear = None
        self._r = torch.tanh(self.h)

    @property
    def act_device(self):
        return self._act_device

    @property
    def input_size(self):
        return self._input_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def eval_dt(self):
        return self._eval_dt

    @property
    def omega(self):
        return self._omega

    @property
    def eta(self):
        return self._eta

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, h):
        self._h = h

    @property
    def out(self):
        return self._out

    @out.setter
    def out(self, out):
        self._out = out

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, d):
        self._d = d

    @property
    def Pinv(self):
        return self._Pinv

    @Pinv.setter
    def Pinv(self, Pinv):
        self._Pinv = Pinv

    # @property
    # def Tanh(self):
    #     return self._Tanh
    #
    # @property
    # def linear(self):
    #     return self._linear

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self._r = r

    def forward(self):
        self.r = torch.tanh(self.h)
        dh = -self.h + torch.inner(self.omega, self.r) + torch.inner(self.eta, self.out)
        self.h = self.h + self.eval_dt * dh
        self.out = torch.einsum('ji, j', self.d, self.r)
        return self.out

    def step(self, target):
        err = self.out - target
        q = torch.inner(self.Pinv, self.r)
        self.Pinv = self.Pinv - (torch.outer(q, q)) / (1 + torch.inner(self.r, q))
        self.d = self.d - torch.outer(torch.inner(self.Pinv, self.r), err)
        return err

    def sample(self, out_len):
        # self.eval()  # eval mode
        out_list = torch.zeros((out_len, self.input_size), device=self.act_device)
        th = torch.clone(self.h)
        tout = torch.clone(self.out)
        for i0 in range(out_len):
            tr = torch.tanh(th)
            dh = -th + torch.inner(self.omega, tr) + torch.inner(self.eta, tout)
            th = th + self.eval_dt * dh
            tout = torch.einsum('ji, j', self.d, tr)
            out_list[i0] = tout
        return out_list
