import sys
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc
import os
from datetime import datetime
# from time import time
# # import dill
import pickle
# import glob
# import importlib
import numpy as np
# import scipy as sp
# import scipy.misc
# import pandas as pd
# import re
# import itertools
# import natsort
# import shutil
from scanf import scanf
import matplotlib
from matplotlib import pyplot as plt
# import matplotlib.ticker as mtick
from matplotlib import colors as mcolors
# from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm, Normalize
from matplotlib.colors import Normalize
# from mpl_toolkits.mplot3d import axes3d, Axes3D
# from matplotlib import animation
# from scipy.optimize import leastsq, curve_fit
# from scipy import interpolate, integrate, optimize, sparse
# from scipy.interpolate import interp1d, interp2d
# from IPython.display import display, HTML, Math
# from scipy import interpolate
# from tqdm.notebook import tqdm as tqdm_notebook
# from tqdm.contrib import tzip
#
# # from act_act_src import baseClass
# from act_src import particleClass
# from act_src import interactionClass
# from act_src import problemClass
# from act_src import relationClass
# from act_codeStore.support_class import *
from act_codeStore import support_fun as spf
# from act_codeStore import support_fun_calculate as spc
from act_codeStore import support_fun_show as sps
from act_src import rnnClass
# pytorch
import torch
from torch import nn

# Masoliver2022 case
data_name = '/home/zhangji/DownCode/RNN_CHIMERAS/theta_A_0.1_N_3_beta_0.025_ic_7_TC_1light.txt'
data2_name = '/home/zhangji/DownCode/RNN_CHIMERAS/phi_A_0.1_N_3_beta_0.025_ic_7_TC_1light.txt'
data = np.loadtxt(data_name)
data2 = np.loadtxt(data2_name)
time = data[:, 0]
dt = time[4] - time[3]
ti_file = 3000
nti_file = int(ti_file / dt)
T = 5e3  # Total training time
nt = int(T / dt)
imin = int(600 / dt)  # start RLS
imax = int(1.5 * T / dt)  # stop RLS
# 
eval_dt = dt
train_ign, train_ini, train_fns, train_max = nti_file, imin, imax, nt
hidden_size, dens_w0, rate_w0, rate_w1, lmd = 1500, 0.1, 1.5, 1, 1
num_layers, nonlinearity, lr = 1, 'tanh', 0.00001
n_epochs = 3
figsize, dpi = np.array((16, 9)) * 0.5, 200
seed0 = 3
#
theta = np.transpose(data[train_ign:, 1:])
phi = np.transpose(data2[train_ign:, 1:])
input_size = 4 * theta.shape[0]
output_size = input_size
# x_inp.shape -> torch.Size([70001, 12])
x_inp = torch.from_numpy(np.vstack((np.cos(theta), np.cos(phi),
                                    np.sin(theta), np.sin(phi)))
                         ).type(torch.float32).T

batch_size = 1000
inpt_loader = torch.utils.data.DataLoader(x_inp[:-1], batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(x_inp[1:], batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
model = rnnClass.chimeraRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                            num_layers=num_layers, nonlinearity=nonlinearity, )
model.set_seed(seed0)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# torch.set_num_interop_threads(8)
torch.set_num_threads(6)
# out_seq, hidden = model(ini_seq)
# print(out_seq.shape)

# Training 
for i0 in np.arange(n_epochs):
    t1 = datetime.now()
    for inpt_seq, test_seq in zip(inpt_loader, test_loader):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        oupt_seq, hidden = model(inpt_seq)
        loss = criterion(oupt_seq, test_seq)
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly
        t2 = datetime.now()
        print('Epoch: {}/{}.............'.format(i0 + 1, n_epochs), end=' ')
        print("Loss: {:03.4f}".format(loss.item()), end=' ')
        print('usage time: %s' % str(t2 - t1))

#     t1 = out_seq.detach().numpy()
#     fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
#     fig.patch.set_facecolor('white')
#     axi.plot(t1)
