import os
from datetime import datetime
from time import time
# import dill
import pickle
import glob
import importlib
import numpy as np
import scipy as sp
import scipy.misc
# import pandas as pd
import re
import itertools
import natsort
import shutil
from scanf import scanf
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm, Normalize
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import animation
from scipy.optimize import leastsq, curve_fit
from scipy import interpolate, integrate, optimize, sparse
from scipy.interpolate import interp1d, interp2d
from IPython.display import display, HTML, Math
from scipy import interpolate
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.contrib import tzip

# from act_act_src import baseClass
# from act_src import particleClass
# from act_src import interactionClass
# from act_src import problemClass
# from act_src import relationClass
from act_src import rnnClass
# from act_codeStore.support_class import *
# from act_codeStore import support_fun as spf
# from act_codeStore import support_fun_calculate as spc
# from act_codeStore import support_fun_show as sps

# pytorch
import torch
from torch import nn
import random

PWD = os.getcwd()
np.set_printoptions(linewidth=110, precision=5)

params = {
    'animation.html':        'html5',
    'animation.embed_limit': 2 ** 128,
    'font.family':           'sans-serif',
    'font.size':             15,
}
preamble = r' '
preamble = preamble + '\\usepackage{bm} '
preamble = preamble + '\\usepackage{amsmath} '
preamble = preamble + '\\usepackage{amssymb} '
preamble = preamble + '\\usepackage{mathrsfs} '
preamble = preamble + '\\DeclareMathOperator{\\Tr}{Tr} '
params['text.latex.preamble'] = preamble
params = {}
params['text.usetex'] = True
plt.rcParams.update(params)

# import emd
# from scipy import signal, stats
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy.optimize import leastsq, curve_fit


data_name = '/home/zhangji/DownCode/RNN_CHIMERAS/theta_A_0.1_N_3_beta_0.025_ic_7_TC_1light.txt'
data2_name = '/home/zhangji/DownCode/RNN_CHIMERAS/phi_A_0.1_N_3_beta_0.025_ic_7_TC_1light.txt'
data = np.loadtxt(data_name)
data2 = np.loadtxt(data2_name)
train_ign = 29999
num_layers, nonlinearity, bias = 1, 'tanh', False
seed0 = 3
batch_size = 1
hidden_size = 1500
num_threads = 6

theta = torch.Tensor(data[train_ign:, 1:])
phi = torch.Tensor(data2[train_ign:, 1:])
Lin, input_size = theta.shape[0], theta.shape[1] * 4
# x_inp.shape -> torch.Size([1, 70000, 12]) (N, L, Hin)
x_inp = torch.cat((torch.cos(theta), torch.cos(phi), torch.sin(theta), torch.sin(phi)),
                  dim=-1).view(batch_size, Lin, input_size)
model = rnnClass.chimeraFORCE(input_size=input_size, hidden_size=hidden_size, eval_dt=0.1,
                              dens_w0=0.1, rate_w0=1.5, rate_w1=1, lmd=1)
torch.cuda.empty_cache()
torch.set_num_threads(1 if model.act_device.type == 'cuda' else num_threads)
model.set_seed(seed0)
test_seq = x_inp.to(model.act_device)

# Training
n_epochs = 1
for i0 in np.arange(n_epochs):
    t1 = datetime.now()
    for ttest in tqdm_notebook(test_seq[0, :10]):
        model.forward()
        model.step(ttest)
    #         torch.cuda.empty_cache()
    t2 = datetime.now()
    print('Epoch: {}/{}.............'.format(i0 + 1, n_epochs), end=' ')
    #     print("Loss: {:03.4f}".format(loss.item()), end=' ')
    print('usage time: %s' % str(t2 - t1))


print(model.forward())
print(model.r)
print(model.h)
print(model.out)
pred_seq = model.sample(3)
print()
print(model.r)
print(model.h)
print(model.out)
print(model.forward())