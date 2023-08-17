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
import pandas as pd
import re
import itertools
from scanf import scanf
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, PowerNorm, Normalize
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.optimize import leastsq, curve_fit
from scipy import interpolate, integrate, optimize, sparse
from scipy.interpolate import interp1d, interp2d
from IPython.display import display, HTML, Math
from scipy import interpolate
from tqdm.notebook import tqdm as tqdm_notebook

# from act_act_src import baseClass
from act_src import particleClass
from act_src import interactionClass
from act_src import problemClass
from act_src import relationClass
from act_codeStore.support_class import *
from act_codeStore import support_fun as spf
from act_codeStore import support_fun_calculate as spc

PWD = os.getcwd()
np.set_printoptions(linewidth=110, precision=5)

params = {'animation.html': 'html5',
          'font.family':    'sans-serif',
          'font.size':      15, }
preamble = r' '
preamble = preamble + '\\usepackage{bm} '
preamble = preamble + '\\usepackage{amsmath} '
preamble = preamble + '\\usepackage{amssymb} '
preamble = preamble + '\\usepackage{mathrsfs} '
preamble = preamble + '\\DeclareMathOperator{\\Tr}{Tr} '
params['text.latex.preamble'] = preamble
params['text.usetex'] = True
plt.rcParams.update(params)

from act_codeStore.support_class import *
from act_codeStore import support_fun as spf
from act_codeStore import support_fun_calculate as spc
from act_codeStore import support_fun_show as sps
from collectiveFish.do_calculate import calculate_fun_dict, prbHandle_dict, rltHandle_dict, ptcHandle_dict

update_fun, update_order, eval_dt = '1fe', (0, 0), 0.1
nptc, max_t, calculate_fun = 1, eval_dt * 100, 'do_ackermann'
l_steer, w_steer = 3, 0

problem_kwargs = {
    'ini_t':           np.float64(0),
    'max_t':           eval_dt * 1e3,
    'update_fun':      '1fe',
    'update_order':    (0, 0),
    'eval_dt':         eval_dt,
    'calculate_fun':   calculate_fun_dict[calculate_fun],
    'prbHandle':       prbHandle_dict[calculate_fun],
    'rltHandle':       rltHandle_dict[calculate_fun],
    'ptcHandle':       ptcHandle_dict[calculate_fun],
    'fileHandle':      'try_Behavior2DProblem',
    'save_every':      np.int64(1),
    'nptc':            np.int64(nptc),
    'overlap_epsilon': np.float64(1e-100),
    'un':              np.float64(0.02) / 0.024,
    'ln':              np.float64(-1),
    'Xlim':            np.float64(6),
    'attract':         np.float64(0.41),
    'align':           np.float64(2.7),
    'l_steer':         np.float64(l_steer),
    'w_steer':         np.float64(w_steer),
    'viewRange':       np.float64(1),
    'seed':            1,
    'tqdm_fun':        tqdm_notebook,
}

doPrb1 = problem_kwargs['calculate_fun'](**problem_kwargs)
prb1 = doPrb1.ini_calculate()
obji = prb1.obj_list[0]
obji.X = np.array((10, 10))
obji.phi = np.float64(0)
obji.phi_steer = np.float64(0)
obji.goal = np.array((10, 20, 3.1415926))
obji.goal_threshold = 0.2
obji.v_max = np.float64(5)
obji.w_max = np.float64(2)
obji.phi_steer_limit = np.pi / 4
prb1.update_self(t0=problem_kwargs['ini_t'], t1=max_t, eval_dt=eval_dt, )

print(np.vstack((np.linalg.norm(obji.U_hist, axis=1), obji.W_steer_hist)).T[1:])
print(np.vstack((obji.X_hist.T, obji.phi_hist, obji.phi_steer_hist)).T[1:])

# ################################################################################3
#
# figsize=np.array((9, 9))*0.5
# dpi = 500 if 'inline' in matplotlib.get_backend() else 100
# plt_tmin = -1
# nmarker = 0.1
#
# fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
# fig.patch.set_facecolor('white')
# cmap = plt.get_cmap('viridis')
# for obji in prb1.obj_list:
#     color = cmap(obji.index / prb1.n_obj)
#     tidx = prb1.t_hist >= plt_tmin
#     X_hist = obji.X_hist[tidx]
#     axi.scatter(X_hist[0, 0], X_hist[0, 1], color=color, marker='s')
#     axi.plot(X_hist[:, 0], X_hist[:, 1], '.-', color=color, markevery=nmarker)
# sps.set_axes_equal(axi)
# plt.show()