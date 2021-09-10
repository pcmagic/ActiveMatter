# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20210906

@author: Zhang Ji
"""

import matplotlib
import subprocess
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import interpolate, integrate, spatial, signal
from scipy.optimize import leastsq, curve_fit
from matplotlib import animation
from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1 import colorbar
from matplotlib import colorbar
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
import matplotlib.ticker as mtick
from matplotlib import colors as mcolors
import importlib
import inspect
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import glob
import natsort
from time import time
import pickle
import re
import shutil
import multiprocessing
import warnings

from act_codeStore import support_fun as spf
from act_codeStore.support_class import *
# from act_act_src import baseClass
from act_src import particleClass
from act_src import interactionClass
from act_src import problemClass
from act_src import relationClass

PWD = os.getcwd()
np.set_printoptions(linewidth=110, precision=5)

params = {'animation.html': 'html5',
          'font.family': 'sans-serif',
          'font.size': 15, }
preamble = r' '
preamble = preamble + '\\usepackage{bm} '
preamble = preamble + '\\usepackage{amsmath} '
preamble = preamble + '\\usepackage{amssymb} '
preamble = preamble + '\\usepackage{mathrsfs} '
preamble = preamble + '\\DeclareMathOperator{\\Tr}{Tr} '
params['text.latex.preamble'] = preamble
params['text.usetex'] = True
plt.rcParams.update(params)


def resampling_data(t, X, resampling_fct=2, t_use=None, interp1d_kind='quadratic'):
    if t_use is None:
        t_use = np.linspace(t.min(), t.max(), np.around(t.size * resampling_fct))
    else:
        war_msg = 'size of t_use is %d, resampling_fct is IGNORED' % t_use.size
        warnings.warn(war_msg)

    intp_fun1d = interpolate.interp1d(t, X, kind=interp1d_kind, copy=False, axis=0,
                                      bounds_error=True)
    return intp_fun1d(t_use)


def make2D_X_video(t, obj_list: list, figsize=(9, 9), dpi=100, stp=1, interval=50, resampling_fct=2,
                   interp1d_kind='quadratic'):
    # percentage = 0
    def update_fun(num, line_list, data_list):
        num = num * stp
        # print(num)
        tqdm_fun.update(1)
        # percentage += 1
        for linei, datai in zip(line_list, data_list):
            linei.set_data((datai[:num, 0], datai[:num, 1]))
        return line_list

    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    axi.set_xlabel('$x_1$')
    axi.set_ylabel('$x_2$')

    line_list = [axi.plot(obji.X_hist[0, 0], obji.X_hist[0, 1])[0] for obji in obj_list]
    data_list = [resampling_data(t, obji.X_hist, resampling_fct=resampling_fct, interp1d_kind=interp1d_kind)
                 for obji in obj_list]
    t_rsp = np.linspace(t.min(), t.max(), np.around(t.size * resampling_fct))
    frames = t_rsp.size // stp
    tqdm_fun = tqdm_notebook(total=frames + 2)
    # plt.show()
    anim = animation.FuncAnimation(fig, update_fun, frames, interval=interval, blit=False,
                                   fargs=(line_list, data_list), )
    # tqdm_fun.update(100 - percentage)
    # tqdm_fun.close()
    return anim
