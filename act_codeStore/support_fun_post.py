# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-12-13

@author: zhangji
"""

# from petsc4py import PETSc
# import os
# from shutil import copyfile
# import glob
import numpy as np
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm
from act_codeStore import support_fun_show as sps
import multiprocessing
# import matplotlib
# import re
# from scanf import scanf
# from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from matplotlib.collections import LineCollection
# from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from scipy import interpolate  # , integrate, optimize
from p_tqdm import p_map


# from mpi4py import MPI
# import cProfile

def NoneFun(t1, *args, **kwargs):
    return t1


def fun_msdi_hist(_):
    t_hist, X_hist, (intp_t, dt, nt, tidx, interp1d_kind) = _
    msdi_hist = []
    intp_X_fun = interpolate.interp1d(t_hist, X_hist, kind=interp1d_kind, axis=0, bounds_error=False, fill_value='extrapolate')
    for i0, ti in enumerate(intp_t):
        intp_t0 = np.arange(t_hist.min(), t_hist.max() - ti + dt * 1e-10, dt / 10)
        intp_t1 = np.arange(t_hist.min() + ti, t_hist.max() + dt * 1e-10, dt / 10)
        msdi = np.sum(np.mean((intp_X_fun(intp_t1) - intp_X_fun(intp_t0)) ** 2, axis=0))
        msdi_hist.append(msdi)
    return msdi_hist


def targ_fun_base(prb1, intp_t_info, sort_idx, t_tmin, t_tmax):
    if sort_idx is None:
        t_plot, W_avg, phi_avg = sps.cal_avrInfo(problem=prb1, t_tmin=t_tmin, t_tmax=t_tmax,
                                                 resampling_fct=1, interp1d_kind='linear',
                                                 tavr=None)
        sort_idx = sps.fun_sort_idx(t_plot, W_avg, phi_avg, sort_type='normal', sort_idx=sort_idx)
    targs = [(prb1.t_hist, obji.X_hist, intp_t_info) for obji in prb1.obj_list[sort_idx]]
    return targs, sort_idx, prb1.n_obj


def targ_fun_ForseSphere(prb1, intp_t_info, *args, **kwargs):
    targs = [(prb1.t_hist, X_hist, intp_t_info)
             for X_hist in prb1.obj_list[0].X_hist.reshape(prb1.t_hist.size, prb1.n_sphere, prb1.dimension).transpose(1, 0, 2)]
    sort_idx = [0, ]
    return targs, sort_idx, prb1.n_sphere


def msd_diff_fun(prb1, intp_fct=None, interp1d_kind='quadratic', intp_method='linear', t_tmin=-np.inf, t_tmax=np.inf,
                 sort_idx=None, tqdm_fun=tqdm_notebook, use_cpu=None, targ_fun=targ_fun_base):
    tidx = (prb1.t_hist >= t_tmin) * (prb1.t_hist <= t_tmax)
    t_hist = prb1.t_hist[tidx]
    
    if intp_fct is None:
        intp_n = t_hist.size
    elif intp_fct < 1:
        intp_n = int(t_hist.size * intp_fct)
    elif intp_fct > 1:
        intp_n = intp_fct
    else:
        raise ValueError('wrong intp_fct, current: %s' % str(intp_fct))
    #
    if intp_method == 'linear':
        intp_t, dt = np.linspace(t_hist.min(), t_hist.max(), intp_n + 1, retstep=True)
        intp_t = intp_t[1:]
    elif intp_method == 'log':
        dt = (t_hist.max() - t_hist.min()) / intp_n
        intp_t = np.exp(np.linspace(np.log(dt), np.log(t_hist.max() - t_hist.min()), intp_n))
    elif intp_method == 'log10':
        dt = (t_hist.max() - t_hist.min()) / intp_n
        intp_t = 10 ** (np.linspace(np.log10(dt), np.log10(t_hist.max() - t_hist.min()), intp_n))
    else:
        raise ValueError('wrong intp_method, current: %s' % str(intp_method))
    intp_t_info = (intp_t, dt, intp_t.size, tidx, interp1d_kind)
    tqdm_fun = NoneFun if tqdm_fun is None else tqdm_fun
    targs, sort_idx, total = targ_fun(prb1, intp_t_info, sort_idx, t_tmin, t_tmax)
    use_cpu = np.min((total, multiprocessing.cpu_count())) if use_cpu is None else use_cpu
    if tqdm_fun is not NoneFun:
        print('number of intp_t nt=%d' % intp_t.size)
        print('used cpu ncpu=%d' % use_cpu)
    
    with multiprocessing.Pool(use_cpu) as pool:
        msd_hist = np.array(list(tqdm_fun(pool.imap(fun_msdi_hist, targs), total=total)))
    #
    # msd_hist = []
    # for obji in tqdm_fun(prb1.obj_list[sort_idx]):
    #     msdi_hist = fun_msdi_hist((prb1.t_hist, obji.X_hist, intp_t_info))
    #     msd_hist.append(msdi_hist)
    # msd_hist = np.array(msd_hist)
    
    diff_hist = []
    for msdi_hist in msd_hist:
        diff = np.polyfit(intp_t, msdi_hist, 1)[0] / (2 * prb1.dimension)
        diff_hist.append(diff)
    diff_hist = np.hstack(diff_hist)
    
    return intp_t, msd_hist, diff_hist
