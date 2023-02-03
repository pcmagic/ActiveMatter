# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:05:23 2017

@author: zhangji
"""

# plt.rcParams['figure.figsize'] = (18.5, 10.5)
# fontsize = 40

# import os
# import glob
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
# import matplotlib
# import re
# from scanf import scanf
# from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from matplotlib.collections import LineCollection
# from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
# from scipy import interpolate  # , integrate, optimize
# from mpi4py import MPI
# import cProfile
#
from act_src import problemClass
from act_codeStore import support_fun as spf

deta1_fun = lambda meat1, meta2, alpha, align: spf.warpToPi(-align / 2 * (
        2 * np.sin(meat1) * np.cos(alpha) + np.sin(meta2 + alpha) + np.sin(meat1 - meta2 - alpha)))
deta2_fun = lambda meat1, meta2, alpha, align: spf.warpToPi(-align / 2 * (
        2 * np.sin(meta2) * np.cos(alpha) + np.sin(meat1 + alpha) + np.sin(meta2 - meat1 - alpha)))


def Jacobian_Kuramoto(phi_list, align):
    # kuramoto model, \phi_i^(t+1) = \phi_i^t + \sigma / (N-1) * \sum_(j!=i){ sin(\phi_j^t - \phi_i^t) }
    n_obj = phi_list.size
    Jac = np.vstack([align / (n_obj - 1) * np.cos(phi_list - phii) for phii in phi_list])
    for i0, phii in enumerate(phi_list):
        Jac[i0, i0] = 1 - align / (n_obj - 1) * (np.sum(np.cos(phi_list - phii)) - 1)
    return Jac


def Lyp_jacobian_kuramoto(prb1: problemClass.behavior2DProblem, tqdm_fun=tqdm_notebook):
    align = prb1.align
    n_obj = prb1.n_obj
    t0, t1, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    jac_pro = np.eye(n_obj, n_obj)
    lmd_list = []
    for i0, ti in enumerate(tqdm_fun(prb1.t_hist)):
        try:
            Tdur = ti - t0 + eval_dt
            phi_list = np.hstack([obji.phi_hist[i0] for obji in prb1.obj_list])
            jac = Jacobian_Kuramoto(phi_list, align)
            jac_pro = jac @ jac_pro
            mu = np.linalg.eig(jac_pro.T @ jac_pro)[0]
            lmd = np.log(mu) / (2 * Tdur)
            lmd_list.append(lmd.real)
        except:
            break
    lmd_list = np.sort(np.vstack(lmd_list), axis=-1)
    return lmd_list

def Lyp_QR_kuramoto(prb1: problemClass.behavior2DProblem, tqdm_fun=tqdm_notebook):
    align = prb1.align
    n_obj = prb1.n_obj
    t0, t1, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt

    sum_lnR = 0
    lmd_list = []
    q = np.eye(n_obj, n_obj)
    for i0, ti in enumerate(tqdm_fun(prb1.t_hist)):
        Tdur = ti - t0 + eval_dt
        phi_list = np.hstack([obji.phi_hist[i0] + 0j for obji in prb1.obj_list])
        q, r = np.linalg.qr(Jacobian_Kuramoto(phi_list, align) @ q)
        sum_lnR = sum_lnR + np.real(np.log(np.diag(r)))
        lmd_list.append(sum_lnR / Tdur)
    lmd_list = np.sort(np.vstack(lmd_list), axis=-1)
    return lmd_list
