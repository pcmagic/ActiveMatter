# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:05:23 2017

@author: zhangji
"""

# plt.rcParams['figure.figsize'] = (18.5, 10.5)
# fontsize = 40

import os
# import glob
import natsort
import pickle
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
from scipy import optimize
# from mpi4py import MPI
# import cProfile
#
from act_src import problemClass
from act_codeStore import support_fun as spf

deta1_fun = lambda meat1, meta2, alpha, align: spf.warpToPi(-align / 2 * (
        2 * np.sin(meat1) * np.cos(alpha) + np.sin(meta2 + alpha) + np.sin(meat1 - meta2 - alpha)))
deta2_fun = lambda meat1, meta2, alpha, align: spf.warpToPi(-align / 2 * (
        2 * np.sin(meta2) * np.cos(alpha) + np.sin(meat1 + alpha) + np.sin(meta2 - meat1 - alpha)))


def NoneFun(t1):
    return t1


def subtract_method_fun(prb1: problemClass.behavior2DProblem, subtract_method='random'):
    # subtract_idx == 'random'
    def sub_method_random(*args):
        subtract_idx = np.random.randint(0, n_obj)
        obj_idx = np.ones(prb1.n_obj, dtype=bool)
        obj_idx[subtract_idx] = False
        return subtract_idx, obj_idx

    # subtract_idx == 'min'
    def sub_method_min(*args):
        phi_list = np.hstack([obji.phi_hist + 0j for obji in prb1.obj_list])
        subtract_idx = np.argsort(np.abs(np.sum(phi_list) - n_obj * phi_list))[0]
        obj_idx = np.ones(prb1.n_obj, dtype=bool)
        obj_idx[subtract_idx] = False
        return subtract_idx, obj_idx

    # subtract_idx is an int \in (0, nobj-1)
    def sub_method_idx(subtract_idx):
        obj_idx = np.ones(prb1.n_obj, dtype=bool)
        obj_idx[subtract_idx] = False
        return subtract_idx, obj_idx

    n_obj = prb1.n_obj
    subtract_method_fun = 'idx' if not isinstance(subtract_method, str) else subtract_method
    sub_method_dct = {'random': sub_method_random,
                      'min':    sub_method_min,
                      'idx':    sub_method_idx, }
    return sub_method_dct[subtract_method_fun](subtract_method)


def phi_Kuramoto(phi_list, align):
    n_phi = phi_list.size
    phi_next = np.array([phii + align / (n_phi - 1) * np.sum(np.sin(phi_list - phii))
                         for phii in phi_list])
    return phi_next


def Jacobian_Kuramoto(phi_list, align):
    # kuramoto model, \phi_i^(t+1) = \phi_i^t + \sigma / (N-1) * \sum_(j!=i){ sin(\phi_j^t - \phi_i^t) }
    n_phi = phi_list.size
    Jac = np.vstack([align / (n_phi - 1) * np.cos(phi_list - phii) for phii in phi_list])
    for i0, phii in enumerate(phi_list):
        Jac[i0, i0] = 1 - align / (n_phi - 1) * (np.sum(np.cos(phi_list - phii)) - 1)
    return Jac


def phi_Sakaguchi_Kuramoto(phi_list, align, phaseLag2D):
    n_phi = phi_list.size
    phi_next = np.array([phii + align / (n_phi - 1) * np.sum(np.sin(phi_list - phii - phaseLag2D))
                         for phii in phi_list])
    return phi_next


def Jacobian_Sakaguchi_Kuramoto(phi_list, align, phaseLag2D):
    # Sakaguchi-kuramoto model
    #   \phi_i^(t+1) = \phi_i^t + \sigma / (N-1) * \sum_(j!=i){ sin(\phi_j^t - \phi_i^t - phaseLag2D) }
    n_phi = phi_list.size
    Jac = np.vstack([align / (n_phi - 1) * np.cos(phi_list - phii - phaseLag2D) for phii in phi_list])
    for i0, phii in enumerate(phi_list):
        Jac[i0, i0] = 1 - align / (n_phi - 1) * (np.sum(np.cos(phi_list - phii - phaseLag2D)) - np.cos(phaseLag2D))
    return Jac


def eta_Kuramoto_reduce(eta_list, align):
    n_eta = eta_list.size
    eta_next = np.array([etai - align / n_eta * (np.sin(etai) +
                                                 np.sum(np.sin(eta_list) + np.sin(etai - eta_list)))
                         for etai in eta_list])
    return eta_next


def Jacobian_Kuramoto_reduce(eta_list, align):
    # kuramoto model, \phi_i^(t+1) = \phi_i^t + \sigma / (N-1) * \sum_(j!=i){ sin(\phi_j^t - \phi_i^t) }
    n_eta = eta_list.size
    Jac = np.vstack([-align / n_eta * (np.cos(etai) - np.cos(eta_list - etai)) for etai in eta_list])
    for i0, etai in enumerate(eta_list):
        Jac[i0, i0] = 1 - align / n_eta * (2 * np.cos(etai) + np.sum(np.cos(eta_list - etai)) - 1)
    return Jac


def Lyp_obj_kuramoto(prb1: problemClass.behavior2DProblem, tqdm_fun=tqdm_notebook,
                     Lyp_ign=-np.inf, use_idx=None):
    align = prb1.align
    n_obj = prb1.n_obj
    t0, t1, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    tqdm_fun = NoneFun if tqdm_fun is None else tqdm_fun
    use_idx = np.isfinite(prb1.t_hist) if use_idx is None else use_idx

    sum_lnR = 0
    lmd_list = []
    for i0, ti in enumerate(tqdm_fun(prb1.t_hist)):
        if ti < Lyp_ign:
            lmd_list.append(np.zeros(n_obj))
        else:
            Tdur = ti - t0 + eval_dt
            phi_list = np.hstack([obji.phi_hist[i0] for obji in prb1.obj_list])
            t1 = [1 - align / (n_obj - 1) * (np.sum(np.cos(phi_list - phii)) - 1)
                  # t1 = [1 - align / (n_obj - 1) * np.sum(np.cos(phi_list - phii)) + align / (n_obj - 1)
                  for phii in phi_list]
            print(t1)
            sum_lnR = sum_lnR + np.log(np.abs(t1))
            lmd_list.append(sum_lnR / Tdur)
    lmd_list = np.vstack(lmd_list)
    obj_Lyp_idx = np.argsort(np.mean(lmd_list[use_idx], axis=0))
    lmd_list = lmd_list[use_idx][:, obj_Lyp_idx]
    return lmd_list


def Lyp_obj_Sakaguchi_kuramoto(prb1: problemClass.behavior2DProblem, tqdm_fun=tqdm_notebook,
                               Lyp_ign=-np.inf, use_idx=None):
    align = prb1.align
    n_obj = prb1.n_obj
    phaseLag2D = prb1.kwargs['phaseLag2D']
    t0, t1, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    tqdm_fun = NoneFun if tqdm_fun is None else tqdm_fun
    use_idx = np.isfinite(prb1.t_hist) if use_idx is None else use_idx

    sum_lnR = 0
    lmd_list = []
    for i0, ti in enumerate(tqdm_fun(prb1.t_hist)):
        if ti < Lyp_ign:
            lmd_list.append(np.zeros(n_obj))
        else:
            Tdur = ti - t0 + eval_dt
            phi_list = np.hstack([obji.phi_hist[i0] for obji in prb1.obj_list])
            t1 = [1 - align / (n_obj - 1) * (np.sum(np.cos(phi_list - phii - phaseLag2D)) - np.cos(phaseLag2D))
                  for phii in phi_list]
            sum_lnR = sum_lnR + np.log(np.abs(t1))
            lmd_list.append(sum_lnR / Tdur)
    lmd_list = np.vstack(lmd_list)
    obj_Lyp_idx = np.argsort(np.mean(lmd_list[use_idx], axis=0))
    lmd_list = lmd_list[use_idx][:, obj_Lyp_idx]
    return lmd_list


def Lyp_obj_kuramoto_reduce(prb1: problemClass.behavior2DProblem, tqdm_fun=tqdm_notebook,
                            Lyp_ign=-np.inf, use_idx=None, subtract_method='random'):
    align = prb1.align
    n_eta = prb1.n_obj - 1
    t0, t1, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    tqdm_fun = NoneFun if tqdm_fun is None else tqdm_fun
    use_idx = np.isfinite(prb1.t_hist) if use_idx is None else use_idx

    sum_lnR = 0
    lmd_list = []
    for i0, ti in enumerate(tqdm_fun(prb1.t_hist)):
        if ti < Lyp_ign:
            lmd_list.append(np.zeros(n_eta))
        else:
            subtract_idx, obj_idx = subtract_method_fun(prb1, subtract_method)
            Tdur = ti - t0 + eval_dt
            eta_list = np.hstack([obji.phi_hist[i0] - prb1.obj_list[subtract_idx].phi_hist[i0]
                                  for obji in prb1.obj_list[obj_idx]])
            eta_list = spf.warpToPi(eta_list)
            t1 = [1 - align / n_eta * (2 * np.cos(etai) + np.sum(np.cos(eta_list - etai)) - 1)
                  # t1 = [1 - align / (n_eta - 1) * (np.sum(np.cos(eta_list - etai) - 1))
                  for etai in eta_list]
            sum_lnR = sum_lnR + np.log(np.abs(t1))
            lmd_list.append(sum_lnR / Tdur)
    lmd_list = np.vstack(lmd_list)
    obj_Lyp_idx = np.argsort(np.mean(lmd_list[use_idx], axis=0))
    lmd_list = lmd_list[use_idx][:, obj_Lyp_idx]
    return lmd_list


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


def Lyp_QR_kuramoto(prb1: problemClass.behavior2DProblem, tqdm_fun=tqdm_notebook,
                    Lyp_ign=-np.inf, use_idx=None):
    align = prb1.align
    n_obj = prb1.n_obj
    t0, t1, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    tqdm_fun = NoneFun if tqdm_fun is None else tqdm_fun
    use_idx = np.isfinite(prb1.t_hist) if use_idx is None else use_idx

    sum_lnR = 0
    lmd_list = []
    q = np.eye(n_obj, n_obj)
    for i0, ti in enumerate(tqdm_fun(prb1.t_hist)):
        if ti < Lyp_ign:
            lmd_list.append(np.zeros(n_obj - 1) * np.nan)
        else:
            Tdur = ti - t0 + eval_dt
            phi_list = np.hstack([obji.phi_hist[i0] + 0j for obji in prb1.obj_list])
            q, r = np.linalg.qr(Jacobian_Kuramoto(phi_list, align) @ q)
            sum_lnR = sum_lnR + np.real(np.log(np.diag(r)))
            lmd_list.append(sum_lnR[np.argsort(np.abs(sum_lnR))[1:]] / Tdur)
    lmd_list = np.sort(np.vstack(lmd_list)[use_idx], axis=-1)
    return lmd_list


def Lyp_QR_Sakaguchi_kuramoto(prb1: problemClass.behavior2DProblem, tqdm_fun=tqdm_notebook,
                              Lyp_ign=-np.inf, use_idx=None):
    align = prb1.align
    phaseLag2D = prb1.kwargs['phaseLag2D']
    n_obj = prb1.n_obj
    t0, t1, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    tqdm_fun = NoneFun if tqdm_fun is None else tqdm_fun
    use_idx = np.isfinite(prb1.t_hist) if use_idx is None else use_idx

    sum_lnR = 0
    lmd_list = []
    q = np.eye(n_obj, n_obj)
    for i0, ti in enumerate(tqdm_fun(prb1.t_hist)):
        if ti < Lyp_ign:
            lmd_list.append(np.zeros(n_obj - 1) * np.nan)
        else:
            Tdur = ti - t0 + eval_dt
            phi_list = np.hstack([obji.phi_hist[i0] + 0j for obji in prb1.obj_list])
            q, r = np.linalg.qr(Jacobian_Sakaguchi_Kuramoto(phi_list, align, phaseLag2D) @ q)
            sum_lnR = sum_lnR + np.real(np.log(np.diag(r)))
            lmd_list.append(sum_lnR[np.argsort(np.abs(sum_lnR))[1:]] / Tdur)
    lmd_list = np.sort(np.vstack(lmd_list)[use_idx], axis=-1)
    return lmd_list


def Lyp_QR_kuramoto_reduce(prb1: problemClass.behavior2DProblem, tqdm_fun=tqdm_notebook,
                           Lyp_ign=-np.inf, use_idx=None, subtract_method='random'):
    align = prb1.align
    n_obj = prb1.n_obj
    t0, t1, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    tqdm_fun = NoneFun if tqdm_fun is None else tqdm_fun
    use_idx = np.isfinite(prb1.t_hist) if use_idx is None else use_idx

    sum_lnR = 0
    lmd_list = []
    q = np.eye(n_obj - 1, n_obj - 1)
    for i0, ti in enumerate(tqdm_fun(prb1.t_hist)):
        if ti < Lyp_ign:
            lmd_list.append(np.zeros(n_obj - 1) * np.nan)
        else:
            subtract_idx, obj_idx = subtract_method_fun(prb1, subtract_method)
            Tdur = ti - t0 + eval_dt
            eta_list = np.hstack([obji.phi_hist[i0] - prb1.obj_list[subtract_idx].phi_hist[i0]
                                  for obji in prb1.obj_list[obj_idx]])
            eta_list = spf.warpToPi(eta_list) + 0j
            # eta_list = np.hstack([eta_list[0] - eta_list[1], eta_list[1] - eta_list[0]])
            # print(np.hstack([obji.phi_hist[i0] for obji in prb1.obj_list]), eta_list)
            # print(phi_list, subtract_idx, eta_list)
            q, r = np.linalg.qr(Jacobian_Kuramoto_reduce(eta_list, align) @ q)
            sum_lnR = sum_lnR + np.real(np.log(np.diag(r)))
            lmd_list.append(sum_lnR / Tdur)
    lmd_list = np.sort(np.vstack(lmd_list)[use_idx], axis=-1)
    return lmd_list


def Lyapunov_std_fun(PWD, folder_name_list, pickle_name,
                     plt_tmin_fct, plt_tmax_fct, Lyp_ign_fct,
                     tqdm_fun=tqdm_notebook):
    tqdm_fun = NoneFun if tqdm_fun is None else tqdm_fun

    Lyp_all_dct = {}  # [align_list, lmdi(all)_vs_time]
    Lyp_obj_dct = {}  # [align_list, lmdi(obj)_vs_time]
    for folder_name in folder_name_list:
        print(folder_name)
        folder_path = os.path.join(PWD, folder_name)
        job_name_list = natsort.natsorted([job_name for job_name in os.listdir(folder_path)
                                           if os.path.isdir(os.path.join(folder_path, job_name))])
        for job_name in tqdm_fun(job_name_list):
            pick_name = os.path.join(PWD, folder_name, job_name, 'pickle.%s' % job_name)
            hdf5_name = os.path.join(PWD, folder_name, job_name, 'hdf5.%s' % job_name)
            try:
                with open(pick_name, 'rb') as handle:
                    prb1 = pickle.load(handle)
                prb1.hdf5_load(hdf5_name=hdf5_name)
                align = prb1.align
                ini_t, max_t, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
                plt_tmin = ini_t + (max_t - ini_t) * plt_tmin_fct
                plt_tmax = ini_t + (max_t - ini_t) * plt_tmax_fct
                Lyp_ign = ini_t + (max_t - ini_t) * Lyp_ign_fct
                assert Lyp_ign < plt_tmin
                tidx = (prb1.t_hist <= plt_tmax) * (prb1.t_hist >= plt_tmin)
                # stastic process 1: prepare dicts
                if not align in Lyp_all_dct.keys():
                    Lyp_all_dct[align] = []
                    Lyp_obj_dct[align] = []
                # Lyapunov exponent
                #   1. lyapunov exponent of the whole system
                lmd_list = Lyp_QR_kuramoto(prb1, tqdm_fun=None, Lyp_ign=Lyp_ign, use_idx=tidx)
                Lyp_all_dct[align].append(lmd_list)
                #   2. lyapunov exponent of the object itself
                lmd_list = Lyp_obj_kuramoto(prb1, tqdm_fun=None, Lyp_ign=Lyp_ign, use_idx=tidx)
                Lyp_obj_dct[align].append(lmd_list)
                # finish
                prb1.destroy_self()
            except:
                print('wrong job folder: ', job_name)

    align_list = np.sort(np.fromiter(Lyp_obj_dct.keys(), dtype=float))
    # n_case = len(Lyp_obj_dct[align_list[0]])
    n_obj = Lyp_obj_dct[align_list[0]][0].shape[1]
    Lyp_all_std = [[] for i0 in
                   np.arange(n_obj - 1)]  # [lmdi_std], lmdi_std: [align_list, (mean, median, std, max, min)]
    Lyp_all_case_std = [[] for i0 in np.arange(n_obj - 1)]  # [align_list, lmdi_std]
    Lyp_obj_std = [[] for i0 in np.arange(n_obj)]  # [obji_std], obji_std: [align_list, (mean, median, std, max, min)]
    Lyp_obj_case_std = [[] for i0 in np.arange(n_obj)]  # [align_list, obji_std]
    for align in align_list:
        # Lyapunov exponent, 1. Lyapunov exponent of the whole system
        for Lyp_idx, tLyp_all_std, tLyp_all_case_std in zip(np.arange(n_obj - 1), Lyp_all_std, Lyp_all_case_std):
            # 1. Lyapunov exponent of the whole system
            tLyp = np.hstack([i0[:, Lyp_idx] for i0 in Lyp_all_dct[align]])
            tLyp_all_std.append((np.mean(tLyp), np.median(tLyp), np.std(tLyp),
                                 np.max(tLyp), np.min(tLyp),))
            tstd = []  # [align_list, case_std], case_std: (mean, median, std, max, min)
            for tLyp in Lyp_all_dct[align]:
                case_lyp = tLyp[:, Lyp_idx]
                tstd.append((np.mean(case_lyp), np.median(case_lyp), np.std(case_lyp),
                             np.max(case_lyp), np.min(case_lyp),))
            tstd = np.vstack(tstd)
            tLyp_all_case_std.append(tstd)
        # Lyapunov exponent, 2. Lyapunov exponent of the object itself
        for Lyp_idx, tLyp_obj_std, tLyp_obj_case_std in zip(np.arange(n_obj), Lyp_obj_std, Lyp_obj_case_std):
            tLyp = np.hstack([i0[:, Lyp_idx] for i0 in Lyp_obj_dct[align]])
            tLyp_obj_std.append((np.mean(tLyp), np.median(tLyp), np.std(tLyp),
                                 np.max(tLyp), np.min(tLyp),))
            tstd = []  # [align_list, case_std], case_std: (mean, median, std, max, min)
            for tLyp in Lyp_obj_dct[align]:
                case_lyp = tLyp[:, Lyp_idx]
                tstd.append((np.mean(case_lyp), np.median(case_lyp), np.std(case_lyp),
                             np.max(case_lyp), np.min(case_lyp),))
            tstd = np.vstack(tstd)
            tLyp_obj_case_std.append(tstd)
    Lyp_all_std = [np.array(i0) for i0 in Lyp_all_std]  # (lmd_idx, align, std_info)
    Lyp_all_case_std = [np.array(i0) for i0 in Lyp_all_case_std]  # (lmd_idx, align, case_idx, std_info)
    Lyp_obj_std = [np.array(i0) for i0 in Lyp_obj_std]  # (obj_idx, align, std_info)
    Lyp_obj_case_std = [np.array(i0) for i0 in Lyp_obj_case_std]  # (obj_idx, align, case_idx, std_info)

    pickle_dct = {'align_list':       align_list,
                  # 'n_case':           n_case,
                  'n_obj':            n_obj,
                  'Lyp_all_dct':      Lyp_all_dct,
                  'Lyp_obj_dct':      Lyp_obj_dct,
                  'Lyp_all_std':      Lyp_all_std,
                  'Lyp_all_case_std': Lyp_all_case_std,
                  'Lyp_obj_std':      Lyp_obj_std,
                  'Lyp_obj_case_std': Lyp_obj_case_std, }
    with open(pickle_name, 'wb') as handle:
        pickle.dump(pickle_dct, handle, protocol=4)
    print('pick static information to file %s ' % pickle_name)
    return True


# -------------------------------------------------------------------------------
G_fun = lambda phi_list, align, phaseLag2D: phi_Sakaguchi_Kuramoto(phi_list=phi_list, align=align, phaseLag2D=phaseLag2D)
J_fun = lambda phi_list, align, phaseLag2D: Jacobian_Sakaguchi_Kuramoto(phi_list=phi_list, align=align, phaseLag2D=phaseLag2D)
def lyp_all_list_fun(xt, align, phaseLag2D, t_itera, nptc):
    q, r = np.linalg.qr(J_fun(xt, align, phaseLag2D))
    lyp_list = []
    sum_lnR = 0
    for i0 in np.arange(t_itera):
        xt = G_fun(xt, align, phaseLag2D)
        q, r = np.linalg.qr(J_fun(xt, align, phaseLag2D) @ q)
        sum_lnR = sum_lnR + np.real(np.log(np.diag(r)))
        lyp_list.append(sum_lnR / (i0 + 1))
    lyp_list = np.vstack(lyp_list)
    return lyp_list


def lyp_obj_list_fun(xt, align, phaseLag2D, t_itera, nptc=3):
    lyp_list = []
    sum_lnR = 0
    for i0 in np.arange(t_itera):
        xt = G_fun(xt, align, phaseLag2D)
        t1 = [1 - align / (nptc - 1) * (np.sum(np.cos(xt - xti - phaseLag2D)) - np.cos(phaseLag2D)) for xti in xt]
        sum_lnR = sum_lnR + np.log(np.abs(t1))
        lyp_list.append(sum_lnR / (i0 + 1))
    lyp_list = np.vstack(lyp_list)
    return lyp_list


def lyp_std_fun(align, phaseLag2D, t_trans, t_itera, n_case, nptc=3, lyp_list_fun=lyp_all_list_fun):
    lyp_std_all = []
    for _ in np.arange(n_case):
        # transient process
        xt = np.random.sample(nptc) + 0j
        for _ in np.arange(t_trans):
            xt = G_fun(xt, align, phaseLag2D)
            # Lyp, interation process
        lyp_list = lyp_list_fun(xt, align, phaseLag2D, t_itera, nptc=nptc)
        obj_Lyp_idx = np.argsort(np.mean(lyp_list, axis=0))
        lyp_std_all.append(lyp_list[(-t_itera // 10):, obj_Lyp_idx])
        # std process
    tlyp_std = np.vstack([(np.mean(t1), np.median(t1), np.std(t1), np.max(t1), np.min(t1))
                          for t1 in np.array(lyp_std_all).T])
    return tlyp_std

# -------------------------------------------------------------------------------------------
# calculate the first two bifurcations of Sakaguchi-Kuramoto model.
deta_fun = lambda t1, t2, align, phaseLag2D, nptc: spf.warpToPi(
    (np.sin(t1 - t2 + phaseLag2D) + np.sin(t1 + phaseLag2D) +
     np.sin(t1 - phaseLag2D) + np.sin(t2 - phaseLag2D)) * (-align / (nptc - 1)))
par_eta_ii_fun = lambda t1, t2, align, phaseLag2D, nptc: 1 - align / (nptc - 1) * (
        np.cos(t1 - t2 + phaseLag2D) + np.cos(t1 + phaseLag2D) + np.cos(t1 - phaseLag2D))
par_eta_ij_fun = lambda t1, t2, align, phaseLag2D, nptc: - align / (nptc - 1) * (
        -np.cos(t1 - t2 + phaseLag2D) + np.cos(t2 - phaseLag2D))
alpha_bifu_fun = lambda talign, kfct, nptc: spf.warpToPi(np.arccos((kfct * (nptc - 1)) / (nptc * talign)))
align_bifu_fun = lambda talpha, kfct, nptc: (kfct * (nptc - 1)) / (nptc * np.cos(talpha))


def Jacobian_matix_fun(eta1_t1, eta2_t1, align, phaseLag2D, nptc):
    eta1_t2 = eta1_t1 + deta_fun(eta1_t1, eta2_t1, align, phaseLag2D, nptc)
    eta2_t2 = eta2_t1 + deta_fun(eta2_t1, eta1_t1, align, phaseLag2D, nptc)
    Jacobian = np.array([
        (par_eta_ii_fun(eta1_t2, eta2_t2, align, phaseLag2D, nptc) *
         par_eta_ii_fun(eta1_t1, eta2_t1, align, phaseLag2D, nptc),

         par_eta_ij_fun(eta1_t2, eta2_t2, align, phaseLag2D, nptc) *
         par_eta_ij_fun(eta2_t1, eta1_t1, align, phaseLag2D, nptc)),

        (par_eta_ii_fun(eta2_t2, eta1_t2, align, phaseLag2D, nptc) *
         par_eta_ii_fun(eta1_t1, eta2_t1, align, phaseLag2D, nptc),

         par_eta_ij_fun(eta2_t2, eta2_t2, align, phaseLag2D, nptc) *
         par_eta_ij_fun(eta2_t1, eta2_t1, align, phaseLag2D, nptc)),
    ])
    return Jacobian


def eta_wapper_fun(eta12, align, phaseLag2D, nptc):
    eta1, eta2 = eta12
    deta1 = deta_fun(eta1, eta2, align, phaseLag2D, nptc)
    deta2 = deta_fun(eta2, eta1, align, phaseLag2D, nptc)
    return np.hstack((deta1 + deta2, eta2 - eta1 - deta1))


def J_egv_fun(align, phaseLag2D, nptc):
    sol = optimize.root(eta_wapper_fun, (0.6424, -0.30667), args=(align, phaseLag2D, nptc))
    eta1_t1, eta2_t1 = sol.x
    J_eta = Jacobian_matix_fun(eta1_t1, eta2_t1, align, phaseLag2D, nptc)
    w, v = np.linalg.eig(J_eta)
    return 1 - np.max(np.abs(w))
# -------------------------------------------------------------------------------------------

#
