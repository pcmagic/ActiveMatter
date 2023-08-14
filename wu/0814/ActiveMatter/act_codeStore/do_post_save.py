import sys
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc
import os
# from datetime import datetime
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
from act_codeStore.support_fun_kuramoto import deta1_fun, deta2_fun

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
params['text.usetex'] = True
plt.rcParams.update(params)


def main_small3_b():
    output_handle = os.getcwd()
    PWD = '/home/zhangji/fishSchool/chimera'
    folder_name = 'small3_b'

    OptDB = PETSc.Options()
    align = np.float64(OptDB.getReal('align', 1))
    alpha = np.float64(OptDB.getReal('alpha', 0.49))
    idx = np.int64(OptDB.getInt('idx', 0))
    job_name = 'small3_b_align%05.2f_alpha%04.2f_%04d' % (align, alpha, idx)

    pickle_name = os.path.join(PWD, folder_name, job_name, 'pickle.%s' % job_name)
    hdf5_name = os.path.join(PWD, folder_name, job_name, 'hdf5.%s' % job_name)
    with open(pickle_name, 'rb') as handle:
        prb1 = pickle.load(handle)
    prb1.hdf5_load(hdf5_name=hdf5_name)

    output_path = os.path.join(output_handle, job_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('make folder %s' % output_path)
    else:
        print('exist folder %s' % output_path)

    ################################################################################3
    dpi = 1000
    ini_t, max_t, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * 0.5, ini_t + (max_t - ini_t) * 0.52, 1e-10
    resampling_fct, interp1d_kind = 1, 'linear'
    sort_dict = {'normal':    lambda: np.argsort(np.mean(W_avg[:, t_plot > t_plot.max() / 2], axis=-1)),
                 'traveling': lambda: np.argsort(phi_avg[:, -1])}
    t_plot, W_avg, phi_avg = sps.cal_avrInfo(problem=prb1, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                             resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                             tavr=tavr)
    sort_idx = sort_dict['normal']()

    ################################################################################
    # save
    # ----------------------------------
    figsize = np.array((16, 9)) * 0.3
    cmap = plt.get_cmap('viridis')
    filename = os.path.join(output_path, 'avrW_%s.png' % prb1.name)
    sps.save_fig_fun(filename, prb1, sps.core_avrPhaseVelocity, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax, resampling_fct=resampling_fct, cmap=cmap,
                     tavr=tavr, sort_idx=sort_idx)

    # ----------------------------------
    figsize = np.array((16, 9)) * 0.3
    markevery, linestyle = 0.3, '-k',
    filename = os.path.join(output_path, 'orderR_%s.png' % prb1.name)
    sps.save_fig_fun(filename, prb1, sps.core_polar_order, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                     markevery=markevery, linestyle=linestyle)

    # ----------------------------------
    figsize = np.array((16, 9)) * 0.3
    cmap = plt.get_cmap('twilight_shifted')
    filename = os.path.join(output_path, 'avrPhi1_%s.png' % prb1.name)
    sps.save_fig_fun(filename, prb1, sps.core_avrPhase, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                     resampling_fct=resampling_fct, cmap=cmap,
                     tavr=0.01, sort_type='normal', sort_idx=sort_idx)
    filename = os.path.join(output_path, 'avrPhi2_%s.png' % prb1.name)
    sps.save_fig_fun(filename, prb1, sps.core_avrPhase, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                     resampling_fct=resampling_fct, cmap=cmap,
                     tavr=0.01, sort_type='traveling', sort_idx=sort_idx)
    return True


def main_subname():
    output_handle = os.getcwd()
    PWD = '/home/zhangji/fishSchool/chimera/small3'

    OptDB = PETSc.Options()
    folder_name = OptDB.getString('folder_name', 'small3_alpha0_idx0000')
    job_name = OptDB.getString('job_name', ' ')
    pickle_name = os.path.join(PWD, folder_name, job_name, 'pickle.%s' % job_name)
    hdf5_name = os.path.join(PWD, folder_name, job_name, 'hdf5.%s' % job_name)
    with open(pickle_name, 'rb') as handle:
        prb1 = pickle.load(handle)
    prb1.hdf5_load(hdf5_name=hdf5_name)

    align = prb1.align
    idx = scanf('_%d', prb1.name)[0]
    save_name = 'alpha0_align%08.5f_%04d' % (align, idx)
    output_path = os.path.join(output_handle, save_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('make folder %s' % output_path)
    else:
        print('exist folder %s' % output_path)

    ################################################################################3
    dpi = 1000
    ini_t, max_t, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * 0.5, ini_t + (max_t - ini_t) * 0.52, 1e-10
    resampling_fct, interp1d_kind = 1, 'linear'
    sort_dict = {'normal':    lambda: np.argsort(np.mean(W_avg[:, t_plot > t_plot.max() / 2], axis=-1)),
                 'traveling': lambda: np.argsort(phi_avg[:, -1])}
    t_plot, W_avg, phi_avg = sps.cal_avrInfo(problem=prb1, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                             resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                             tavr=tavr)
    sort_idx = sort_dict['normal']()

    ################################################################################
    # save
    # ----------------------------------
    figsize = np.array((9, 9)) * 0.5
    filename = os.path.join(output_path, 'trj_%s.png' % save_name)
    sps.save_fig_fun(filename, prb1, sps.core_trajectory2D, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax, resampling_fct=resampling_fct,
                     show_idx=None, range_full_obj=False, plt_full_time=False)
    # ----------------------------------
    figsize = np.array((16, 9)) * 0.3
    cmap = plt.get_cmap('viridis')
    filename = os.path.join(output_path, 'avrW_%s.png' % save_name)
    sps.save_fig_fun(filename, prb1, sps.core_avrPhaseVelocity, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax, resampling_fct=resampling_fct, cmap=cmap,
                     tavr=tavr, sort_idx=sort_idx)

    # ----------------------------------
    figsize = np.array((16, 9)) * 0.3
    markevery, linestyle = 0.3, '-k',
    filename = os.path.join(output_path, 'orderR_%s.png' % save_name)
    sps.save_fig_fun(filename, prb1, sps.core_polar_order, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                     markevery=markevery, linestyle=linestyle)

    # ----------------------------------
    figsize = np.array((16, 9)) * 0.3
    cmap = plt.get_cmap('twilight_shifted')
    filename = os.path.join(output_path, 'avrPhi1_%s.png' % save_name)
    sps.save_fig_fun(filename, prb1, sps.core_avrPhase, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                     resampling_fct=resampling_fct, cmap=cmap,
                     tavr=0.01, sort_type='normal', sort_idx=sort_idx)
    filename = os.path.join(output_path, 'avrPhi2_%s.png' % save_name)
    sps.save_fig_fun(filename, prb1, sps.core_avrPhase, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                     resampling_fct=resampling_fct, cmap=cmap,
                     tavr=0.01, sort_type='traveling', sort_idx=sort_idx)
    return True


def main_gradientMap():
    def do_plot(problem, figsize=np.array((13, 9)) * 0.3, dpi=100):
        prb1 = problem
        ini_t, max_t, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
        eta1_hist = spf.warpToPi(prb1.obj_list[2].phi_hist - prb1.obj_list[0].phi_hist)
        eta2_hist = spf.warpToPi(prb1.obj_list[2].phi_hist - prb1.obj_list[1].phi_hist)
        # deta1_hist = spf.warpToPi(prb1.obj_list[2].W_hist - prb1.obj_list[0].W_hist)
        # deta2_hist = spf.warpToPi(prb1.obj_list[2].W_hist - prb1.obj_list[1].W_hist)
        deta1_hist = deta1_fun(eta1_hist, eta2_hist, alpha, align)
        deta2_hist = deta2_fun(eta1_hist, eta2_hist, alpha, align)
        #
        eta1, eta2 = np.meshgrid(np.linspace(-1, 1, 15) * np.pi, np.linspace(-1, 1, 15) * np.pi)
        deta1 = deta1_fun(eta1, eta2, alpha, align)
        deta2 = deta2_fun(eta1, eta2, alpha, align)
        #
        plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * 0.8, ini_t + (max_t - ini_t) * 0.9, eval_dt
        t_hist = prb1.t_hist
        tidx = (t_hist <= plt_tmax) * (t_hist >= plt_tmin)

        ################################################################################3
        fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
        fig.patch.set_facecolor('white')
        # gradient field ------------------------------------------------------------------------
        tnorm = np.sqrt(deta1 ** 2 + deta2 ** 2)
        vmax = 1
        cmap = plt.get_cmap('gray_r')
        norm = Normalize(vmin=0, vmax=vmax)
        cqu = axi.quiver(eta1 / np.pi, eta2 / np.pi, deta1 / tnorm, deta2 / tnorm, tnorm / align,
                         norm=norm, cmap=cmap, angles='xy', pivot='mid', scale=30)
        fig.colorbar(cqu).ax.set_title('$|\\eta| / \\sigma$')
        # direction vs time ---------------------------------------------------------------------
        tnorm = np.sqrt(deta1_hist ** 2 + deta2_hist ** 2)
        cmap = plt.get_cmap('jet')
        norm = Normalize(vmin=plt_tmin, vmax=plt_tmax)
        # avg_stp = np.min((30, tidx.sum()))
        # cplx_R = np.mean(np.array([(np.cos(tobj.phi_hist[tidx]), np.sin(tobj.phi_hist[tidx]))
        #                            for tobj in prb1.obj_list]), axis=0)
        # para_R = np.linalg.norm(cplx_R, axis=0)
        # t1 = sps.moving_avg(para_R, avg_stp=avg_stp) ** 10
        # quiver_alpha = (t1 - t1.min()) / (t1.max() - t1.min())
        quiver_alpha = np.ones_like(t_hist[tidx])
        axi.quiver(eta1_hist[tidx] / np.pi, eta2_hist[tidx] / np.pi,
                   deta1_hist[tidx] / tnorm[tidx], deta2_hist[tidx] / tnorm[tidx], t_hist[tidx],
                   norm=norm, cmap=cmap, angles='xy', scale=30, alpha=quiver_alpha)
        # location (in phase map) vs time -------------------------------------------------------
        csc = axi.scatter(eta1_hist[tidx] / np.pi, eta2_hist[tidx] / np.pi, 1,
                          c=t_hist[tidx], cmap=cmap, norm=norm, alpha=quiver_alpha)
        fig.colorbar(csc).ax.set_title('$t$')
        # ----------------------------------------------------------------------------------------
        axi.set_xlabel('$\\eta_1 / \\pi$')
        axi.set_ylabel('$\\eta_2 / \\pi$')
        axi.set_title('$\\sigma = %f$' % align)
        return fig, axi

    output_handle = os.getcwd()
    PWD = '/home/zhangji/fishSchool/chimera/small3'

    OptDB = PETSc.Options()
    folder_name = OptDB.getString('folder_name', 'small3_alpha0_idx0000')
    job_name = OptDB.getString('job_name', ' ')
    pickle_name = os.path.join(PWD, folder_name, job_name, 'pickle.%s' % job_name)
    hdf5_name = os.path.join(PWD, folder_name, job_name, 'hdf5.%s' % job_name)
    with open(pickle_name, 'rb') as handle:
        prb1 = pickle.load(handle)
    prb1.hdf5_load(hdf5_name=hdf5_name)
    align = prb1.align
    alpha = prb1.kwargs['phaseLag2D']
    assert np.isclose(alpha, 0)
    idx = scanf('_%d', prb1.name)[0]
    save_name = 'alpha0_align%08.5f_%04d' % (align, idx)
    output_path = os.path.join(output_handle, save_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('make folder %s' % output_path)
    else:
        print('exist folder %s' % output_path)

    ################################################################################
    # save
    # ----------------------------------
    figsize = np.array((13, 9)) * 0.5
    dpi = 1000
    filename = os.path.join(output_path, 'GM_%s.png' % save_name)
    sps.save_fig_fun(filename, prb1, do_plot, figsize=figsize, dpi=dpi)
    return True


def main_trajectory():
    output_handle = os.getcwd()
    PWD = '/home/zhangji/fishSchool/chimera/small3'

    OptDB = PETSc.Options()
    folder_name = OptDB.getString('folder_name', 'small3_alpha0_idx0000')
    job_name = OptDB.getString('job_name', ' ')
    pickle_name = os.path.join(PWD, folder_name, job_name, 'pickle.%s' % job_name)
    hdf5_name = os.path.join(PWD, folder_name, job_name, 'hdf5.%s' % job_name)
    with open(pickle_name, 'rb') as handle:
        prb1 = pickle.load(handle)
    prb1.hdf5_load(hdf5_name=hdf5_name)
    align = prb1.align
    alpha = prb1.kwargs['phaseLag2D']
    assert np.isclose(alpha, 0)
    idx = scanf('_%d', prb1.name)[0]
    save_name = 'alpha0_align%08.5f_%04d' % (align, idx)
    output_path = os.path.join(output_handle, save_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print('make folder %s' % output_path)
    else:
        print('exist folder %s' % output_path)

    ################################################################################
    # save
    # ----------------------------------
    figsize = np.array((9, 9)) * 1
    dpi = 1000
    ini_t, max_t, eval_dt = prb1.t0, prb1.t1, prb1.eval_dt
    resampling_fct, interp1d_kind = 1, 'linear'
    #
    plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * 0, ini_t + (max_t - ini_t) * 1, eval_dt
    filename = os.path.join(output_path, 'tja_%s.png' % save_name)
    sps.save_fig_fun(filename, prb1, sps.core_trajectory2D, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax, resampling_fct=resampling_fct,
                     show_idx=None, range_full_obj=False, plt_full_time=False)
    #
    plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * 0.5, ini_t + (max_t - ini_t) * 0.6, eval_dt
    filename = os.path.join(output_path, 'tjb_%s.png' % save_name)
    sps.save_fig_fun(filename, prb1, sps.core_trajectory2D, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax, resampling_fct=resampling_fct,
                     show_idx=None, range_full_obj=False, plt_full_time=False)
    #
    plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * 0.51, ini_t + (max_t - ini_t) * 0.52, eval_dt
    filename = os.path.join(output_path, 'tjc_%s.png' % save_name)
    sps.save_fig_fun(filename, prb1, sps.core_trajectory2D, figsize=figsize, dpi=dpi,
                     plt_tmin=plt_tmin, plt_tmax=plt_tmax, resampling_fct=resampling_fct,
                     show_idx=None, range_full_obj=False, plt_full_time=False)
    return True


def main_fun():
    print('do noting')
    return True


if __name__ == '__main__':
    OptDB = PETSc.Options()
    if OptDB.getBool('main_small3_b', False):
        OptDB.setValue('main_fun', False)
        main_small3_b()

    if OptDB.getBool('main_subname', False):
        OptDB.setValue('main_fun', False)
        main_subname()

    if OptDB.getBool('main_gradientMap', False):
        OptDB.setValue('main_fun', False)
        main_gradientMap()

    if OptDB.getBool('main_trajectory', False):
        OptDB.setValue('main_fun', False)
        main_trajectory()

    if OptDB.getBool('main_fun', True):
        main_fun()
