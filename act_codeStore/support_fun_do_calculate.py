import sys

# import numpy as np
import petsc4py

# import matplotlib
# matplotlib.use('agg')
petsc4py.init(sys.argv)

# import numpy as np
# import pickle
# from time import time
from petsc4py import PETSc

# from datetime import datetime
from tqdm import tqdm
# import shutil
# import os
# from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from act_src import problemClass
from act_src import relationClass
from act_src import particleClass

# from act_src import interactionClass
from act_codeStore.support_fun import *
from act_codeStore import support_fun_calculate as spc
from act_codeStore import support_fun_show as sps


def do_pickle(prb1: problemClass._baseProblem, **kwargs):
    prb1.pick_prepare()
    prb1.pick_myself(**kwargs)
    return True


def do_hdf5(prb1: problemClass._baseProblem, **kwargs):
    prb1.hdf5_pick(**kwargs)
    prb1.empty_hist()
    prb1.pick_myself(**kwargs)
    return True


def export_trajectory2D(prb1: problemClass._baseProblem, **kwargs):
    OptDB = PETSc.Options()
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    
    # setup
    figsize = np.array((10, 10)) * 1
    dpi = 100
    resampling_fct, interp1d_kind = None, "linear"
    save_fig = OptDB.getBool("save_fig", True)
    save_sub_fig = OptDB.getBool("save_sub_fig", True)
    
    if rank == 0:
        if save_fig:
            filename = "%s/fig_%s.png" % (prb1.name, prb1.name)
            sps.save_fig_fun(filename, prb1, sps.core_trajectory2D, figsize=figsize, dpi=dpi, plt_tmin=-np.inf, plt_tmax=np.inf,
                             resampling_fct=resampling_fct, plt_full_obj=True, plt_full_time=False, )
        #
        if save_sub_fig:
            t1 = np.linspace(prb1.t0, prb1.t1, 11)
            for i0, (plt_tmin, plt_tmax) in enumerate(zip(t1[:-1], t1[1:])):
                filename = "%s/fig_%s_%d.png" % (prb1.name, prb1.name, i0)
                sps.save_fig_fun(filename, prb1, sps.core_trajectory2D, figsize=figsize, dpi=dpi, plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                                 resampling_fct=resampling_fct, plt_full_obj=True, plt_full_time=False, )
    return True


def export_avrPhaseVelocity(prb1: problemClass._baseProblem, tavr=10, **kwargs):
    OptDB = PETSc.Options()
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    
    # setup
    export_avrPhaseVelocity = OptDB.getBool("export_avrPhaseVelocity", True)
    if rank == 0 and export_avrPhaseVelocity:
        figsize = np.array((16, 9)) * 0.4
        dpi = 400
        plt_tmin, plt_tmax = -1, prb1.t_hist.max()
        resampling_fct, interp1d_kind = 1, "linear"
        cmap = plt.get_cmap("viridis")
        avrW_name = "%s/avrW_%s.png" % (prb1.name, prb1.name)
        sps.save_fig_fun(avrW_name, prb1, sps.core_avrPhaseVelocity, figsize=figsize, dpi=dpi, plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                         resampling_fct=resampling_fct, cmap=cmap, tavr=tavr, )
    return True


def export_20220629(prb1: problemClass._baseProblem, tavr=10, **kwargs):
    OptDB = PETSc.Options()
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    
    # setup
    export_20220629 = OptDB.getBool("export_20220629", False)
    if rank == 0 and export_20220629:
        figsize = np.array((16, 9)) * 0.3
        dpi, plt_tmin, plt_tmax = 400, -1, prb1.t_hist.max()
        resampling_fct, interp1d_kind = 1, "linear"
        # ----------------------------------
        cmap = sps.twilight_diverging()
        fig_name = "%s/avrW_%s.png" % (prb1.name, prb1.name)
        sps.save_fig_fun(fig_name, prb1, sps.core_avrPhaseVelocity, figsize=figsize, dpi=dpi, plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                         resampling_fct=resampling_fct, cmap=cmap, vmin=None, vmax=None, npabs=False, tavr=tavr, )
        # ----------------------------------
        markevery, linestyle = (0.3, "o-C1",)
        fig_name = "%s/orderR_%s.png" % (prb1.name, prb1.name)
        sps.save_fig_fun(fig_name, prb1, sps.core_polar_order, figsize=figsize, dpi=dpi, plt_tmin=plt_tmin, plt_tmax=plt_tmax, markevery=markevery,
                         linestyle=linestyle, )
        # ----------------------------------
        cmap = plt.get_cmap("twilight_shifted")
        fig_name = "%s/avrPhi1_%s.png" % (prb1.name, prb1.name)
        sps.save_fig_fun(fig_name, prb1, sps.core_avrPhase, figsize=figsize, dpi=dpi, plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                         resampling_fct=resampling_fct, cmap=cmap, tavr=0.01,
                         sort_type="normal", )  # fig_name = '%s/avrPhi2_%s.png' % (prb1.name, prb1.name)  # sps.save_fig_fun(fig_name, prb1, sps.core_avrPhase, figsize=figsize, dpi=dpi,  #                  plt_tmin=plt_tmin, plt_tmax=plt_tmax,  #                  resampling_fct=resampling_fct, cmap=cmap,  #                  tavr=0.01, sort_type='traveling')
    return True


def export_ackermann(prb1: problemClass._baseProblem, tavr=10, **kwargs):
    OptDB = PETSc.Options()
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    
    # setup
    export_ackermann = OptDB.getBool("export_ackermann", False)
    if rank == 0 and export_ackermann:
        figsize = np.array((16, 9)) * 0.3
        dpi, plt_tmin, plt_tmax = 400, -1, prb1.t_hist.max()
        resampling_fct, interp1d_kind = 1, "linear"
        # ----------------------------------
        markevery, linestyle = (0.3, "o-C1",)
        fig_name = "%s/orderR_%s.png" % (prb1.name, prb1.name)
        sps.save_fig_fun(fig_name, prb1, sps.core_polar_order, figsize=figsize, dpi=dpi,
                         plt_tmin=plt_tmin, plt_tmax=plt_tmax,
                         markevery=markevery, linestyle=linestyle, )
        # # ----------------------------------
        # cmap_W = plt.get_cmap('bwr')
        # cmap_phi = plt.get_cmap("twilight_shifted")
        # filenames = ["%s/avrW_%s.png" % (prb1.name, prb1.name), "%s/avrPhi1_%s.png" % (prb1.name, prb1.name)]
        # sps.save_figs_fun(filenames, prb1, sps.core_phi_W, figsize=figsize, dpi=dpi, plt_tmin=plt_tmin,
        #                   resampling_fct=resampling_fct, interp1d_kind='linear',
        #                   tavr=tavr, sort_type='normal', sort_idx=None,
        #                   cmap_phi=cmap_phi, cmap_W=cmap_W,
        #                   vmin_W=None, vmax_W=None, npabs_W=False, norm_W='Normalize', )
        # # ----------------------------------
        # cmap_Ws = plt.get_cmap('bwr')
        # cmap_phis = plt.get_cmap("twilight_shifted")
        # filenames = ["%s/avrWs_%s.png" % (prb1.name, prb1.name), "%s/avrPhis_%s.png" % (prb1.name, prb1.name)]
        # sps.save_figs_fun(filenames, prb1, sps.core_phis_Ws, figsize=figsize, dpi=dpi, plt_tmin=plt_tmin,
        #                   resampling_fct=resampling_fct, interp1d_kind='linear',
        #                   tavr=tavr, sort_type='normal', sort_idx=None,
        #                   cmap_phis=cmap_phis, cmap_Ws=cmap_Ws,
        #                   vmin_Ws=None, vmax_Ws=None, npabs_Ws=False, norm_Ws='Normalize', )
        # ----------------------------------
        cmap_W = plt.get_cmap('bwr')
        cmap_phi = plt.get_cmap("twilight_shifted")
        cmap_Ws = plt.get_cmap('bwr')
        cmap_phis = plt.get_cmap("twilight_shifted")
        filenames = ["%s/avrW_%s.png" % (prb1.name, prb1.name),
                     "%s/avrPhi1_%s.png" % (prb1.name, prb1.name),
                     "%s/avrWs_%s.png" % (prb1.name, prb1.name),
                     "%s/avrPhis_%s.png" % (prb1.name, prb1.name)]
        sps.save_figs_fun(filenames, prb1, sps.core_phi_W_phis_Ws, figsize=figsize, dpi=dpi, plt_tmin=plt_tmin,
                          resampling_fct=resampling_fct, interp1d_kind='linear',
                          tavr=tavr, sort_type='normal', sort_idx=None,
                          cmap_phi=cmap_phi, cmap_W=cmap_W,
                          vmin_W=None, vmax_W=None, npabs_W=False, norm_W='Normalize',
                          cmap_phis=cmap_phis, cmap_Ws=cmap_Ws,
                          vmin_Ws=None, vmax_Ws=None, npabs_Ws=False, norm_Ws='Normalize', )
    return True


#
@profile(filename="profile_out")
def main_profile(**main_kwargs):
    return main_fun(**main_kwargs)
