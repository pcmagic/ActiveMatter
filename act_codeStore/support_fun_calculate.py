# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20210910

@author: Zhang Ji
"""
import abc
# import matplotlib
# import subprocess
# import os
from ctypes import CDLL

RAND_MAX = 2147483647

from petsc4py import PETSc
import numpy as np
import poisson_disc as psd
# import pickle
# import re
# from tqdm.notebook import tqdm as tqdm_notebook
# from tqdm import tqdm
# from scipy import interpolate
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
# from matplotlib import colors as mcolors

from act_src import baseClass
from act_src import problemClass
# from act_src import relationClass
# from act_src import particleClass
from act_src import interactionClass
from act_codeStore import support_fun as spf


# from act_codeStore import support_fun as spf


class _base_doCalculate(baseClass.baseObj):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._kwargs_necessary = []

    @abc.abstractmethod
    def ini_kwargs(self):
        return True

    @abc.abstractmethod
    def do_calculate(self, **kwargs):
        return


class _base_do2D(_base_doCalculate):
    def ini_kwargs(self):
        super().ini_kwargs()
        self._kwargs_necessary = self._kwargs_necessary + ['update_fun', 'update_order', 'save_every', 'tqdm_fun']
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ini_kwargs()
        for tkwargs in self._kwargs_necessary:
            err_msg = 'lost necessary parameter: %s' % tkwargs
            assert tkwargs in kwargs.keys(), err_msg

        self._nptc = kwargs['nptc']
        self._overlap_epsilon = kwargs['overlap_epsilon']
        self._un = kwargs['un']
        self._ln = kwargs['ln']
        self._Xlim = kwargs['Xlim']
        self._seed = kwargs['seed']
        self._fileHandle = kwargs['fileHandle']
        self._prbHandle = kwargs['prbHandle']
        self._rltHandle = kwargs['rltHandle']
        self._ptcHandle = kwargs['ptcHandle']
        self._problem = None

    @property
    def problem(self):
        return self._problem

    @property
    def nptc(self):
        return self._nptc

    @property
    def overlap_epsilon(self):
        return self._overlap_epsilon

    @property
    def un(self):
        return self._un

    @property
    def ln(self):
        return self._ln

    @property
    def Xlim(self):
        return self._Xlim

    @property
    def seed(self):
        return self._seed

    @property
    def fileHandle(self):
        return self._fileHandle

    @property
    def prbHandle(self):
        return self._prbHandle

    @property
    def rltHandle(self):
        return self._rltHandle

    @property
    def ptcHandle(self):
        return self._ptcHandle

    def _test_para(self, para, para_name, nptc, err_msg):
        if np.alltrue(np.isfinite(para)):
            para = np.array(para)
            if para.size == 1:
                spf.petscInfo(self.problem.logger, '  All the particles have a unified %s=%f, ' % (para_name, para))
                para = np.ones(nptc) * para
            else:
                assert para.size == nptc, err_msg
                spf.petscInfo(self.problem.logger, '  The %s of each particle is given. ' % para_name, )
        elif para == 'random':
            para = np.random.sample(nptc)
            spf.petscInfo(self.problem.logger,
                          '  The %s of each particle following an uniform distribution. ' % para_name, )
        else:
            raise Exception(err_msg)
        return para

    def _set_problem(self, **kwargs):
        self.problem.update_fun = kwargs['update_fun']
        self.problem.update_order = kwargs['update_order']
        self.problem.save_every = kwargs['save_every']
        self.problem.tqdm_fun = kwargs['tqdm_fun']
        self.problem.do_save = True
        return True

    def _set_relation(self):
        self.problem.relationHandle = self.rltHandle(name='Relation2D', **self.kwargs)
        return True

    def _set_particle(self):
        np.random.seed(self.seed)
        for tun, tln in zip(self.un, self.ln):
            tptc = self.ptcHandle(length=tln, name='ptc2D')
            tptc.phi = (np.random.sample((1,))[0] - 0.5) * 2 * np.pi
            # tptc.X = np.random.uniform(-self.Xlim, self.Xlim, (2,))
            tptc.X = np.random.uniform(-self.Xlim / 2, self.Xlim / 2, (2,))
            tptc.u = tun
            tptc.w = 0
            self.problem.add_obj(tptc)
        spf.petscInfo(self.problem.logger, '  All the particles have a unified %s=%f, ' % ('spin', 0))
        spf.petscInfo(self.problem.logger, '  Generate %d particles with random seed %s' % (self.un.size, self.seed), )
        spf.petscInfo(self.problem.logger, '  Generate method: random_sample. ')
        return True

    @abc.abstractmethod
    def addInteraction(self):
        return

    def ini_calculate(self):
        kwargs = self.kwargs
        self._problem = self.prbHandle(name=self.fileHandle, **kwargs)
        spf.petscInfo(self.problem.logger, '#' * 72)
        spf.petscInfo(self.problem.logger, 'Generate Problem. ')
        self._un = self._test_para(self.un, 'speed', self.nptc, 'wrong parameter un. ')
        self._ln = self._test_para(self.ln, 'length', self.nptc, 'wrong parameter ln. ')
        self._set_problem(**kwargs)
        self._set_relation()
        self._set_particle()
        self.addInteraction()
        return self.problem

    def do_calculate(self, max_t, ini_t=0, eval_dt=0.1, ):
        err_msg = 'wrong parameter eval_dt, eval_dt>0. '
        assert eval_dt > 0, err_msg
        # err_msg = 'wrong parameter nptc, at least 5 particles (nptc > 4).  '
        # assert nptc > 4, err_msg

        problem = self.ini_calculate()
        problem.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
        return problem


class do_FiniteDipole2D(_base_do2D):
    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.FiniteDipole2D(name='FiniteDipole2D')
        self.problem.add_act(act2)
        return True


class do_LimFiniteDipole2D(do_FiniteDipole2D):
    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.limFiniteDipole2D(name='FiniteDipole2D')
        self.problem.add_act(act2)
        return True


class do_behaviorParticle2D(_base_do2D):
    def ini_kwargs(self):
        super().ini_kwargs()
        err_msg = 'wrong parameter ln, only -1 is acceptable. '
        assert np.isclose(self.kwargs['ln'], -1), err_msg
        self._kwargs_necessary = self._kwargs_necessary + ['align', 'attract', 'viewRange']
        return True

    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        # act3 = interactionClass.Attract2D(name='Attract2D')
        # self.problem.add_act(act3)
        # act4 = interactionClass.Align2D(name='Align2D')
        # self.problem.add_act(act4)
        act6 = interactionClass.AlignAttract2D(name='AlignAttract2D')
        self.problem.add_act(act6)
        return True

    def _set_problem(self, **kwargs):
        super()._set_problem(**kwargs)
        self.problem.align = kwargs['align']
        self.problem.attract = kwargs['attract']
        self.problem.viewRange = kwargs['viewRange']
        return True


class dbg_behaviorParticle2D(do_behaviorParticle2D):
    def nothing(self):
        pass


class do_behaviorWienerParticle2D(do_behaviorParticle2D):
    def ini_kwargs(self):
        super().ini_kwargs()
        err_msg = 'wrong parameter update_fun, only "1fe" is acceptable. '
        assert self.kwargs['update_fun'] == "1fe", err_msg
        err_msg = 'wrong parameter update_order, only (0, 0) is acceptable. '
        assert self.kwargs['update_order'] == (0, 0), err_msg
        self._kwargs_necessary = self._kwargs_necessary + ['rot_noise', 'trs_noise']
        return True

    def addInteraction(self):
        super().addInteraction()
        act5 = interactionClass.Wiener2D(name='Wiener2D')
        self.problem.add_act(act5)
        return True

    def _set_problem(self, **kwargs):
        super()._set_problem(**kwargs)
        self.problem.rot_noise = kwargs['rot_noise']
        self.problem.trs_noise = kwargs['trs_noise']
        return True


class do_dbgBokaiZhang(do_behaviorParticle2D):
    def _set_particle(self):
        libc = CDLL("libc.so.6")
        libc.srand(self.seed)
        for tun, tln in zip(self.un, self.ln):
            tptc = self.ptcHandle(length=tln, name='ptc2D')
            tptc.X = np.array((self.Xlim / 2 * libc.rand() / (RAND_MAX + 1.0),
                               self.Xlim / 2 * libc.rand() / (RAND_MAX + 1.0)))
            tptc.phi = np.float64(spf.warpToPi(2 * np.pi * libc.rand() / (RAND_MAX + 1.0)))
            tptc.u = tun
            self.problem.add_obj(tptc)
            # t1 = tptc.phi if tptc.phi > 0 else 2 * np.pi + tptc.phi
            # print("%3d, %15.10f, %15.10f, %15.10f" %
            #       (tptc.index, tptc.X[0], tptc.X[1], t1))
        spf.petscInfo(self.problem.logger, '  Generate %d particles with random seed %s' % (self.un.size, self.seed), )
        return True

    def dbg_Attract2D(self):
        act3 = interactionClass.Attract2D(name='Attract2D')
        self.problem.add_act(act3)

        self.problem.Xall = np.vstack([objj.X for objj in self.problem.obj_list])
        self.problem.relationHandle.update_relation()
        self.problem.relationHandle.update_neighbor()
        self.problem.relationHandle.check_self()
        act3.update_prepare()
        Uall, Wall = act3.update_action_numpy()
        for tptc, wi in zip(self.problem.obj_list, Wall):
            t1 = tptc.phi if tptc.phi > 0 else 2 * np.pi + tptc.phi
            print("%3d, %15.10f, %15.10f, %15.10f, %15.10f" %
                  (tptc.index, tptc.X[0], tptc.X[1], t1, wi))
        return True

    def dbg_Align2D(self):
        act4 = interactionClass.Align2D(name='Align2D')
        self.problem.add_act(act4)

        self.problem.Xall = np.vstack([objj.X for objj in self.problem.obj_list])
        self.problem.relationHandle.update_relation()
        self.problem.relationHandle.update_neighbor()
        self.problem.relationHandle.check_self()
        act4.update_prepare()
        Uall, Wall = act4.update_action_numpy()
        for tptc, wi in zip(self.problem.obj_list, Wall):
            t1 = tptc.phi if tptc.phi > 0 else 2 * np.pi + tptc.phi
            print("%3d, %15.10f, %15.10f, %15.10f, %15.10f" %
                  (tptc.index, tptc.X[0], tptc.X[1], t1, wi))
        return True

    def dbg_AlignAttract2D(self):
        act6 = interactionClass.AlignAttract2D(name='AlignAttract2D')
        self.problem.add_act(act6)

        self.problem.Xall = np.vstack([objj.X for objj in self.problem.obj_list])
        self.problem.relationHandle.update_relation()
        self.problem.relationHandle.update_neighbor()
        self.problem.relationHandle.check_self()
        act5.update_prepare()
        _, Wall = act5.update_action_numpy()
        for tptc, wi in zip(self.problem.obj_list, Wall):
            t1 = tptc.phi if tptc.phi > 0 else 2 * np.pi + tptc.phi
            print("%3d, %15.10f, %15.10f, %15.10f, %15.10f" %
                  (tptc.index, tptc.X[0], tptc.X[1], t1, wi))
        return True


class do_actLimFiniteDipole2D(do_behaviorParticle2D, do_LimFiniteDipole2D):
    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.limFiniteDipole2D(name='FiniteDipole2D')
        self.problem.add_act(act2)
        act6 = interactionClass.AlignAttract2D(name='AlignAttract2D')
        self.problem.add_act(act6)
        return True


class do_actLight2D(do_behaviorParticle2D):
    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        actLight = interactionClass.lightAttract2D(name='lightAttract2D')
        self.problem.add_act(actLight)
        return True


class do_phaseLag2D(do_behaviorParticle2D):
    def ini_kwargs(self):
        super().ini_kwargs()
        self._kwargs_necessary = self._kwargs_necessary + ['phaseLag2D', 'localRange']
        return True

    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.phaseLag2D(name='phaseLag2D', phaseLag=self.kwargs['phaseLag2D'])
        self.problem.add_act(act2)
        return True


class do_phaseLag2D_dbg(do_phaseLag2D):
    def ini_kwargs(self):
        super().ini_kwargs()
        err_msg = 'wrong parameter update_fun, only "1fe" is acceptable. '
        assert self.kwargs['update_fun'] == "1fe", err_msg
        err_msg = 'wrong parameter update_order, only (0, 0) is acceptable. '
        assert self.kwargs['update_order'] == (0, 0), err_msg
        self._kwargs_necessary = self._kwargs_necessary + ['rot_noise', 'trs_noise']
        return True

    def addInteraction(self):
        act1 = interactionClass.selfSpeed2D(name='selfSpeed2D')
        self.problem.add_act(act1)
        act2 = interactionClass.phaseLag2D(name='phaseLag2D', phaseLag=self.kwargs['phaseLag2D'])
        self.problem.add_act(act2)
        act5 = interactionClass.Wiener2D(name='Wiener2D')
        self.problem.add_act(act5)
        return True

    def _set_problem(self, **kwargs):
        super()._set_problem(**kwargs)
        self.problem.rot_noise = kwargs['rot_noise']
        self.problem.trs_noise = kwargs['trs_noise']
        return True


class do_phaseLagWiener2D(do_phaseLag2D):
    def ini_kwargs(self):
        super().ini_kwargs()
        err_msg = 'wrong parameter update_fun, only "1fe" is acceptable. '
        assert self.kwargs['update_fun'] == "1fe", err_msg
        err_msg = 'wrong parameter update_order, only (0, 0) is acceptable. '
        assert self.kwargs['update_order'] == (0, 0), err_msg
        self._kwargs_necessary = self._kwargs_necessary + ['rot_noise', 'trs_noise']
        return True

    def addInteraction(self):
        super().addInteraction()
        act5 = interactionClass.Wiener2D(name='Wiener2D')
        self.problem.add_act(act5)
        return True

    def _set_problem(self, **kwargs):
        super()._set_problem(**kwargs)
        self.problem.rot_noise = kwargs['rot_noise']
        self.problem.trs_noise = kwargs['trs_noise']
        return True


class do_phaseLag2D_Wiener(do_phaseLag2D):
    def ini_kwargs(self):
        super().ini_kwargs()
        err_msg = 'wrong parameter update_fun, only "1fe" is acceptable. '
        assert self.kwargs['update_fun'] == "1fe", err_msg
        err_msg = 'wrong parameter update_order, only (0, 0) is acceptable. '
        assert self.kwargs['update_order'] == (0, 0), err_msg
        self._kwargs_necessary = self._kwargs_necessary + ['phaseLag_rdm_fct']
        return True

    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.phaseLag2D_Wiener(name='phaseLag2D_Wiener',
                                                  phaseLag=self.kwargs['phaseLag2D'],
                                                  phaseLag_rdm_fct=self.kwargs['phaseLag_rdm_fct'], )
        self.problem.add_act(act2)
        return True


class do_actPeriodic2D(do_behaviorParticle2D):
    def ini_kwargs(self):
        super().ini_kwargs()
        err_msg = 'wrong parameter ln, Xrange >= Xlim. '
        assert self.kwargs['Xrange'] >= self.kwargs['Xlim'], err_msg
        self._kwargs_necessary = self._kwargs_necessary + ['Xrange']
        return True

    def _set_problem(self, **kwargs):
        super()._set_problem(**kwargs)
        err_msg = 'wrong problem handle, current: %s ' % str(self.problem)
        assert isinstance(self.problem, problemClass.actPeriodic2DProblem), err_msg
        return True


class do_phaseLagPeriodic2D(do_phaseLag2D, do_actPeriodic2D):
    def chimera_cycle(self, chimera_cycle_radius):
        err_msg = 'chimera_cycle_radius=%f does not in the range (0, %f)' % (chimera_cycle_radius, self.Xlim)
        assert chimera_cycle_radius <= self.Xlim, err_msg

        np.random.seed(self.seed)
        for tun, tln in zip(self.un, self.ln):
            tptc = self.ptcHandle(length=tln, name='ptc2D')
            tptc.phi = (np.random.sample((1,))[0] - 0.5) * 2 * np.pi
            t1 = tptc.phi - np.pi / 2
            tptc.X = np.hstack((np.cos(t1), np.sin(t1))) * chimera_cycle_radius
            tptc.u = tun
            self.problem.add_obj(tptc)
        spf.petscInfo(self.problem.logger, '  Generate %d particles with random seed %s' % (self.un.size, self.seed), )
        spf.petscInfo(self.problem.logger,
                      '  Generate method: chimera_cycle, chimera_cycle_radius=%f' % chimera_cycle_radius)

        # # dbg
        # self.problem.Xall = np.vstack([objj.X for objj in self.problem.obj_list])
        # self.problem.Phiall = np.vstack([objj.phi for objj in self.problem.obj_list])
        # # self.problem.show_particle_location()
        # self.problem.show_particle_directon()
        # assert 1 == 2
        return True

    def chimera_overlap(self):
        points = psd.Bridson_sampling(dims=np.ones(2) * self.Xlim, radius=self.overlap_epsilon * 2, k=30,
                                      hypersphere_sample=psd.hypersphere_surface_sample)
        # print('dbg', points.shape)
        tidx_list = np.random.choice(points.shape[0], self.nptc, replace=False)
        for tun, tln, tidx in zip(self.un, self.ln, tidx_list):
            tptc = self.ptcHandle(length=tln, name='ptc2D')
            tptc.phi = (np.random.sample((1,))[0] - 0.5) * 2 * np.pi
            tptc.X = points[tidx]
            tptc.u = tun
            self.problem.add_obj(tptc)
        spf.petscInfo(self.problem.logger, '  Generate %d particles with random seed %s' % (self.un.size, self.seed), )
        spf.petscInfo(self.problem.logger, '  Generate method: chimera_overlap. ')
        return True

    def _set_particle(self):
        OptDB = PETSc.Options()
        chimera_particle_method = OptDB.getString('chimera_particle_method', 'random_sample')
        # chimera_particle_method = OptDB.getString('chimera_particle_method', 'chimera_overlap')

        if chimera_particle_method == 'random_sample':
            super()._set_particle()
        elif chimera_particle_method == 'chimera_cycle':
            chimera_cycle_radius = np.float64(OptDB.getReal('chimera_cycle_radius', 1))
            self.chimera_cycle(chimera_cycle_radius)
        elif chimera_particle_method == 'chimera_overlap':
            self.chimera_overlap()
        else:
            raise ValueError('wrong chimera_particle_method, current: %s' % chimera_particle_method)
        return True


class do_phaseLagPeriodic2D_LJ(do_phaseLagPeriodic2D):
    def ini_kwargs(self):
        super().ini_kwargs()
        self._kwargs_necessary = self._kwargs_necessary + ['LJ_A', 'LJ_B', 'LJ_a', 'LJ_b', ]
        return True

    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.phaseLag2D(name='phaseLag2D', phaseLag=self.kwargs['phaseLag2D'])
        self.problem.add_act(act2)
        act3 = interactionClass.LennardJonePotential2D_point(name='LJ2D_point',
                                                             A=self.kwargs['LJ_A'],
                                                             B=self.kwargs['LJ_B'],
                                                             a=self.kwargs['LJ_a'],
                                                             b=self.kwargs['LJ_b'])
        self.problem.add_act(act3)
        return True


class do_phaseLag2D_AR(do_phaseLag2D):
    def ini_kwargs(self):
        super().ini_kwargs()
        self._kwargs_necessary = self._kwargs_necessary + ['AR_k1', 'AR_k2', 'AR_k3', 'AR_k4', ]
        return True

    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.phaseLag2D(name='phaseLag2D', phaseLag=self.kwargs['phaseLag2D'])
        self.problem.add_act(act2)
        act3 = interactionClass.AttractRepulsion2D_point(name='AR2D_point',
                                                         k1=self.kwargs['AR_k1'],
                                                         k2=self.kwargs['AR_k2'],
                                                         k3=self.kwargs['AR_k3'],
                                                         k4=self.kwargs['AR_k4'])
        self.problem.add_act(act3)
        return True


# class do_phaseLag2D_AR_Voronoi(do_phaseLag2D):
#     def ini_kwargs(self):
#         super().ini_kwargs()
#         self._kwargs_necessary = self._kwargs_necessary + ['AR_k1', 'AR_k2', 'AR_k3', 'AR_k4', ]
#         return True
#
#     def addInteraction(self):
#         act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
#         self.problem.add_act(act1)
#         act2 = interactionClass.phaseLag2D(name='phaseLag2D', phaseLag=self.kwargs['phaseLag2D'])
#         self.problem.add_act(act2)
#         act3 = interactionClass.AttractRepulsion2D_point(name='AR2D_point',
#                                                          k1=self.kwargs['AR_k1'],
#                                                          k2=self.kwargs['AR_k2'],
#                                                          k3=self.kwargs['AR_k3'],
#                                                          k4=self.kwargs['AR_k4'])
#         self.problem.add_act(act3)
#         return True


class do_phaseLagWiener2D_AR(do_phaseLag2D_AR):
    def ini_kwargs(self):
        super().ini_kwargs()
        err_msg = 'wrong parameter update_fun, only "1fe" is acceptable. '
        assert self.kwargs['update_fun'] == "1fe", err_msg
        err_msg = 'wrong parameter update_order, only (0, 0) is acceptable. '
        assert self.kwargs['update_order'] == (0, 0), err_msg
        self._kwargs_necessary = self._kwargs_necessary + ['rot_noise', 'trs_noise']
        return True

    def addInteraction(self):
        super().addInteraction()
        act5 = interactionClass.Wiener2D(name='Wiener2D')
        self.problem.add_act(act5)
        return True

    def _set_problem(self, **kwargs):
        super()._set_problem(**kwargs)
        self.problem.rot_noise = kwargs['rot_noise']
        self.problem.trs_noise = kwargs['trs_noise']
        return True


# class do_phaseLagWiener2D_AR_Voronoi(do_phaseLagWiener2D_AR):
#     # def ini_kwargs(self):
#     #     super().ini_kwargs()
#     #     err_msg = 'wrong parameter update_fun, only "1fe" is acceptable. '
#     #     assert self.kwargs['update_fun'] == "1fe", err_msg
#     #     err_msg = 'wrong parameter update_order, only (0, 0) is acceptable. '
#     #     assert self.kwargs['update_order'] == (0, 0), err_msg
#     #     self._kwargs_necessary = self._kwargs_necessary + ['rot_noise', 'trs_noise']
#     #     return True
#
#     def addInteraction(self):
#         super().addInteraction()
#         act5 = interactionClass.Wiener2D(name='Wiener2D')
#         self.problem.add_act(act5)
#         return True
#
#     def _set_problem(self, **kwargs):
#         super()._set_problem(**kwargs)
#         self.problem.rot_noise = kwargs['rot_noise']
#         self.problem.trs_noise = kwargs['trs_noise']
#         return True

class do_dbg_action(_base_do2D):
    def _set_particle(self):
        np.random.seed(self.seed)
        # robots
        for tun, tln in zip(self.un, self.ln):
            tptc = self.ptcHandle(length=tln, name='ptc2D')
            tptc.phi = (np.random.sample((1,))[0] - 0.5) * 2 * np.pi
            # tptc.X = np.random.uniform(-self.Xlim, self.Xlim, (2,))
            tptc.X = np.random.uniform(-self.Xlim / 2, self.Xlim / 2, (2,))
            tptc.u = tun
            self.problem.add_obj(tptc)
        spf.petscInfo(self.problem.logger, '  Generate %d particles with random seed %s' % (self.un.size, self.seed), )
        spf.petscInfo(self.problem.logger, '  Generate method: random_sample. ')

        # target
        ##############
        return True

    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.FiniteDipole2D(name='FiniteDipole2D')
        self.problem.add_act(act2)
        return True

    def do_calculate(self, max_t, ini_t=0, eval_dt=0.1, ):
        err_msg = 'wrong parameter eval_dt, eval_dt>0. '
        assert eval_dt > 0, err_msg

        self.addInteraction()
        self.problem.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
        return self.problem
