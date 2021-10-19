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
from act_src import relationClass
from act_src import particleClass
from act_src import interactionClass
from act_codeStore import support_fun as spf


# from act_codeStore import support_fun as spf


class _base_doCalculate(baseClass.baseObj):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def ini_kwargs(self, **kwargs):
        return

    @abc.abstractmethod
    def do_calculate(self, **kwargs):
        return


class _base_do2D(_base_doCalculate):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun']
    def __init__(self, nptc, overlap_epsilon=1e-2, un=1, ln=1, Xlim=3, seed=None, fileHandle='...',
                 prbHandle=problemClass._base2DProblem,
                 rltHandle=relationClass._baseRelation2D,
                 ptcHandle=particleClass.particle2D,
                 **kwargs):
        super().__init__(**kwargs)
        err_msg = 'wrong parameter nptc, at least 5 particles (nptc > 4).  '
        assert nptc > 4, err_msg
        self._problem = prbHandle(name=fileHandle)
        spf.petscInfo(self.problem.logger, '#' * 72)
        spf.petscInfo(self.problem.logger, 'Generate Problem. ')

        self._nptc = nptc
        self._overlap_epsilon = overlap_epsilon
        self._un = self._test_para(un, 'speed', nptc, 'wrong parameter un. ')
        self._ln = self._test_para(ln, 'length', nptc, 'wrong parameter ln. ')
        self._Xlim = Xlim
        self._seed = seed
        self._fileHandle = fileHandle
        self._prbHandle = prbHandle
        self._rltHandle = rltHandle
        self._ptcHandle = ptcHandle

        self._set_problem(**kwargs)
        self._set_relation()
        self._set_particle()

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
        return True

    def _set_relation(self):
        self.problem.relationHandle = self.rltHandle(name='Relation2D')
        self.problem.relationHandle.overlap_epsilon = self.overlap_epsilon
        return True

    def _set_particle(self):
        np.random.seed(self.seed)
        for tun, tln in zip(self.un, self.ln):
            tptc = self.ptcHandle(length=tln, name='ptc2D')
            tptc.phi = (np.random.sample((1,))[0] - 0.5) * 2 * np.pi
            tptc.X = np.random.uniform(-self.Xlim, self.Xlim, (2,))
            tptc.u = tun
            self.problem.add_obj(tptc)
        return True

    @abc.abstractmethod
    def addInteraction(self):
        return

    def do_calculate(self, max_t, ini_t=0, eval_dt=0.1, ):
        err_msg = 'wrong parameter eval_dt, eval_dt>0. '
        assert eval_dt > 0, err_msg

        self.addInteraction()
        self.problem.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
        return self.problem


class do_FiniteDipole2D(_base_do2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun']
    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.FiniteDipole2D(name='FiniteDipole2D')
        self.problem.add_act(act2)
        return True


class do_LimFiniteDipole2D(do_FiniteDipole2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun']
    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        act2 = interactionClass.limFiniteDipole2D(name='FiniteDipole2D')
        self.problem.add_act(act2)
        return True


class do_behaviorParticle2D(_base_do2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun', 'align', 'attract']
    def __init__(self, *args, **kwargs):
        err_msg = 'wrong parameter update_fun, only "1fe" is acceptable. '
        assert kwargs['update_fun'] == "1fe", err_msg
        err_msg = 'wrong parameter update_order, only (0, 0) is acceptable. '
        assert kwargs['update_order'] == (0, 0), err_msg
        err_msg = 'wrong parameter ln, only -1 is acceptable. '
        assert np.isclose(kwargs['ln'], -1), err_msg

        super().__init__(*args, **kwargs)

    def addInteraction(self):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        self.problem.add_act(act1)
        # act3 = interactionClass.Attract2D(name='Attract2D')
        # self.problem.add_act(act3)
        # act4 = interactionClass.Align2D(name='Align2D')
        # self.problem.add_act(act4)
        act5 = interactionClass.AlignAttract2D(name='AlignAttract2D')
        self.problem.add_act(act5)
        return True

    def _set_problem(self, **kwargs):
        super()._set_problem(**kwargs)
        self.problem.align = kwargs['align']
        self.problem.attract = kwargs['attract']
        return True


class do_behaviorWienerParticle2D(do_behaviorParticle2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun',
    #                     'align', 'attract',
    #                     'rot_noise', 'trs_noise']
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
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun',
    #                     'align', 'attract',
    #                     'rot_noise', 'trs_noise']

    # void initialize_particles() {
    #     int i, j;
    #     double dx, dy, dr, dr2;
    #     double tempx, tempy, temp_ori;
    #     srand(1);
    #
    #     particles = (struct particles_struct * ) calloc(N, sizeof(struct particles_struct));
    #
    #     for (i = 0; i < N; i++) {
    #         particles[i].ID = i + 1;
    #         particles[i].x = SX * rand() / (RAND_MAX + 1.0);
    #         particles[i].y = SY * rand() / (RAND_MAX + 1.0);
    #         particles[i].ori = 2 * PI * rand() / (RAND_MAX + 1.0);
    #         printf("%3d, %15.10f, %15.10f, %15.10f \n",
    #             particles[i].ID, particles[i].x, particles[i].y, particles[i].ori);
    #     }
    # }

    def _set_particle(self):
        libc = CDLL("libc.so.6")
        libc.srand(self.seed)
        for tun, tln in zip(self.un, self.ln):
            tptc = self.ptcHandle(length=tln, name='ptc2D')
            tptc.X = np.array((self.Xlim * libc.rand() / (RAND_MAX + 1.0),
                               self.Xlim * libc.rand() / (RAND_MAX + 1.0)))
            tptc.phi = np.float64(spf.warpToPi(2 * np.pi * libc.rand() / (RAND_MAX + 1.0)))
            tptc.u = tun
            self.problem.add_obj(tptc)
            # t1 = tptc.phi if tptc.phi > 0 else 2 * np.pi + tptc.phi
            # print("%3d, %15.10f, %15.10f, %15.10f" %
            #       (tptc.index, tptc.X[0], tptc.X[1], t1))
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
        act5 = interactionClass.AlignAttract2D(name='AlignAttract2D')
        self.problem.add_act(act5)

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


class do_actLimFiniteDipole2D(do_LimFiniteDipole2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun']
    def addInteraction(self, ):
        super().addInteraction()
        # act3 = interactionClass.Attract2D(name='Attract2D')
        # self.problem.add_act(act3)
        # act4 = interactionClass.Align2D(name='Align2D')
        # self.problem.add_act(act4)
        act5 = interactionClass.AlignAttract2D(name='AlignAttract2D')
        self.problem.add_act(act5)
        return True
