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
#
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


# from act_codeStore import support_fun as spf


class _base_doCalculate(baseClass.baseObj):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        PETSc.Sys.Print()
        PETSc.Sys.Print('Collective motion solve, Zhang Ji, 2021. ')
        PETSc.Sys.Print('Generate Problem. ')

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
        un = self._test_para(un, 'speed', nptc, 'wrong parameter un. ')
        ln = self._test_para(ln, 'length', nptc, 'wrong parameter ln. ')

        prb1 = prbHandle(name=fileHandle)
        self._set_problem_property(prb1, **kwargs)

        rlt1 = rltHandle(name='Relation2D')
        rlt1.overlap_epsilon = overlap_epsilon
        prb1.relationHandle = rlt1

        np.random.seed(seed)
        for tun, tln in zip(un, ln):
            tptc = ptcHandle(length=tln, name='ptc2D')
            tptc.phi = (np.random.sample((1,))[0] - 0.5) * 2 * np.pi
            tptc.X = np.random.uniform(-Xlim, Xlim, (2,))
            tptc.u = tun
            prb1.add_obj(tptc)
        self._problem = prb1

    @property
    def problem(self):
        return self._problem

    def _set_problem_property(self, prb1, **kwargs):
        prb1.update_fun = kwargs['update_fun']
        prb1.update_order = kwargs['update_order']
        prb1.save_every = kwargs['save_every']
        prb1.tqdm_fun = kwargs['tqdm_fun']
        return True

    def _test_para(self, para, para_name, nptc, err_msg):
        if np.alltrue(np.isfinite(para)):
            para = np.array(para)
            if para.size == 1:
                PETSc.Sys.Print('  All the particles have a unified %s=%f, ' % (para_name, para))
                para = np.ones(nptc) * para
            else:
                assert para.size == nptc, err_msg
                PETSc.Sys.Print('  The %s of each particle is given. ' % para_name, )
        elif para == 'random':
            para = np.random.sample(nptc)
            PETSc.Sys.Print('  The %s of each particle following an uniform distribution. ' % para_name, )
        else:
            raise Exception(err_msg)
        return para

    @abc.abstractmethod
    def addInteraction(self, prb1):
        return

    def do_calculate(self, max_t, ini_t=0, eval_dt=0.1, ):
        err_msg = 'wrong parameter eval_dt, eval_dt>0. '
        assert eval_dt > 0, err_msg

        prb1 = self.problem
        self.addInteraction(prb1)
        # prb1.update_prepare()
        prb1.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
        return prb1


class do_FiniteDipole2D(_base_do2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun']
    def addInteraction(self, prb1):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        prb1.add_act(act1)
        act2 = interactionClass.FiniteDipole2D(name='FiniteDipole2D')
        prb1.add_act(act2)
        return True


class do_LimFiniteDipole2D(do_FiniteDipole2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun']
    def addInteraction(self, prb1):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        prb1.add_act(act1)
        act2 = interactionClass.limFiniteDipole2D(name='FiniteDipole2D')
        prb1.add_act(act2)
        return True


class do_behaviorParticle2D(_base_do2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun', 'align', 'attract']
    def __init__(self, **kwargs):
        err_msg = 'wrong parameter update_fun, only "1fe" is acceptable. '
        assert kwargs['update_fun'] == "1fe", err_msg
        err_msg = 'wrong parameter update_order, only (0, 0) is acceptable. '
        assert kwargs['update_order'] == (0, 0), err_msg
        err_msg = 'wrong parameter ln, only -1 is acceptable. '
        assert np.isclose(kwargs['ln'], -1), err_msg

        super().__init__(**kwargs)

    def addInteraction(self, prb1):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        prb1.add_act(act1)
        act3 = interactionClass.Attract2D(name='Attract2D')
        prb1.add_act(act3)
        act4 = interactionClass.Align2D(name='Align2D')
        prb1.add_act(act4)
        return True

    def _set_problem_property(self, prb1: 'problemClass.behavior2DProblem', **kwargs):
        super()._set_problem_property(prb1, **kwargs)
        prb1.align = kwargs['align']
        prb1.attract = kwargs['attract']
        return True


class do_behaviorWienerParticle2D(do_behaviorParticle2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun',
    #                     'align', 'attract',
    #                     'rot_noise', 'trs_noise']
    def addInteraction(self, prb1):
        super().addInteraction(prb1)
        act5 = interactionClass.Wiener2D(name='Wiener2D')
        prb1.add_act(act5)
        return True

    def _set_problem_property(self, prb1: 'problemClass.behavior2DProblem', **kwargs):
        super()._set_problem_property(prb1, **kwargs)
        prb1.rot_noise = kwargs['rot_noise']
        prb1.trs_noise = kwargs['trs_noise']
        return True


class do_actLimFiniteDipole2D(do_LimFiniteDipole2D):
    # kwargs_necessary = ['update_fun', 'update_order', 'save_every', 'tqdm_fun']
    def addInteraction(self, prb1):
        super().addInteraction(prb1)
        act3 = interactionClass.Attract2D(name='Attract2D')
        prb1.add_act(act3)
        act4 = interactionClass.Align2D(name='Align2D')
        prb1.add_act(act4)
        return True
