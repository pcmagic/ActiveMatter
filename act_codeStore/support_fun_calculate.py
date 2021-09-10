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
# from petsc4py import PETSc
import numpy as np
# import pickle
# import re
from tqdm.notebook import tqdm as tqdm_notebook
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


class _base_doCalculate(baseClass.baseObj):
    @abc.abstractmethod
    def ini_kwargs(self, **kwargs):
        return

    @abc.abstractmethod
    def do_calculate(self, **kwargs):
        return


class _base_do2D(_base_doCalculate):
    def __init__(self, nptc, update_fun='5bs', update_order=(1e-3, 1e-5),
                 overlap_epsilon=1e-2, attract=1, align=1,
                 un=1, ln=1, Xlim=3, seed=None,
                 save_every=1, tqdm_fun=tqdm_notebook,
                 prbHandle=problemClass._base2DProblem,
                 rltHandle=relationClass.relation2D,
                 ptcHandle=particleClass.particle2D,
                 **kwargs):
        super().__init__(**kwargs)

        err_msg = 'wrong parameter nptc. '
        assert nptc > 4, err_msg
        un = self._test_para(un, nptc, 'wrong parameter un. ')
        ln = self._test_para(ln, nptc, 'wrong parameter ln. ')

        prb1 = prbHandle(name='prb1')
        prb1.attract = attract
        prb1.align = align
        prb1.update_fun = update_fun
        prb1.update_order = update_order
        prb1.save_every = save_every
        prb1.tqdm_fun = tqdm_fun

        rlt1 = rltHandle(name='rlt1')
        rlt1.overlap_epsilon = overlap_epsilon
        prb1.relationHandle = rlt1

        np.random.seed(seed)
        for tun, tln in zip(un, ln):
            tptc = ptcHandle(length=tln, name='ptc2D')
            tptc.phi = (np.random.sample((1,)) - 0.5) * 2 * np.pi
            tptc.X = np.random.uniform(-Xlim, Xlim, (2,))
            tptc.u = tun
            prb1.add_obj(tptc)
        self._problem = prb1

    @property
    def problem(self):
        return self._problem

    def _test_para(self, para, nptc, err_msg):
        if np.alltrue(np.isfinite(para)):
            para = np.array(para)
            if para.size == 1:
                para = np.ones(nptc) * para
            assert para.size == nptc, err_msg
        elif para == 'random':
            para = np.random.sample(nptc)
        else:
            raise Exception(err_msg)
        return para

    @abc.abstractmethod
    def addInteraction(self, prb1):
        return

    def do_calculate(self, max_t, ini_t=0, eval_dt=0.1, ):
        prb1 = self.problem
        self.addInteraction(prb1)
        prb1.update_prepare()
        prb1.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
        return prb1


class do_FiniteDipole2D(_base_do2D):
    def addInteraction(self, prb1):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        prb1.add_act(act1)
        act2 = interactionClass.FiniteDipole2D(name='FiniteDipole2D')
        prb1.add_act(act2)
        return True


class do_LimFiniteDipole2D(do_FiniteDipole2D):
    def addInteraction(self, prb1):
        act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
        prb1.add_act(act1)
        act2 = interactionClass.limFiniteDipole2D(name='FiniteDipole2D')
        prb1.add_act(act2)
        return True


class do_actLimFiniteDipole2D(do_LimFiniteDipole2D):
    def addInteraction(self, prb1):
        super().addInteraction(prb1)
        act3 = interactionClass.Attract2D(name='Attract2D')
        prb1.add_act(act3)
        act4 = interactionClass.Align2D(name='Align2D')
        prb1.add_act(act4)
        return True
