"""
20210810
Zhang Ji

calculate the interactions
"""
import abc

import numpy as np
from petsc4py import PETSc

from act_src import baseClass
from act_src import particleClass
from act_src import problemClass
from act_src import relationClass
from act_codeStore.support_class import *


# from act_codeStore.support_class import *


class _baseAction(baseClass.baseObj):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._obj_list = uniqueList(acceptType=particleClass.baseParticle)  # contain objects
        self._dimension = -1  # -1 for undefined, 2 for 2D, 3 for 3D
        self._dmda = None

    @property
    def obj_list(self):
        return self._obj_list

    @property
    def n_obj(self):
        return len(self.obj_list)

    @property
    def dimension(self):
        return self._dimension

    @property
    def dmda(self):
        return self._dmda

    def _check_add_obj(self, obj):
        err_msg = 'wrong object type'
        assert isinstance(obj, particleClass.baseParticle), err_msg
        err_msg = 'wrong dimension'
        assert np.isclose(self.dimension, obj.dimension), err_msg
        return True

    def add_obj(self, obj):
        self._check_add_obj(obj)
        self._obj_list.append(obj)
        return True

    def check_self(self):
        pass
        return True

    def update_finish(self):
        pass
        return True

    def set_dmda(self):
        self._dmda = PETSc.DMDA().create(sizes=(self.n_obj,), dof=1, stencil_width=0, comm=PETSc.COMM_WORLD)
        self._dmda.setFromOptions()
        self._dmda.setUp()
        return True

    @abc.abstractmethod
    def update_each_action(self, obji: "particleClass.baseParticle", **kwargs):
        return 0, 0

    @abc.abstractmethod
    def update_action(self):
        return 0, 0

    # def update_action(self):
    #     Uall, Wall = [], []
    #     for obji in self.obj_list:
    #         u, w = self.update_each_action(obji)
    #         Uall.append(u)
    #         Wall.append(w)
    #     Uall = np.hstack(Uall)
    #     Wall = np.hstack(Wall)
    #     return Uall, Wall


class _baseAction2D(_baseAction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dimension = 2  # 2 for 2D
        self._obj_list = uniqueList(acceptType=particleClass.particle2D)  # contain objects

    def update_action(self):
        # Uall, Wall = [], []
        # for obji in self.obj_list:
        #     u, w = self.update_each_action(obji)
        #     Uall.append(u)
        #     Wall.append(w)
        # Uall = np.hstack(Uall)
        # Wall = np.hstack(Wall)

        nobj = self.n_obj
        dimension = self.dimension
        dmda = self.dmda
        obj_list = self.obj_list

        Uall = np.zeros((nobj * dimension))
        Wall = np.zeros((nobj))
        for i0 in range(dmda.getRanges()[0][0], dmda.getRanges()[0][1]):
            # print(i0)
            obji = obj_list[i0]
            i1 = obji.index
            Uall[2 * i1:2 * i1 + 2], Wall[i1] = self.update_each_action(obji)
        # assert 1 == 2
        return Uall, Wall


class selfPropelled2D(_baseAction2D):
    def update_each_action(self, obji: "particleClass.baseParticle", **kwargs):
        U = obji.u * obji.P1
        return U, 0


class Dipole2D(_baseAction2D):
    def check_self(self):
        prb = self.father  # type: problemClass.active2DProblem
        err_msg = 'action %s needs at least two particles. ' % type(self).__name__
        assert prb.n_obj > 1, err_msg
        return True

    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.active2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass.relation2D
        theta_ij = relationHandle.theta_ij
        rho_ij = relationHandle.rho_ij
        e_rho_ij = relationHandle.e_rho_ij

        Ui = 0
        for objj in obji.neighbor_list:  # type: particleClass.particle2D
            objj_idx = objj.index
            tth = theta_ij[obji_idx, objj_idx]
            trho = rho_ij[obji_idx, objj_idx]
            te_rho = e_rho_ij[obji_idx, objj_idx]
            t1 = np.dot(np.array(((np.cos(tth), -np.sin(tth)),
                                  (np.sin(tth), np.cos(tth)))), te_rho)
            uij = (objj.dipole * obji.u / np.pi) * (t1 / trho ** 2)
            Ui += uij
        assert 1 == 2
        return Ui


class FiniteDipole2D(Dipole2D):
    def check_self(self):
        super().check_self()

        prb = self.father  # type: problemClass.active2DProblem
        err_handle = 'wrong particle type. particle name: %s, current type: %s, expect type: %s. '
        # todo: modify prb.obj_list
        for obji in prb.obj_list:  # type: particleClass.baseParticle
            err_msg = err_handle % (obji.index, obji.type, 'finiteDipole2D')
            assert isinstance(obji, particleClass.finiteDipole2D), err_msg
        return True

    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.active2DProblem
        l = obji.length
        phi = obji.phi  # alpha in the theory.
        z0 = obji.X[0] + 1j * obji.X[1]  # center in complex plane
        t1 = l / 2 * np.exp(1j * (phi + np.pi / 2))
        zl = z0 + t1  # left vortex
        zr = z0 - t1  # right vortex

        wol, wor = 0, 0
        for objj in prb.obj_list:  # type: particleClass.finiteDipole2D
            if objj is not obji:
                wol += objj.UDipole2Dat(zl)
                wor += objj.UDipole2Dat(zr)
        omega = (wol + wor) / 2  # omega = u - i * v: complex velocity
        Ui = np.array((np.real(omega), -np.imag(omega)))
        Wi = np.real((wor - wol) * np.exp(1j * phi)) / l
        return Ui, Wi


class limFiniteDipole2D(FiniteDipole2D):
    def check_self(self):
        super().check_self()

        prb = self.father  # type: problemClass.active2DProblem
        err_handle = 'wrong particle type. particle name: %s, current type: %s, expect type: %s. '
        # todo: modify prb.obj_list
        for obji in prb.obj_list:  # type: particleClass.baseParticle
            err_msg = err_handle % (obji.index, obji.type, 'limFiniteDipole2D')
            assert isinstance(obji, particleClass.limFiniteDipole2D), err_msg
        return True

    def update_each_action(self, obji: "particleClass.limFiniteDipole2D", **kwargs):
        prb = self.father  # type: problemClass.active2DProblem
        Ui, Wi = 0, 0
        for objj in prb.obj_list:  # type: particleClass.limFiniteDipole2D
            if objj is not obji:
                Ui += objj.UDipole2Dof(obji)
                Wi += objj.WDipole2Dof(obji)
        return Ui, Wi


class Attract2D(_baseAction2D):
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.active2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass.VoronoiRelation2D
        # relationHandle.dbg_showVoronoi()
        theta_ij = relationHandle.theta_ij
        rho_ij = relationHandle.rho_ij

        Wi1 = 0
        Wi2 = 0
        for objj in obji.neighbor_list:  # type: particleClass.particle2D
            objj_idx = objj.index
            tth = theta_ij[obji_idx, objj_idx]
            trho = rho_ij[obji_idx, objj_idx]
            t2 = 1 + np.cos(tth)
            Wi1 += obji.attract * trho * np.sin(tth) * t2
            Wi2 += t2
            # print()
            # print(obji_idx, objj_idx, tth, objj.attract * trho * np.sin(tth) * t2, t2)
        Wi = Wi1 / Wi2
        return np.zeros(2), Wi


class Align2D(_baseAction2D):
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.active2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass.VoronoiRelation2D
        # relationHandle.dbg_showVoronoi()
        theta_ij = relationHandle.theta_ij

        Wi1 = 0
        Wi2 = 0
        for objj in obji.neighbor_list:  # type: particleClass.particle2D
            objj_idx = objj.index
            tth = theta_ij[obji_idx, objj_idx]
            tphi_ij = objj.phi - obji.phi
            t2 = 1 + np.cos(tth)
            Wi1 += obji.align * obji.u * np.sin(tphi_ij) * t2
            Wi2 += t2
            # print()
            # print(obji_idx, objj_idx, tphi_ij, obji.align * obji.u * np.sin(tphi_ij) * t2, t2)
        Wi = Wi1 / Wi2
        return np.zeros(2), Wi
