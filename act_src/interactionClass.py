"""
20210810
Zhang Ji

calculate the interactions
"""
import abc

import numpy as np
from petsc4py import PETSc
from time import time

from act_src import baseClass
from act_src import particleClass
from act_src import problemClass
from act_src import relationClass
from act_codeStore.support_class import *
from act_codeStore import support_fun as spf
from src import stokes_flow as sf
from src import geo
from src.myio import *
from src import forceSphere2d as fs2


# from act_codeStore.support_class import *


class _baseAction(baseClass.baseObj):
    def __init__(self, name="...", **kwargs):
        super().__init__(name, **kwargs)
        self._obj_list = uniqueList(acceptType=particleClass._baseParticle)  # contain objects
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
        err_msg = "wrong object type"
        assert isinstance(obj, particleClass._baseParticle), err_msg
        err_msg = "wrong dimension"
        assert np.isclose(self.dimension, obj.dimension), err_msg
        return True
    
    def add_obj(self, obj):
        self._check_add_obj(obj)
        self._obj_list.append(obj)
        return True
    
    def check_self(self, **kwargs):
        pass
        return True
    
    def update_prepare(self):
        self.set_dmda()
    
    def update_finish(self):
        pass
        return True
    
    def set_dmda(self):
        self._dmda = PETSc.DMDA().create(sizes=(self.n_obj,), dof=1, stencil_width=0, comm=PETSc.COMM_WORLD)
        self._dmda.setFromOptions()
        self._dmda.setUp()
        return True
    
    @abc.abstractmethod
    def update_each_action(self, obji: "particleClass._baseParticle", **kwargs):
        return 0, 0
    
    @abc.abstractmethod
    def Jacobian(self):
        return None
    
    @abc.abstractmethod
    def update_action(self, **kwargs):
        return 0, 0
    
    def update_action_numpy(self):
        Uall, Wall = [], []
        for obji in self.obj_list:
            u, w = self.update_each_action(obji)
            Uall.append(u)
            Wall.append(w)
        Uall = np.hstack(Uall)
        Wall = np.hstack(Wall)
        return Uall, Wall
    
    def destroy_self(self, **kwargs):
        super().destroy_self(**kwargs)
        if self.dmda is not None:
            self._dmda.destroy()
            self._dmda = None
        return True
    
    def print_info(self):
        super().print_info()
        spf.petscInfo(self.father.logger, "  None")
        return True


class _baseAction2D(_baseAction):
    def __init__(self, name="...", **kwargs):
        super().__init__(name=name, **kwargs)
        self._dimension = 2  # 2 for 2D
        self._obj_list = uniqueList(acceptType=particleClass.particle2D)  # contain objects
    
    def update_action(self, F):
        nobj = self.n_obj
        dimension = self.dimension
        dmda = self.dmda
        obj_list = self.obj_list
        
        # Uall = np.zeros((nobj * dimension))
        # Wall = np.zeros((nobj))
        idxW0 = dimension * nobj
        for i0 in range(dmda.getRanges()[0][0], dmda.getRanges()[0][1]):
            # print(i0)
            obji = obj_list[i0]
            i1 = obji.index
            u, w = self.update_each_action(obji)
            # print('dbg', i1, obji, u, w)
            F.setValues((dimension * i1, dimension * i1 + 1), u, addv=True)
            F.setValue(idxW0 + i0, w, addv=True)
            # print('dbg', i0, [obji.X for obji in obj_list])
        # if self.type == 'phaseLag2D':
        #     print(F[:])
        return True


class selfPropelled2D(_baseAction2D):
    def update_each_action(self, obji: "particleClass._baseParticle", **kwargs):
        U = obji.u * obji.P1
        return U, 0


class selfSpeed2D(_baseAction2D):
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        U = obji.u * obji.P1
        return U, obji.w


class Dipole2D(_baseAction2D):
    def check_self(self, **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        err_msg = "action %s needs at least two particles. " % type(self).__name__
        assert prb.n_obj > 1, err_msg
        return True
    
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass._baseRelation2D
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
                                  (np.sin(tth), np.cos(tth)))),
                        te_rho, )
            uij = (objj.dipole * obji.u / np.pi) * (t1 / trho ** 2)
            Ui += uij
        assert 1 == 2
        return Ui


class FiniteDipole2D(Dipole2D):
    def check_self(self, **kwargs):
        super().check_self()
        
        prb = self.father  # type: problemClass.behavior2DProblem
        err_handle = 'wrong particle type. particle name: %s, current type: %s, expect type: %s. '
        # todo: modify prb.obj_list
        for obji in prb.obj_list:  # type: particleClass._baseParticle
            err_msg = err_handle % (obji.index, obji.type, 'finiteDipole2D')
            assert isinstance(obji, particleClass.finiteDipole2D), err_msg
        return True
    
    def update_each_action(self, obji: "particleClass.finiteDipole2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
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
    def check_self(self, **kwargs):
        super().check_self()
        
        prb = self.father  # type: problemClass.behavior2DProblem
        err_handle = "wrong particle type. particle name: %s, current type: %s, expect type: %s. "
        # todo: modify prb.obj_list
        for obji in prb.obj_list:  # type: particleClass._baseParticle
            err_msg = err_handle % (obji.index, obji.type, "limFiniteDipole2D")
            assert isinstance(obji, particleClass.limFiniteDipole2D), err_msg
        return True
    
    def update_each_action(self, obji: "particleClass.limFiniteDipole2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        Ui, Wi = 0, 0
        for objj in prb.obj_list:  # type: particleClass.limFiniteDipole2D
            if objj is not obji:
                ui, wi = objj.UWDipole2Dof(obji)
                Ui += ui
                Wi += wi
        return Ui, Wi


class Attract2D(_baseAction2D):
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass.VoronoiBaseRelation2D
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
            Wi1 += trho * np.sin(tth) * t2
            Wi2 += t2
            # print()
            # print(obji_idx, objj_idx, tth, objj.attract * trho * np.sin(tth) * t2, t2)
        Wi = obji.attract * Wi1 / Wi2
        return np.zeros(2), Wi


class lightAttract2D(_baseAction2D):
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        lightDecayFct = prb.lightDecayFct
        obji_idx = obji.index
        viewRange = obji.viewRange
        relationHandle = prb.relationHandle  # type: relationClass.VoronoiBaseRelation2D
        theta_ij = relationHandle.theta_ij
        rho_ij = relationHandle.rho_ij
        
        Wi1 = 0
        Wi2 = 0
        for objj in obji.neighbor_list:  # type: particleClass.particle2D
            objj_idx = objj.index
            tth = theta_ij[obji_idx, objj_idx] / viewRange * np.pi
            if -np.pi < tth < np.pi:
                trho = rho_ij[obji_idx, objj_idx]
                t2 = 1 + np.cos(tth)
                Wi1 += np.exp(-lightDecayFct * trho) * t2
                Wi2 += t2
        Wi = obji.attract * Wi1 / Wi2
        return np.zeros(2), Wi


class Align2D(_baseAction2D):
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass.VoronoiBaseRelation2D
        # relationHandle.dbg_showVoronoi()
        theta_ij = relationHandle.theta_ij
        
        Wi1 = 0
        Wi2 = 0
        for objj in obji.neighbor_list:  # type: particleClass.particle2D
            objj_idx = objj.index
            tth = theta_ij[obji_idx, objj_idx]
            tphi_ij = objj.phi - obji.phi
            t2 = 1 + np.cos(tth)
            Wi1 += np.sin(tphi_ij) * t2
            Wi2 += t2
            # print()
            # print(obji_idx, objj_idx, tphi_ij, obji.align  * np.sin(tphi_ij) * t2, t2)
        Wi = obji.align * Wi1 / Wi2
        return np.zeros(2), Wi


class AlignAttract2D(_baseAction2D):
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass.VoronoiBaseRelation2D
        # relationHandle.dbg_showVoronoi()
        theta_ij = relationHandle.theta_ij
        rho_ij = relationHandle.rho_ij
        
        Wi1_align = 0
        Wi1_attract = 0
        Wi2 = 0
        for objj in obji.neighbor_list:  # type: particleClass.particle2D
            objj_idx = objj.index
            tth = theta_ij[obji_idx, objj_idx]
            tphi_ij = objj.phi - obji.phi
            trho = rho_ij[obji_idx, objj_idx]
            t2 = 1 + np.cos(tth)
            Wi1_align += np.sin(tphi_ij) * t2
            Wi1_attract += trho * np.sin(tth) * t2
            Wi2 += t2
            # print()
            # print(obji_idx, objj_idx, tphi_ij, obji.align  * np.sin(tphi_ij) * t2, t2)
        Wi1 = obji.align * Wi1_align + obji.attract * Wi1_attract
        Wi = Wi1 / Wi2
        return np.zeros(2), Wi


class Wiener2D(_baseAction2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sqrt_dt = np.nan
    
    @property
    def sqrt_dt(self):
        return self._sqrt_dt
    
    def update_prepare(self):
        super().update_prepare()
        prb = self.father  # type: problemClass._base2DProblem
        self._sqrt_dt = np.sqrt(prb.eval_dt)
    
    def check_self(self, **kwargs):
        prb = self.father  # type: problemClass._base2DProblem
        update_fun = prb.update_fun
        err_msg = 'wrong parameter update_fun, only "1fe" is acceptable. '
        assert update_fun == "1fe", err_msg
    
    def update_each_action(self, obji: "particleClass._baseParticle", **kwargs):
        return np.zeros(2), obji.rot_noise * np.random.normal() / self.sqrt_dt
    
    def print_info(self):
        baseClass.baseObj.print_info(self)
        spf.petscInfo(self.father.logger, "  sqrt_dt=%f" % self.sqrt_dt)
        return True


class phaseLag2D(_baseAction2D):
    def __init__(self, phaseLag=0, **kwargs):
        super().__init__(**kwargs)
        self._phaseLag = phaseLag
    
    @property
    def phaseLag(self):
        return self._phaseLag
    
    @phaseLag.setter
    def phaseLag(self, phaseLag):
        self._phaseLag = phaseLag
    
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        phaseLag = self.phaseLag
        Ui = np.zeros(2)
        Wi = (obji.align * np.mean([np.sin(objj.phi - obji.phi - phaseLag)
                                    for objj in obji.neighbor_list])
              if obji.neighbor_list else 0)
        return Ui, Wi
    
    def print_info(self):
        baseClass.baseObj.print_info(self)
        spf.petscInfo(self.father.logger, "  phaseLag=%f" % self.phaseLag)
        return True


class phaseLag2D_Wiener(phaseLag2D):
    def __init__(self, phaseLag_rdm_fct=0, **kwargs):
        super().__init__(**kwargs)
        self._phaseLag_rdm_fct = phaseLag_rdm_fct
    
    @property
    def phaseLag_rdm_fct(self):
        return self._phaseLag_rdm_fct
    
    @phaseLag_rdm_fct.setter
    def phaseLag_rdm_fct(self, phaseLag_rdm_fct):
        self._phaseLag_rdm_fct = phaseLag_rdm_fct
    
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass._baseRelation2D
        # theta_ij = relationHandle.theta_ij
        rho_ij = relationHandle.rho_ij
        phaseLag = self.phaseLag
        phaseLag_rdm_fct = self.phaseLag_rdm_fct
        
        Ui = np.zeros(2)
        # --------------------- trivial case
        # Wi = obji.align * np.mean(
        #     [np.sin(objj.phi - obji.phi - phaseLag)
        #      for objj in obji.neighbor_list]) \
        #     if obji.neighbor_list else 0
        #
        phaseLag_random = phaseLag_rdm_fct * np.random.normal()
        Wi = (obji.align * np.mean(
                [np.sin(objj.phi - obji.phi - phaseLag - phaseLag_random)
                 for objj in obji.neighbor_list])
              if obji.neighbor_list else 0)
        #
        # phaseLag_random = phaseLag_rdm_fct * np.random.normal()
        # Wi = obji.align  * np.mean(
        #     [np.random.normal() * np.sin(objj.phi - obji.phi - phaseLag - phaseLag_random)
        #      for objj in obji.neighbor_list]) \
        #     if obji.neighbor_list else 0
        #
        # phaseLag_random = phaseLag_rdm_fct
        # Wi = obji.align * np.mean(
        #     [np.sin(objj.phi - obji.phi - phaseLag - phaseLag_random / rho_ij[obji_idx, objj.index])
        #      for objj in obji.neighbor_list]) \
        #     if obji.neighbor_list else 0
        
        # print(111)
        return Ui, Wi
    
    def print_info(self):
        baseClass.baseObj.print_info(self)
        spf.petscInfo(self.father.logger, "  phaseLag=%f, phaseLag_random=%f"
                      % (self.phaseLag, self.phaseLag_rdm_fct), )
        return True


class LennardJonePotential2D_point(_baseAction2D):
    def __init__(self, A=0, B=0, a=12, b=6, **kwargs):
        super().__init__(**kwargs)
        self._A = A
        self._B = B
        self._a = a
        self._b = b
    
    @property
    def A(self):
        return self._A
    
    @A.setter
    def A(self, A):
        self._A = A
    
    @property
    def B(self):
        return self._B
    
    @B.setter
    def B(self, B):
        self._B = B
    
    @property
    def a(self):
        return self._a
    
    @a.setter
    def a(self, a):
        self._a = a
    
    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, b):
        self._b = b
    
    #
    # def fun_fLJ(self, r):
    #     rnm = np.linalg.norm()
    #     t1 = self.a * self.A * r ** (-1 - self.a) - self.b * self.B * r ** (-1 - self.b)
    
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass._baseRelation2D
        theta_ij = relationHandle.theta_ij
        rho_ij = relationHandle.rho_ij
        
        trho = rho_ij[obji_idx, :]
        trho[obji_idx] = np.inf
        tphi = theta_ij[obji_idx, :] + obji.phi
        
        t1 = self.a * self.A * trho ** (-1 - self.a) - self.b * self.B * trho ** (-1 - self.b)
        # t1[obji_idx] = 0
        Ui = -np.array((np.mean(t1 * np.cos(tphi)), np.mean(t1 * np.sin(tphi))))
        Wi = np.zeros(1)
        return Ui, Wi
    
    def print_info(self):
        baseClass.baseObj.print_info(self)
        spf.petscInfo(self.father.logger, "  A=%e, B=%e, a=%e, b=%e" % (self.A, self.B, self.a, self.b), )
        return True


class AttractRepulsion2D_point(_baseAction2D):
    def __init__(self, k1=-1, k2=-1, k3=1, k4=2, **kwargs):
        super().__init__(**kwargs)
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._k4 = k4
    
    @property
    def k1(self):
        return self._k1
    
    @k1.setter
    def k1(self, k1):
        self._k1 = k1
    
    @property
    def k2(self):
        return self._k2
    
    @k2.setter
    def k2(self, k2):
        self._k2 = k2
    
    @property
    def k3(self):
        return self._k3
    
    @k3.setter
    def k3(self, k3):
        self._k3 = k3
    
    @property
    def k4(self):
        return self._k4
    
    @k4.setter
    def k4(self, k4):
        self._k4 = k4
    
    def fun_fAR(self, r, obji_idx):
        # k1 * r ** k2 + k3 * r ** k4
        r[obji_idx] = np.nan
        v = self.k1 * r ** self.k2 + self.k3 * r ** self.k4
        v[obji_idx] = 0
        return v
    
    def update_each_action(self, obji: "particleClass.particle2D", **kwargs):
        prb = self.father  # type: problemClass.behavior2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass._baseRelation2D
        rho_ij = relationHandle.rho_ij
        theta_ij = relationHandle.theta_ij
        neighbor_idx_list = [objj.index for objj in obji.neighbor_list]
        
        if len(obji.neighbor_list) > 0:
            tphi = (np.array([theta_ij[obji_idx, index] for index in neighbor_idx_list]) + obji.phi)
            Vi = np.array([self.k1 * rho_ij[obji_idx, index] ** self.k2 +
                           self.k3 * rho_ij[obji_idx, index] ** self.k4
                           for index in neighbor_idx_list])
            Ui = np.array((np.mean(Vi * np.cos(tphi)), np.mean(Vi * np.sin(tphi)))) + np.zeros(2)
        else:
            Ui = np.zeros(2)
        Wi = np.zeros(1)
        return Ui, Wi
    
    def print_info(self):
        baseClass.baseObj.print_info(self)
        spf.petscInfo(self.father.logger, "  v = k1 * r ** k2 + k3 * r ** k4")
        spf.petscInfo(self.father.logger, "  k1=%e, k2=%e, k3=%e, k4=%e" % (self.k1, self.k2, self.k3, self.k4), )
        return True


# this is a reproduce of '03car_world' of ir_sim.
# see https://github.com/hanruihua/intelligent-robot-simulator/blob/main/ir_sim/usage/03car_world/car_world.py for detial.
class Ackermann2D(_baseAction2D):
    def __init__(self, radian_tolerance=0, **kwargs):
        super().__init__(**kwargs)
        self._obj_list = uniqueList(acceptType=particleClass.ackermann2D)  # contain objects
        self._radian_tolerance = radian_tolerance
    
    @property
    def radian_tolerance(self):
        return self._radian_tolerance
    
    @radian_tolerance.setter
    def radian_tolerance(self, radian_tolerance):
        self._radian_tolerance = radian_tolerance
    
    def update_action(self, F):
        nobj = self.n_obj
        dimension = self.dimension
        dmda = self.dmda
        obj_list = self.obj_list
        
        idxW0 = dimension * nobj
        idxW_steer0 = (dimension + 1) * nobj
        for i0 in range(dmda.getRanges()[0][0], dmda.getRanges()[0][1]):
            obji = obj_list[i0]
            i1 = obji.index
            u, w, w_steer = self.update_each_action(obji)
            # print('dbg', i1, obji, u, w)
            F.setValues((dimension * i1, dimension * i1 + 1), u, addv=True)
            F.setValue(idxW0 + i0, w, addv=True)
            F.setValue(idxW_steer0 + i0, w_steer, addv=True)
        # assert 1 == 2
        return True
    
    def cal_des_vel(self, obji: "particleClass.ackermann2D", **kwargs):
        def relative(state1, state2):
            dif = state2[0:2] - state1[0:2]
            dis = np.linalg.norm(dif)
            radian = np.arctan2(dif[1], dif[0])
            return dis, radian
        
        state1 = obji.X
        state2 = obji.goal
        v_max = obji.v_max
        w_max = obji.w_max
        radian_tolerance = self.radian_tolerance
        
        dis, radian = relative(state1, state2)
        car_radian = obji.phi + obji.phi_steer
        diff_radian = spf.warpToPi(radian - car_radian)
        
        if diff_radian > radian_tolerance:
            w_opti = np.float32(w_max)
        elif diff_radian < -radian_tolerance:
            w_opti = -np.float32(w_max)
        else:
            w_opti = np.float32(0)
        
        if dis < obji.goal_threshold:
            v_opti = np.float32(0)
            w_opti = np.float32(0)
        else:
            v_opti = np.clip(v_max * np.cos(diff_radian), 0, None)
        pass
        return v_opti, w_opti
    
    def update_each_action(self, obji: "particleClass.ackermann2D", **kwargs):
        # \def\SetClass{article}
        # \documentclass{\SetClass}
        # \usepackage[linesnumbered,lined,boxed,commentsnumbered]{algorithm2e}
        # \begin{document}
        # \IncMargin{1em}
        # \begin{algorithm}
        # \SetKwData{Left}{left}\SetKwData{This}{this}\SetKwData{Up}{up}
        # \SetKwFunction{Union}{Union}\SetKwFunction{FindCompress}{FindCompress}
        # \BlankLine
        # \ U$\leftarrow$ \ {$obji\cdot\ u*obji\cdot\ p1 $}\;
        # \ W$\leftarrow$ \ {$np\cdot\ tan(obji\cdot\ phi \_steer )* obji\cdot\ u / obji\cdot\ l\_steer $}\;
        # \caption{Ackermann2D}\label{algo_disjdecomp}
        # \end{algorithm}\DecMargin{1em}
        # \end{document}
        
        # U -> vehicle velocity
        # w -> vehicle spin
        # obj.w -> steer spin
        obji.u, obji.w_steer = self.cal_des_vel(obji)
        
        U = obji.u * obji.P1
        W = np.tan(obji.phi_steer) * obji.u / obji.l_steer
        W_steer = obji.w_steer
        return U, W, W_steer


class Ackermann_AlignAttract2D(Ackermann2D):
    def cal_des_vel(self, obji: "particleClass.ackermann2D", **kwargs):
        prb = self.father  # type: problemClass.Ackermann2DProblem
        obji_idx = obji.index
        relationHandle = prb.relationHandle  # type: relationClass._baseRelation2D
        # relationHandle.dbg_showVoronoi()
        theta_ij = relationHandle.theta_ij
        rho_ij = relationHandle.rho_ij
        
        Wi1_align = 0
        Wi1_attract = 0
        Wi2 = 0
        for objj in obji.neighbor_list:  # type: particleClass.particle2D
            objj_idx = objj.index
            tth = theta_ij[obji_idx, objj_idx]
            tphi_ij = objj.phi - obji.phi
            trho = rho_ij[obji_idx, objj_idx]
            t2 = 1 + np.cos(tth)
            Wi1_align += np.sin(tphi_ij) * t2
            Wi1_attract += trho * np.sin(tth) * t2
            Wi2 += t2
        Wi1 = obji.align * Wi1_align + obji.attract * Wi1_attract
        Wi = Wi1 / Wi2
        return obji.u, Wi


class Ackermann_phaseLag2D(phaseLag2D, Ackermann2D):
    def cal_des_vel(self, obji: "particleClass.ackermann2D", **kwargs):
        phaseLag = self.phaseLag
        Wi = (obji.align * np.mean([np.sin(objj.phi - obji.phi - phaseLag)
                                    for objj in obji.neighbor_list])
              if obji.neighbor_list else 0)
        return obji.u, Wi

    def update_each_action(self, obji: "particleClass.ackermann2D", **kwargs):
        return Ackermann2D.update_each_action(self, obji, **kwargs)


class ForceSphere2D(_baseAction2D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._obj_list = uniqueList(acceptType=particleClass.ForceSphere2D)  # contain objects
    
    def update_prepare(self):
        pass
    
    def update_action(self, F):
        assert self.dimension == 2
        
        obji = self.obj_list[0]
        prb_MR = obji.prb_MR  # type:sf.ForceSphere2DProblem
        # prb_MR.set_velocity_petsc(F)
        prb_MR.create_matrix_light()
        prb_MR.solve_resistance(PETScSysPrint=False)
        # F[:] = prb_MR.get_velocity_petsc()
        pass
        return True


class ForceSphere2D_matrixAct(_baseAction2D):
    def __init__(self, name="...", **kwargs):
        super().__init__(name=name, **kwargs)
        self._obj_list = uniqueList(acceptType=particleClass.ForceSphere2D_matrixObj)  # contain objects
        self._ksp = PETSc.KSP()
        self._force_petsc = None  # F = R U, for ksp solver
        self._Minf_petsc = None  # R matrix, lubrication part
        self._Rlub_petsc = None  # R matrix, lubrication part
        self._Rtol_petsc = None  # full modified R matrix, Rtol = I + Minf^-1 * Rlub
        self._Fmdf_petsc = None  # modified force vector, Fmdf = Minf * F
        self._velocity = np.zeros([0])  # velocity information
        self._spin = np.zeros([0])  # spin information
        # self._re_fmdf = np.zeros([0])  # resolved information of modified force
    
    @property
    def ksp(self):
        return self._ksp
    
    @property
    def Minf_petsc(self):
        return self._Minf_petsc
    
    @property
    def Rlub_petsc(self):
        return self._Rlub_petsc
    
    @property
    def Rtol_petsc(self):
        return self._Rtol_petsc
    
    @property
    def Fmdf_petsc(self):
        return self._Fmdf_petsc
    
    @property
    def force_petsc(self):
        return self._force_petsc
    
    @property
    # F = R U and U = ts_f (U for ksp solver, and ts_f for ts solver)
    def velocity_petsc(self):
        return self.father.ts_f
    
    @property
    def velocity(self):
        return self._velocity
    
    @property
    def spin(self):
        return self._spin
    
    def ksp_prepare(self):
        # t0 = time()
        problem = self.father
        kwargs = problem.kwargs
        solve_method = kwargs['solve_method']
        precondition_method = kwargs['precondition_method']
        # self.velocity_petsc[:] = ini_guess[:]
        # self.velocity_petsc.assemble()
        
        ksp = self.ksp
        ksp.create(comm=PETSc.COMM_WORLD)
        ksp.setType(solve_method)
        ksp.getPC().setType(precondition_method)
        # OptDB = PETSc.Options()
        ksp.setFromOptions()
        ksp.setGMRESRestart(ksp.getTolerances()[-1])
        ksp.setInitialGuessNonzero(True)
        ksp.setOperators(self.Rtol_petsc)
        ksp.setUp()
        return True
    
    def update_prepare(self):
        assert self.dimension == 2
        problem = self.father  # type:problemClass.ForceSphere2D_matrix
        dmda = problem.dmda
        # ts = problem.ts
        # ts_f = problem.ts_f
        # ts_y = problem.ts_y
        # obj = problem.obj_list[0]
        # fileHandle = kwargs['fileHandle']
        
        self._force_petsc = dmda.createGlobalVector()
        self._Fmdf_petsc = dmda.createGlobalVector()
        self._Minf_petsc = dmda.createMatrix()
        self._Rlub_petsc = dmda.createMatrix()
        self._Rtol_petsc = dmda.createMatrix()
        for tpetsc_obj, type in zip((self.force_petsc, self.Fmdf_petsc, self.Minf_petsc, self.Rlub_petsc, self.Rtol_petsc),
                                    ('standard', 'standard', 'dense', 'dense', 'dense')):
            tpetsc_obj.setType(type)
            tpetsc_obj.setFromOptions()
            tpetsc_obj.setUp()
            # print(tpetsc_obj.getSizes())
        
        self.ksp_prepare()
        return True
    
    def create_matrix(self):
        problem = self.father  # type:problemClass.ForceSphere2D_matrixPro
        kwargs = problem.kwargs
        dmda = problem.dmda
        # ts = problem.ts
        # ts_f = problem.ts_f
        ts_y = problem.ts_y  # ts_y = [(x0, x1, phi)]
        obj = problem.obj_list[0]
        sphere_R = obj.r
        # sphere_R = problem.sphere_R
        diag_err = kwargs['diag_err']  # Avoiding errors introduced by nan values. (避免nan值引入的误差)
        # rs2 = kwargs['rs2']
        sdis = kwargs['sdis']
        length = kwargs['length']
        width = kwargs['width']
        mu = kwargs['mu']
        
        fs2.M_R_petsc_simp(self.Minf_petsc, self.Rlub_petsc, self.Rtol_petsc, dmda,
                           sphere_R, ts_y, sdis, length, width,
                           mu=mu, diag_err=diag_err)
        return True
    
    def solve_velocity(self, F):
        problem = self.father  # type:problemClass.ForceSphere2D_matrixPro
        kwargs = problem.kwargs
        mu = kwargs['mu']
        ksp = self.ksp
        frac = 1 / (np.pi * mu)  # 前置系数
        
        # self.Minf_petsc.view()
        self.Minf_petsc.mult(self.force_petsc, self.Fmdf_petsc)
        self.Fmdf_petsc.scale(frac)
        ksp.solve(self.Fmdf_petsc, F)
        
        # the following codes are commanded for speedup.
        # # reorder force from petsc index to normal index, and separate to each object.
        # t_velocity = self.vec_scatter(self._velocity_petsc, destroy=False)
        # tmp_velocity = []
        # tmp_spin = []
        # for obj0 in self.get_obj_list():  # type: geo.sphere_particle_2d
        #     ugeo = obj0.get_u_geo()
        #     _, u_glbIdx_all = ugeo.get_glbIdx()
        #     obj0.set_velocity(t_velocity[u_glbIdx_all[:ugeo.get_dof() * ugeo.get_n_nodes()]])
        #     obj0.set_spin(t_velocity[u_glbIdx_all[ugeo.get_dof() * ugeo.get_n_nodes():]])
        #     tmp_velocity.append(obj0.get_velocity())
        #     tmp_spin.append(obj0.get_spin())
        # self._velocity = np.hstack(tmp_velocity)
        # self._spin = np.hstack(tmp_spin)
        return True
    
    def update_action(self, F):
        assert self.dimension == 2
        problem = self.father  # type:problemClass.ForceSphere2D_matrix
        kwargs = problem.kwargs
        For = kwargs['For']
        Tor = kwargs['Tor']
        phi = problem.Phiall
        dmda = problem.dmda
        # print(ts_f.getSize())
        # print(ts_y.getArray())
        
        # calculate force of ksp solver
        t0 = time()
        fs2.F_petsc(dmda, self.force_petsc, For, Tor, phi)
        t1 = time()
        spf.petscInfo(problem.logger, '    F_petsc time:         %fs' % (t1 - t0))
        self.create_matrix()
        t2 = time()
        spf.petscInfo(problem.logger, '    create_matrix time:   %fs' % (t2 - t1))
        self.solve_velocity(F)
        t3 = time()
        spf.petscInfo(problem.logger, '    solve_velocity time:  %fs' % (t3 - t2))
        spf.petscInfo(problem.logger, '    update_action time:   %fs' % (t3 - t0))
        spf.petscInfo(problem.logger, ' ')
        return True


class ForceSphere2D_noMult(ForceSphere2D_matrixAct):
    def create_matrix(self):
        problem = self.father  # type:problemClass.ForceSphere2D_matrixPro
        kwargs = problem.kwargs
        dmda = problem.dmda
        ts_y = problem.ts_y  # ts_y = [(x0, x1, phi)]
        obj = problem.obj_list[0]
        sphere_R = obj.r
        diag_err = kwargs['diag_err']  # Avoiding errors introduced by nan values. (避免nan值引入的误差)
        sdis = kwargs['sdis']
        length = kwargs['length']
        width = kwargs['width']
        mu = kwargs['mu']
        
        fs2.M_R_petsc_simp_noMult(self.Minf_petsc, self.Rlub_petsc, dmda,
                                  sphere_R, ts_y, sdis, length, width,
                                  mu=mu, diag_err=diag_err)
        return True
    
    def solve_velocity(self, F):
        problem = self.father  # type:problemClass.ForceSphere2D_matrixPro
        kwargs = problem.kwargs
        mu = kwargs['mu']
        ksp = self.ksp
        frac = 1 / (np.pi * mu)  # 前置系数
        
        # self.Minf_petsc.view()
        self.Minf_petsc.mult(self.force_petsc, self.Fmdf_petsc)
        self.Fmdf_petsc.scale(frac)
        ksp.solve(self.Fmdf_petsc, F)
        
        # the following codes are commanded for speedup.
        # # reorder force from petsc index to normal index, and separate to each object.
        # t_velocity = self.vec_scatter(self._velocity_petsc, destroy=False)
        # tmp_velocity = []
        # tmp_spin = []
        # for obj0 in self.get_obj_list():  # type: geo.sphere_particle_2d
        #     ugeo = obj0.get_u_geo()
        #     _, u_glbIdx_all = ugeo.get_glbIdx()
        #     obj0.set_velocity(t_velocity[u_glbIdx_all[:ugeo.get_dof() * ugeo.get_n_nodes()]])
        #     obj0.set_spin(t_velocity[u_glbIdx_all[ugeo.get_dof() * ugeo.get_n_nodes():]])
        #     tmp_velocity.append(obj0.get_velocity())
        #     tmp_spin.append(obj0.get_spin())
        # self._velocity = np.hstack(tmp_velocity)
        # self._spin = np.hstack(tmp_spin)
        return True
