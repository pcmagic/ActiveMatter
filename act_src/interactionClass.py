"""
20210810
Zhang Ji

calculate the interactions
"""
import abc

import numpy as np
from petsc4py import PETSc

from act_codeStore.support_fun import warpToPi
from act_src import baseClass
from act_src import particleClass
from act_src import problemClass
from act_src import relationClass
from act_codeStore.support_class import *
from act_codeStore import support_fun as spf


# from act_codeStore.support_class import *


class _baseAction(baseClass.baseObj):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._obj_list = uniqueList(
            acceptType=particleClass._baseParticle
        )  # contain objects
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
        self._dmda = PETSc.DMDA().create(
            sizes=(self.n_obj,), dof=1, stencil_width=0, comm=PETSc.COMM_WORLD
        )
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dimension = 2  # 2 for 2D
        self._obj_list = uniqueList(
            acceptType=particleClass.particle2D
        )  # contain objects

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
            t1 = np.dot(
                np.array(((np.cos(tth), -np.sin(tth)), (np.sin(tth), np.cos(tth)))),
                te_rho,
            )
            uij = (objj.dipole * obji.u / np.pi) * (t1 / trho**2)
            Ui += uij
        assert 1 == 2
        return Ui


class FiniteDipole2D(Dipole2D):
    def check_self(self, **kwargs):
        super().check_self()

        prb = self.father  # type: problemClass.behavior2DProblem
        err_handle = "wrong particle type. particle name: %s, current type: %s, expect type: %s. "

        # todo: modify prb.obj_list
        def __init__(self, name="...", **kwargs):
            super().__init__(name, **kwargs)
            # self._type = 'particle2D'
            self._dimension = 2  # 2 for 2D
            self._viewRange = np.ones(1) * np.pi  # how large the camera can view.
            self._phi_steer = 0  # particle self-spin speed
            self._P1 = np.array((1, 0))  # major norm P1, for 2D version
            self._phi = 0  # angular coordinate of P1
            self._phi_hist = []  # major norm P1, for 2D version

            self._X = np.array((0, 0))  # particle center coordinate
            self._U = np.nan * np.array(
                (0, 0)
            )  # particle translational velocity in global coordinate
            self._phi_steer = np.nan * np.array(
                (0,)
            )  # particle rotational velocity in global coordinate
            self._neighbor_list = uniqueList(acceptType=type(self))
            self.update_phi()

        @_baseAction.father.setter
        def father(self, father):
            assert isinstance(father, problemClass._base2DProblem)
            self._father = father

        @property
        def viewRange(self):
            return self._viewRange

        @property
        def phi_steer(self):
            return self._phi_steer

        @phi_steer.setter
        def w(self, phi_steer):
            self._phi_steer = phi_steer

        @_baseAction.P1.setter
        def P1(self, P1):
            # err_msg = 'wrong array size'
            # assert P1.size == 2, err_msg
            _baseAction.P1.fset(self, P1)
            self.update_phi()

        @property
        def phi(self):
            return self._phi

        @phi.setter
        def phi(self, phi):
            # phi = np.hstack((phi,))
            err_msg = "phi is a scale. "
            assert phi.size == 1, err_msg
            assert -np.pi <= phi <= np.pi, phi
            self._phi = phi
            self.update_P1()

        @_baseAction.phi_steer.setter
        def phi_steer(self, phi_steer):
            err_msg = "phi_steer is a scale. "
            assert phi_steer.size == 1, err_msg
            _baseAction.phi_steer.fset(self, phi_steer)

        @property
        def phi_hist(self):
            return self._phi_hist

        def update_phi(self):
            self._phi = np.arctan2(self._P1[1], self._P1[0])
            return True

        def update_P1(self):
            phi = self._phi
            self._P1 = np.array((np.cos(phi), np.sin(phi)))
            return True

        def update_position(self, X, phi, **kwargs):
            self.X = X
            self.phi = warpToPi(phi)
            return True

        def do_store_data(self, **kwargs):
            super().do_store_data()
            if self.rank0:
                self.phi_hist.append(self.phi)  # phi is a float, no necessary to copy.
            return True

        def empty_hist(self, **kwargs):
            super().empty_hist()
            self._phi_hist = np.nan
            return True

        def hdf5_pick(self, handle, **kwargs):
            hdf5_kwargs = self.father.hdf5_kwargs
            obji_hist = super().hdf5_pick(handle, **kwargs)
            obji_hist.create_dataset("phi_hist", data=self.phi_hist, **hdf5_kwargs)
            return obji_hist

        def hdf5_load(self, handle, **kwargs):
            obji_hist = super().hdf5_load(handle, **kwargs)
            self._phi_hist = obji_hist["phi_hist"][:]
            return obji_hist

        def check_self(self, **kwargs):
            super().check_self()
            err_msg = "wrong parameter value: %s "
            assert self.dimension in (2,), err_msg % "dimension"
            assert isinstance(self.father, problemClass._base2DProblem), (
                err_msg % "father"
            )
            assert self.X.shape == (2,), err_msg % "X"
            assert self.U.shape == (2,), err_msg % "U"
            assert self.phi_steer.size == 1, err_msg % "W"
            for obji in self.neighbor_list:
                assert isinstance(obji, interaction2D), err_msg % "neighbor_list"
            assert np.isfinite(self.P1).all(), err_msg % "P1"
            assert self.P1.shape == (2,), err_msg % "P1"
            assert isinstance(self.phi, np.float64), err_msg % "phi"
            assert np.isfinite(self.phi), err_msg % "phi"
            return True

        def update_finish(self):
            super().update_finish()
            if self.rank0:
                self._phi_steer_hist = np.hstack(self.phi_steer_hist)
                self._phi_hist = np.hstack(self.phi_hist)
            return True

        for obji in prb.obj_list:  # type: particleClass._baseParticle
            err_msg = err_handle % (obji.index, obji.type, "finiteDipole2D")
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
        Wi = (
            obji.align
            * np.mean(
                [np.sin(objj.phi - obji.phi - phaseLag) for objj in obji.neighbor_list]
            )
            if obji.neighbor_list
            else 0
        )
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
        Wi = (
            obji.align
            * np.mean(
                [
                    np.sin(objj.phi - obji.phi - phaseLag - phaseLag_random)
                    for objj in obji.neighbor_list
                ]
            )
            if obji.neighbor_list
            else 0
        )
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
        spf.petscInfo(
            self.father.logger,
            "  phaseLag=%f, phaseLag_random=%f"
            % (self.phaseLag, self.phaseLag_rdm_fct),
        )
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

        t1 = self.a * self.A * trho ** (-1 - self.a) - self.b * self.B * trho ** (
            -1 - self.b
        )
        # t1[obji_idx] = 0
        Ui = -np.array((np.mean(t1 * np.cos(tphi)), np.mean(t1 * np.sin(tphi))))
        Wi = np.zeros(1)
        return Ui, Wi

    def print_info(self):
        baseClass.baseObj.print_info(self)
        spf.petscInfo(
            self.father.logger,
            "  A=%e, B=%e, a=%e, b=%e" % (self.A, self.B, self.a, self.b),
        )
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
        v = self.k1 * r**self.k2 + self.k3 * r**self.k4
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
            tphi = (
                np.array([theta_ij[obji_idx, index] for index in neighbor_idx_list])
                + obji.phi
            )
            Vi = np.array(
                [
                    self.k1 * rho_ij[obji_idx, index] ** self.k2
                    + self.k3 * rho_ij[obji_idx, index] ** self.k4
                    for index in neighbor_idx_list
                ]
            )
            Ui = np.array(
                (np.mean(Vi * np.cos(tphi)), np.mean(Vi * np.sin(tphi)))
            ) + np.zeros(2)
        else:
            Ui = np.zeros(2)
        Wi = np.zeros(1)
        return Ui, Wi

    def print_info(self):
        baseClass.baseObj.print_info(self)
        spf.petscInfo(self.father.logger, "  v = k1 * r ** k2 + k3 * r ** k4")
        spf.petscInfo(
            self.father.logger,
            "  k1=%e, k2=%e, k3=%e, k4=%e" % (self.k1, self.k2, self.k3, self.k4),
        )
        return True


class Ackermann2D(_baseAction2D):
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
        return True

    def update_each_action(self, obji: "particleClass.ackermann2D", **kwargs):
        # todo: latex version, eq kinetic... wu
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
        U = obji.u * obji.P1
        W = np.tan(obji.phi_steer) * obji.u / obji.l_steer
        W_steer = obji.w_steer
        return U, W, W_steer
