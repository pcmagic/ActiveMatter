# coding=utf-8
"""
20210810
Zhang Ji

particle itself, i.e. a fish, a microswimmer, or a bird.
"""
import abc

import numpy as np
# from petsc4py import PETSc
from act_codeStore.support_fun import warpToPi
from act_src import baseClass
from act_src import problemClass
from act_src.interactionClass import _baseAction
from act_codeStore.support_class import *


class _baseParticle(baseClass.baseObj):
    def __init__(self, name="...", **kwargs):
        super().__init__(name, **kwargs)
        self._index = -1  # object index
        self._dimension = -1  # -1 for undefined, 2 for 2D, 3 for 3D
        self._u = 0  # particle speed
        self._X = np.nan  # particle center position
        self._U = np.nan  # particle translational velocity in global coordinate
        self._W = np.nan  # particle rotational velocity in global coordinate
        self._P1 = np.nan  # major norm P1
        # self._phi = np.nan  # angular coordinate of P1. For 2D version
        # self._q = np.nan # q is a quaternion(w, x, y, z), where w=cos(theta/2). For 3D version
        self._attract = 0  # attract intensity
        self._align = 0  # align intensity
        self._rot_noise = 0  # rotational noise
        self._trs_noise = 0  # translational noise
        self._dipole = 0  # dipole intensity
        self._neighbor_list = uniqueList(acceptType=_baseParticle)
        self._action_list = uniqueList(acceptType=_baseAction)
        # print(self._name)
        # print(self._type)
        # print(self._kwargs)
        
        # historical information
        self._X_hist = []  # center location
        # self._phi_hist = []  # angular coordinate of P1, for 2D version
        # self._q_hist = []  # q is a quaternion(w, x, y, z), where w=cos(theta/2). For 3D version
        self._U_hist = []  # total velocity in global coordinate.
        self._W_hist = []  # total rotational velocity in global coordinate.
    
    def __repr__(self):
        return "%s_%04d" % (self._type, self.index)
    
    @property
    def name(self):
        t1 = "%s_%04d" % (self._name, self.index)
        return t1
    
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index):
        self._index = index
    
    @property
    def dimension(self):
        return self._dimension
    
    @property
    def father(self):
        return self._father
    
    @father.setter
    def father(self, father):
        assert isinstance(father, problemClass._baseProblem)
        self._father = father
    
    @property
    def u(self):
        return self._u
    
    @u.setter
    def u(self, u):
        err_msg = "u is a scale. "
        assert u.size == 1, err_msg
        self._u = u
    
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, X):
        err_msg = "X is a vector with %d components. " % self.dimension
        assert self._X.size == self.dimension, err_msg
        self._X = np.array(X)
    
    @property
    def U(self):
        return self._U
    
    @U.setter
    def U(self, U):
        err_msg = "U is a vector with %d components. " % self.dimension
        assert self._U.size == self.dimension, err_msg
        self._U = np.array(U)
    
    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, W):
        self._W = np.array(W)
    
    @property
    def P1(self):
        return self._P1
    
    @P1.setter
    def P1(self, P1):
        P1 = np.array(P1)
        self._P1 = P1.ravel() / np.linalg.norm(P1)
        err_msg = "P1 is a vector with %d components. " % self.dimension
        assert self._P1.size == self.dimension, err_msg
    
    @property
    def attract(self):
        return self._attract
    
    @attract.setter
    def attract(self, attract):
        err_msg = "attract is a scale. "
        assert attract.size == 1, err_msg
        self._attract = attract
    
    @property
    def align(self):
        return self._align
    
    @align.setter
    def align(self, align):
        err_msg = "align is a scale. "
        assert align.size == 1, err_msg
        self._align = align
    
    @property
    def rot_noise(self):
        return self._rot_noise
    
    @rot_noise.setter
    def rot_noise(self, rot_noise):
        self._rot_noise = rot_noise
    
    @property
    def trs_noise(self):
        return self._trs_noise
    
    @trs_noise.setter
    def trs_noise(self, trs_noise):
        self._trs_noise = trs_noise
    
    @property
    def dipole(self):
        return self._dipole
    
    @property
    def neighbor_list(self):
        return self._neighbor_list
    
    @property
    def X_hist(self):
        return self._X_hist
    
    @property
    def U_hist(self):
        return self._U_hist
    
    @property
    def W_hist(self):
        return self._W_hist
    
    # @abc.abstractmethod
    # def update_self(self, **kwargs):
    #     return
    
    def set_X_zero(self):
        self.X = np.zeros(self.dimension)
    
    @abc.abstractmethod
    def update_position(self, **kwargs):
        return
    
    def update_velocity(self, U, W, **kwargs):
        self.U = U
        self.W = W
        return True
    
    def do_store_data(self, **kwargs):
        if self.rank0:
            self.X_hist.append(self.X.copy())
            self.U_hist.append(self.U.copy())
            self.W_hist.append(self.W.copy())
        return True
    
    def destroy_self(self, **kwargs):
        super().destroy_self(**kwargs)
        self.neighbor_list.clear()
        return True
    
    def empty_hist(self, **kwargs):
        self._X_hist = np.nan
        self._U_hist = np.nan
        self._W_hist = np.nan
        return True
    
    def hdf5_pick(self, handle, **kwargs):
        hdf5_kwargs = self.father.hdf5_kwargs
        obji_hist = handle.create_group(self.name)
        obji_hist.create_dataset("X_hist", data=self.X_hist, **hdf5_kwargs)
        obji_hist.create_dataset("U_hist", data=self.U_hist, **hdf5_kwargs)
        obji_hist.create_dataset("W_hist", data=self.W_hist, **hdf5_kwargs)
        return obji_hist
    
    def hdf5_load(self, handle, **kwargs):
        obji_hist = handle[self.name]
        self._X_hist = obji_hist["X_hist"][:]
        self._U_hist = obji_hist["U_hist"][:]
        self._W_hist = obji_hist["W_hist"][:]
        return obji_hist
    
    def check_self(self, **kwargs):
        err_msg = "wrong parameter value: %s "
        
        assert self.index >= 0, err_msg % "index"
        assert self.dimension in (2, 3), err_msg % "dimension"
        assert isinstance(self.father, problemClass._baseProblem), err_msg % "father"
        assert np.isfinite(self.u).all(), err_msg % "u"
        assert np.isfinite(self.X).all(), err_msg % "X"
        # assert np.isfinite(self.U).all(), err_msg % 'U'
        # assert np.isfinite(self.W).all(), err_msg % 'W'
        assert np.isfinite(self.attract), err_msg % "attract"
        assert np.isfinite(self.align), err_msg % "align"
        assert np.isfinite(self.dipole), err_msg % "dipole"
        for obji in self.neighbor_list:
            assert isinstance(obji, _baseParticle), err_msg % "neighbor_list"
        return True
    
    def update_finish(self):
        if self.rank0:
            self._X_hist = np.vstack(self.X_hist)
            self._U_hist = np.vstack(self.U_hist)
        return True


class particle2D(_baseParticle):
    def __init__(self, name="...", **kwargs):
        super().__init__(name, **kwargs)
        # self._type = 'particle2D'
        self._dimension = 2  # 2 for 2D
        self._viewRange = np.ones(1) * np.pi  # how large the camera can view.
        self._w = 0  # particle self-spin speed
        self._P1 = np.array((1, 0))  # major norm P1, for 2D version
        self._phi = 0  # angular coordinate of P1
        self._phi_hist = []  # major norm P1, for 2D version
        
        self._X = np.array((0, 0))  # particle center coordinate
        self._U = np.nan * np.array((0, 0))  # particle translational velocity in global coordinate
        self._W = np.nan * np.array((0,))  # particle rotational velocity in global coordinate
        self._neighbor_list = uniqueList(acceptType=type(self))
        self.update_phi()
    
    @_baseParticle.father.setter
    def father(self, father):
        assert isinstance(father, problemClass._base2DProblem)
        self._father = father
    
    @property
    def viewRange(self):
        return self._viewRange
    
    @viewRange.setter
    def viewRange(self, viewRange):
        err_msg = "viewRange is a scale. "
        assert viewRange.size == 1, err_msg
        assert -np.pi <= viewRange <= np.pi
        self._viewRange = viewRange
    
    @property
    def w(self):
        return self._w
    
    @w.setter
    def w(self, w):
        # phi = np.hstack((phi,))
        err_msg = "\omega is a scale. "
        assert w.size == 1, err_msg
        self._w = w
    
    @_baseParticle.P1.setter
    def P1(self, P1):
        # err_msg = 'wrong array size'
        # assert P1.size == 2, err_msg
        _baseParticle.P1.fset(self, P1)
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
    
    @_baseParticle.W.setter
    def W(self, W):
        err_msg = "W is a scale. "
        assert W.size == 1, err_msg
        _baseParticle.W.fset(self, W)
    
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
        assert isinstance(self.father, problemClass._base2DProblem), err_msg % "father"
        assert self.X.shape == (2,), err_msg % "X"
        assert self.U.shape == (2,), err_msg % "U"
        assert self.W.size == 1, err_msg % "W"
        for obji in self.neighbor_list:
            assert isinstance(obji, particle2D), err_msg % "neighbor_list"
        assert np.isfinite(self.P1).all(), err_msg % "P1"
        assert self.P1.shape == (2,), err_msg % "P1"
        assert isinstance(self.phi, np.float64), err_msg % "phi"
        assert np.isfinite(self.phi), err_msg % "phi"
        return True
    
    def update_finish(self):
        super().update_finish()
        if self.rank0:
            self._W_hist = np.hstack(self.W_hist)
            self._phi_hist = np.hstack(self.phi_hist)
        return True


class ackermann2D(particle2D):
    # todo: Add command, annotate references, and define all variables wu
    # reference:May 2017 preprint of Modern Robotics, Lynch and Park, Cambridge U. Press, 2017 http://modernrobotics.org
    """
    l_steer: wheelbase between the front and rear wheels
    phi_steer: steering angle
    """
    
    # ref, and variable's definition.
    # deninition of l_steer, phi_steer
    def __init__(self, l_steer, name="...", **kwargs):
        super().__init__(name, **kwargs)
        self._l_steer = l_steer  # wheelbase
        self._phi_steer = 0  # Car orientation angle
        self._w_steer = 0  # steer self-spin speed
        self._W_steer = 0  # steer rotational velocity in global coordinate
        
        # for dbg only, compare with ir_sim, see car_robot.py of ir_sim project for detail.
        self._goal = np.ones(3)  # the tragic location of ackermann car
        self._v_max = np.inf  # Maximum linear speed
        self._w_max = np.inf  # Maximum angular velocity
        self._goal_threshold = 0  # Target threshold
        self._phi_steer_limit = np.inf  # Car orientation angle limit
        
        # historical information
        self._phi_steer_hist = []  # Car orientation angle history list
        self._W_steer_hist = []  # steer rotational velocity in global coordinate history list
    
    @property
    def phi_steer_hist(self):
        return self._phi_steer_hist
    
    @property
    def W_steer_hist(self):
        return self._W_steer_hist
    
    @property
    def l_steer(self):
        return self._l_steer
    
    @l_steer.setter
    def l_steer(self, l_steer):
        # todo: some check wu
        l_steer = np.array(l_steer)
        err_msg = "wrong parameter value: %s "
        assert self.l_steer.size == 1, err_msg % "l_steer"
        err_msg = "wheelbase must > 0 . "
        assert l_steer > 0, err_msg
        self._l_steer = l_steer
    
    @property
    def phi_steer(self):
        return self._phi_steer
    
    @phi_steer.setter
    def phi_steer(self, phi_steer):
        # todo:some check wu
        # err_msg = 'steering angle must be between -pi and +pi'
        # assert phi_steer > -np.pi, err_msg
        # assert phi_steer < np.pi, err_msg
        self._phi_steer = phi_steer
    
    @property
    def w_steer(self):
        return self._w_steer
    
    @w_steer.setter
    def w_steer(self, w_steer):
        self._w_steer = w_steer
    
    @property
    def W_steer(self):
        return self._W_steer
    
    @W_steer.setter
    def W_steer(self, W_steer):
        self._W_steer = W_steer
    
    @property
    def goal(self):
        return self._goal
    
    @goal.setter
    def goal(self, goal):
        goal = np.array(goal)
        err_msg = "Each parameter in goal is greater than or equal to ."
        assert np.all(goal >= 0), err_msg
        self._goal = goal
    
    @property
    def v_max(self):
        return self._v_max
    
    @v_max.setter
    def v_max(self, v_max):
        v_max = np.array(v_max)
        err_msg = "v_max must >= 0 . "
        assert v_max >= 0, err_msg
        self._v_max = v_max
    
    @property
    def w_max(self):
        return self._w_max
    
    @w_max.setter
    def w_max(self, w_max):
        w_max = np.array(w_max)
        err_msg = "w_max must >= 0 . "
        assert w_max >= 0, err_msg
        self._w_max = w_max
    
    @property
    def goal_threshold(self):
        return self._goal_threshold
    
    @goal_threshold.setter
    def goal_threshold(self, goal_threshold):
        goal_threshold = np.array(goal_threshold)
        err_msg = "goal_threshold must >= 0 . "
        assert goal_threshold >= 0, err_msg
        self._goal_threshold = goal_threshold
    
    @property
    def phi_steer_limit(self):
        return self._phi_steer_limit
    
    @phi_steer_limit.setter
    def phi_steer_limit(self, phi_steer_limit):
        phi_steer_limit = np.array(phi_steer_limit)
        err_msg = "phi_steer_limit must >= 0 . "
        assert phi_steer_limit >= 0, err_msg
        self._phi_steer_limit = phi_steer_limit
    
    def update_position(self, X, phi, phi_steer=0, **kwargs):
        self.X = X
        self.phi = warpToPi(phi)
        self.phi_steer = np.clip(phi_steer, -self.phi_steer_limit, self.phi_steer_limit)
        return True
    
    def update_velocity(self, U, W, W_steer=0, **kwargs):
        self.U = U
        self.W = W
        self.W_steer = W_steer
        return True
    
    def do_store_data(self, **kwargs):
        super().do_store_data()
        if self.rank0:
            self.W_steer_hist.append(self.w_steer)  # w_steer is a float, no necessary to copy.
            self.phi_steer_hist.append(self.phi_steer)  # phi_steer is a float, no necessary to copy.
        return True
    
    def update_finish(self):
        super().update_finish()
        if self.rank0:
            self._phi_steer_hist = np.hstack(self.phi_steer_hist)
        return True


class singleForceSphere2D(particle2D):
    def __init__(self, name="...", **kwargs):
        super().__init__(name, **kwargs)
        self._prb_MR = None
    
    @property
    def prb_MR(self):
        return self._prb_MR
    
    @prb_MR.setter
    def prb_MR(self, prb_MR):
        from src import stokes_flow as sf
        err_msg = "wrong problem type of prb_MR, current: %s " % repr(prb_MR)
        assert isinstance(prb_MR, sf.ForceSphere2DProblem), err_msg
        self._prb_MR = prb_MR
    
    def check_self(self, **kwargs):
        from src import stokes_flow as sf
        
        super().check_self(**kwargs)
        err_msg = "wrong parameter value: %s "
        assert self.dimension in (2,), err_msg % "dimension"
        assert isinstance(self.father, problemClass.singleForceSphere2DProblem), err_msg % "father"
        assert isinstance(self.prb_MR, sf.ForceSphere2DProblem), err_msg % "prb_MR"
        return True


class ForceSphere2D(singleForceSphere2D):
    def __init__(self, name="...", **kwargs):
        super().__init__(name, **kwargs)
        self._prb_MR = None
        self._r = np.nan  # particle radius
    
    @property
    def prb_MR(self):
        return self._prb_MR
    
    @prb_MR.setter
    def prb_MR(self, prb_MR):
        from src import stokes_flow as sf
        err_msg = "wrong problem type for mobility and resistance matrices calculation, current: %s " % repr(prb_MR)
        assert isinstance(prb_MR, sf.ForceSphere2DProblem)
        self._prb_MR = prb_MR
    
    @particle2D.X.setter
    def X(self, X):
        self._X = np.hstack(X)
    
    @particle2D.phi.setter
    def phi(self, phi):
        self._phi = phi
    
    @particle2D.u.setter
    def u(self, u):
        self._u = u
    
    @particle2D.U.setter
    def U(self, U):
        self._U = np.array(U)
    
    @particle2D.w.setter
    def w(self, w):
        self._w = w
    
    @particle2D.W.setter
    def W(self, W):
        self._W = np.array(W)
    
    @property
    def r(self):
        return self._r
    
    @r.setter
    def r(self, r):
        self._r = r
    
    def check_self(self, **kwargs):
        from src import stokes_flow as sf
        err_msg = "wrong parameter value: %s "
        
        assert self.index >= 0, err_msg % "index"
        assert isinstance(self.father, problemClass._baseProblem), err_msg % "father"
        assert np.isfinite(self.u).all(), err_msg % "u"
        assert np.isfinite(self.X).all(), err_msg % "X"
        assert np.isfinite(self.U).all(), err_msg % 'U'
        assert np.isfinite(self.W).all(), err_msg % 'W'
        
        assert self.dimension in (2,), err_msg % "dimension"
        assert isinstance(self.father, problemClass.ForceSphere2DProblem), err_msg % "father"
        assert isinstance(self.prb_MR, sf.ForceSphere2DProblem), err_msg % "prb_MR"
        # assert self.X.shape == (2,), err_msg % "X"
        # assert self.U.shape == (2,), err_msg % "U"
        # assert self.W.size == 1, err_msg % "W"
        # for obji in self.neighbor_list:
        #     assert isinstance(obji, particle2D), err_msg % "neighbor_list"
        # assert np.isfinite(self.P1).all(), err_msg % "P1"
        # assert self.P1.shape == (2,), err_msg % "P1"
        # assert isinstance(self.phi, np.float64), err_msg % "phi"
        # assert np.isfinite(self.phi), err_msg % "phi"
        return True


class ForceSphere2D_matrixObj(ForceSphere2D):
    # def __init__(self, name="...", **kwargs):
    #     super().__init__(name, **kwargs)
    #     self._prb_MR = None
    #     self._r = np.nan  # particle radius
    
    def check_self(self, **kwargs):
        from src import stokes_flow as sf
        err_msg = "wrong parameter value: %s "
        
        assert self.index >= 0, err_msg % "index"
        assert isinstance(self.father, problemClass._baseProblem), err_msg % "father"
        assert np.isfinite(self.u).all(), err_msg % "u"
        assert np.isfinite(self.X).all(), err_msg % "X"
        assert np.isfinite(self.U).all(), err_msg % 'U'
        assert np.isfinite(self.W).all(), err_msg % 'W'
        
        assert self.dimension in (2,), err_msg % "dimension"
        assert isinstance(self.father, problemClass.ForceSphere2D_matrixPro), err_msg % "father"
        # assert isinstance(self.prb_MR, sf.ForceSphere2DProblem), err_msg % "prb_MR"
        # assert self.X.shape == (2,), err_msg % "X"
        # assert self.U.shape == (2,), err_msg % "U"
        # assert self.W.size == 1, err_msg % "W"
        # for obji in self.neighbor_list:
        #     assert isinstance(obji, particle2D), err_msg % "neighbor_list"
        # assert np.isfinite(self.P1).all(), err_msg % "P1"
        # assert self.P1.shape == (2,), err_msg % "P1"
        # assert isinstance(self.phi, np.float64), err_msg % "phi"
        # assert np.isfinite(self.phi), err_msg % "phi"
        return True


class finiteDipole2D(particle2D):
    def __init__(self, length, name="...", **kwargs):
        super().__init__(name, **kwargs)
        self._Z = np.nan
        self._Zl = np.nan
        self._Zr = np.nan
        self._length = length  # length of particle
        self.update_dipole()
        self.update_Z()
    
    @property
    def length(self):
        return self._length
    
    @length.setter
    def length(self, length):
        err_msg = "length is a scale. "
        assert length.size == 1, err_msg
        self._length = length
        self.update_Z()
        self.update_dipole()
    
    @_baseParticle.X.setter
    def X(self, X):
        _baseParticle.X.fset(self, X)
        self.update_Z()
    
    @_baseParticle.father.setter
    def father(self, father):
        assert isinstance(father, problemClass.finiteDipole2DProblem)
        self._father = father
    
    @particle2D.phi.setter
    def phi(self, phi):
        particle2D.phi.fset(self, phi)
        self.update_Z()
    
    @property
    def Z(self):
        return self._Z  # center in complex plane
    
    @property
    def Zl(self):
        return self._Zl  # left vortex
    
    @property
    def Zr(self):
        return self._Zr  # right vortex
    
    @_baseParticle.u.setter
    def u(self, u):
        _baseParticle.u.fset(self, u)
        self.update_dipole()
    
    def update_Z(self):
        self._Z = self.X[0] + 1j * self.X[1]
        l = self.length
        phi = self.phi  # alpha in the theory.
        t1 = l / 2 * np.exp(1j * (phi + np.pi / 2))
        self._Zl = self.Z + t1
        self._Zr = self.Z - t1
        return True
    
    def update_dipole(self):
        # \tau in my theory, see the draft for detail.
        self._dipole = 2 * np.pi * self.length * self.u
        return True
    
    def UDipole2Dat(self, Z):
        tau = self.dipole
        Zl = self.Zl  # left vortex
        Zr = self.Zr  # right vortex
        wo = (1j * tau / (2 * np.pi) * (1 / (Z - Zr) - 1 / (Z - Zl)))  # W = u - i * v: complex velocity
        return wo
    
    def UselfPropelled2D(self):
        U = self.u * self.P1
        return U
    
    def check_self(self, **kwargs):
        super().check_self()
        err_msg = "wrong parameter value: %s "
        assert np.isfinite(self.length), err_msg % "length"


class limFiniteDipole2D(finiteDipole2D):
    @_baseParticle.father.setter
    def father(self, father):
        assert isinstance(father, problemClass.limFiniteDipole2DProblem)
        self._father = father
    
    def UDipole2Dat(self, Zn):
        tau = self.dipole
        l = self.length
        phi = self.phi
        dZ = Zn - self.Z
        t1 = ((tau * l) / (2 * np.pi)) * (np.exp(1j * phi) / dZ ** 2)
        # Ui = np.dstack((np.real(t1), -np.imag(t1)))
        # print(Ui.shape)
        # print(self, Zn, self.Z, np.real(t1), -np.imag(t1))
        return np.real(t1), -np.imag(t1)
    
    def UDipole2Dof(self, obji: "finiteDipole2D"):
        # print(self, obji, obji.Z)
        # print(self, self, self.Z)
        return np.array(self.UDipole2Dat(obji.Z))
    
    def WDipole2Dat(self, Zn, phin):
        tau = self.dipole
        l = self.length
        phi = self.phi
        dZ = Zn - self.Z
        Wi = np.real(1j * np.exp(1j * (2 * phin + phi)) / dZ ** 3 * (tau * l / np.pi))
        return Wi
    
    def WDipole2Dof(self, obji: "finiteDipole2D"):
        return self.WDipole2Dat(obji.Z, obji.phi)
    
    def UWDipole2Dat(self, Zn, phin):
        tau = self.dipole
        l = self.length
        phi = self.phi
        dZ = Zn - self.Z
        t1 = ((tau * l) / (2 * np.pi)) * (np.exp(1j * phi) / dZ ** 2)
        Wi = np.real(1j * np.exp(1j * (2 * phin + phi)) / dZ ** 3 * (tau * l / np.pi))
        return np.real(t1), -np.imag(t1), Wi
    
    def UWDipole2Dof(self, obji: "finiteDipole2D"):
        u1, u2, w = self.UWDipole2Dat(obji.Z, obji.phi)
        return np.array((u1, u2)), w
