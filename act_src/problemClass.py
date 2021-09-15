# coding=utf-8
"""
20210810
Zhang Ji

problem class, organize the problem.
"""
import abc
import pickle
from tqdm import tqdm
import numpy as np
from tqdm.notebook import tqdm as tqdm_notebook
from petsc4py import PETSc
from datetime import datetime

from act_src import baseClass
from act_src import particleClass
from act_src import interactionClass
from act_src import relationClass
from act_codeStore.support_class import *


class _baseProblem(baseClass.baseObj):
    def __init__(self, name='...', tqdm_fun=tqdm_notebook, **kwargs):
        super().__init__(name, **kwargs)
        # self._type = 'baseProblem'
        self._dimension = -1  # -1 for undefined, 2 for 2D, 3 for 3D
        self._rot_noise = 0  # rotational noise
        self._trs_noise = 0  # translational noise
        self._obj_list = uniqueList(acceptType=particleClass._baseParticle)  # contain objects
        self._action_list = uniqueList(
            acceptType=interactionClass._baseAction)  # contain rotational interactions
        self._Xall = np.nan  # location at current time
        self._Wall = np.nan  # rotational velocity at current time
        self._Uall = np.nan  # translational velocity at current time
        self._relationHandle = relationClass._baseRelation()
        self._pick_filename = '...'

        # parameters for temporal evaluation.
        self._comm = PETSc.COMM_WORLD
        self._save_every = 1
        self._tqdm_fun = tqdm_fun
        self._tqdm = None
        self._update_fun = '3bs'  # funHandle and order
        self._update_order = (1e-6, 1e-9)  # rtol, atol
        self._t0 = 0  # simulation time in the range (t0, t1)
        self._t1 = -1  # simulation time in the range (t0, t1)
        self._eval_dt = -1  # \delta_t, simulation
        self._max_it = -1  # iteration loop no more than max_it
        self._percentage = 0  # percentage of time depend solver.
        self._t_hist = []
        self._dt_hist = []
        self._tmp_idx = []  # temporary globe idx

    @property
    def dimension(self):
        return self._dimension

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
    def action_list(self):
        return self._action_list

    @property
    def obj_list(self):
        return self._obj_list

    @property
    def Uall(self):
        return self._Uall

    @Uall.setter
    def Uall(self, Uall):
        self._Uall = Uall

    @property
    def Wall(self):
        return self._Wall

    @Wall.setter
    def Wall(self, Wall):
        self._Wall = Wall

    @property
    def Xall(self):
        return self._Xall

    @Xall.setter
    def Xall(self, Xall):
        self._Xall = Xall

    @property
    def relationHandle(self):
        return self._relationHandle

    @relationHandle.setter
    def relationHandle(self, relationHandle):
        self._check_relationHandle(relationHandle)
        relationHandle.father = self
        self._relationHandle = relationHandle

    @property
    def comm(self):
        return self._comm

    @property
    def save_every(self):
        return self._save_every

    @save_every.setter
    def save_every(self, save_every):
        self._save_every = int(save_every)

    @property
    def tqdm_fun(self):
        return self._tqdm_fun

    @tqdm_fun.setter
    def tqdm_fun(self, tqdm_fun):
        self._tqdm_fun = tqdm_fun

    @property
    def tqdm(self):
        return self._tqdm

    @tqdm.setter
    def tqdm(self, mytqdm):
        err_msg = 'wrong parameter type tqdm. '
        assert isinstance(mytqdm, tqdm), err_msg
        self._tqdm = mytqdm

    @property
    def update_fun(self):
        return self._update_fun

    @update_fun.setter
    def update_fun(self, update_fun):
        assert update_fun in ("1fe", "2a", "3", "3bs", "4", "5f",
                              "5dp", "5bs", "6vr", "7vr", "8vr")
        self._update_fun = update_fun

    @property
    def update_order(self):
        return self._update_order

    @update_order.setter
    def update_order(self, update_order):
        assert len(update_order) == 2
        self._update_order = update_order

    @property
    def rtol(self):
        return self.update_order[0]

    @property
    def atol(self):
        return self.update_order[1]

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, t0):
        self._t0 = t0

    @property
    def t1(self):
        return self._t1

    @t1.setter
    def t1(self, t1):
        self._t1 = t1

    @property
    def eval_dt(self):
        return self._eval_dt

    @eval_dt.setter
    def eval_dt(self, eval_dt):
        self._eval_dt = eval_dt

    @property
    def max_it(self):
        return self._max_it

    @max_it.setter
    def max_it(self, max_it):
        self._max_it = max_it

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, percentage):
        self._percentage = percentage

    @property
    def t_hist(self):
        return self._t_hist

    @property
    def dt_hist(self):
        return self._dt_hist

    @property
    def tmp_idx(self):
        return self._tmp_idx

    @property
    def n_obj(self):
        return len(self.obj_list)

    @property
    def polar(self) -> np.asarray:
        polar = np.linalg.norm(np.sum(np.vstack([obji.P1 for obji in self.obj_list]), axis=0)) / self.n_obj
        return polar

    @property
    def milling_Daniel2014(self) -> np.asarray:
        t1 = np.vstack([np.cross(obji.X, obji.P1) / np.linalg.norm(obji.X) for obji in self.obj_list])
        milling = np.linalg.norm(np.sum(t1, axis=0)) / self.n_obj
        return milling

    def _check_add_obj(self, obj):
        err_msg = 'wrong object type'
        assert isinstance(obj, particleClass._baseParticle), err_msg
        err_msg = 'wrong dimension'
        assert np.isclose(self.dimension, obj.dimension), err_msg
        return True

    def add_obj(self, obj):
        self._check_add_obj(obj)
        obj.index = self.n_obj
        obj.father = self
        self._obj_list.append(obj)
        return True

    def _check_add_act(self, act):
        # err_msg = 'wrong object type'
        # assert isinstance(act, interaction.WAction), err_msg
        pass

    def add_act(self, act: "interactionClass._baseAction", add_all_obj=True):
        self._check_add_act(act)
        self.action_list.append(act)
        act.father = self
        if add_all_obj:
            for obji in self.obj_list:
                act.add_obj(obji)
        return True

    @staticmethod
    def _check_relationHandle(pos: "relationClass._baseRelation"):
        err_msg = 'wrong object type'
        assert isinstance(pos, relationClass._baseRelation), err_msg
        # pass

    def update_prepare(self):
        self.Xall = np.vstack([objj.X for objj in self.obj_list])
        # self.Uall = np.vstack([objj.U for objj in self.obj_list])
        # self.Wall = np.vstack([objj.W for objj in self.obj_list])
        self.update_step()
        for acti in self.action_list:  # type: interactionClass._baseAction
            acti.update_prepare()
        self.check_self()
        self.print_info()
        return True

    def update_step(self):
        self.relationHandle.update_relation()
        self.relationHandle.update_neighbor()
        self.relationHandle.check_self()
        return True

    def update_UWall(self, F):
        self.update_step()
        F.zeroEntries()
        # PETSc.Sys.Print(F.getArray())
        # print(F.getArray())
        for action in self.action_list:
            action.update_action(F)
            # F.assemble()
        F.assemble()
        # PETSc.Sys.Print(F.getArray())
        return True

    def check_self(self, **kwargs):
        self._obj_list = np.array(self.obj_list)
        # todo: check all parameters.
        err_msg = 'wrong parameter value: %s '
        assert self.dimension in (2, 3), err_msg % 'dimension'

        for obji in self.obj_list:  # type: particleClass._baseParticle
            obji.check_self()
        for acti in self.action_list:  # type: interactionClass._baseAction
            acti.check_self()
        self.relationHandle.check_self()
        return True

    def update_self(self, t1, t0=0, max_it=10 ** 9, eval_dt=0.001):
        comm = self.comm
        (rtol, atol) = self.update_order
        update_fun = self.update_fun
        tqdm_fun = self.tqdm_fun
        self.t1 = t0
        self.t1 = t1
        self.eval_dt = eval_dt
        self.max_it = max_it
        self.percentage = 0
        self.update_prepare()
        self.tqdm = tqdm_fun(total=100)

        # do simulation
        y0 = self._get_y0()
        y = PETSc.Vec().createWithArray(y0, comm=comm)
        f = y.duplicate()
        # print(f)
        # print(1111)
        ts = PETSc.TS().create(comm=comm)
        ts.setProblemType(ts.ProblemType.NONLINEAR)
        ts.setType(ts.Type.RK)
        ts.setRKType(update_fun)
        ts.setRHSFunction(self._rhsfunction, f)
        ts.setTime(t0)
        ts.setMaxTime(t1)
        ts.setMaxSteps(max_it)
        ts.setTimeStep(eval_dt)
        ts.setMonitor(self._monitor)
        ts.setPostStep(self._postfunction)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        # ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)
        ts.setFromOptions()
        ts.setSolution(y)
        ts.setTolerances(rtol, atol)
        ts.setUp()
        # self._do_store_data(ts, 0, 0, y)
        ts.solve(y)

        # finish simulation
        self.update_finish(ts)
        return True

    @abc.abstractmethod
    def update_position(self, **kwargs):
        return

    @abc.abstractmethod
    def update_velocity(self, **kwargs):
        return

    def update_hist(self, **kwargs):
        for obji in self.obj_list:
            obji.do_store_data()
        return True

    @abc.abstractmethod
    def _get_y0(self, **kwargs):
        return

    @abc.abstractmethod
    def _rhsfunction(self, ts, t, Y, F):
        return

    def _do_store_data(self, ts, i, t, Y):
        if t > self.max_it:
            return False
        else:
            dt = ts.getTimeStep()
            self.t_hist.append(t)
            self.dt_hist.append(dt)
            self.update_hist()
            return True

    def _monitor(self, ts, i, t, Y):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        save_every = self._save_every
        # print(ts.getTimeStep())
        if not i % save_every:
            percentage = np.clip(t / self._t1 * 100, 0, 100)
            dp = int(percentage - self._percentage)
            if (dp >= 1) and (rank == 0):
                self._tqdm.update(dp)
                self._percentage = self._percentage + dp
            self._do_store_data(ts, i, t, Y)
        return True

    @abc.abstractmethod
    def _postfunction(self, ts):
        return

    def update_finish(self, ts):
        # comm = PETSc.COMM_WORLD.tompi4py()
        # rank = comm.Get_rank()
        # if rank == 0:
        self._tqdm.update(100 - self._percentage)
        self._tqdm.close()
        self._t_hist = np.hstack(self.t_hist)
        self._dt_hist = np.hstack(self.dt_hist)
        # i = ts.getStepNumber()
        # t = ts.getTime()
        # Y = ts.getSolution()
        # self._do_store_data(ts, i, t, Y)

        for obji in self.obj_list:  # type: particleClass._baseParticle
            obji.update_finish()
        for acti in self.action_list:  # type: interactionClass._baseAction
            acti.update_finish()
        self.relationHandle.check_self()

        PETSc.Sys.Print()
        PETSc.Sys.Print('Solve, finish time: %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return True

    def _destroy_problem(self):
        self._comm = None
        self._tqdm = None
        pass

    def destroy_self(self):
        self._destroy_problem()
        for obji in self.obj_list:  # type: particleClass._baseParticle
            obji.destroy_self()
        for acti in self.action_list:  # type: interactionClass._baseAction
            acti.destroy_self()
        self.relationHandle.destroy_self()
        return True

    def pickmyself(self, filename: str):
        comm = PETSc.COMM_WORLD.tompi4py()
        rank = comm.Get_rank()
        self.destroy_self()
        self._pick_filename = filename

        # dbg
        # print('dbg')
        # self._obj_list = None
        # self._action_list = None
        # self._relationHandle = None
        if rank == 0:
            with open(filename, 'wb') as handle:
                pickle.dump(self, handle, protocol=4)
        return True

    def print_self_info(self):
        PETSc.Sys.Print('  rotational noise: %f, translational noise: %f' %
                        (self.rot_noise, self.trs_noise))

    def print_info(self):
        # OptDB = PETSc.Options()
        PETSc.Sys.Print()
        PETSc.Sys.Print('Information about %s (%s): ' % (str(self), self.type,))
        PETSc.Sys.Print('  This is a %d dimensional problem, contain %d objects. ' %
                        (self.dimension, self.n_obj))
        PETSc.Sys.Print('  update function: %s, update order: %s, max loop: %d' %
                        (self.update_fun, self.update_order, self.max_it))
        PETSc.Sys.Print('  t0=%f, t1=%f, dt=%f' %
                        (self.t0, self.t1, self.eval_dt))
        self.print_self_info()

        for acti in self.action_list:  # type: interactionClass._baseAction
            acti.print_info()
        self.relationHandle.print_info()

        PETSc.Sys.Print()
        PETSc.Sys.Print('Solve, start time: %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return True


class _base2DProblem(_baseProblem):
    def __init__(self, name='...', **kwargs):
        super().__init__(name, **kwargs)
        self._Phiall = np.nan
        self._dimension = 2  # 2 for 2D
        self._action_list = uniqueList(acceptType=interactionClass._baseAction2D)  # contain rotational interactions

    @property
    def Phiall(self):
        return self._Phiall

    @Phiall.setter
    def Phiall(self, Phiall):
        self._Phiall = Phiall

    def _check_add_obj(self, obj):
        super()._check_add_obj(obj)
        err_msg = 'wrong object type'
        assert isinstance(obj, particleClass.particle2D), err_msg
        return True

    def update_prepare(self):
        self.Phiall = np.vstack([objj.phi for objj in self.obj_list])
        super().update_prepare()
        return True

    def update_position(self, **kwargs):
        obji: particleClass.particle2D
        Xi: np.ndarray
        phii: np.ndarray
        for obji, Xi, phii in zip(self.obj_list, self.Xall, self.Phiall):
            obji.update_position(Xi, phii)
        return True

    def update_velocity(self, **kwargs):
        obji: particleClass.particle2D
        Uall: np.ndarray
        Wall: np.ndarray
        for obji, Ui, Wi in zip(self.obj_list, self.Uall, self.Wall):
            obji.update_velocity(Ui, Wi)
        return True

    def _get_y0(self, **kwargs):
        X_all, phi_all = [], []
        for obji in self.obj_list:
            X_all.append(obji.X)
            phi_all.append(obji.phi)
        y0 = np.hstack([np.hstack(X_all), np.hstack(phi_all)])
        return y0

    def Y2Xphi(self, Y):
        y = self.vec_scatter(Y, destroy=False)
        nobj = self.n_obj
        dim = self.dimension
        X_size = dim * nobj
        X_all = y[0:X_size]
        phi_all = y[X_size:]
        return X_all, phi_all

    def _rhsfunction(self, ts, t, Y, F):
        # structure:
        #   Y = [X_all, phi_all]
        #   F = [U_all, W_all]
        X_all, phi_all = self.Y2Xphi(Y)
        self.Xall, self.Phiall = X_all.reshape((-1, 2)), phi_all
        self.update_position()
        self.update_UWall(F)
        tF = self.vec_scatter(F)
        self.Uall = tF[:self.dimension * self.n_obj].reshape((-1, 2))
        self.Wall = tF[self.dimension * self.n_obj:]
        self.update_velocity()
        # F.assemble()
        # PETSc.Sys.Print()
        # PETSc.Sys.Print('dbg', t)
        # PETSc.Sys.Print('%+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, ' % (
        #     Y.getArray()[0], Y.getArray()[1], Y.getArray()[2],
        #     Y.getArray()[3], Y.getArray()[4], Y.getArray()[5], ))
        # PETSc.Sys.Print('%+.10f, %+.10f, %+.10f, %+.10f, %+.10f, %+.10f, ' % (
        #     F.getArray()[0], F.getArray()[1], F.getArray()[2],
        #     F.getArray()[3], F.getArray()[4], F.getArray()[5], ))
        return True

    def _postfunction(self, ts):
        return True


class finiteDipole2DProblem(_base2DProblem):
    def _check_add_obj(self, obj):
        super()._check_add_obj(obj)
        err_msg = 'wrong object type'
        assert isinstance(obj, particleClass.finiteDipole2D), err_msg
        return True


class limFiniteDipole2DProblem(_base2DProblem):
    def _check_add_obj(self, obj):
        super()._check_add_obj(obj)
        err_msg = 'wrong object type'
        assert isinstance(obj, particleClass.limFiniteDipole2D), err_msg
        return True


class behavior2DProblem(_base2DProblem):
    def __init__(self, name='...', **kwargs):
        super().__init__(name, **kwargs)
        self._attract = np.nan  # attract intensity
        self._align = np.nan  # align intensity

    @property
    def attract(self):
        return self._attract

    @attract.setter
    def attract(self, attract):
        self._attract = attract

    @property
    def align(self):
        return self._align

    @align.setter
    def align(self, align):
        self._align = align

    def add_obj(self, obj: "particleClass._baseParticle"):
        super().add_obj(obj)
        obj.attract = self.attract
        obj.align = self.align
        obj.rot_noise = self.rot_noise
        obj.trs_noise = self.trs_noise
        return True

    def print_self_info(self):
        super().print_self_info()
        PETSc.Sys.Print('  align: %f, attract: %f' %
                        (self.align, self.attract))


class behaviorFiniteDipole2DProblem(behavior2DProblem, finiteDipole2DProblem):
    def _nothing(self):
        pass


class actLimFiniteDipole2DProblem(behavior2DProblem, limFiniteDipole2DProblem):
    def _nothing(self):
        pass
