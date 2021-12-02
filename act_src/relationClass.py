# coding=utf-8
"""
20210810
Zhang Ji

global perspective, handle the positional (and other external) relationships.
"""
from petsc4py import PETSc
import abc
import numpy as np
import scipy as scp
from scipy.spatial import Voronoi
# from petsc4py import PETSc
from act_src import baseClass
from act_src import particleClass
from act_src import problemClass
from act_codeStore import support_fun as spf


# from act_codeStore import support_fun as spf


# from act_codeStore.support_class import *


# from act_codeStore.support_class import *

class _baseRelation(baseClass.baseObj):
    @abc.abstractmethod
    def check_self(self, **kwargs):
        return

    def update_prepare(self, **kwargs):
        return True

    # @abc.abstractmethod
    def update_relation(self, **kwargs):
        return

    @abc.abstractmethod
    def update_neighbor(self, **kwargs):
        return

    def print_info(self):
        super().print_info()
        spf.petscInfo(self.father.logger, '  None')
        return True


class _baseRelation2D(_baseRelation):
    def __init__(self, name='...', **kwargs):
        super().__init__(name, **kwargs)
        self._theta_ij = np.nan  # angle between ei and (rj - ri)
        self._e_rho_ij = np.nan  # unite direction (ri - rj) / |ri - rj|
        self._rho_ij = np.nan  # inter-swimmer distance |ri - rj|
        self._phi_rho_ij = np.nan  # angle of e_rho_ij, np.arctan2(e_rho_ij[1], e_rho_ij[0])
        self._overlap_epsilon = 0

    @property
    def theta_ij(self):
        return self._theta_ij

    @property
    def e_rho_ij(self):
        return self._e_rho_ij

    @property
    def phi_rho_ij(self):
        return self._phi_rho_ij

    @property
    def rho_ij(self):
        return self._rho_ij

    @property
    def overlap_epsilon(self):
        return self._overlap_epsilon

    @overlap_epsilon.setter
    def overlap_epsilon(self, e):
        assert e >= 0
        self._overlap_epsilon = e

    def _check_overlap(self):
        err_msg = 'some particles overlap'
        t1 = self.rho_ij > self.overlap_epsilon
        np.fill_diagonal(t1, True)
        assert t1.all(), err_msg
        return True

    def check_self(self, **kwargs):
        self._check_overlap()
        return True

    # this is a temple, most time do not use this function during simulation.
    @staticmethod
    def theta_phi_rho_each(obji: "particleClass.particle2D",
                           objj: "particleClass.particle2D"):
        ei, ri = obji.P1, obji.X
        ej, rj = objj.P1, objj.X
        Pij = rj - ri
        Pji = ri - rj
        t1 = np.linalg.norm(Pij)
        theta_ij = np.arccos(np.dot(ei, Pij) / (np.linalg.norm(ei) * t1))
        theta_ji = np.arccos(np.dot(ej, Pji) / (np.linalg.norm(ej) * t1))
        phi_ij = np.arccos(np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej)))
        return theta_ij, theta_ji, phi_ij

    def cal_theta_rho(self):
        prb = self.father  # type: problemClass._baseProblem
        theta_ij, e_rho_ij, phi_rho_ij, rho_ij = [], [], [], []
        obji: particleClass.particle2D
        for i0, obji in enumerate(prb.obj_list):
            tPij = prb.Xall - obji.X
            # tPji = -tPij
            trhoij = np.linalg.norm(tPij, axis=-1)
            tphi_rhoij = np.arctan2(tPij[:, 1], tPij[:, 0])
            e_rho_ij.append(tPij)
            rho_ij.append(trhoij)
            phi_rho_ij.append(tphi_rhoij)
            theta_ij.append(obji.phi - tphi_rhoij)
        self._e_rho_ij = np.array(e_rho_ij)
        self._phi_rho_ij = np.vstack(phi_rho_ij)
        self._rho_ij = np.vstack(rho_ij)
        self._theta_ij = np.vstack(theta_ij)
        return True

    # def cal_theta_rho(self):
    #     prb = self.father  # type: problemClass.baseProblem
    #     theta_ij, e_rho_ij, rho_ij = [], [], []
    #     for obji in prb.obj_list:
    #         # ei, ri = obji.P1, obji.X
    #         tPij = prb.Xall - obji.X
    #         e_rho_ij.append(tPij)
    #         # tPji = -tPij
    #         trhoij = np.linalg.norm(tPij, axis=-1)
    #         rho_ij.append(trhoij)
    #         theta_ij.append(np.arccos(np.einsum('i,ji', obji.P1, tPij) / (np.linalg.norm(obji.P1) * trhoij)))
    #         # theta_ji.append(np.arccos(np.einsum('i,ji', obji.P1, tPij)
    #         #                           / (np.linalg.norm(obji.P1) * trhoij)))
    #     self._e_rho_ij = np.array(e_rho_ij)
    #     self._rho_ij = np.vstack(rho_ij)
    #     self._theta_ij = np.vstack(theta_ij)
    #     return True

    # def update_relation(self, **kwargs):
    #     self.cal_theta_rho()
    #     return True

    # def update_neighbor(self, **kwargs):
    #     prb = self.father  # type: problemClass._baseProblem
    #     for obji in prb.obj_list:  # type: particleClass.particle2D
    #         neighbor_list = obji.neighbor_list
    #         neighbor_list.clear()
    #         for objj in prb.obj_list:  # type: particleClass.particle2D
    #             if obji is not objj:
    #                 neighbor_list.append_noCheck(objj)
    #     return True

    def update_relation(self, **kwargs):
        pass

    def update_neighbor(self, **kwargs):
        pass

    def print_info(self):
        baseClass.baseObj.print_info(self)
        spf.petscInfo(self.father.logger, '  overlap_epsilon=%f' % self.overlap_epsilon)
        return True


class finiteRelation2D(_baseRelation2D):
    def cal_rho(self):
        prb = self.father  # type: problemClass._baseProblem
        rho_ij = []
        for obji in prb.obj_list:
            tPij = prb.Xall - obji.X
            trhoij = np.linalg.norm(tPij, axis=-1)
            rho_ij.append(trhoij)
        self._rho_ij = np.vstack(rho_ij)
        return True

    def update_relation(self, **kwargs):
        self.cal_rho()
        return True


class limFiniteRelation2D(finiteRelation2D):
    def _noting(self):
        pass


class VoronoiBaseRelation2D(_baseRelation2D):
    def __init__(self, name='...', **kwargs):
        super().__init__(name=name, **kwargs)
        self._fun_update_neighbor = None

    def cal_theta_rho(self):
        prb = self.father  # type: problemClass._baseProblem
        theta_ij, rho_ij = [], []
        theta_ij, rho_ij = np.zeros((prb.n_obj, prb.n_obj)), np.zeros((prb.n_obj, prb.n_obj)),
        obji: particleClass.particle2D
        for i0, obji in enumerate(prb.obj_list):
            tPij = prb.Xall - obji.X
            # tPij = prb.Xall - obji.X
            # tPji = -tPij
            tphi_rhoij = np.arctan2(tPij[:, 1], tPij[:, 0])
            trhoij = np.linalg.norm(tPij, axis=-1)
            # trhoij = scp.linalg.norm(tPij, axis=-1)
            # rho_ij.append(trhoij)
            # theta_ij.append(tphi_rhoij - obji.phi)
            rho_ij[i0, :] = trhoij
            theta_ij[i0, :] = tphi_rhoij - obji.phi
        self._rho_ij = np.vstack(rho_ij)
        self._theta_ij = np.vstack(theta_ij)
        return True

    def update_prepare(self):
        if self.father.n_obj > 4:
            self._fun_update_neighbor = self.update_neighbor_voronoi
        else:
            self._fun_update_neighbor = self.update_neighbor_all
        return True

    def update_relation(self, **kwargs):
        self.cal_theta_rho()
        return True

    def update_neighbor_voronoi(self, **kwargs):
        prb = self.father  # type: problemClass._baseProblem
        X_list = prb.Xall
        vor = Voronoi(X_list)
        idx_X2ridge = [[] for _ in vor.points]
        idx_ridge: int
        for idx_ridge, (X0, X1) in enumerate(vor.ridge_points):
            idx_X2ridge[X0].append(idx_ridge)
            idx_X2ridge[X1].append(idx_ridge)

        idx_ridge2X = vor.ridge_points
        obji: particleClass.particle2D
        for obji, idxi_X2ridge in zip(prb.obj_list, idx_X2ridge):
            obji.neighbor_list.clear()
            # idx_X = np.hstack([idx_ridge2X[i0] for i0 in idxi_X2ridge])
            # tidx = np.logical_not(np.isclose(idx_X, obji.index))
            # for i0 in idx_X[tidx]:
            #     obji.neighbor_list.append_noCheck(prb.obj_list[i0])
            # for i0 in [idx_ridge2X[i0] for i0 in idxi_X2ridge]:
            for i0 in idxi_X2ridge:
                t1 = idx_ridge2X[i0]
                if t1[0] == obji.index:
                    obji.neighbor_list.append_noCheck(prb.obj_list[t1[1]])
                else:
                    obji.neighbor_list.append_noCheck(prb.obj_list[t1[0]])
        return True

    def update_neighbor_all(self, **kwargs):
        prb = self.father  # type: problemClass._baseProblem
        for obji in prb.obj_list:
            obji.neighbor_list.clear()
            for objj in prb.obj_list:
                obji.neighbor_list.append_noCheck(objj)
        return True

    def update_neighbor(self, **kwargs):
        return self._fun_update_neighbor(**kwargs)

    def dbg_showVoronoi(self, **kwargs):
        # from matplotlib import pyplot as plt
        prb = self.father  # type: problemClass._baseProblem
        X_list = prb.Xall
        vor = Voronoi(X_list)

        ##########################################
        from scipy.spatial import voronoi_plot_2d
        fig = voronoi_plot_2d(vor)
        axi = fig.gca()
        axi.set_xlabel('x')
        axi.set_ylabel('y')
        for obji in prb.obj_list:
            axi.arrow(*obji.X, *obji.P1 * 3, head_width=0.3)
            axi.text(*obji.X, obji.index)
        fig.show()

        # print()
        # for obji in prb.obj_list:
        #     print(obji.index, [objj.index for objj in obji.neighbor_list])
        # print(idx_X2ridge)

        return True


class AllBaseRelation2D(VoronoiBaseRelation2D):
    def update_prepare(self):
        self._fun_update_neighbor = self.update_neighbor_all
        return True
