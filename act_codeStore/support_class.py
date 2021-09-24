import sys
import copy
from collections import UserList
import numpy as np
# from scipy.spatial.transform import Rotation as spR
from act_codeStore.support_fun import Rloc2glb

__all__ = ['uniqueList', 'typeList', 'intList', 'floatList',
           'abs_comp', 'abs_construct_matrix',
           'fullprint',
           'coordinate_transformation',
           'Quaternion']


class uniqueList(UserList):
    def __init__(self, liste=None, acceptType=None):
        if liste is None:
            liste = []
        self._acceptType = acceptType
        super().__init__(liste)

    def check(self, other):
        err_msg = 'only type %s is accepted. ' % (self._acceptType)
        assert self._acceptType is None or isinstance(other, self._acceptType), err_msg

        err_msg = 'item ' + repr(other) + ' add muilt times. '
        assert self.count(other) == 0, err_msg

    def __add__(self, other):
        self.check(other)
        super().__add__(other)

    def append(self, item):
        self.check(item)
        super().append(item)

    def append_noCheck(self, item):
        super().append(item)


class typeList(UserList):
    def __init__(self, acceptType):
        self._acceptType = acceptType
        super().__init__()

    def check(self, other):
        err_msg = 'only type %s is accepted. ' % (self._acceptType)
        assert self._acceptType is None or isinstance(other, self._acceptType), err_msg

    def __add__(self, other):
        self.check(other)
        super().__add__(other)

    def append(self, item):
        self.check(item)
        super().append(item)


class intList(typeList):
    def __init__(self):
        super().__init__(int)


class floatList(typeList):
    def __init__(self):
        super().__init__(float)


class abs_comp:
    def __init__(self, **kwargs):
        need_args = ['name']
        opt_args = {'childType': abs_comp}
        self._kwargs = kwargs
        self.check_args(need_args, opt_args)

        self._father = None
        self._name = kwargs['name']
        self._child_list = uniqueList(acceptType=kwargs['childType'])
        self._create_finished = False
        self._index = -1

    def checkmyself(self):
        for child in self._child_list:
            self._create_finished = child.checkmyself and \
                                    self._create_finished
        return self._create_finished

    def myself_info(self):
        if self._create_finished:
            str = ('%s: %d, %s, create sucessed' % (
                self.__class__.__name__, self._index, self._name))
        else:
            str = ('%s: %d, %s, create not finished' % (
                self.__class__.__name__, self._index, self._name))
        return str

    def printmyself(self):
        spf.petscInfo(self.father.logger, self.myself_info())
        return True

    def savemyself(self, file_name):
        fid = open(file_name, 'w')
        fid.write(self.myself_info())
        return True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def type(self):
        return self.__class__.__name__

    @property
    def father(self):
        return self._father

    @father.setter
    def father(self, father):
        self._father = father

    def save_context(self):
        pass

    def restore_context(self):
        pass

    def check_child_index(self, index):
        err_msg = 'wrong index %d, last index is %d. ' % (index, len(self._child_list))
        assert index <= len(self._child_list), err_msg

        err_msg = 'wrong index %d, first index is 0' % index
        assert index > 0, err_msg
        return True

    def get_child(self, index):
        self.check_child_index(index)
        return self._child_list[index]

    @property
    def child_list(self):
        return self._child_list

    def __repr__(self):
        return self.myself_info()

    def check_need_args(self, need_args=None):
        if need_args is None:
            need_args = list()
        kwargs = self._kwargs
        for key in need_args:
            err_msg = "information about '%s' is necessary for %s.%s. " \
                      % (key, self.__class__.__name__, sys._getframe(2).f_code.co_name)
            assert key in kwargs, err_msg
        return True

    def check_opt_args(self, opt_args=None):
        if opt_args is None:
            opt_args = dict()
        kwargs = self._kwargs
        for key, value in opt_args.items():
            if key not in kwargs:
                kwargs[key] = opt_args[key]
        self._kwargs = kwargs
        return kwargs

    def check_args(self, need_args=None, opt_args=None):
        if opt_args is None:
            opt_args = dict()
        if need_args is None:
            need_args = list()
        self.check_need_args(need_args)
        kwargs = self.check_opt_args(opt_args)
        self._kwargs = kwargs
        return kwargs


# abstract class for matrix construction.
class abs_construct_matrix(abs_comp):
    def __init__(self):
        super().__init__(childType=abs_comp)


class coordinate_transformation:
    @staticmethod
    def vector_rotation(f, R):
        fx, fy, fz = f.T[[0, 1, 2]]
        fx1 = R[0][0] * fx + R[0][1] * fy + R[0][2] * fz
        fy1 = R[1][0] * fx + R[1][1] * fy + R[1][2] * fz
        fz1 = R[2][0] * fx + R[2][1] * fy + R[2][2] * fz
        return np.dstack((fx1, fy1, fz1))[0]


class fullprint:
    # context manager for printing full numpy arrays

    def __init__(self, **kwargs):
        kwargs.setdefault('threshold', np.inf)
        self.opt = kwargs

    def __enter__(self):
        self._opt = np.get_printoptions()
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self._opt)


class Quaternion:
    """docstring for Quaternion"""

    def __init__(self, axis=np.array([0, 0, 1.0]), angle=0):
        axis = np.array(axis)

        xyz = np.sin(.5 * angle) * axis / np.linalg.norm(axis)
        self.q = np.array([
            np.cos(.5 * angle),
            xyz[0],
            xyz[1],
            xyz[2]
        ])

    def __add__(self, other):
        Q = Quaternion()
        Q.q = self.q + other
        return Q

    def __str__(self):
        return str(self.q)

    def __repr__(self):
        return str(self.q)

    def copy(self):
        q2 = copy.deepcopy(self)
        return q2

    def mul(self, other):
        # print(type(other))
        # print(type(other) is Quaternion)
        # assert (type(other) is Quaternion)

        W = self.q[0]
        X = self.q[1]
        Y = self.q[2]
        Z = self.q[3]

        w = other.q[0]
        x = other.q[1]
        y = other.q[2]
        z = other.q[3]

        Q = Quaternion()
        Q.q = np.array([
            w * W - x * X - y * Y - z * Z,
            W * x + w * X + Y * z - y * Z,
            W * y + w * Y - X * z + x * Z,
            X * y - x * Y + W * z + w * Z
        ])
        return Q

    def set_wxyz(self, w, x, y, z):
        self.q = np.array([w, x, y, z])

    def from_matrix(self, rotM):
        assert np.isclose(np.linalg.det(rotM), 1)

        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        t = rotM[0, 0] + rotM[1, 1] + rotM[2, 2]
        r = np.sqrt(1 + t)
        s = 1 / 2 / r
        w = 1 / 2 * r
        x = (rotM[2, 1] - rotM[1, 2]) * s
        y = (rotM[0, 2] - rotM[2, 0]) * s
        z = (rotM[1, 0] - rotM[0, 1]) * s
        self.q = np.array([w, x, y, z])

        # # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        # w = np.sqrt(1 + rotM[0, 0] + rotM[1, 1] + rotM[2, 2]) / 2
        # x = (rotM[2, 1] - rotM[1, 2]) / (4 * w)
        # y = (rotM[0, 2] - rotM[2, 0]) / (4 * w)
        # z = (rotM[1, 0] - rotM[0, 1]) / (4 * w)
        # self.q = np.array([w, x, y, z])

    def from_thphps(self, theta, phi, psi):
        self.from_matrix(Rloc2glb(theta, phi, psi))

        # # dbg code
        # rotM = Rloc2glb(theta, phi, psi)
        # tq1 = spR.from_matrix(rotM).as_quat()
        # print(tq1[3], tq1[0], tq1[1], tq1[2])
        # print(self.q[0], self.q[1], self.q[2], self.q[3])

    def normalize(self):
        self.q = self.q / np.linalg.norm(self.q)

    # def rot_by(self, q1: "Quaternion"):
    #     # q2 = q1 q q1^-1
    #     q = self
    #     q2 = q1.mul(q.mul(q1.get_inv()))
    #     return q2
    #
    # def rot_from(self, q1: "Quaternion"):
    #     # q2 = q q1 q^-1
    #     q = self
    #     # q2 = q.mul(q1.mul(q.get_inv()))
    #     q2 = q1.mul(q)
    #     return q2

    def get_E(self):
        W = self.q[0]
        X = self.q[1]
        Y = self.q[2]
        Z = self.q[3]

        return np.array([
            [-X, W, -Z, Y],
            [-Y, Z, W, -X],
            [-Z, -Y, X, W]
        ])

    def get_G(self):
        W = self.q[0]
        X = self.q[1]
        Y = self.q[2]
        Z = self.q[3]

        return np.array([
            [-X, W, Z, -Y],
            [-Y, -Z, W, X],
            [-Z, Y, -X, W]
        ])

    def get_R(self):
        return np.matmul(self.get_E(), self.get_G().T)

    def get_inv(self):
        q_inv = Quaternion()
        q_inv.set_wxyz(self.q[0], -self.q[1], -self.q[2], -self.q[3])
        return q_inv

    def get_thphps(self):
        W = self.q[0]
        X = self.q[1]
        Y = self.q[2]
        Z = self.q[3]
        # print('dbg, theta', W, X, Y, Z, theta, Z ** 2 - Y ** 2 - X ** 2 + W ** 2)
        theta = np.arccos(np.clip(Z ** 2 - Y ** 2 - X ** 2 + W ** 2, -1, 1))
        phi = np.arctan2(2 * Y * Z - 2 * W * X, 2 * X * Z + 2 * W * Y)
        phi = phi if phi > 0 else phi + 2 * np.pi  # (-pi,pi) -> (0, 2pi)
        psi = np.arctan2(2 * Y * Z + 2 * W * X, -2 * X * Z + 2 * W * Y)
        psi = psi if psi > 0 else psi + 2 * np.pi  # (-pi,pi) -> (0, 2pi)
        return theta, phi, psi

    def get_rotM(self):
        return Rloc2glb(*self.get_thphps())


def tube_flatten(container):
    for i in container:
        if isinstance(i, (uniqueList, list, tuple)):
            for j in tube_flatten(i):
                yield j
        else:
            yield i