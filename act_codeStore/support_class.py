import sys
import copy
from collections import UserList
import numpy as np
# from scipy.spatial.transform import Rotation as spR
from matplotlib.ticker import Locator
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import Normalize
from petsc4py import PETSc
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


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()
        view_interval = self.axis.get_view_interval()
        if view_interval[-1] > majorlocs[-1]:
            majorlocs = np.hstack((majorlocs, view_interval[-1]))
        assert np.all(majorlocs >= 0)
        if np.isclose(majorlocs[0], 0):
            majorlocs = majorlocs[1:]

        # # iterate through minor locs, handle the lowest part, old version
        # minorlocs = []
        # for i in range(1, len(majorlocs)):
        #     majorstep = majorlocs[i] - majorlocs[i - 1]
        #     if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
        #         ndivs = 10
        #     else:
        #         ndivs = 9
        #     minorstep = majorstep / ndivs
        #     locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
        #     minorlocs.extend(locs)

        # iterate through minor locs, handle the lowest part, my version
        minorlocs = []
        for i in range(1, len(majorlocs)):
            tloc = majorlocs[i - 1]
            tgap = majorlocs[i] - majorlocs[i - 1]
            tstp = majorlocs[i - 1] * self.linthresh * 10
            while tloc < tgap and not np.isclose(tloc, tgap):
                tloc = tloc + tstp
                minorlocs.append(tloc)
        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


class midPowerNorm(Normalize):
    # user define color norm
    def __init__(self, gamma=10, midpoint=1, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)
        assert gamma > 1
        self.gamma = gamma
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        midpoint = self.midpoint
        logmid = np.log(midpoint) / np.log(gamma)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                     mask=mask)
            resdat = result.data
            tidx1 = resdat < midpoint
            tidx2 = np.logical_not(tidx1)
            resdat1 = np.log(resdat[tidx1]) / np.log(gamma)
            v1 = np.log(vmin) / np.log(gamma)
            tx, ty = [v1, logmid], [0, 0.5]
            #             print(resdat1, tx, ty)
            tuse1 = np.interp(resdat1, tx, ty)
            resdat2 = np.log(resdat[tidx2]) / np.log(gamma)
            v2 = np.log(vmax) / np.log(gamma)
            tx, ty = [logmid, v2], [0.5, 1]
            tuse2 = np.interp(resdat2, tx, ty)
            resdat[tidx1] = tuse1
            resdat[tidx2] = tuse2
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        return result


# class zeroPowerNorm(Normalize):
#     def __init__(self, gamma=10, linthresh=1, linscale=1, vmin=None, vmax=None, clip=False):
#         Normalize.__init__(self, vmin, vmax, clip)
#         assert gamma > 1
#         self.gamma = gamma
#         self.midpoint = 0
#         assert vmin < 0
#         assert vmax > 0
#         self.linthresh = linthresh
#         self.linscale = linscale
#
#     def __call__(self, value, clip=None):
#         if clip is None:
#             clip = self.clip
#         result, is_scalar = self.process_value(value)
#
#         self.autoscale_None(result)
#         gamma = self.gamma
#         midpoint = self.midpoint
#         linthresh = self.linthresh
#         linscale = self.linscale
#         vmin, vmax = self.vmin, self.vmax
#
#         if clip:
#             mask = np.ma.getmask(result)
#             result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
#                                  mask=mask)
#         assert result.max() > 0
#         assert result.min() < 0
#
#         mag0 = np.log(result.max()) / np.log(linthresh)
#         mag2 = np.log(-result.min()) / np.log(linthresh)
#         mag1 = linscale / (linscale + mag0 + mag2)
#         b0 = mag0 / (mag0 + mag1 + mag2)
#         b1 = (mag0 + mag1) / (mag0 + mag1 + mag2)
#
#         resdat = result.data
#         tidx0 = (resdat > -np.inf) * (resdat <= -linthresh)
#         tidx1 = (resdat > -linthresh) * (resdat <= linthresh)
#         tidx2 = (resdat > linthresh) * (resdat <= np.inf)
#         resdat0 = np.log(-resdat[tidx0]) / np.log(gamma)
#         resdat1 = resdat[tidx1]
#         resdat2 = np.log(resdat[tidx2]) / np.log(gamma)
#         #
#         tx, ty = [np.log(-vmin) / np.log(gamma), np.log(linthresh) / np.log(gamma)], [0, b0]
#         tuse0 = np.interp(resdat0, tx, ty)
#         #
#         tx, ty = [-linthresh, linthresh], [b0, b1]
#         tuse1 = np.interp(resdat1, tx, ty)
#
#         tx, ty = [v1, logmid], [0, 0.5]
#         #             print(resdat1, tx, ty)
#         tuse1 = np.interp(resdat1, tx, ty)
#         resdat2 = np.log(resdat[tidx2]) / np.log(gamma)
#         v2 = np.log(vmax) / np.log(gamma)
#         tx, ty = [logmid, v2], [0.5, 1]
#         tuse2 = np.interp(resdat2, tx, ty)
#         resdat[tidx1] = tuse1
#         resdat[tidx2] = tuse2
#         result = np.ma.array(resdat, mask=result.mask, copy=False)
#         return result


# user define color norm
class midLinearNorm(Normalize):
    def __init__(self, midpoint=1, vmin=None, vmax=None, clip=False):
        # clip: see np.clip, Clip (limit) the values in an array.
        # assert 1 == 2
        Normalize.__init__(self, vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        result, is_scalar = self.process_value(value)
        # print(type(result))

        self.autoscale_None(result)
        midpoint = self.midpoint
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)
            resdat = result.data
            tidx1 = resdat < midpoint
            tidx2 = np.logical_not(tidx1)
            resdat1 = resdat[tidx1]
            if vmin < midpoint:
                tx, ty = [vmin, midpoint], [0, 0.5]
                tuse1 = np.interp(resdat1, tx, ty)
            else:
                tuse1 = np.zeros_like(resdat1)
            resdat2 = resdat[tidx2]
            if vmax > midpoint:
                tx, ty = [midpoint, vmax], [0.5, 1]
                tuse2 = np.interp(resdat2, tx, ty)
            else:
                tuse2 = np.zeros_like(resdat2)
            resdat[tidx1] = tuse1
            resdat[tidx2] = tuse2
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        return result


class TwoSlopeNorm(Normalize):
    # noinspection PyMissingConstructor
    def __init__(self, vcenter, vmin=None, vmax=None):
        """
        Normalize data with a set center.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the the max value of the dataset.

        Examples
        --------
        This maps data value -4000 to 0., 0 to 0.5, and +10000 to 1.0; data
        between is linearly interpolated::

            >>> import matplotlib.colors as mcolors
            >>> offset = mcolors.TwoSlopeNorm(vmin=-4000.,
                                              vcenter=0., vmax=10000)
            >>> data = [-4000., -2000., 0., 2500., 5000., 7500., 10000.]
            >>> offset(data)
            array([0., 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
        """

        self.vcenter = vcenter
        self.vmin = vmin
        self.vmax = vmax
        if vcenter is not None and vmax is not None and vcenter >= vmax:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')
        if vcenter is not None and vmin is not None and vcenter <= vmin:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')

    def autoscale_None(self, A):
        """
        Get vmin and vmax, and then clip at vcenter
        """
        super().autoscale_None(A)
        if self.vmin > self.vcenter:
            self.vmin = self.vcenter
        if self.vmax < self.vcenter:
            self.vmax = self.vcenter

    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The clip argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # sets self.vmin, self.vmax if None

        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        result = np.ma.masked_array(
            np.interp(result, [self.vmin, self.vcenter, self.vmax],
                      [0, 0.5, 1.]), mask=np.ma.getmask(result))
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


def tube_flatten(container):
    for i in container:
        if isinstance(i, (uniqueList, list, tuple)):
            for j in tube_flatten(i):
                yield j
        else:
            yield i