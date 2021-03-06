# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20210906

@author: Zhang Ji
"""

# import subprocess
import os
import numpy as np
# from scipy.io import loadmat
from scipy import interpolate  # , integrate, spatial, signal
# from scipy.optimize import leastsq, curve_fit
# from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
# from time import time
# import pickle
import warnings
from petsc4py import PETSc

import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.ticker import Locator
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import Normalize, ListedColormap, LogNorm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from mpl_toolkits.axes_grid1 import colorbar
# from matplotlib import colorbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # , zoomed_inset_axes
# import matplotlib.ticker as mtick
# from matplotlib import colors as mcolors

from act_codeStore import support_fun as spf
from act_src import problemClass
from act_src import particleClass

# from act_codeStore.support_class import *
# from act_act_src import baseClass
# from act_src import interactionClass
# from act_src import relationClass

PWD = os.getcwd()
np.set_printoptions(linewidth=110, precision=5)

params = {
    'animation.html': 'html5',
    'font.family':    'sans-serif',
    'font.size':      15,
}
preamble = r' '
preamble = preamble + '\\usepackage{bm} '
preamble = preamble + '\\usepackage{amsmath} '
preamble = preamble + '\\usepackage{amssymb} '
preamble = preamble + '\\usepackage{mathrsfs} '
preamble = preamble + '\\DeclareMathOperator{\\Tr}{Tr} '
params['text.latex.preamble'] = preamble
params['text.usetex'] = True
plt.rcParams.update(params)


def set_axes_equal(ax, rad_fct=0.5):
    figsize = ax.figure.get_size_inches()
    l1, l2 = ax.get_position().bounds[2:] * figsize
    lmax = np.max((l1, l2))

    if ax.name == "3d":
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])

        origin = np.mean(limits, axis=1)
        radius = rad_fct * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        radius_x = l1 / lmax * radius
        radius_y = l1 / lmax * radius
        radius_z = l2 / lmax * radius
        ax.set_xlim3d([origin[0] - radius_x, origin[0] + radius_x])
        ax.set_ylim3d([origin[1] - radius_y, origin[1] + radius_y])
        ax.set_zlim3d([origin[2] - radius_z, origin[2] + radius_z])
    else:
        limits = np.array([
            ax.get_xlim(),
            ax.get_ylim(),
        ])

        origin = np.mean(limits, axis=1)
        radius = rad_fct * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        radius_x = l1 / lmax * radius
        radius_y = l2 / lmax * radius
        ax.set_xlim([origin[0] - radius_x, origin[0] + radius_x])
        ax.set_ylim([origin[1] - radius_y, origin[1] + radius_y])
    return ax


# Topics: line, color, LineCollection, cmap, colorline, codex
'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
'''


# Data manipulation:
def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


# Interface to LineCollection:
def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), ax=None, norm=plt.Normalize(0.0, 1.0),
              label=' ', linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, x.size)
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.patch.set_facecolor('white')
    else:
        plt.sca(ax)
        # fig = plt.gcf()

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax.add_collection(lc)
    return lc


def colorline3d(tnodes, tcl, quiver_length_fct=None, clb_title=' ', show_project=False, tu=None,
                nu_show=50, return_fig=False, ax0=None, tcl_lim=None, tcl_fontsize=10,
                cmap=plt.get_cmap('jet')):
    if ax0 is None:
        fig = plt.figure(figsize=(8, 8), dpi=100)
        fig.patch.set_facecolor('white')
        ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        assert hasattr(ax0, 'get_zlim')
        plt.sca(ax0)
        fig = plt.gcf()
    if tcl_lim is None:
        tcl_lim = (tcl.min(), tcl.max())
    ax0.plot(tnodes[:, 0], tnodes[:, 1], tnodes[:, 2]).pop(0).remove()
    cax1 = inset_axes(ax0, width="80%", height="5%", bbox_to_anchor=(0.1, 0.1, 0.8, 1),
                      loc=9, bbox_transform=ax0.transAxes, borderpad=0, )
    norm = plt.Normalize(*tcl_lim)
    cmap = cmap
    # Create the 3D-line collection object
    points = tnodes.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(tcl)
    ax0.add_collection3d(lc, zs=points[:, :, 2].flatten(), zdir='z')
    clb = fig.colorbar(lc, cax=cax1, orientation="horizontal")
    clb.ax.tick_params(labelsize=tcl_fontsize)
    clb.ax.set_title(clb_title)
    clb_ticks = np.linspace(*tcl_lim, 5)
    clb.set_ticks(clb_ticks)
    clb.ax.set_yticklabels(clb_ticks)
    set_axes_equal(ax0)
    if show_project:
        ax0.plot(np.ones_like(tnodes[:, 0]) * ax0.get_xlim()[0], tnodes[:, 1], tnodes[:, 2], '--k',
                 alpha=0.2)
        ax0.plot(tnodes[:, 0], np.ones_like(tnodes[:, 1]) * ax0.get_ylim()[1], tnodes[:, 2], '--k',
                 alpha=0.2)
        ax0.plot(tnodes[:, 0], tnodes[:, 1], np.ones_like(tnodes[:, 0]) * ax0.get_zlim()[0], '--k',
                 alpha=0.2)
    if not tu is None:
        assert not quiver_length_fct is None
        t_stp = np.max((1, tu.shape[0] // nu_show))
        color_len = tnodes[::t_stp, 0].size
        quiver_length = np.max(tnodes.max(axis=0) - tnodes.min(axis=0)) * quiver_length_fct
        # colors = [cmap(1.0 * i / color_len) for i in range(color_len)]
        # ax0.quiver(tnodes[::t_stp, 0], tnodes[::t_stp, 1], tnodes[::t_stp, 2],
        #            tu[::t_stp, 0], tu[::t_stp, 1], tu[::t_stp, 2],
        #            length=quiver_length, arrow_length_ratio=0.2, pivot='tail', normalize=False,
        #            colors=colors)
        ax0.quiver(tnodes[::t_stp, 0], tnodes[::t_stp, 1], tnodes[::t_stp, 2],
                   tu[::t_stp, 0], tu[::t_stp, 1], tu[::t_stp, 2],
                   length=quiver_length, arrow_length_ratio=0.2, pivot='tail', normalize=False,
                   colors='k')
    plt.sca(ax0)
    ax0.set_xlabel('$X_1$')
    ax0.set_ylabel('$X_2$')
    ax0.set_zlabel('$X_3$')
    # for spine in ax0.spines.values():
    #     spine.set_visible(False)
    # plt.tight_layout()

    t1 = fig if return_fig else True
    return t1


def add_inset(ax0, rect, *args, **kwargs):
    box = ax0.get_position()
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()
    inptx = interpolate.interp1d(xlim, (0, box.x1 - box.x0))
    inpty = interpolate.interp1d(ylim, (0, box.y1 - box.y0))
    left = inptx(rect[0]) + box.x0
    bottom = inpty(rect[1]) + box.y0
    width = inptx(rect[2] + rect[0]) - inptx(rect[0])
    height = inpty(rect[3] + rect[1]) - inpty(rect[1])
    new_rect = np.hstack((left, bottom, width, height))
    return ax0.figure.add_axes(new_rect, *args, **kwargs)


def multicolor_ylabel(ax, list_of_strings, list_of_colors, axis='x', anchorpad=0, **kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""

    # x-axis label
    if axis == 'x' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', **kw))
                 for text, color in zip(list_of_strings, list_of_colors)]
        xbox = HPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc='lower left', child=xbox, pad=anchorpad,
                                          frameon=False, bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis == 'y' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom',
                                               rotation=90, **kw))
                 for text, color in zip(list_of_strings[::-1], list_of_colors)]
        ybox = VPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc='lower left', child=ybox, pad=anchorpad,
                                          frameon=False, bbox_to_anchor=(-0.105, 0.25),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)


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


def RBGColormap(color: np.asarray, ifcheck=True):
    if ifcheck:
        if color.size == 3:
            color = np.hstack((color, 1))
        err_mg = 'color is an array contain (R, B, G) or (R, B, G, A) information. '
        assert color.size == 4, err_mg

    N = 256
    vals = np.ones((N, 4)) * color
    vals[:, 3] = np.linspace(0.1 * color[3], 0.5 * color[3], N)
    newcmp = ListedColormap(vals)
    return newcmp


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


def resampling_data(t, X, resampling_fct=2, t_use=None, interp1d_kind='quadratic'):
    if t_use is None:
        t_use = np.linspace(t.min(), t.max(), int(t.size * resampling_fct))
    else:
        war_msg = 'size of t_use is %d, resampling_fct is IGNORED' % t_use.size
        warnings.warn(war_msg)

    intp_fun1d = interpolate.interp1d(t, X, kind=interp1d_kind, copy=False, axis=0,
                                      bounds_error=True)
    return intp_fun1d(t_use)


def get_increase_angle(ty1):
    ty = ty1.copy()
    for i0, dt in enumerate(np.diff(ty)):
        if dt > np.pi:
            ty[i0 + 1:] = ty[i0 + 1:] - 2 * np.pi
        elif dt < -np.pi:
            ty[i0 + 1:] = ty[i0 + 1:] + 2 * np.pi
    return ty


def get_continue_angle(tx, ty1, t_use=None):
    if t_use is None:
        t_use = np.linspace(tx.min(), tx.max(), 2 * tx.size)
    if np.array(t_use).size == 1:
        t_use = np.linspace(tx.min(), tx.max(), t_use * tx.size)

    ty = get_increase_angle(ty1)
    intp_fun1d = interpolate.interp1d(tx, ty, kind='quadratic', copy=False, axis=0,
                                      bounds_error=True)
    ty = intp_fun1d(t_use) % (2 * np.pi)
    ty[ty > np.pi] = ty[ty > np.pi] - 2 * np.pi
    return ty


def separate_idx(ty, spt_fct=0.5):
    # for periodic boundary condition, separate to small components to avoid the jump of the trajectory.
    idx_list = []
    idx_list.append(-1)  # first idx is 0, but later will plus 1.
    for tyi in ty:
        dtyi = np.diff(tyi)
        tol = (tyi.max() - tyi.min()) * spt_fct
        idx_list.append(np.argwhere(dtyi > tol).flatten())
        idx_list.append(np.argwhere(dtyi < -tol).flatten())
    idx_list.append(ty.shape[1] - 1)  # last idx is (size-1).
    # print(idx_list)
    t1 = np.unique(np.hstack(idx_list))
    # print(t1)
    return np.vstack((t1[:-1] + 1, t1[1:])).T


def make2D_X_video(t, obj_list: list, figsize=(9, 9), dpi=100, stp=1, interval=50, resampling_fct=2,
                   interp1d_kind='quadratic', tmin=-np.inf, tmax=np.inf, plt_range=None, t0_marker='s'):
    # percentage = 0
    def update_fun(num, line_list, data_list):
        num = num * stp
        # print(num)
        tqdm_fun.update(1)
        # percentage += 1
        for linei, datai in zip(line_list, data_list):
            linei.set_data((datai[:num, 0], datai[:num, 1]))
        return line_list

    tidx = (t >= tmin) * (t <= tmax)
    data_list = np.array([resampling_data(t[tidx], obji.X_hist[tidx], resampling_fct=resampling_fct,
                                          interp1d_kind=interp1d_kind)
                          for obji in obj_list])
    data_max = data_list.max(axis=0).max(axis=0)
    data_min = data_list.min(axis=0).min(axis=0)
    data_mid = (data_max + data_min) / 2
    if plt_range is None:
        plt_range = np.max(data_max - data_min)
    print('plt_range is', plt_range)

    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    axi.set_xlabel('$x_1$')
    axi.set_xlim([data_mid[0] - plt_range, data_mid[0] + plt_range])
    axi.set_ylabel('$x_2$')
    axi.set_ylim([data_mid[1] - plt_range, data_mid[1] + plt_range])

    [axi.plot(obji.X_hist[tidx, 0], obji.X_hist[tidx, 1], linestyle='None', ) for obji in obj_list]
    [axi.scatter(obji.X_hist[tidx, 0][0], obji.X_hist[tidx, 1][0], color='k', marker=t0_marker) for obji in obj_list]
    axi.axis('equal')
    # tticks = np.around(np.linspace(*axi.get_xlim(), 21), decimals=2)[1::6]
    # axi.set_xticks(tticks)
    # axi.set_xticklabels(tticks)
    # tticks = np.around(np.linspace(*axi.get_ylim(), 21), decimals=2)[1::6]
    # axi.set_yticks(tticks)
    # axi.set_yticklabels(tticks)
    # plt.tight_layout()
    # plt.show()

    t_rsp = np.linspace(t[tidx].min(), t[tidx].max(), int(t[tidx].size * resampling_fct))
    frames = t_rsp.size // stp
    tqdm_fun = tqdm_notebook(total=frames + 2)
    line_list = [axi.plot(obji.X_hist[tidx][0, 0], obji.X_hist[tidx][0, 1])[0] for obji in obj_list]
    anim = animation.FuncAnimation(fig, update_fun, frames, interval=interval, blit=False,
                                   fargs=(line_list, data_list), )
    # tqdm_fun.update(100 - percentage)
    # tqdm_fun.close()
    return anim


def show_fig_fun(problem, fig_handle, return_info=False, *args, **kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    t1 = True
    if rank == 0:
        fig, axi = fig_handle(problem=problem, *args, **kwargs)
        t1 = fig, axi if return_info else True
    return t1


def save_fig_fun(filename, problem, fig_handle, dpi=100, *args, **kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    extension = os.path.splitext(filename)[1]
    if extension == '':
        filename = '%s.png' % filename
    else:
        filetype = list(plt.gcf().canvas.get_supported_filetypes().keys())
        err_msg = 'wrong file extension, support: %s' % filetype
        assert extension[1:] in filetype, err_msg
    filenameHandle, extension = os.path.splitext(filename)
    # if extension[1:] in ('png', 'pdf', 'svg'):
    #     metadata = {
    #         'Title':  filenameHandle,
    #         'Author': 'Zhang Ji'
    #     }
    # elif extension[1:] in ('eps', 'ps',):
    #     metadata = {'Creator': 'Zhang Ji'}
    # else:
    #     metadata = None

    if rank == 0:
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        fig, axi = fig_handle(problem=problem, *args, **kwargs)
        # fig.savefig(fname=filename, dpi=dpi, metadata=metadata)
        fig.savefig(fname=filename, dpi=dpi)
        plt.close(fig)
        matplotlib.use(backend)

    logger = problem.logger
    spf.petscInfo(logger, ' ')
    spf.petscInfo(logger, 'save 2D trajectory to %s' % filename)
    return True


def _show_prepare(figsize, dpi, problem, plt_tmin, plt_tmax, show_idx, plt_full_obj):
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    tidx = (problem.t_hist >= plt_tmin) * (problem.t_hist <= plt_tmax)
    if show_idx is None:
        show_idx = np.arange(problem.n_obj)
    show_idx = np.array(show_idx)
    err_msg = 'wrong parameter show_idx: %s' % str(show_idx)
    assert show_idx.max() < problem.n_obj, err_msg
    assert np.all([np.issubdtype(i0, np.integer) for i0 in show_idx]), err_msg
    show_list = problem.obj_list[show_idx]
    range_list = problem.obj_list if plt_full_obj else show_list
    # print('dbg range_list', range_list)
    return tidx, show_list, range_list, fig, axi


def _trajectory2D_fun_fun(obji: 'particleClass.particle2D', tidx, axi, cmap, t0_marker,
                          interp1d_kind, resampling_fct, **line_fun_kwargs):
    # tcolor = cmap(obji.index / problem.n_obj)
    # color = np.ones((X_hist.shape[0], 4)) * tcolor
    # color[:, 3] = np.linspace(0, tcolor[3], X_hist.shape[0])
    # axi.plot(X_hist[:, 0], X_hist[:, 1], '-', color=color)

    problem = obji.father
    t_hist = problem.t_hist[tidx]
    norm = plt.Normalize(0.0, 1.0)
    X_hist = obji.X_hist[tidx]

    if resampling_fct is not None:
        X_hist = resampling_data(t_hist, X_hist, resampling_fct=resampling_fct, interp1d_kind=interp1d_kind)
    axi.scatter(X_hist[0, 0], X_hist[0, 1], color='k', marker=t0_marker)

    lc = LineCollection(make_segments(X_hist[:, 0], X_hist[:, 1]),
                        array=np.linspace(0.0, 1.0, X_hist.shape[0]),
                        cmap=RBGColormap(cmap(obji.index / problem.n_obj), ifcheck=False),
                        norm=norm)
    axi.add_collection(lc)
    return axi


def _segments_fun(obji: 'particleClass.particle2D', tidx, axi, cmap, t0_marker,
                  interp1d_kind, resampling_fct, **line_fun_kwargs):
    problem = obji.father
    t_hist = problem.t_hist[tidx]
    X_hist = obji.X_hist[tidx]

    if resampling_fct is not None:
        X_hist = resampling_data(t_hist, X_hist, resampling_fct=resampling_fct, interp1d_kind=interp1d_kind)
    axi.scatter(X_hist[0, 0], X_hist[0, 1], color='k', marker=t0_marker)

    spr_idx = separate_idx(X_hist.T)
    for tidx0, tidx1 in spr_idx:
        axi.plot(X_hist[tidx0:tidx1, 0], X_hist[tidx0:tidx1, 1], c=cmap(obji.index / problem.n_obj),
                 **line_fun_kwargs)
    return axi


def _core_X2D_fun(problem: 'problemClass._base2DProblem', line_fun, show_idx=None,
                  figsize=np.array((50, 50)) * 5, dpi=100, plt_tmin=-np.inf, plt_tmax=np.inf,
                  resampling_fct=None, interp1d_kind='quadratic',
                  t0_marker='s', cmap=plt.get_cmap('brg'),
                  plt_full_obj=True, plt_full_time=True, **line_fun_kwargs):
    _ = _show_prepare(figsize, dpi, problem, plt_tmin, plt_tmax, show_idx, plt_full_obj)
    tidx, show_list, range_list, fig, axi = _
    if np.any(tidx):
        # plot
        for obji in show_list:
            line_fun(obji, tidx, axi, cmap, t0_marker, interp1d_kind, resampling_fct, **line_fun_kwargs)
        # set range
        tidx = np.ones_like(tidx) if plt_full_time else tidx
        Xmax = np.array([obji.X_hist[tidx].max(axis=0) for obji in range_list]).max(axis=0)
        Xmin = np.array([obji.X_hist[tidx].min(axis=0) for obji in range_list]).min(axis=0)
        Xrng = (Xmax - Xmin).max() * 0.55
        Xmid = (Xmax + Xmin) / 2
        axi.set_xlim(Xmid[0] - Xrng, Xmid[0] + Xrng)
        axi.set_ylim(Xmid[1] - Xrng, Xmid[1] + Xrng)
        # # set_axes_equal(axi)
    else:
        logger = problem.logger
        logger.info(' ')
        logger.warn('Problem %s has no time in range (%f, %f)' % (str(problem), plt_tmin, plt_tmax))
    return fig, axi


def core_trajectory2D(*args, **kwargs):
    return _core_X2D_fun(line_fun=_trajectory2D_fun_fun, *args, **kwargs)


def core_segments2D(*args, **kwargs):
    return _core_X2D_fun(line_fun=_segments_fun, *args, **kwargs)


def cal_avrPhaseVelocity(problem: 'problemClass._base2DProblem',
                         t_tmin=-np.inf, t_tmax=np.inf,
                         resampling_fct=1, interp1d_kind='quadratic',
                         tavr=1):
    tidx = (problem.t_hist >= t_tmin) * (problem.t_hist <= t_tmax)
    if np.isnan(problem.obj_list[0].W_hist[tidx][0]):
        tidx[0] = False
    t_hist = problem.t_hist[tidx]
    t_use, dt_res = np.linspace(t_hist.min(), t_hist.max(), int(t_hist.size * resampling_fct), retstep=True)
    avg_stp = np.ceil(tavr / dt_res).astype('int')
    weights = np.ones(avg_stp) / avg_stp

    avg_all = []
    for obji in problem.obj_list:  # type:particleClass.particle2D
        W_hist = interpolate.interp1d(t_hist, obji.W_hist[tidx], kind=interp1d_kind, copy=False)(t_use)
        avg_all.append(np.convolve(weights, W_hist, mode='same'))
    avg_all = np.abs(np.vstack(avg_all))
    avg_stpD2 = avg_stp // 2
    for i0 in np.arange(avg_stpD2):
        avg_all[:, i0] = avg_all[:, i0] / (avg_stpD2 + i0 + avg_stp % 2) * avg_stp
        i1 = - i0 - 1
        avg_all[:, i1] = avg_all[:, i1] / (avg_stpD2 + i0 + 1) * avg_stp
    return t_use, avg_all


def cal_avrInfo(problem: 'problemClass._base2DProblem',
                t_tmin=-np.inf, t_tmax=np.inf,
                resampling_fct=1, interp1d_kind='quadratic',
                tavr=1):
    tidx = (problem.t_hist >= t_tmin) * (problem.t_hist <= t_tmax)
    if np.isnan(problem.obj_list[0].W_hist[tidx][0]):
        tidx[0] = False
    t_hist = problem.t_hist[tidx]
    t_use, dt_res = np.linspace(t_hist.min(), t_hist.max(), int(t_hist.size * resampling_fct), retstep=True)
    avg_stp = np.ceil(tavr / dt_res).astype('int')
    t_return = t_use
    weights = np.ones(avg_stp) / avg_stp

    W_avg = []
    phi_avg = []
    for obji in problem.obj_list:  # type:particleClass.particle2D
        W_hist = interpolate.interp1d(t_hist, obji.W_hist[tidx], kind=interp1d_kind, copy=False)(t_use)
        phi_hist = get_continue_angle(t_hist, obji.phi_hist[tidx], t_use=t_use)
        W_avg.append(np.convolve(weights, W_hist, mode='same'))
        phi_avg.append(np.convolve(weights, phi_hist, mode='same'))
    W_avg = np.abs(np.vstack(W_avg))
    phi_avg = np.vstack(phi_avg)
    return t_return, W_avg, phi_avg


def core_avrPhaseVelocity(problem: 'problemClass._base2DProblem',
                          figsize=np.array((50, 50)) * 5, dpi=100,
                          plt_tmin=-np.inf, plt_tmax=np.inf,
                          resampling_fct=1, interp1d_kind='quadratic',
                          vmin=None, vmax=None,
                          norm=None,
                          cmap=plt.get_cmap('bwr'), tavr=1):
    # _ = _show_prepare(figsize, dpi, problem, plt_tmin, plt_tmax, show_idx=None, plt_full_obj=True)
    # tidx, show_list, range_list, fig, axi = _
    # if np.isnan(problem.obj_list[0].W_hist[tidx][0]):
    #     tidx[0] = False
    # t_hist = problem.t_hist[tidx]
    # t_use, dt_res = np.linspace(t_hist.min(), t_hist.max(), int(t_hist.size * resampling_fct), retstep=True)
    # avg_stp = np.ceil(tavr / dt_res).astype('int')
    # idx_min = 0
    # idx_max = (t_use.size // avg_stp) * avg_stp + idx_min
    # t_plot = np.mean(t_use[idx_min:idx_max].reshape((-1, avg_stp)), axis=-1)
    #
    # avg_all = []
    # for obji in show_list:  # type:particleClass.particle2D
    #     W_hist = interpolate.interp1d(t_hist, obji.W_hist[tidx], kind=interp1d_kind, copy=False)(t_use)
    #     avg_all.append(np.mean(W_hist[idx_min:idx_max].reshape((-1, avg_stp)), axis=-1))
    # avg_all = np.vstack(avg_all)

    t_plot, avg_all = cal_avrPhaseVelocity(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                           resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                           tavr=tavr)
    sort_idx = np.argsort(np.mean(avg_all[:, t_plot > t_plot.max() / 2], axis=-1))

    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    # if avg_all.min() * avg_all.max() < 0:
    #     norm = midLinearNorm(midpoint=0, vmin=avg_all.min(), vmax=avg_all.max())
    # else:
    #     norm = Normalize(vmin=avg_all.min(), vmax=avg_all.max())
    if norm is None:
        vmin = avg_all.min() if vmin is None else vmin
        vmax = avg_all.max() if vmax is None else vmax
        print('norm in range (%f, %f)' % (vmin, vmax))
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        print('ignore parameters vmin and vmax. ')
    # norm = LogNorm(vmin=vmin, vmax=vmax)
    obj_idx = np.arange(1, problem.n_obj + 1)
    # c = axi.pcolor(t_plot, obj_idx, avg_all[sort_idx, :], cmap=cmap, norm=norm, shading='auto')
    c = axi.pcolorfast(t_plot, obj_idx, avg_all[sort_idx, :], cmap=cmap, norm=norm)
    clb = fig.colorbar(c, ax=axi)
    clb.ax.set_title('$ | \\langle \\dot{\\phi} \\rangle | $', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(1, problem.n_obj)

    # # dbg
    print(t_plot.shape, np.arange(problem.n_obj).shape, avg_all.shape)
    return fig, axi


def cal_polar_order(problem: 'problemClass._base2DProblem',
                    t_tmin=-np.inf, t_tmax=np.inf, ):
    tidx = (problem.t_hist >= t_tmin) * (problem.t_hist <= t_tmax)
    if np.isnan(problem.obj_list[0].W_hist[tidx][0]):
        tidx[0] = False
    t_hist = problem.t_hist[tidx]

    cplx_R = np.mean(np.array([(np.cos(tobj.phi_hist[tidx]), np.sin(tobj.phi_hist[tidx]))
                               for tobj in problem.obj_list]), axis=0).T
    # avg_all = np.vstack(avg_all)
    return t_hist, cplx_R
