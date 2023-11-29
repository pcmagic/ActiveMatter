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
# from scipy import sparse
from colorspacious import cspace_converter

import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.ticker import Locator
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from mpl_toolkits.axes_grid1 import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    'animation.html':        'html5',
    'animation.embed_limit': 2 ** 128,
    'font.family':           'sans-serif',
    'font.size':             15,
    }
preamble = r' '
preamble = preamble + '\\usepackage{bm} '
preamble = preamble + '\\usepackage{amsmath} '
preamble = preamble + '\\usepackage{amssymb} '
preamble = preamble + '\\usepackage{mathrsfs} '
preamble = preamble + '\\usepackage{nicefrac, xfrac} '
preamble = preamble + '\\DeclareMathOperator{\\Tr}{Tr} '
params['text.latex.preamble'] = preamble
params['text.usetex'] = True
plt.rcParams.update(params)


def plt_all_colormap(figsize=None, dpi=100):
    # Indices to step through colormap.
    x = np.linspace(0.0, 1.0, 100)
    
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    def plot_color_gradients(cmap_category, cmap_list):
        fig, axs = plt.subplots(figsize=figsize, dpi=dpi, nrows=len(cmap_list), ncols=2)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99,
                            wspace=0.05)
        fig.suptitle(cmap_category + ' colormaps', fontsize=14, y=1.0, x=0.6)
        
        for ax, name in zip(axs, cmap_list):
            # Get RGB values for colormap.
            rgb = plt.get_cmap(name)(x)[np.newaxis, :, :3]
            
            # Get colormap in CAM02-UCS colorspace. We want the lightness.
            lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
            L = lab[0, :, 0]
            L = np.float64(np.vstack((L, L, L)))
            
            ax[0].imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            ax[1].imshow(L, aspect='auto', cmap='binary_r', vmin=0., vmax=100.)
            pos = list(ax[0].get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
        
        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs.flat:
            ax.set_axis_off()
    
    plot_color_gradients('Perceptually Uniform Sequential',
                         ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
    
    plot_color_gradients('Sequential',
                         ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])
    
    plot_color_gradients('Sequential (2)',
                         ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                          'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                          'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'])
    
    plot_color_gradients('Diverging',
                         ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                          'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'])
    
    plot_color_gradients('Cyclic', ['twilight', 'twilight_shifted', 'hsv'])
    
    plot_color_gradients('Qualitative',
                         ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                          'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                          'tab20c'])
    
    plot_color_gradients('Miscellaneous',
                         ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                          'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                          'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                          'turbo', 'nipy_spectral', 'gist_ncar'])
    return True


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
class midLinearNorm(TwoSlopeNorm):
    def __init__(self, midpoint=1, vmin=None, vmax=None, clip=False):
        super().__init__(vcenter=midpoint, vmin=vmin, vmax=vmax)


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


def twilight_diverging():
    colors = plt.get_cmap('twilight_shifted')(np.linspace(0.2, 0.8, 256))
    cmap = LinearSegmentedColormap.from_list('my_colormap', colors)
    return cmap


def binary_diverging():
    colors = plt.get_cmap('binary')(np.linspace(0.2, 1, 256))
    cmap = LinearSegmentedColormap.from_list('my_colormap', colors)
    return cmap


def mid_cmap():
    colors11 = plt.get_cmap('Blues')
    colors12 = plt.get_cmap('Reds')
    colors1 = np.vstack((colors11(np.linspace(1, 0.2, 256)), colors12(np.linspace(0.2, 1, 256))))
    cmap1 = LinearSegmentedColormap.from_list('my_colormap', colors1)
    # colors21 = plt.get_cmap('bone')
    # colors22 = plt.get_cmap('spring')
    # # colors2 = np.vstack((colors21(np.linspace(0, 0.5, 256)), colors22(np.linspace(0, 1, 256))))
    # colors2 = colors22(np.linspace(0, 1, 256))
    # cmap2 = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors2)
    return cmap1


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
        if resampling_fct is not None:
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


def get_continue_angle(tx, ty1, t_use=None, interp1d_kind='quadratic', bounds_error=True):
    if t_use is None:
        t_use = np.linspace(tx.min(), tx.max(), 2 * tx.size)
    if np.array(t_use).size == 1:
        t_use = np.linspace(tx.min(), tx.max(), t_use * tx.size)
    
    ty = get_increase_angle(ty1)
    intp_fun1d = interpolate.interp1d(tx, ty, kind=interp1d_kind, copy=False, axis=0,
                                      bounds_error=bounds_error)
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


def moving_avg(y0, avg_stp):
    weights = np.ones(avg_stp) / avg_stp
    y1 = np.convolve(weights, y0, mode='same')
    avg_stpD2 = avg_stp // 2
    for i0 in np.arange(avg_stpD2):
        y1[i0] = y1[i0] / (avg_stpD2 + i0 + avg_stp % 2) * avg_stp
        i1 = - i0 - 1
        y1[i1] = y1[i1] / (avg_stpD2 + i0 + 1) * avg_stp
    return y1


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
    t_use = np.linspace(t.min(), t.max(), int(t.size * resampling_fct))
    data_list = np.array([resampling_data(t[tidx], obji.X_hist[tidx], resampling_fct=resampling_fct,
                                          interp1d_kind=interp1d_kind, t_use=t_use)
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
    #
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


def make2D_Xomega_video(problem: 'problemClass._base2DProblem',
                        figsize=np.array((9.2, 9)) * 1, dpi=100,
                        plt_tmin=-np.inf, plt_tmax=np.inf,
                        stp=1, interval=50,
                        resampling_fct=1, interp1d_kind='quadratic',
                        vmin=None, vmax=None, norm=None,
                        cmap=plt.get_cmap('viridis'), tavr=1,
                        plt_range=None, ):
    def update_fun(num, scat, title, data_list, avg_all, t_plot, cax):
        num = num * stp
        tqdm_fun.update(1)
        scat.set_offsets(data_list[:, num, :])
        scat.set_array(avg_all[:, num])
        title.set_text(title_fmt % (num, t_plot[num]))
        # clb = fig.colorbar(scat, cax=cax)
        # clb.ax.set_title('$\\varphi / \\pi$')
        return scat
    
    t = problem.t_hist
    tidx = (t >= plt_tmin) * (t <= plt_tmax)
    tidx[0] = False
    obj_list = problem.obj_list
    marker = 'o'
    title_fmt = '$ | \\langle \\dot{\\varphi} \\rangle | $, idx=%08d, t=%10.4f'
    
    ax_title = 1
    t_plot, avg_all = cal_avrPhaseVelocity(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax, tavr=tavr,
                                           resampling_fct=resampling_fct, interp1d_kind=interp1d_kind, )
    # sort_idx = np.argsort(np.mean(avg_all[:, t_plot > t_plot.max() / 2], axis=-1))
    if norm is None:
        vmin = avg_all.min() if vmin is None else vmin
        vmax = avg_all.max() if vmax is None else vmax
        print('norm in range (%f, %f)' % (vmin, vmax))
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        if (vmin is not None) or (vmax is not None):
            war_msg = 'ignore parameters vmin and vmax. '
            warnings.warn(war_msg)
    #
    data_list = np.array([resampling_data(t[tidx], obji.X_hist[tidx], interp1d_kind=interp1d_kind,
                                          resampling_fct=None, t_use=t_plot)
                          for obji in obj_list])
    data_max = data_list.max(axis=0).max(axis=0)
    data_min = data_list.min(axis=0).min(axis=0)
    data_mid = (data_max + data_min) / 2
    if plt_range is None:
        plt_range = np.max(data_max - data_min)
    print('plt_range is', plt_range)
    # # dbg
    # print('dbg', data_list.shape, avg_all.shape)
    
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    crt_idx = 0
    # cax = inset_axes(axi, width="5%", height="90%", bbox_to_anchor=(0.08, 0, 1, 1),
    #                  loc='right', bbox_transform=axi.transAxes, borderpad=0, )
    cax = make_axes_locatable(axi).append_axes('right', '5%', '5%')
    axi.set_xlabel('$x_1$')
    axi.set_xlim([data_mid[0] - plt_range, data_mid[0] + plt_range])
    axi.set_ylabel('$x_2$')
    axi.set_ylim([data_mid[1] - plt_range, data_mid[1] + plt_range])
    ax_title = title_fmt % (crt_idx, t_plot[crt_idx])
    axi.set_title("   ")
    axi.axis('equal')
    scat = axi.scatter(data_list[:, crt_idx, 0], data_list[:, crt_idx, 1], c=avg_all[:, crt_idx],
                       marker=marker, cmap=cmap, norm=norm)
    clb = fig.colorbar(scat, cax=cax)
    # clb.ax.set_title('$ | \\langle \\dot{\\varphi} \\rangle | $', fontsize='small')
    title = axi.text(0.5, 1, ax_title, bbox={
        'facecolor': (0, 0, 0, 0),
        'edgecolor': (0, 0, 0, 0),
        'pad':       5
        },
                     transform=axi.transAxes, ha="center", va='bottom')
    
    frames = t_plot.size // stp
    tqdm_fun = tqdm_notebook(total=frames + 2)
    anim = animation.FuncAnimation(fig, update_fun, frames, interval=interval, blit=False,
                                   fargs=(scat, title, data_list, avg_all, t_plot, cax), )
    return anim


def make2D_Xphi_video(problem: 'problemClass._base2DProblem',
                      figsize=np.array((9.2, 9)) * 1, dpi=100,
                      plt_tmin=-np.inf, plt_tmax=np.inf,
                      stp=1, interval=50,
                      resampling_fct=1, interp1d_kind='quadratic',
                      cmap=plt.get_cmap('viridis'),
                      plt_range=None, ):
    assert resampling_fct is None
    print('dbg, resampling_fct of current method is prohibit. ')
    
    def update_fun(num, scat, title, data_list, phi_list, t_plot, cax):
        num = num * stp
        tqdm_fun.update(1)
        scat.set_offsets(data_list[:, num, :])
        scat.set_array(phi_list[:, num] / np.pi)
        title.set_text(title_fmt % (num, t_plot[num]))
        # clb = fig.colorbar(scat, cax=cax)
        # clb.ax.set_title('$\\varphi / \\pi$')
        return scat
    
    t = problem.t_hist
    tidx = (t >= plt_tmin) * (t <= plt_tmax)
    tidx[0] = False
    obj_list = problem.obj_list
    marker = 'o'
    title_fmt = '$\\varphi / \\pi$, idx=%08d, t=%10.4f'
    
    # t_plot, W_avg, phi_list = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
    #                                      resampling_fct=1, interp1d_kind=interp1d_kind,
    #                                      tavr=0.001)
    t_plot = problem.t_hist[tidx]
    phi_list = np.array([obji.phi_hist[tidx] for obji in obj_list])
    print(phi_list.max() / np.pi, phi_list.min() / np.pi, )
    vmin, vmax = -1, 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    #
    # data_list = np.array([resampling_data(t[tidx], obji.X_hist[tidx], interp1d_kind=interp1d_kind,
    #                                       resampling_fct=None, t_use=t_plot)
    #                       for obji in obj_list])
    data_list = np.array([obji.X_hist[tidx] for obji in obj_list])
    data_max = data_list.max(axis=0).max(axis=0)
    data_min = data_list.min(axis=0).min(axis=0)
    data_mid = (data_max + data_min) / 2
    if plt_range is None:
        plt_range = np.max(data_max - data_min)
    print('plt_range is', plt_range)
    # # dbg
    # print('dbg', data_list.shape, avg_all.shape)
    
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    crt_idx = 0
    # cax = inset_axes(axi, width="5%", height="90%", bbox_to_anchor=(0.08, 0, 1, 1),
    #                  loc='right', bbox_transform=axi.transAxes, borderpad=0, )
    cax = make_axes_locatable(axi).append_axes('right', '5%', '5%')
    axi.set_xlabel('$x_1$')
    axi.set_xlim([data_mid[0] - plt_range, data_mid[0] + plt_range])
    axi.set_ylabel('$x_2$')
    axi.set_ylim([data_mid[1] - plt_range, data_mid[1] + plt_range])
    ax_title = title_fmt % (crt_idx, t_plot[crt_idx])
    axi.set_title("   ")
    axi.axis('equal')
    scat = axi.scatter(data_list[:, crt_idx, 0], data_list[:, crt_idx, 1], c=phi_list[:, crt_idx] / np.pi,
                       marker=marker, cmap=cmap, norm=norm)
    clb = fig.colorbar(scat, cax=cax)
    # clb.ax.set_title('$\\varphi / \\pi$', fontsize='small')
    title = axi.text(0.5, 1, ax_title, bbox={
        'facecolor': (0, 0, 0, 0),
        'edgecolor': (0, 0, 0, 0),
        'pad':       5
        },
                     transform=axi.transAxes, ha="center", va='bottom')
    
    frames = t_plot.size // stp
    tqdm_fun = tqdm_notebook(total=frames + 2)
    anim = animation.FuncAnimation(fig, update_fun, frames, interval=interval, blit=False,
                                   fargs=(scat, title, data_list, phi_list, t_plot, cax), )
    return anim


# def dbg_make2D_Xphi_video(problem: 'problemClass._base2DProblem',
#                           figsize=np.array((9.2, 9)) * 1, dpi=100,
#                           plt_tmin=-np.inf, plt_tmax=np.inf,
#                           stp=1, interval=50,
#                           resampling_fct=1, interp1d_kind='quadratic',
#                           cmap=plt.get_cmap('viridis'),
#                           plt_range=None, ):
#     assert resampling_fct is None
#     print('dbg, resampling_fct of current method is prohibit. ')
#
#     t = problem.t_hist
#     tidx = (t >= plt_tmin) * (t <= plt_tmax)
#     tidx[0] = False
#     obj_list = problem.obj_list
#     marker = 'o'
#     anim_name = problem.name
#
#     # t_plot, W_avg, phi_list = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
#     #                                      resampling_fct=1, interp1d_kind=interp1d_kind,
#     #                                      tavr=0.001)
#     t_plot = problem.t_hist[tidx]
#     phi_list = np.array([obji.phi_hist[tidx] for obji in obj_list])
#     print(phi_list.max() / np.pi, phi_list.min() / np.pi, phi_list.shape)
#     vmin, vmax = -1, 1
#     norm = Normalize(vmin=vmin, vmax=vmax)
#     #
#     # data_list = np.array([resampling_data(t[tidx], obji.X_hist[tidx], interp1d_kind=interp1d_kind,
#     #                                       resampling_fct=None, t_use=t_plot)
#     #                       for obji in obj_list])
#     data_list = np.array([obji.X_hist[tidx] for obji in obj_list])
#     data_max = data_list.max(axis=0).max(axis=0)
#     data_min = data_list.min(axis=0).min(axis=0)
#     data_mid = (data_max + data_min) / 2
#     if plt_range is None:
#         plt_range = np.max(data_max - data_min)
#     print('plt_range is', plt_range)
#     # # dbg
#     # print('dbg', data_list.shape, avg_all.shape)
#
#     fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
#     fig.patch.set_facecolor('white')
#     cax = make_axes_locatable(axi).append_axes('right', '5%', '5%')
#     # cax = inset_axes(axi, width="5%", height="90%", bbox_to_anchor=(0.08, 0, 1, 1),
#     #                  loc='right', bbox_transform=axi.transAxes, borderpad=0, )
#     axi.set_xlabel('$x_1$')
#     axi.set_xlim([data_mid[0] - plt_range, data_mid[0] + plt_range])
#     axi.set_ylabel('$x_2$')
#     axi.set_ylim([data_mid[1] - plt_range, data_mid[1] + plt_range])
#     axi.set_title('  ')
#     axi.axis('equal')
#     for i0 in tqdm_notebook(np.arange(0, phi_list.shape[1], stp)):
#         ax_title = '$\\varphi / \\pi$, idx=%04d, t=%f' % (i0, t_plot[i0])
#         axi.set_title(ax_title)
#         scat = axi.scatter(data_list[:, i0, 0], data_list[:, i0, 1], c=phi_list[:, i0] / np.pi,
#                            marker=marker, cmap=cmap, norm=norm)
#         clb = fig.colorbar(scat, cax=cax)
#         # clb.ax.set_title(ax_title, fontsize='small')
#         filename = os.path.join(PWD, 'animation', 'dbg', '%s_%04d.png' % (anim_name, i0))
#         fig.savefig(fname=filename, dpi=dpi)
#         plt.sca(axi)
#         plt.cla()
#         plt.sca(cax)
#         plt.cla()
#     return True


def make2D_Xphiomega_video(problem: 'problemClass._base2DProblem',
                           figsize=np.array((11, 9)) * 0.5, dpi=100,
                           plt_tmin=-np.inf, plt_tmax=np.inf,
                           stp=1, interval=50,
                           resampling_fct=1, interp1d_kind='quadratic',
                           vmin=None, vmax=None, norm=None,
                           cmap=plt.get_cmap('viridis'), tavr=1,
                           plt_range=None, ):
    assert resampling_fct is None
    print('dbg, resampling_fct of current method is prohibit. ')
    
    def update_fun(num, qr, title, phi_list, W_list, data_list, t_plot):
        num = num * stp
        tqdm_fun.update(1)
        qr.set_offsets(data_list[:, num, :])
        qr.set_UVC(np.cos(phi_list[:, num]), np.sin(phi_list[:, num]), W_list[:, num])
        title.set_text(title_fmt % (num, t_plot[num]))
        return True
    
    t = problem.t_hist
    tidx = (t >= plt_tmin) * (t <= plt_tmax)
    tidx[0] = False
    obj_list = problem.obj_list
    title_fmt = '$ | \\langle \\dot{\\varphi} \\rangle | $, idx=%08d, t=%10.4f'
    t_plot = problem.t_hist[tidx]
    dt_res = np.mean(np.diff(t_plot))
    avg_stp = np.ceil(tavr / dt_res).astype('int')
    avg_stpD2 = avg_stp // 2
    weights = np.ones(avg_stp) / avg_stp
    err_msg = 'tavr <= %f, current: %f' % (t_plot.max() - t_plot.min(), tavr)
    assert avg_stp <= t_plot.size, err_msg
    
    phi_list = np.array([obji.phi_hist[tidx] for obji in obj_list])
    W_list = np.abs(np.array([np.convolve(weights, obji.W_hist[1:], mode='same') for obji in obj_list]))
    for i0 in np.arange(avg_stpD2):
        W_list[:, i0] = W_list[:, i0] / (avg_stpD2 + i0 + avg_stp % 2) * avg_stp
        i1 = - i0 - 1
        W_list[:, i1] = W_list[:, i1] / (avg_stpD2 + i0 + 1) * avg_stp
    W_list = W_list[:, tidx[1:]]
    data_list = np.array([obji.X_hist[tidx] for obji in obj_list])
    data_max = data_list.max(axis=0).max(axis=0)
    data_min = data_list.min(axis=0).min(axis=0)
    data_mid = (data_max + data_min) / 2
    #
    if norm is None:
        vmin = W_list.min() if vmin is None else vmin
        vmax = W_list.max() if vmax is None else vmax
        print('norm in range (%f, %f)' % (vmin, vmax))
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        if (vmin is not None) or (vmax is not None):
            war_msg = 'ignore parameters vmin and vmax. '
            warnings.warn(war_msg)
    #
    if plt_range is None:
        plt_range = np.max(data_max - data_min)
        print('plt_range is', plt_range)
    
    fig, (axi, cax) = plt.subplots(nrows=1, ncols=2, figsize=figsize, dpi=dpi, constrained_layout=True,
                                   gridspec_kw={
                                       'width_ratios': [10, 1]
                                       })
    fig.patch.set_facecolor('white')
    crt_idx = 0
    axi.set_xlabel('$x_1$')
    axi.set_ylabel('$x_2$')
    axi.plot(np.ones(2) * data_mid[0], [data_mid[1] - plt_range / 2, data_mid[1] + plt_range / 2], ' ')
    axi.plot([data_mid[0] - plt_range / 2, data_mid[0] + plt_range / 2], np.ones(2) * data_mid[1], ' ')
    axi.set_title("   ")
    qr = axi.quiver(data_list[:, crt_idx, 0], data_list[:, crt_idx, 1],
                    np.cos(phi_list[:, crt_idx]), np.sin(phi_list[:, crt_idx]),
                    W_list[:, crt_idx], cmap=cmap, norm=norm)
    fig.colorbar(qr, cax=cax)
    ax_title = title_fmt % (crt_idx, t_plot[crt_idx])
    title = axi.text(0.5, 1, ax_title, bbox={
        'facecolor': (0, 0, 0, 0),
        'edgecolor': (0, 0, 0, 0),
        'pad':       5
        },
                     transform=axi.transAxes, ha="center", va='bottom')
    axi.axis('equal')
    
    frames = t_plot.size // stp
    tqdm_fun = tqdm_notebook(total=frames + 2)
    anim = animation.FuncAnimation(fig, update_fun, frames, interval=interval, blit=False,
                                   fargs=(qr, title, phi_list, W_list, data_list, t_plot), )
    return anim


def show_fig_fun(problem, fig_handle, return_info=False, *args, **kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    t1 = True
    if rank == 0:
        fig, axi = fig_handle(problem=problem, *args, **kwargs)
        t1 = (fig, axi) if return_info else True
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
    # filenameHandle, extension = os.path.splitext(filename)
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
    spf.petscInfo(logger, 'save figure to %s' % filename)
    return True


def save_figs_fun(filenames, problem, fig_handle, dpi=100, *args, **kwargs):
    comm = PETSc.COMM_WORLD.tompi4py()
    rank = comm.Get_rank()
    logger = problem.logger
    spf.petscInfo(logger, ' ')
    
    if rank == 0:
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        figs, axis = fig_handle(problem=problem, *args, **kwargs)
        for filename, fig in zip(filenames, figs):
            extension = os.path.splitext(filename)[1]
            if extension == '':
                filename = '%s.png' % filename
            else:
                filetype = list(plt.gcf().canvas.get_supported_filetypes().keys())
                err_msg = 'wrong file extension, support: %s' % filetype
                assert extension[1:] in filetype, err_msg
            fig.savefig(fname=filename, dpi=dpi)
            plt.close(fig)
            logger.info('save figure to %s' % filename)
        matplotlib.use(backend)
    return True


def _show_prepare(figsize, dpi, problem, plt_tmin, plt_tmax, show_idx, range_full_obj):
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    tidx = (problem.t_hist >= plt_tmin) * (problem.t_hist <= plt_tmax)
    if show_idx is None:
        show_idx = np.arange(problem.n_obj)
    show_idx = np.array((show_idx,)).ravel()
    err_msg = 'wrong parameter show_idx: %s' % str(show_idx)
    assert show_idx.max() < problem.n_obj, err_msg
    assert np.all([np.issubdtype(i0, np.integer) for i0 in show_idx]), err_msg
    show_list = problem.obj_list[show_idx]
    range_list = problem.obj_list if range_full_obj else show_list
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
                  figsize=None, dpi=None, plt_tmin=-np.inf, plt_tmax=np.inf,
                  resampling_fct=None, interp1d_kind='quadratic',
                  t0_marker='s', cmap=plt.get_cmap('brg'),
                  range_full_obj=True, plt_full_time=True, **line_fun_kwargs):
    figsize = np.ones(2) * 5 if figsize is None else figsize
    dpi = 100 if dpi is None else dpi
    _ = _show_prepare(figsize, dpi, problem, plt_tmin, plt_tmax, show_idx, range_full_obj)
    tidx, show_list, range_list, fig, axi = _
    if np.any(tidx):
        # plot
        for obji in show_list:
            line_fun(obji, tidx, axi, cmap, t0_marker, interp1d_kind, resampling_fct, **line_fun_kwargs)
        # set range
        if plt_full_time:
            tidx = np.ones_like(tidx)
            tidx[:1] = 0
        Xmax = np.array([obji.X_hist[tidx].max(axis=0) for obji in range_list]).max(axis=0)
        Xmin = np.array([obji.X_hist[tidx].min(axis=0) for obji in range_list]).min(axis=0)
        Xrng = (Xmax - Xmin).max() * 0.55
        Xmid = (Xmax + Xmin) / 2
        axi.set_xlim(Xmid[0] - Xrng, Xmid[0] + Xrng)
        axi.set_ylim(Xmid[1] - Xrng, Xmid[1] + Xrng)
        axi.set_xlabel('$x_1$')
        axi.set_ylabel('$x_2$')
        # print(Xmin, Xmax, Xrng, Xmid)
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
                         tavr=1, npabs=True):
    tidx = (problem.t_hist >= t_tmin) * (problem.t_hist <= t_tmax)
    if np.isnan(problem.obj_list[0].W_hist[tidx][0]):
        tidx[0] = False
    t_hist = problem.t_hist[tidx]
    t_use, dt_res = np.linspace(t_hist.min(), t_hist.max(), int(t_hist.size * resampling_fct), retstep=True)
    avg_stp = np.ceil(tavr / dt_res).astype('int')
    weights = np.ones(avg_stp) / avg_stp
    err_msg = 'tavr <= %f, current: %f' % (t_use.max() - t_use.min(), tavr)
    assert avg_stp <= t_use.size, err_msg
    
    avg_all = []
    for obji in problem.obj_list:  # type:particleClass.particle2D
        W_hist = interpolate.interp1d(t_hist, obji.W_hist[tidx], kind=interp1d_kind, copy=False)(t_use)
        avg_all.append(np.convolve(weights, W_hist, mode='same'))
    # avg_all = np.abs(np.vstack(avg_all))
    avg_all = np.abs(np.vstack(avg_all)) if npabs else np.vstack(avg_all)
    avg_stpD2 = avg_stp // 2
    for i0 in np.arange(avg_stpD2):
        avg_all[:, i0] = avg_all[:, i0] / (avg_stpD2 + i0 + avg_stp % 2) * avg_stp
        i1 = - i0 - 1
        avg_all[:, i1] = avg_all[:, i1] / (avg_stpD2 + i0 + 1) * avg_stp
    return t_use, avg_all


def cal_avrInfo(problem: 'problemClass._base2DProblem',
                t_tmin=-np.inf, t_tmax=np.inf,
                resampling_fct=1, interp1d_kind='quadratic',
                tavr=1, npabs=True):
    resampling_fct = 1 if resampling_fct is None else resampling_fct
    tidx = problem.t_hist < np.inf
    if np.isnan(problem.obj_list[0].W_hist[tidx][0]):
        tidx[0] = False
    t_hist = problem.t_hist[tidx]
    
    # without interpolation.
    if tavr is None:
        tidx2 = (t_hist >= t_tmin) * (t_hist <= t_tmax)
        t_use = t_hist[tidx2]
        W_avg = np.vstack([obji.W_hist[tidx] for obji in problem.obj_list])[:, tidx2][:, 1:]
        phi_avg = np.vstack([obji.phi_hist[tidx] for obji in problem.obj_list])[:, tidx2][:, 1:]
        W_avg = np.abs(np.vstack(W_avg)) if npabs else np.vstack(W_avg)
        return t_use, W_avg, phi_avg
    
    # interpolation
    t_use, dt_res = np.linspace(t_hist.min(), t_hist.max(), int(t_hist.size * resampling_fct), retstep=True)
    avg_stp = np.ceil(tavr / dt_res).astype('int')
    weights = np.ones(avg_stp) / avg_stp
    err_msg = 'tavr <= %f, current: %f' % (t_use.max() - t_use.min(), tavr)
    assert avg_stp <= t_use.size, err_msg
    #
    W_avg = []
    phi_avg = []
    for obji in problem.obj_list:  # type:particleClass.particle2D
        W_hist = interpolate.interp1d(t_hist, obji.W_hist[tidx], kind=interp1d_kind, copy=False)(t_use)
        phi_hist = get_continue_angle(t_hist, obji.phi_hist[tidx], t_use=t_use)
        W_avg.append(np.convolve(weights, W_hist, mode='same'))
        phi_avg.append(np.convolve(weights, phi_hist, mode='same'))
    W_avg = np.abs(np.vstack(W_avg)) if npabs else np.vstack(W_avg)
    phi_avg = np.vstack(phi_avg)
    avg_stpD2 = avg_stp // 2
    for i0 in np.arange(avg_stpD2):
        W_avg[:, i0] = W_avg[:, i0] / (avg_stpD2 + i0 + avg_stp % 2) * avg_stp
        phi_avg[:, i0] = phi_avg[:, i0] / (avg_stpD2 + i0 + avg_stp % 2) * avg_stp
        i1 = - i0 - 1
        W_avg[:, i1] = W_avg[:, i1] / (avg_stpD2 + i0 + 1) * avg_stp
        phi_avg[:, i1] = phi_avg[:, i1] / (avg_stpD2 + i0 + 1) * avg_stp
    #
    tidx2 = (t_use >= t_tmin) * (t_use <= t_tmax)
    t_use = t_use[tidx2]
    W_avg = W_avg[:, tidx2]
    phi_avg = phi_avg[:, tidx2]
    return t_use, W_avg, phi_avg


def cal_avrInfo_steer(problem: 'problemClass._base2DProblem',
                      t_tmin=-np.inf, t_tmax=np.inf,
                      resampling_fct=1, interp1d_kind='quadratic',
                      tavr=1, npabs=True):
    resampling_fct = 1 if resampling_fct is None else resampling_fct
    tidx = problem.t_hist < np.inf
    if np.isnan(problem.obj_list[0].W_steer_hist[tidx][0]):
        tidx[0] = False
    t_hist = problem.t_hist[tidx]
    
    # without interpolation.
    if tavr is None:
        tidx2 = (t_hist >= t_tmin) * (t_hist <= t_tmax)
        t_use = t_hist[tidx2]
        W_steer_avg = np.vstack([obji.W_steer_hist[tidx] for obji in problem.obj_list])[:, tidx2][:, 1:]
        # print(W_steer_avg)
        phi_steer_avg = np.vstack([obji.phi_steer_hist[tidx] for obji in problem.obj_list])[:, tidx2][:, 1:]
        W_steer_avg = np.abs(np.vstack(W_steer_avg)) if npabs else np.vstack(W_steer_avg)
        return t_use, W_steer_avg, phi_steer_avg
    
    # interpolation
    t_use, dt_res = np.linspace(t_hist.min(), t_hist.max(), int(t_hist.size * resampling_fct), retstep=True)
    avg_stp = np.ceil(tavr / dt_res).astype('int')
    weights = np.ones(avg_stp) / avg_stp
    err_msg = 'tavr <= %f, current: %f' % (t_use.max() - t_use.min(), tavr)
    assert avg_stp <= t_use.size, err_msg
    #
    W_steer_avg = []
    phi_steer_avg = []
    for obji in problem.obj_list:  # type:particleClass.particle2D
        W_steer_hist = interpolate.interp1d(t_hist, obji.W_steer_hist[tidx], kind=interp1d_kind, copy=False)(t_use)
        phi_steer_hist = get_continue_angle(t_hist, obji.phi_steer_hist[tidx], t_use=t_use)
        W_steer_avg.append(np.convolve(weights, W_steer_hist, mode='same'))
        phi_steer_avg.append(np.convolve(weights, phi_steer_hist, mode='same'))
    W_steer_avg = np.abs(np.vstack(W_steer_avg)) if npabs else np.vstack(W_steer_avg)
    phi_steer_avg = np.vstack(phi_steer_avg)
    avg_stpD2 = avg_stp // 2
    for i0 in np.arange(avg_stpD2):
        W_steer_avg[:, i0] = W_steer_avg[:, i0] / (avg_stpD2 + i0 + avg_stp % 2) * avg_stp
        phi_steer_avg[:, i0] = phi_steer_avg[:, i0] / (avg_stpD2 + i0 + avg_stp % 2) * avg_stp
        i1 = - i0 - 1
        W_steer_avg[:, i1] = W_steer_avg[:, i1] / (avg_stpD2 + i0 + 1) * avg_stp
        phi_steer_avg[:, i1] = phi_steer_avg[:, i1] / (avg_stpD2 + i0 + 1) * avg_stp
    #
    tidx2 = (t_use >= t_tmin) * (t_use <= t_tmax)
    t_use = t_use[tidx2]
    W_steer_avg = W_steer_avg[:, tidx2]
    phi_steer_avg = phi_steer_avg[:, tidx2]
    return t_use, W_steer_avg, phi_steer_avg


def fun_sort_idx(t_plot, W_avg, phi_avg, sort_type='normal', sort_idx=None):
    def sort_normal(t_plot, W_avg, phi_avg):
        assert t_plot.size - W_avg.shape[1] in (0, 1)
        t_use = t_plot[-W_avg.shape[1]:]
        t_threshold = (t_use.max() - t_use.min()) / 2 + t_use.min()
        tidx = t_use > t_threshold
        dt = np.hstack((0, np.diff(t_use)))
        sort_idx = np.argsort(np.mean(np.abs(W_avg[:, tidx] * dt[tidx]), axis=-1))
        return sort_idx
    
    sort_dict = {
        'normal':    sort_normal,
        'traveling': lambda t_plot, W_avg, phi_avg: np.argsort(phi_avg[:, -1])
        }
    try:
        sort_idx = sort_dict[sort_type](t_plot, W_avg, phi_avg) if sort_idx is None else sort_idx
    except:
        raise ValueError('wrong sort_type, current: %s, accept: %s' % (sort_type, sort_dict.keys()))
    return sort_idx


def core_avrPhaseVelocity(problem: 'problemClass.behavior2DProblem', figsize=np.array((50, 50)) * 5, dpi=100,
                          plt_tmin=-np.inf, plt_tmax=np.inf, resampling_fct=1, interp1d_kind='quadratic',
                          cmap=plt.get_cmap('bwr'), tavr=1, sort_type='normal', sort_idx=None,
                          vmin='None', vmax=1, npabs=True, norm='Normalize', ):
    if vmin == 'None':
        vmin = 0 if npabs else -1
    align = problem.align
    t_plot, W_avg, phi_avg = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                         resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                         tavr=tavr, npabs=npabs)
    sort_idx = fun_sort_idx(t_plot, W_avg, phi_avg, sort_type=sort_type, sort_idx=sort_idx)
    
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    norm = Normalize(vmin=vmin, vmax=vmax) if norm == 'Normalize' else norm
    obj_idx = np.arange(0, problem.n_obj + 1)
    # t_plot = np.hstack((t_plot, t_plot.max() + problem.eval_dt))
    c = axi.pcolorfast(t_plot, obj_idx, W_avg[sort_idx, :] / align, cmap=cmap, norm=norm)
    clb = fig.colorbar(c, ax=axi)
    if tavr is not None and (tavr > problem.eval_dt):
        if npabs:
            clb.ax.set_title('$\\langle | \\delta \\varphi | \\rangle / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$ \\langle \\delta \\varphi \\rangle / \\sigma $', fontsize='small')
    else:
        if npabs:
            clb.ax.set_title('$| \\delta \\varphi | / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$\\delta \\varphi / \\sigma $', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_title('$\\sigma = %20.10f$' % align)
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        align = problem.align
        axi.set_title('$\\sigma = %20.10f$' % align)
    return fig, axi


def core_avrPhase(problem: 'problemClass._base2DProblem', figsize=np.array((50, 50)) * 5, dpi=100,
                  plt_tmin=-np.inf, plt_tmax=np.inf, resampling_fct=1, interp1d_kind='quadratic',
                  cmap=plt.get_cmap('bwr'), tavr=1, sort_type='normal', sort_idx=None):
    t_plot, W_avg, phi_avg = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                         resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                         tavr=tavr)
    sort_idx = fun_sort_idx(t_plot, W_avg, phi_avg, sort_type=sort_type, sort_idx=sort_idx)
    
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    vmin, vmax = -1, 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    obj_idx = np.arange(0, problem.n_obj + 1)
    # c = axi.pcolor(t_plot, obj_idx, avg_all[sort_idx, :], cmap=cmap, norm=norm, shading='auto')
    c = axi.pcolorfast(t_plot, obj_idx, phi_avg[sort_idx, :] / np.pi, cmap=cmap, norm=norm)
    clb = fig.colorbar(c, ax=axi)
    clb.ax.set_title('$\\varphi / \\pi$', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(1, problem.n_obj)
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        align = problem.align
        axi.set_title('$\\sigma = %20.10f$' % align)
    return fig, axi


def core_phi_W(problem: 'problemClass._base2DProblem', figsize=np.array((50, 50)) * 5, dpi=100,
               plt_tmin=-np.inf, plt_tmax=np.inf, resampling_fct=1, interp1d_kind='quadratic',
               tavr=1, sort_type='normal', sort_idx=None,
               cmap_phi=plt.get_cmap('bwr'), cmap_W=plt.get_cmap('bwr'),
               vmin_W='None', vmax_W=1, npabs_W=True, norm_W='Normalize', ):
    if isinstance(problem, problemClass.behavior2DProblem):
        align = problem.align
    t_plot, W_avg, phi_avg = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                         resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                         tavr=tavr, npabs=npabs_W)
    sort_idx = fun_sort_idx(t_plot, W_avg, phi_avg, sort_type=sort_type, sort_idx=sort_idx)
    figs, axs = [], []
    
    # phi ------------------------------------------------------------------------------------------
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    vmin_phi, vmax_phi = -1, 1
    norm_phi = Normalize(vmin=vmin_phi, vmax=vmax_phi)
    obj_idx = np.arange(0, problem.n_obj + 1)
    # c = axi.pcolor(t_plot, obj_idx, avg_all[sort_idx, :], cmap=cmap, norm=norm, shading='auto')
    c = axi.pcolorfast(t_plot, obj_idx, phi_avg[sort_idx, :] / np.pi, cmap=cmap_phi, norm=norm_phi)
    clb = fig.colorbar(c, ax=axi)
    clb.ax.set_title('$\\varphi / \\pi$', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(1, problem.n_obj)
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        axi.set_title('$\\sigma = %20.10f$' % align)
    figs.append(fig)
    axs.append(axi)
    
    # W ------------------------------------------------------------------------------------------
    if vmin_W == 'None':
        vmin_W = 0 if npabs_W else -1
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    norm_W = Normalize(vmin=vmin_W, vmax=vmax_W) if norm_W == 'Normalize' else norm_W
    obj_idx = np.arange(0, problem.n_obj + 1)
    # t_plot = np.hstack((t_plot, t_plot.max() + problem.eval_dt))
    c = axi.pcolorfast(t_plot, obj_idx, W_avg[sort_idx, :] / align, cmap=cmap_W, norm=norm_W)
    clb = fig.colorbar(c, ax=axi)
    if tavr is not None and (tavr > problem.eval_dt):
        if npabs_W:
            clb.ax.set_title('$\\langle | \\delta \\varphi | \\rangle / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$ \\langle \\delta \\varphi \\rangle / \\sigma $', fontsize='small')
    else:
        if npabs_W:
            clb.ax.set_title('$| \\delta \\varphi | / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$\\delta \\varphi / \\sigma $', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_title('$\\sigma = %20.10f$' % align)
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        align = problem.align
        axi.set_title('$\\sigma = %20.10f$' % align)
    figs.append(fig)
    axs.append(axi)
    return figs, axs


def core_phis_Ws(problem: 'problemClass._base2DProblem', figsize=np.array((50, 50)) * 5, dpi=100,
                 plt_tmin=-np.inf, plt_tmax=np.inf, resampling_fct=1, interp1d_kind='quadratic',
                 tavr=1, sort_type='normal', sort_idx=None,
                 cmap_phis=plt.get_cmap('bwr'), cmap_Ws=plt.get_cmap('bwr'),
                 vmin_Ws='None', vmax_Ws=1, npabs_Ws=True, norm_Ws='Normalize', ):
    if isinstance(problem, problemClass.behavior2DProblem):
        align = problem.align
    t_plot, Ws_avg, phis_avg = cal_avrInfo_steer(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                                 resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                                 tavr=tavr, npabs=npabs_Ws)
    sort_idx = fun_sort_idx(t_plot, Ws_avg, phis_avg, sort_type=sort_type, sort_idx=sort_idx)
    figs, axs = [], []
    
    # phis ------------------------------------------------------------------------------------------
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    vmin_phis, vmax_phis = -1, 1
    norm_phis = Normalize(vmin=vmin_phis, vmax=vmax_phis)
    obj_idx = np.arange(0, problem.n_obj + 1)
    # c = axi.pcolor(t_plot, obj_idx, avg_all[sort_idx, :], cmap=cmap, norm=norm, shading='auto')
    c = axi.pcolorfast(t_plot, obj_idx, phis_avg[sort_idx, :] / np.pi, cmap=cmap_phis, norm=norm_phis)
    clb = fig.colorbar(c, ax=axi)
    clb.ax.set_title('$\\varphi_s / \\pi$', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(1, problem.n_obj)
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        axi.set_title('$\\sigma = %20.10f$' % align)
    figs.append(fig)
    axs.append(axi)
    
    # Ws ------------------------------------------------------------------------------------------
    if vmin_Ws == 'None':
        vmin_Ws = 0 if npabs_Ws else -1
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    norm_Ws = Normalize(vmin=vmin_Ws, vmax=vmax_Ws) if norm_Ws == 'Normalize' else norm_Ws
    obj_idx = np.arange(0, problem.n_obj + 1)
    # t_plot = np.hstack((t_plot, t_plot.max() + problem.eval_dt))
    c = axi.pcolorfast(t_plot, obj_idx, Ws_avg[sort_idx, :] / align, cmap=cmap_Ws, norm=norm_Ws)
    clb = fig.colorbar(c, ax=axi)
    if tavr is not None and (tavr > problem.eval_dt):
        if npabs_Ws:
            clb.ax.set_title('$\\langle | \\delta \\varphi_s | \\rangle / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$ \\langle \\delta \\varphi_s \\rangle / \\sigma $', fontsize='small')
    else:
        if npabs_Ws:
            clb.ax.set_title('$| \\delta \\varphi_s | / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$\\delta \\varphi_s / \\sigma $', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_title('$\\sigma = %20.10f$' % align)
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        align = problem.align
        axi.set_title('$\\sigma = %20.10f$' % align)
    figs.append(fig)
    axs.append(axi)
    return figs, axs


def core_phi_W_phis_Ws(problem: 'problemClass._base2DProblem', figsize=np.array((50, 50)) * 5, dpi=100,
                       plt_tmin=-np.inf, plt_tmax=np.inf, resampling_fct=1, interp1d_kind='quadratic',
                       tavr=1, sort_type='normal', sort_idx=None,
                       cmap_phi=plt.get_cmap('bwr'), cmap_W=plt.get_cmap('bwr'),
                       vmin_W='None', vmax_W=1, npabs_W=True, norm_W='Normalize',
                       cmap_phis=plt.get_cmap('bwr'), cmap_Ws=plt.get_cmap('bwr'),
                       vmin_Ws='None', vmax_Ws=1, npabs_Ws=True, norm_Ws='Normalize', ):
    if isinstance(problem, problemClass.behavior2DProblem):
        align = problem.align
    t_plot, W_avg, phi_avg = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                         resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                         tavr=tavr, npabs=npabs_W)
    t_plot, Ws_avg, phis_avg = cal_avrInfo_steer(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                                 resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                                 tavr=tavr, npabs=npabs_Ws)
    sort_idx = fun_sort_idx(t_plot, W_avg, phi_avg, sort_type=sort_type, sort_idx=sort_idx)
    figs, axs = [], []
    
    # phi ------------------------------------------------------------------------------------------
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    vmin_phi, vmax_phi = -1, 1
    norm_phi = Normalize(vmin=vmin_phi, vmax=vmax_phi)
    obj_idx = np.arange(0, problem.n_obj + 1)
    # c = axi.pcolor(t_plot, obj_idx, avg_all[sort_idx, :], cmap=cmap, norm=norm, shading='auto')
    c = axi.pcolorfast(t_plot, obj_idx, phi_avg[sort_idx, :] / np.pi, cmap=cmap_phi, norm=norm_phi)
    clb = fig.colorbar(c, ax=axi)
    clb.ax.set_title('$\\varphi / \\pi$', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(1, problem.n_obj)
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        axi.set_title('$\\sigma = %20.10f$' % align)
    figs.append(fig)
    axs.append(axi)
    
    # W ------------------------------------------------------------------------------------------
    if vmin_W == 'None':
        vmin_W = 0 if npabs_W else -1
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    norm_W = Normalize(vmin=vmin_W, vmax=vmax_W) if norm_W == 'Normalize' else norm_W
    obj_idx = np.arange(0, problem.n_obj + 1)
    # t_plot = np.hstack((t_plot, t_plot.max() + problem.eval_dt))
    c = axi.pcolorfast(t_plot, obj_idx, W_avg[sort_idx, :] / align, cmap=cmap_W, norm=norm_W)
    clb = fig.colorbar(c, ax=axi)
    if tavr is not None and (tavr > problem.eval_dt):
        if npabs_W:
            clb.ax.set_title('$\\langle | \\delta \\varphi | \\rangle / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$ \\langle \\delta \\varphi \\rangle / \\sigma $', fontsize='small')
    else:
        if npabs_W:
            clb.ax.set_title('$| \\delta \\varphi | / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$\\delta \\varphi / \\sigma $', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_title('$\\sigma = %20.10f$' % align)
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        align = problem.align
        axi.set_title('$\\sigma = %20.10f$' % align)
    figs.append(fig)
    axs.append(axi)
    
    # phis ------------------------------------------------------------------------------------------
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    vmin_phis, vmax_phis = -1, 1
    norm_phis = Normalize(vmin=vmin_phis, vmax=vmax_phis)
    obj_idx = np.arange(0, problem.n_obj + 1)
    # c = axi.pcolor(t_plot, obj_idx, avg_all[sort_idx, :], cmap=cmap, norm=norm, shading='auto')
    c = axi.pcolorfast(t_plot, obj_idx, phis_avg[sort_idx, :] / np.pi, cmap=cmap_phis, norm=norm_phis)
    clb = fig.colorbar(c, ax=axi)
    clb.ax.set_title('$\\varphi_s / \\pi$', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(1, problem.n_obj)
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        axi.set_title('$\\sigma = %20.10f$' % align)
    figs.append(fig)
    axs.append(axi)
    
    # Ws ------------------------------------------------------------------------------------------
    if vmin_Ws == 'None':
        vmin_Ws = 0 if npabs_Ws else -1
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    norm_Ws = Normalize(vmin=vmin_Ws, vmax=vmax_Ws) if norm_Ws == 'Normalize' else norm_Ws
    obj_idx = np.arange(0, problem.n_obj + 1)
    # t_plot = np.hstack((t_plot, t_plot.max() + problem.eval_dt))
    c = axi.pcolorfast(t_plot, obj_idx, Ws_avg[sort_idx, :] / align, cmap=cmap_Ws, norm=norm_Ws)
    clb = fig.colorbar(c, ax=axi)
    if tavr is not None and (tavr > problem.eval_dt):
        if npabs_Ws:
            clb.ax.set_title('$\\langle | \\delta \\varphi_s | \\rangle / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$ \\langle \\delta \\varphi_s \\rangle / \\sigma $', fontsize='small')
    else:
        if npabs_Ws:
            clb.ax.set_title('$| \\delta \\varphi_s | / \\sigma $', fontsize='small')
        else:
            clb.ax.set_title('$\\delta \\varphi_s / \\sigma $', fontsize='small')
    axi.set_xlabel('$t$')
    axi.set_ylabel('index')
    axi.set_title('$\\sigma = %20.10f$' % align)
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    yticks = axi.get_yticks()
    if yticks.size < problem.n_obj:
        yticks[0] = 1
        axi.set_yticks(yticks - 0.5)
        axi.set_yticklabels(['%d' % i0 for i0 in yticks])
    else:
        axi.set_yticks(obj_idx[1:] - 0.5)
        axi.set_yticklabels(obj_idx[1:])
    if isinstance(problem, problemClass.behavior2DProblem):
        align = problem.align
        axi.set_title('$\\sigma = %20.10f$' % align)
    figs.append(fig)
    axs.append(axi)
    return figs, axs


def cal_polar_order(problem: 'problemClass._base2DProblem',
                    t_tmin=-np.inf, t_tmax=np.inf, show_idx=None):
    tidx = (problem.t_hist >= t_tmin) * (problem.t_hist <= t_tmax)
    if np.isnan(problem.obj_list[0].W_hist[tidx][0]):
        tidx[0] = False
    t_hist = problem.t_hist[tidx]
    if show_idx is None:
        show_idx = np.arange(problem.n_obj)
    
    cplx_R = np.mean(np.array([(np.cos(tobj.phi_hist[tidx]), np.sin(tobj.phi_hist[tidx]))
                               for tobj in problem.obj_list[show_idx]]), axis=0).T
    # avg_all = np.vstack(avg_all)
    return t_hist, cplx_R


def cal_polar_order_chimera(problem: 'problemClass._base2DProblem',
                            t_tmin=-np.inf, t_tmax=np.inf, npabs=True,
                            w_Range=(-np.inf, np.inf),
                            resampling_fct=1, interp1d_kind='quadratic', tavr=1, ):
    tidx = (problem.t_hist >= t_tmin) * (problem.t_hist <= t_tmax)
    if np.isnan(problem.obj_list[0].W_hist[tidx][0]):
        tidx[0] = False
    t_hist = problem.t_hist[tidx]
    t_use, dt_res = np.linspace(t_hist.min(), t_hist.max(), int(t_hist.size * resampling_fct), retstep=True)
    avg_stp = np.ceil(tavr / dt_res).astype('int')
    weights = np.ones(avg_stp) / avg_stp
    err_msg = 'tavr <= %f, current: %f' % (t_use.max() - t_use.min(), tavr)
    assert avg_stp <= t_use.size, err_msg
    
    W_avg = []
    phi_hst = []
    for obji in problem.obj_list:  # type:particleClass.particle2D
        W_avg.append(np.convolve(weights,
                                 interpolate.interp1d(t_hist, obji.W_hist[tidx], kind=interp1d_kind, copy=False)(t_use),
                                 mode='same'))
        phi_hst.append(get_continue_angle(t_hist, obji.phi_hist[tidx], t_use=t_use))
    W_avg = np.abs(np.vstack(W_avg)) if npabs else np.vstack(W_avg)
    avg_stpD2 = avg_stp // 2
    for i0 in np.arange(avg_stpD2):
        W_avg[:, i0] = W_avg[:, i0] / (avg_stpD2 + i0 + avg_stp % 2) * avg_stp
        i1 = - i0 - 1
        W_avg[:, i1] = W_avg[:, i1] / (avg_stpD2 + i0 + 1) * avg_stp
    phi_hst = np.vstack(phi_hst)
    #
    cos_phi = np.zeros_like(t_use)
    sin_phi = np.zeros_like(t_use)
    use_phi = np.zeros_like(t_use)
    for tW, tphi in zip(W_avg, phi_hst):
        ttidx = np.logical_and(tW >= w_Range[0], tW <= w_Range[1])
        use_phi = use_phi + ttidx
        cos_phi = cos_phi + np.cos(tphi) * ttidx
        sin_phi = sin_phi + np.sin(tphi) * ttidx
    cplx_R = (cos_phi + 1j * sin_phi) / use_phi
    return t_hist, cplx_R


def core_polar_order(problem: 'problemClass._base2DProblem',
                     figsize=np.array((50, 50)) * 5, dpi=100,
                     plt_tmin=-np.inf, plt_tmax=np.inf,
                     markevery=0.3, linestyle='-C1',
                     xscale='linear', yscale='linear',
                     show_idx=None):
    t_hist, cplx_R = cal_polar_order(problem, t_tmin=plt_tmin, t_tmax=plt_tmax, show_idx=show_idx)
    odp_R = np.linalg.norm(cplx_R, axis=-1)
    xlim, ylim = (t_hist.min(), t_hist.max()), (0, 1.03)
    
    fig, axi = plt.subplots(1, 1, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.patch.set_facecolor('white')
    axi.plot(t_hist, odp_R, linestyle, markevery=markevery, )
    axi.set_xlabel('$t$')
    axi.set_ylabel('$R$')
    axi.set_xscale(xscale)
    axi.set_yscale(yscale)
    axi.set_xlim(*xlim)
    axi.set_ylim(*ylim)
    axi.spines['top'].set_visible(False)
    axi.spines['right'].set_visible(False)
    axi.spines['left'].set_position(('data', xlim[0]))
    axi.spines['bottom'].set_position(('data', ylim[0]))
    axi.tick_params(direction='in')
    axi.plot(1, ylim[0], ">k", transform=axi.get_yaxis_transform(), clip_on=False)
    axi.plot(xlim[0], 1, "^k", transform=axi.get_xaxis_transform(), clip_on=False)
    return fig, axi


deta1_fun = lambda meat1, meta2, alpha, align: spf.warpToPi(
        -align / 2 * (2 * np.sin(meat1) * np.cos(alpha) + np.sin(meta2 + alpha) + np.sin(meat1 - meta2 - alpha)))
deta2_fun = lambda meat1, meta2, alpha, align: spf.warpToPi(
        -align / 2 * (2 * np.sin(meta2) * np.cos(alpha) + np.sin(meat1 + alpha) + np.sin(meta2 - meat1 - alpha)))


def fun_grdmp_bck(fig, axi, alpha, align, order=1, cmap=plt.get_cmap('gray_r'),
                  plt_colorbar=True):
    def _fun_grdmp_nth(etati, alpha, align, order):
        for _ in np.arange(order):
            etati = [spf.warpToPi(deta1_fun(etati[0], etati[1], alpha, align) + etati[0]),
                     spf.warpToPi(deta2_fun(etati[0], etati[1], alpha, align) + etati[1])]
        return etati
    
    # plot background gradient map
    etat0 = np.meshgrid(np.linspace(-1, 1, 21) * np.pi, np.linspace(-1, 1, 21) * np.pi)
    eta_use = _fun_grdmp_nth(etat0, alpha, align, order)
    deta_use = [spf.warpToPi(eta_use[0] - etat0[0]),
                spf.warpToPi(eta_use[1] - etat0[1])]
    axi.set_xlabel('$\\eta_1 / \\pi$')
    axi.set_ylabel('$\\eta_2 / \\pi$')
    axi.set_title('$\\sigma = %f$' % align)
    
    # # gradient field, normalized by sigma
    # norm = LogNorm(vmin=1e-2, vmax=1e0)
    # tnorm = np.sqrt(deta_use[0] ** 2 + deta_use[1] ** 2)
    # cqu = axi.quiver(etat0[0] / np.pi, etat0[1] / np.pi, deta_use[0] / tnorm, deta_use[1] / tnorm, tnorm / align,
    #                  norm=norm, cmap=cmap, angles='xy', pivot='mid', scale=40)
    # fig.colorbar(cqu).ax.set_title('$|\\delta\\eta^{t+%d}| / \\sigma$ \n' % order)
    
    # gradient field, MOD pi
    norm = Normalize(vmin=0, vmax=1)
    tnorm = np.sqrt(deta_use[0] ** 2 + deta_use[1] ** 2)
    cqu = axi.quiver(etat0[0] / np.pi, etat0[1] / np.pi, deta_use[0] / tnorm, deta_use[1] / tnorm, tnorm / np.pi,
                     norm=norm, cmap=cmap, angles='xy', pivot='mid', scale=40)
    if plt_colorbar:
        cbar = fig.colorbar(cqu, shrink=.99)
        cbar.ax.set_title('$\\sfrac{|\\delta\\eta^{t+%d}|}{\\pi} $' % order, fontsize='small')
        cbar.ax.tick_params(labelsize='small')
    return True


def axi_avrPhaseVelocity(problem: 'problemClass.behavior2DProblem', axi,
                         plt_tmin_fct=0, plt_tmax_fct=1, tavr_fct=0.1,
                         resampling_fct=1, interp1d_kind='quadratic',
                         vmin=0, vmax=1, cmap=plt.get_cmap('bwr'), npabs=True,
                         sort_idx=None, substract_mean0=False):
    ini_t, max_t, eval_dt = problem.t0, problem.t1, problem.eval_dt
    plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * plt_tmin_fct, ini_t + (
            max_t - ini_t) * plt_tmax_fct, tavr_fct * eval_dt
    align = problem.align
    t_plot, avg_all, _ = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                     resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                     tavr=tavr, npabs=npabs)
    sort_idx = np.arange(0, 3) if sort_idx is None else sort_idx
    if substract_mean0:
        avg_all = avg_all - np.mean(avg_all[sort_idx[0]])
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    obj_idx = np.arange(0, problem.n_obj + 1)
    c = axi.pcolorfast(t_plot, obj_idx, avg_all[sort_idx, :] / align, cmap=cmap, norm=norm)
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    axi.set_yticks(obj_idx[1:] - 0.5)
    axi.set_yticklabels(obj_idx[1:])
    return c


def axi_rltPhaseVelocity(problem: 'problemClass.behavior2DProblem', axi,
                         plt_tmin_fct=0, plt_tmax_fct=1, tavr_fct=0.1,
                         resampling_fct=1, interp1d_kind='quadratic',
                         vmin=0, vmax=1, cmap=plt.get_cmap('bwr'), npabs=True,
                         sort_idx=None, ):
    ini_t, max_t, eval_dt = problem.t0, problem.t1, problem.eval_dt
    plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * plt_tmin_fct, ini_t + (
            max_t - ini_t) * plt_tmax_fct, tavr_fct * eval_dt
    align = problem.align
    t_plot, avg_all, _ = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                     resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                     tavr=tavr, npabs=npabs)
    sort_idx = np.arange(0, 3) if sort_idx is None else sort_idx
    avg_all = avg_all - avg_all[sort_idx[0]]
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    obj_idx = np.arange(0, problem.n_obj)
    c = axi.pcolorfast(t_plot, obj_idx, avg_all[sort_idx[1:], :] / align, cmap=cmap, norm=norm)
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(obj_idx.min(), problem.n_obj - 1)
    axi.set_yticks(obj_idx[1:] - 0.5)
    axi.set_yticklabels(obj_idx[1:])
    return c


def axi_avrPhase(problem: 'problemClass.behavior2DProblem', axi,
                 plt_tmin_fct=0, plt_tmax_fct=1, tavr_fct=0.1,
                 resampling_fct=1, interp1d_kind='quadratic',
                 cmap=plt.get_cmap('bwr'), npabs=True,
                 sort_idx=None):
    ini_t, max_t, eval_dt = problem.t0, problem.t1, problem.eval_dt
    plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * plt_tmin_fct, ini_t + (
            max_t - ini_t) * plt_tmax_fct, tavr_fct * eval_dt
    align = problem.align
    t_plot, W_avg, phi_avg = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                         resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                         tavr=tavr, npabs=npabs)
    sort_idx = np.arange(0, 3) if sort_idx is None else sort_idx
    phi_avg = phi_avg - np.mean(phi_avg[sort_idx[0]])
    
    vmin, vmax = -1, 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    obj_idx = np.arange(0, problem.n_obj + 1)
    c = axi.pcolorfast(t_plot, obj_idx, phi_avg[sort_idx, :] / align, cmap=cmap, norm=norm)
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(obj_idx.min(), problem.n_obj)
    axi.set_yticks(obj_idx[1:] - 0.5)
    axi.set_yticklabels(obj_idx[1:])
    return c


def axi_rltPhase(problem: 'problemClass.behavior2DProblem', axi,
                 plt_tmin_fct=0, plt_tmax_fct=1, tavr_fct=0.1,
                 resampling_fct=1, interp1d_kind='quadratic',
                 cmap=plt.get_cmap('bwr'), npabs=True,
                 sort_idx=None, ):
    ini_t, max_t, eval_dt = problem.t0, problem.t1, problem.eval_dt
    plt_tmin, plt_tmax, tavr = ini_t + (max_t - ini_t) * plt_tmin_fct, ini_t + (
            max_t - ini_t) * plt_tmax_fct, tavr_fct * eval_dt
    align = problem.align
    t_plot, W_avg, phi_avg = cal_avrInfo(problem=problem, t_tmin=plt_tmin, t_tmax=plt_tmax,
                                         resampling_fct=resampling_fct, interp1d_kind=interp1d_kind,
                                         tavr=tavr, npabs=npabs)
    sort_idx = np.arange(0, 3) if sort_idx is None else sort_idx
    phi_avg = phi_avg - phi_avg[sort_idx[0]]
    
    vmin, vmax = -1, 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    obj_idx = np.arange(0, problem.n_obj)
    c = axi.pcolorfast(t_plot, obj_idx, phi_avg[sort_idx[1:], :] / align, cmap=cmap, norm=norm)
    axi.set_xlim(t_plot.min(), t_plot.max())
    axi.set_ylim(obj_idx.min(), problem.n_obj - 1)
    axi.set_yticks(obj_idx[1:] - 0.5)
    axi.set_yticklabels(obj_idx[1:])
    return c
