# coding: utf-8
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:05:23 2017

@author: zhangji
"""

# plt.rcParams['figure.figsize'] = (18.5, 10.5)
# fontsize = 40

import os
# import glob
import numpy as np
# import matplotlib
import re
from scanf import scanf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from scipy import interpolate  # , integrate, optimize
from mpi4py import MPI
import cProfile

# font = {'size': 20}
# matplotlib.rc('font', **font)
# np.set_printoptions(linewidth=90, precision=5)

markerstyle_list = ['^', 'v', 'o', 's', 'p', 'd', 'H',
                    '1', '2', '3', '4', '8', 'P', '*',
                    'h', '+', 'x', 'X', 'D', '|', '_', ]

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
              '#7f7f7f', '#bcbd22', '#17becf']


def func_line(x, a0, a1):
    y = a0 + a1 * x
    return y


def fit_line(ax, x, y, x0, x1, ifprint=1, linestyle='-.', linewidth=1, extendline=False,
             color='k', alpha=0.7):
    idx = np.array(x >= x0) & np.array(x <= x1) & np.isfinite(x) & np.isfinite(y)
    tx = x[idx]
    ty = y[idx]
    fit_para = np.polyfit(tx, ty, 1)
    pol_y = np.poly1d(fit_para)
    if extendline:
        fit_x = np.linspace(x.min(), x.max(), 100)
    else:
        fit_x = np.linspace(max(x.min(), x0), min(x.max(), x1), 100)
    if ax is not None:
        ax.plot(fit_x, pol_y(fit_x), linestyle, linewidth=linewidth,
                color=color, alpha=alpha)
    if ifprint:
        print('y = %f + %f * x' % (fit_para[1], fit_para[0]), 'in range',
              (x[idx].min(), x[idx].max()))
    return fit_para


def fit_power_law(ax, x, y, x0, x1, ifprint=1, linestyle='-.', linewidth=1, extendline=False,
                  color='k', alpha=0.7):
    idx = np.array(x >= x0) & np.array(x <= x1) & np.isfinite((np.log10(x))) & np.isfinite(
        (np.log10(y)))
    tx = np.log10(x[idx])
    ty = np.log10(y[idx])
    fit_para = np.polyfit(tx, ty, 1)
    pol_y = np.poly1d(fit_para)

    if extendline:
        fit_x = np.log10(np.linspace(x.min(), x.max(), 30))
    else:
        fit_x = np.log10(np.linspace(max(x.min(), x0), min(x.max(), x1), 30))
    if ax is not None:
        ax.loglog(10 ** fit_x, 10 ** pol_y(fit_x), linestyle, linewidth=linewidth,
                  color=color, alpha=alpha)
    if ifprint:
        print('log(y) = %f + %f * log(x)' % (fit_para[1], fit_para[0]), 'in range',
              (10 ** tx.min(), 10 ** tx.max()))
        print('ln(y) = %f + %f * ln(x)' % (fit_para[1] * np.log(10), fit_para[0]), 'in range',
              (10 ** tx.min(), 10 ** tx.max()))
    return fit_para


def fit_semilogy(ax, x, y, x0, x1, ifprint=1, linestyle='-.', linewidth=1, extendline=False,
                 color='k', alpha=0.7):
    idx = np.array(x >= x0) & np.array(x <= x1) & np.isfinite(x) & np.isfinite(np.log10(y))
    tx = x[idx]
    ty = np.log10(y[idx])
    fit_para = np.polyfit(tx, ty, 1)
    pol_y = np.poly1d(fit_para)
    if extendline:
        fit_x = np.linspace(x.min(), x.max(), 30)
    else:
        fit_x = np.linspace(max(x.min(), x0), min(x.max(), x1), 30)
    if ax is not None:
        ax.plot(fit_x, 10 ** pol_y(fit_x), linestyle, linewidth=linewidth,
                color=color, alpha=alpha)
    if ifprint:
        print('log(y) = %f + %f * x' % (fit_para[1], fit_para[0]), 'in range', (tx.min(), tx.max()))
        fit_para = fit_para * np.log(10)
        print('ln(y) = %f + %f * x' % (fit_para[1], fit_para[0]), 'in range', (tx.min(), tx.max()))
    return fit_para


def norm_self(v):
    return v / np.linalg.norm(v)


def angle_2vectors(v1, v2, vct_direct=None):
    v1 = norm_self(np.array(v1).ravel())
    v2 = norm_self(np.array(v2).ravel())
    err_msg = 'inputs are not 3 dimensional vectors. '
    assert v1.size == 3, err_msg
    assert v2.size == 3, err_msg
    t1 = np.dot(v1, v2)
    if vct_direct is None:
        sign = 1
    else:
        vct_direct = norm_self(np.array(vct_direct).ravel())
        assert vct_direct.size == 3, err_msg
        sign = np.sign(np.dot(vct_direct, np.cross(v1, v2)))
    theta = sign * np.arccos(t1)
    return theta


def rot_vec2rot_mtx(rot_vct):
    rot_vct = np.array(rot_vct).flatten()
    err_msg = 'rot_vct is a numpy array contain three components. '
    assert rot_vct.size == 3, err_msg

    def S(vct):
        rot_mtx = np.array([[0, -vct[2], vct[1]],
                            [vct[2], 0, -vct[0]],
                            [-vct[1], vct[0], 0]])
        return rot_mtx

    theta = np.linalg.norm(rot_vct)
    if theta > 1e-6:
        n = rot_vct / theta
        Sn = S(n)
        R = np.eye(3) + np.sin(theta) * Sn + (1 - np.cos(theta)) * np.dot(Sn, Sn)
    else:
        Sr = S(rot_vct)
        theta2 = theta ** 2
        R = np.eye(3) + (1 - theta2 / 6.) * Sr + (.5 - theta2 / 24.) * np.dot(Sr, Sr)
    return R


def get_rot_matrix(norm=np.array([0, 0, 1]), theta=0):
    norm = np.array(norm).reshape((3,))
    theta = -1 * float(theta)
    if np.linalg.norm(norm) > 0:
        norm = norm / np.linalg.norm(norm)
    a = norm[0]
    b = norm[1]
    c = norm[2]
    rotation = np.array([
        [a ** 2 + (1 - a ** 2) * np.cos(theta),
         a * b * (1 - np.cos(theta)) + c * np.sin(theta),
         a * c * (1 - np.cos(theta)) - b * np.sin(theta)],
        [a * b * (1 - np.cos(theta)) - c * np.sin(theta),
         b ** 2 + (1 - b ** 2) * np.cos(theta),
         b * c * (1 - np.cos(theta)) + a * np.sin(theta)],
        [a * c * (1 - np.cos(theta)) + b * np.sin(theta),
         b * c * (1 - np.cos(theta)) - a * np.sin(theta),
         c ** 2 + (1 - c ** 2) * np.cos(theta)]])
    return rotation


def vector_rotation_norm(P2, norm=np.array([0, 0, 1]), theta=0, rotation_origin=np.zeros(3)):
    rotation = get_rot_matrix(norm, theta)
    P20 = np.dot(rotation, (P2 - rotation_origin)) + rotation_origin
    P20 = P20 / np.linalg.norm(P20)
    return P20


def vector_rotation(P2, norm=np.array([0, 0, 1]), theta=0, rotation_origin=np.zeros(3)):
    rotation = get_rot_matrix(norm, theta)
    P20 = np.dot(rotation, (P2 - rotation_origin)) + rotation_origin
    return P20


def rotMatrix_DCM(x0, y0, z0, x, y, z):
    # Diebel, James. "Representing attitude: Euler angles, unit quaternions, and rotation vectors."
    #  Matrix 58.15-16 (2006): 1-35.
    # eq. 17
    # https://arxiv.org/pdf/1705.06997.pdf
    # appendix B
    # Graf, Basile. "Quaternions and dynamics." arXiv preprint arXiv:0811.2889 (2008).
    #
    # A rotation matrix may also be referred to as a direction
    # cosine matrix, because the elements of this matrix are the
    # cosines of the unsigned angles between the body-¯xed axes
    # and the world axes. Denoting the world axes by (x; y; z)
    # and the body-fixed axes by (x0; y0; z0), let \theta_{x';y} be,
    # for example, the unsigned angle between the x'-axis and the y-axis
    # (x0, y0, z0)^T = dot(R, (x, y, z)^T )

    R = np.array(((np.dot(x0, x), np.dot(x0, y), np.dot(x0, z)),
                  (np.dot(y0, x), np.dot(y0, y), np.dot(y0, z)),
                  (np.dot(z0, x), np.dot(z0, y), np.dot(z0, z))))
    return R


def Rloc2glb(theta, phi, psi):
    rotM = np.array(
        ((np.cos(phi) * np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi),
          -(np.cos(psi) * np.sin(phi)) - np.cos(phi) * np.cos(theta) * np.sin(psi),
          np.cos(phi) * np.sin(theta)),
         (np.cos(psi) * np.cos(theta) * np.sin(phi) + np.cos(phi) * np.sin(psi),
          np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
          np.sin(phi) * np.sin(theta)),
         (-(np.cos(psi) * np.sin(theta)),
          np.sin(psi) * np.sin(theta),
          np.cos(theta))))
    return rotM


def mycot(x):
    return 1 / np.tan(x)


def mycsc(x):
    return 1 / np.sin(x)


def mysec(x):
    return 1 / np.cos(x)


def warpToPi(phases):
    # return (phases + np.pi) % (2 * np.pi) - np.pi
    return ((-phases + np.pi) % (2.0 * np.pi) - np.pi) * -1.0


def warpMinMax(x, min, max):
    # wrap x -> [min,max)
    return min + warpMax(x - min, max - min)


def warpMax(x, max):
    # wrap x -> [0,max)
    return np.mod(max + np.mod(x, max), max)


def Adams_Bashforth_Methods(order, f_list, eval_dt):
    def o1(f_list, eval_dt):
        delta = eval_dt * f_list[-1]
        return delta

    def o2(f_list, eval_dt):
        delta = eval_dt * (3 / 2 * f_list[-1] - 1 / 2 * f_list[-2])
        return delta

    def o3(f_list, eval_dt):
        delta = eval_dt * (23 / 12 * f_list[-1] - 16 / 12 * f_list[-2] + 5 / 12 * f_list[-3])
        return delta

    def o4(f_list, eval_dt):
        delta = eval_dt * (
                55 / 24 * f_list[-1] - 59 / 24 * f_list[-2] + 37 / 24 * f_list[-3] - 9 / 24 *
                f_list[-4])
        return delta

    def o5(f_list, eval_dt):
        delta = eval_dt * (
                1901 / 720 * f_list[-1] - 2774 / 720 * f_list[-2] + 2616 / 720 * f_list[-3]
                - 1274 / 720 * f_list[-4] + 251 / 720 * f_list[-5])
        return delta

    def get_order(order):
        return dict([(1, o1),
                     (2, o2),
                     (3, o3),
                     (4, o4),
                     (5, o5),
                     ]).get(order, o1)

    return get_order(order)(f_list, eval_dt)


def Adams_Moulton_Methods(order, f_list, eval_dt):
    def o1(f_list, eval_dt):
        delta = eval_dt * f_list[-1]
        return delta

    def o2(f_list, eval_dt):
        delta = eval_dt * (1 / 2 * f_list[-1] + 1 / 2 * f_list[-2])
        return delta

    def o3(f_list, eval_dt):
        delta = eval_dt * (5 / 12 * f_list[-1] + 8 / 12 * f_list[-2] - 1 / 12 * f_list[-3])
        return delta

    def o4(f_list, eval_dt):
        delta = eval_dt * (
                9 / 24 * f_list[-1] + 19 / 24 * f_list[-2] - 5 / 24 * f_list[-3] + 1 / 24 *
                f_list[-4])
        return delta

    def o5(f_list, eval_dt):
        delta = eval_dt * (251 / 720 * f_list[-1] + 646 / 720 * f_list[-2] - 264 / 720 * f_list[-3]
                           + 106 / 720 * f_list[-4] - 19 / 720 * f_list[-5])
        return delta

    def get_order(order):
        return dict([(1, o1),
                     (2, o2),
                     (3, o3),
                     (4, o4),
                     (5, o5),
                     ]).get(order, o1)

    return get_order(order)(f_list, eval_dt)


def write_pbs_head(fpbs, job_name, nodes=1):
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=24\n' % nodes)
    fpbs.write('#PBS -l walltime=72:00:00\n')
    fpbs.write('#PBS -q common\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')
    return True


def write_pbs_head_dbg(fpbs, job_name, nodes=1):
    assert np.isclose(nodes, 1)
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=24\n' % nodes)
    fpbs.write('#PBS -l walltime=24:00:00\n')
    fpbs.write('#PBS -q debug\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')
    return True


def write_pbs_head_serial(fpbs, job_name, nodes=1):
    assert np.isclose(nodes, 1)
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=1\n' % nodes)
    fpbs.write('#PBS -l walltime=1000:00:00\n')
    fpbs.write('#PBS -q serial\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')
    return True


def write_pbs_head_q03(fpbs, job_name, nodes=1):
    fpbs.write('#! /bin/bash\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=24\n' % nodes)
    fpbs.write('#PBS -l walltime=72:00:00\n')
    fpbs.write('#PBS -q q03\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('\n')
    return True


def write_pbs_head_newturb(fpbs, job_name, nodes=1):
    fpbs.write('#!/bin/sh\n')
    fpbs.write('#PBS -M zhangji@csrc.ac.cn\n')
    fpbs.write('#PBS -l nodes=%d:ppn=24\n' % nodes)
    fpbs.write('#PBS -l walltime=24:00:00\n')
    fpbs.write('#PBS -N %s\n' % job_name)
    fpbs.write('\n')
    fpbs.write('cd $PBS_O_WORKDIR\n')
    fpbs.write('source /storage/zhang/.bashrc\n')
    fpbs.write('\n')
    return True


def write_pbs_head_haiguang(fpbs, job_name, nodes=1):
    fpbs.write('#!/bin/sh\n')
    fpbs.write('# run the job in the main node directly. ')
    fpbs.write('\n')
    return True


def write_pbs_head_JiUbuntu(fpbs, job_name, nodes=1):
    fpbs.write('#!/bin/sh\n')
    fpbs.write('# run the job in the main node directly. ')
    fpbs.write('\n')
    return True


def _write_main_run_top(frun, main_hostname='ln0'):
    frun.write('t_dir=$PWD \n')
    frun.write('bash_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" \n\n')

    # check if the script run on the main node.
    frun.write('if [ $(hostname) == \'%s\' ]; then\n' % main_hostname)
    frun.write('    echo \'this node is %s. \' \n' % main_hostname)
    frun.write('else \n')
    frun.write('    echo \'please run in the node %s. \' \n' % main_hostname)
    frun.write('    exit \n')
    frun.write('fi \n\n')
    return True


def write_main_run(write_pbs_head, job_dir, ncase):
    tname = os.path.join(job_dir, 'main_run.pbs')
    print('ncase =', ncase)
    print('write parallel pbs file to %s' % tname)
    with open(tname, 'w') as fpbs:
        write_pbs_head(fpbs, job_dir, nodes=ncase)
        fpbs.write('seq 0 %d | parallel -j 1 -u --sshloginfile $PBS_NODEFILE \\\n' % (ncase - 1))
        fpbs.write('\"cd $PWD;echo $PWD;bash myscript.csh {}\"')
    return True


def write_main_run_comm_list(comm_list, txt_list, use_node, njob_node, job_dir,
                             write_pbs_head000=write_pbs_head, n_job_pbs=None,
                             random_order=False, ):
    def _parallel_pbs_ln0(n_use_comm, njob_node, csh_name):
        t2 = 'seq 0 %d | parallel -j %d -u ' % (n_use_comm - 1, njob_node)
        t2 = t2 + ' --sshloginfile $PBS_NODEFILE --sshdelay 0.1 '
        t2 = t2 + ' "cd $PWD; echo $PWD; echo; bash %s {} true " \n\n ' % csh_name
        return t2

    def _parallel_pbs_newturb(n_use_comm, njob_node, csh_name):
        t2 = 'seq 0 %d | parallel -j %d -u ' % (n_use_comm - 1, njob_node)
        t2 = t2 + ' --sshdelay 0.1 '
        t2 = t2 + ' "cd $PWD; echo $PWD; echo; bash %s {} true " \n\n ' % csh_name
        return t2

    PWD = os.getcwd()
    comm_list = np.array(comm_list)
    txt_list = np.array(txt_list)
    t_path = os.path.join(PWD, job_dir)
    if not os.path.exists(t_path):
        os.makedirs(t_path)
        print('make folder %s' % t_path)
    else:
        print('exist folder %s' % t_path)
    n_case = len(comm_list)
    if n_job_pbs is None:
        n_job_pbs = use_node * njob_node
    n_pbs = (n_case // n_job_pbs) + np.sign(n_case % n_job_pbs)
    if random_order:
        tidx = np.arange(n_case)
        np.random.shuffle(tidx)
        comm_list = comm_list[tidx]
        txt_list = txt_list[tidx]

    # generate comm_list.sh
    t_name0 = os.path.join(t_path, 'comm_list.sh')
    with open(t_name0, 'w') as fcomm:
        for i0, ts, f in zip(range(n_case), comm_list, txt_list):
            fcomm.write('%s > %s.txt 2> %s.err \n' % (ts, f, f))
            fcomm.write('echo \'%d / %d, %s start.\'  \n\n' % (i0 + 1, n_case, f))

    assert callable(write_pbs_head000)
    if write_pbs_head000 is write_pbs_head:
        main_hostname = 'ln0'
        _parallel_pbs_use = _parallel_pbs_ln0
        run_fun = 'qsub %s\n\n'
    # elif write_pbs_head000 is write_pbs_head_q03:
    #     main_hostname = 'ln0'
    #     _parallel_pbs_use = _parallel_pbs_ln0
    #     run_fun = 'qsub %s\n\n'
    elif write_pbs_head000 is write_pbs_head_dbg:
        main_hostname = 'ln0'
        _parallel_pbs_use = _parallel_pbs_ln0
        run_fun = 'qsub %s\n\n'
    elif write_pbs_head000 is write_pbs_head_q03:
        main_hostname = 'ln0'
        _parallel_pbs_use = _parallel_pbs_ln0
        run_fun = 'qsub %s\n\n'
    elif write_pbs_head000 is write_pbs_head_serial:
        main_hostname = 'ln0'
        _parallel_pbs_use = _parallel_pbs_ln0
        run_fun = 'qsub %s\n\n'
    elif write_pbs_head000 is write_pbs_head_newturb:
        main_hostname = 'newturb'
        _parallel_pbs_use = _parallel_pbs_newturb
        run_fun = 'qsub %s\n\n'
        assert np.isclose(use_node, 1)
    elif write_pbs_head000 is write_pbs_head_haiguang:
        main_hostname = 'bogon'
        _parallel_pbs_use = _parallel_pbs_newturb
        run_fun = 'cd $bash_dir \nnohup bash %s &\ncd $t_dir\n\n'
        assert np.isclose(use_node, 1)
    elif write_pbs_head000 is write_pbs_head_JiUbuntu:
        main_hostname = 'JiUbuntu'
        _parallel_pbs_use = _parallel_pbs_newturb
        run_fun = 'cd $bash_dir \nnohup bash %s &\ncd $t_dir\n\n'
        assert np.isclose(use_node, 1)
    else:
        raise ValueError('wrong write_pbs_head000')
    # generate .pbs file and .csh file
    t_name0 = os.path.join(t_path, 'main_run.sh')
    with open(t_name0, 'w') as frun:
        _write_main_run_top(frun, main_hostname=main_hostname)
        # noinspection PyTypeChecker
        for t1 in np.arange(n_pbs, dtype='int'):
            use_comm = comm_list[t1 * n_job_pbs: np.min(((t1 + 1) * n_job_pbs, n_case))]
            use_txt = txt_list[t1 * n_job_pbs: np.min(((t1 + 1) * n_job_pbs, n_case))]
            n_use_comm = len(use_comm)
            tnode = np.min((use_node, np.ceil(n_use_comm / njob_node)))
            pbs_name = 'run%03d.pbs' % t1
            csh_name = 'run%03d.csh' % t1
            # generate .pbs file
            t_name = os.path.join(t_path, pbs_name)
            with open(t_name, 'w') as fpbs:
                # pbs_head = '%s_%s' % (job_dir, pbs_name)
                pbs_head = '%s_%d' % (job_dir, t1)
                write_pbs_head000(fpbs, pbs_head, nodes=tnode)
                fpbs.write(_parallel_pbs_use(n_use_comm, njob_node, csh_name))
            # generate .csh file for submit
            t_name = os.path.join(t_path, csh_name)
            with open(t_name, 'w') as fcsh:
                fcsh.write('#!/bin/csh -fe \n\n')
                t2 = 'comm_list=('
                for t3 in use_comm:
                    t2 = t2 + '"%s" ' % t3
                t2 = t2 + ') \n\n'
                fcsh.write(t2)
                t2 = 'txt_list=('
                for t3 in use_txt:
                    t2 = t2 + '"%s" ' % t3
                t2 = t2 + ') \n\n'
                fcsh.write(t2)
                fcsh.write('echo ${comm_list[$1]} \'>\' ${txt_list[$1]}.txt'
                           ' \'2>\' ${txt_list[$1]}.err \n')
                fcsh.write('echo $(expr $1 + 1) / %d, ${txt_list[$1]} start.  \n' % n_case)
                fcsh.write('echo \n')
                fcsh.write('if [ ${2:-false} = true ]; then \n')
                fcsh.write('    ${comm_list[$1]} > ${txt_list[$1]}.txt 2> ${txt_list[$1]}.err \n')
                fcsh.write('fi \n\n')
            frun.write(run_fun % pbs_name)
        frun.write('\n')
    print('input %d cases.' % n_case)
    print('generate %d pbs files in total.' % n_pbs)
    if random_order:
        print(' --->>random order mode is ON. ')
    print('Command of first case is:')
    print(comm_list[0])
    return True


def write_myscript(job_name_list, job_dir):
    t1 = ' '.join(['\"%s\"' % job_name for job_name in job_name_list])
    tname = os.path.join(job_dir, 'myscript.csh')
    print('write myscript csh file to %s' % tname)
    with open(tname, 'w') as fcsh:
        fcsh.write('#!/bin/sh -fe\n')
        fcsh.write('job_name_list=(%s)\n' % t1)
        fcsh.write('\n')
        fcsh.write('echo ${job_name_list[$1]}\n')
        fcsh.write('cd ${job_name_list[$1]}\n')
        fcsh.write('bash ${job_name_list[$1]}.sh\n')
    return True


def write_main_run_local(comm_list, njob_node, job_dir, random_order=False, ):
    PWD = os.getcwd()
    comm_list = np.array(comm_list)
    n_comm = comm_list.size
    pbs_name = 'main_run.pbs'
    csh_name = 'csh.main_run'

    t_path = os.path.join(PWD, job_dir)
    if not os.path.exists(t_path):
        os.makedirs(t_path)
        print('make folder %s' % t_path)
    else:
        print('exist folder %s' % t_path)

    if random_order:
        tidx = np.arange(n_comm)
        np.random.shuffle(tidx)
        comm_list = comm_list[tidx]

    # generate comm_list.sh
    t_name0 = os.path.join(t_path, 'comm_list.sh')
    with open(t_name0, 'w') as fcomm:
        for i0, ts in enumerate(comm_list):
            fcomm.write('%s \n' % ts)
            fcomm.write('echo \'%d / %d start.\'  \n\n' % (i0 + 1, n_comm))

    # generate .pbs file
    t_name = os.path.join(t_path, pbs_name)
    with open(t_name, 'w') as fpbs:
        fpbs.write('#!/bin/sh\n')
        fpbs.write('# run the job locally. ')
        fpbs.write('\n')
        t2 = 'seq 0 %d | parallel -j %d -u ' % (n_comm - 1, njob_node)
        t2 = t2 + ' --sshdelay 0.1 '
        t2 = t2 + ' "cd $PWD; echo $PWD; echo; bash %s {} true " \n\n ' % csh_name
        fpbs.write(t2)

    # generate .csh file for submit
    t_name = os.path.join(t_path, csh_name)
    with open(t_name, 'w') as fcsh:
        fcsh.write('#!/bin/csh -fe \n\n')
        t2 = 'comm_list=('
        for t3 in comm_list:
            t2 = t2 + '"%s" ' % t3
        t2 = t2 + ') \n\n'
        fcsh.write(t2)
        fcsh.write('echo ${comm_list[$1]} \n')
        fcsh.write('echo $(expr $1 + 1) / %d start.  \n' % n_comm)
        fcsh.write('echo \n')
        fcsh.write('if [ ${2:-false} = true ]; then \n')
        fcsh.write('    ${comm_list[$1]} \n')
        fcsh.write('fi \n\n')
    print('Input %d cases. ' % n_comm)
    print('Random order mode is %s. ' % random_order)
    print('Command of first case is:')
    print(comm_list[0])
    return True


def profile(filename=None, comm=MPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.rank)
                pr.dump_stats(filename_r)
            return result

        return wrap_f

    return prof_decorator


def read_array(text_headle, FILE_DATA, array_length=6):
    t_match = re.search(text_headle, FILE_DATA)
    if t_match is not None:
        t1 = t_match.end()
        myformat = ('%f ' * array_length)[:-1]
        temp1 = np.array(scanf(myformat, FILE_DATA[t1:]))
    else:
        temp1 = np.ones(array_length)
        temp1[:] = np.nan
    return temp1


def mpiprint(*args, **kwargs):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        print(*args, **kwargs)


def petscInfo(logger, msg):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        logger.info(msg)


def check_file_extension(filename, extension):
    if filename[-len(extension):] != extension:
        filename = filename + extension
    return filename
