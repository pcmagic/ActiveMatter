import sys
import petsc4py

# import matplotlib
# matplotlib.use('agg')
petsc4py.init(sys.argv)

import numpy as np
# import pickle
# from time import time
from petsc4py import PETSc
# from datetime import datetime
from tqdm import tqdm

from act_src import problemClass
from act_src import relationClass
from act_src import particleClass
# from act_src import interactionClass
from act_codeStore import support_fun_calculate as spc


# get kwargs
def get_problem_kwargs(**main_kwargs):
    calculate_fun_dict = {
        'do_FiniteDipole2D':       spc.do_FiniteDipole2D,
        'do_LimFiniteDipole2D':    spc.do_LimFiniteDipole2D,
        'do_actLimFiniteDipole2D': spc.do_actLimFiniteDipole2D,
    }
    prbHandle_dict = {
        'do_FiniteDipole2D':       problemClass.finiteDipole2DProblem,
        'do_LimFiniteDipole2D':    problemClass.limFiniteDipole2DProblem,
        'do_actLimFiniteDipole2D': problemClass.actLimFiniteDipole2DProblem,
    }
    rltHandle_dict = {
        'do_FiniteDipole2D':       relationClass.finiteRelation2D,
        'do_LimFiniteDipole2D':    relationClass.limFiniteRelation2D,
        'do_actLimFiniteDipole2D': relationClass.VoronoiRelation2D,
    }
    ptcHandle_dict = {
        'do_FiniteDipole2D':       particleClass.finiteDipole2D,
        'do_LimFiniteDipole2D':    particleClass.limFiniteDipole2D,
        'do_actLimFiniteDipole2D': particleClass.limFiniteDipole2D,
    }

    OptDB = PETSc.Options()

    ini_t = OptDB.getReal('ini_t', 0)
    max_t = OptDB.getReal('max_t', 1)
    update_fun = OptDB.getString('update_fun', '5bs')
    rtol = OptDB.getReal('rtol', 1e-3)
    atol = OptDB.getReal('atol', 1e-6)
    eval_dt = OptDB.getReal('eval_dt', 0.01)
    calculate_fun = OptDB.getString('calculate_fun', 'do_FiniteDipole2D')
    fileHandle = OptDB.getString('f', '')
    save_every = OptDB.getReal('save_every', 1)

    nptc = OptDB.getInt('nptc', 5)
    overlap_epsilon = OptDB.getReal('overlap_epsilon', 0)
    un = OptDB.getReal('un', 1)
    ln = OptDB.getReal('ln', 1)
    Xlim = OptDB.getReal('Xlim', 3)
    attract = OptDB.getReal('attract', 0)
    align = OptDB.getReal('align', 0)
    seed0 = OptDB.getBool('seed0', False)

    err_msg = 'wrong parameter nptc, at least 5 particles (nptc > 4).  '
    assert nptc > 4, err_msg
    seed = 0 if seed0 else None
    np.random.seed(seed0)

    problem_kwargs = {
        'ini_t':           ini_t,
        'max_t':           max_t,
        'update_fun':      update_fun,
        'update_order':    (rtol, atol),
        'eval_dt':         eval_dt,
        'calculate_fun':   calculate_fun_dict[calculate_fun],
        'prbHandle':       prbHandle_dict[calculate_fun],
        'rltHandle':       rltHandle_dict[calculate_fun],
        'ptcHandle':       ptcHandle_dict[calculate_fun],
        'fileHandle':      fileHandle,
        'save_every':      save_every,
        'nptc':            nptc,
        'overlap_epsilon': overlap_epsilon,
        'un':              un,
        'ln':              ln,
        'Xlim':            Xlim,
        'attract':         attract,
        'align':           align,
        'seed':            seed,
        'tqdm_fun':        tqdm,
    }

    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    max_t = problem_kwargs['max_t']
    ini_t = problem_kwargs['ini_t']
    eval_dt = problem_kwargs['eval_dt']

    PETSc.Sys.Print(problem_kwargs)
    doPrb1 = problem_kwargs['calculate_fun'](**problem_kwargs)
    prb1 = doPrb1.do_calculate(ini_t=ini_t, max_t=max_t, eval_dt=eval_dt, )
    # for obji in prb1.obj_list:
    #     PETSc.Sys.Print('dbg, obji.X', obji.X)


if __name__ == '__main__':
    OptDB = PETSc.Options()

    if OptDB.getBool('main_fun', True):
        main_fun()
