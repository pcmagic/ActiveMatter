import numpy as np
from tqdm import tqdm

tqdm_notebook = tqdm  # for dbg
# from act_act_src import baseClass
from act_src import particleClass
from act_src import interactionClass
from act_src import problemClass
from act_src import relationClass
# from collectiveFish.do_calculate import calculate_fun_dict, prbHandle_dict, rltHandle_dict, ptcHandle_dict
from collectiveFish.do_calculate import *

update_fun, update_order, eval_dt = '4', (0, 0), 0.01
nptc, calculate_fun = 10, 'do_phaseLagPeriodic2D'
ini_t, max_t, Xlim = np.float64(0), eval_dt * 2, 1
seed = 1

problem_kwargs = {
    'ini_t':           ini_t,
    'max_t':           max_t,
    'update_fun':      update_fun,
    'update_order':    update_order,
    'eval_dt':         eval_dt,
    'calculate_fun':   calculate_fun_dict[calculate_fun],
    'prbHandle':       prbHandle_dict[calculate_fun],
    'rltHandle':       rltHandle_dict[calculate_fun],
    'ptcHandle':       ptcHandle_dict[calculate_fun],
    'fileHandle':      'try_phaseLag2D',
    'save_every':      np.int64(1),
    'nptc':            np.int64(nptc),
    'overlap_epsilon': np.float64(1e-100),
    'un':              np.float64(1),
    'ln':              np.float64(-1),
    'Xlim':            np.float64(Xlim),
    'Xrange':          np.float64(Xlim),
    'attract':         np.float64(0),
    'align':           np.float64(1),
    'viewRange':       np.float64(1),
    'localRange':      np.float64(0.3),
    'phaseLag2D':      np.float64(1.54 / np.pi),
    'seed':            seed,
    'tqdm_fun':        tqdm_notebook,
}

doPrb1 = problem_kwargs['calculate_fun'](**problem_kwargs)
prb1 = doPrb1.do_calculate(ini_t=ini_t, max_t=max_t, eval_dt=eval_dt, )
do_hdf5(prb1, **problem_kwargs)
prb1.hdf5_load(showInfo=False)

print(11111111111111111111)
