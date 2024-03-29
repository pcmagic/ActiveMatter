import sys

# import numpy as np
import petsc4py

# import matplotlib
# matplotlib.use('agg')
petsc4py.init(sys.argv)

# import numpy as np
# import pickle
# from time import time
from petsc4py import PETSc

# from datetime import datetime
from tqdm import tqdm
# import shutil
# import os
# from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

from act_src import problemClass
from act_src import relationClass
from act_src import particleClass

# from act_src import interactionClass
from act_codeStore.support_fun import *
from act_codeStore import support_fun_calculate as spc
from act_codeStore import support_fun_show as sps
from act_codeStore.support_fun_do_calculate import *

calculate_fun_dict = {
    "do_FiniteDipole2D":              spc.do_FiniteDipole2D,
    "do_LimFiniteDipole2D":           spc.do_LimFiniteDipole2D,
    "do_behaviorParticle2D":          spc.do_behaviorParticle2D,
    "dbg_behaviorParticle2D":         spc.dbg_behaviorParticle2D,
    "do_behaviorWienerParticle2D":    spc.do_behaviorWienerParticle2D,
    "do_dbgBokaiZhang":               spc.do_dbgBokaiZhang,
    "do_actLimFiniteDipole2D":        spc.do_actLimFiniteDipole2D,
    "do_actLight2D":                  spc.do_actLight2D,
    "do_phaseLag2D":                  spc.do_phaseLag2D,
    "do_phaseLag2D_dbg":              spc.do_phaseLag2D_dbg,
    "do_phaseLagWiener2D":            spc.do_phaseLagWiener2D,
    "do_phaseLagPeriodic2D":          spc.do_phaseLagPeriodic2D,
    "do_phaseLagPeriodic2D_LJ":       spc.do_phaseLagPeriodic2D_LJ,
    "do_phaseLag2D_AR":               spc.do_phaseLag2D_AR,
    "do_phaseLag2D_AR_Voronoi":       spc.do_phaseLag2D_AR,
    "do_phaseLagWiener2D_AR":         spc.do_phaseLagWiener2D_AR,
    "do_phaseLagWiener2D_AR_Voronoi": spc.do_phaseLagWiener2D_AR,
    "do_phaseLag2D_Wiener":           spc.do_phaseLag2D_Wiener,
    "do_dbg_action":                  spc.do_dbg_action,
    "do_ackermann_goal":              spc.do_ackermann_goal,
    "do_ackermann_alignAttract":      spc.do_ackermann_alignattract,
    "do_ackermann_phaseLag2D":        spc.do_ackermann_phaseLag2D,
    "do_ackermann_smallSigma":        spc.do_ackermann_smallSigma,
    }

prbHandle_dict = {
    "do_FiniteDipole2D":              problemClass.finiteDipole2DProblem,
    "do_LimFiniteDipole2D":           problemClass.limFiniteDipole2DProblem,
    "do_behaviorParticle2D":          problemClass.behavior2DProblem,
    "dbg_behaviorParticle2D":         problemClass.behavior2DProblem,
    "do_behaviorWienerParticle2D":    problemClass.behavior2DProblem,
    "do_dbgBokaiZhang":               problemClass.behavior2DProblem,
    "do_actLimFiniteDipole2D":        problemClass.actLimFiniteDipole2DProblem,
    "do_actLight2D":                  problemClass.actLimFiniteDipole2DProblem,
    "do_phaseLag2D":                  problemClass.behavior2DProblem,
    "do_phaseLag2D_dbg":              problemClass.behavior2DProblem,
    "do_phaseLagWiener2D":            problemClass.behavior2DProblem,
    "do_phaseLagPeriodic2D":          problemClass.actPeriodic2DProblem,
    "do_phaseLagPeriodic2D_LJ":       problemClass.actPeriodic2DProblem,
    "do_phaseLag2D_AR":               problemClass.behavior2DProblem,
    "do_phaseLag2D_AR_Voronoi":       problemClass.behavior2DProblem,
    "do_phaseLagWiener2D_AR":         problemClass.behavior2DProblem,
    "do_phaseLagWiener2D_AR_Voronoi": problemClass.behavior2DProblem,
    "do_phaseLag2D_Wiener":           problemClass.behavior2DProblem,
    "do_dbg_action":                  problemClass.finiteDipole2DProblem,
    "do_ackermann_goal":              problemClass.Ackermann2DProblem_goal,
    "do_ackermann_alignAttract":      problemClass.Ackermann2DProblem,
    "do_ackermann_phaseLag2D":        problemClass.Ackermann2DProblem,
    "do_ackermann_smallSigma":        problemClass.Ackermann2DProblem,
    }

rltHandle_dict = {
    "do_FiniteDipole2D":              relationClass.finiteRelation2D,
    "do_LimFiniteDipole2D":           relationClass.limFiniteRelation2D,
    "do_behaviorParticle2D":          relationClass.VoronoiBaseRelation2D,
    "dbg_behaviorParticle2D":         relationClass.AllBaseRelation2D,
    "do_behaviorWienerParticle2D":    relationClass.VoronoiBaseRelation2D,
    "do_dbgBokaiZhang":               relationClass.VoronoiBaseRelation2D,
    "do_actLimFiniteDipole2D":        relationClass.VoronoiBaseRelation2D,
    "do_actLight2D":                  relationClass.AllBaseRelation2D,
    "do_phaseLag2D":                  relationClass.localBaseRelation2D,  # 'do_phaseLag2D':                  relationClass.AllBaseRelation2D,
    "do_phaseLag2D_dbg":              relationClass.localBaseRelation2D,
    "do_phaseLagWiener2D":            relationClass.localBaseRelation2D,
    "do_phaseLagPeriodic2D":          relationClass.periodicLocalRelation2D,
    "do_phaseLagPeriodic2D_LJ":       relationClass.periodicLocalRelation2D,
    "do_phaseLag2D_AR":               relationClass.localBaseRelation2D,
    "do_phaseLag2D_AR_Voronoi":       relationClass.VoronoiBaseRelation2D,
    "do_phaseLagWiener2D_AR":         relationClass.localBaseRelation2D,
    "do_phaseLagWiener2D_AR_Voronoi": relationClass.VoronoiBaseRelation2D,
    "do_phaseLag2D_Wiener":           relationClass.localBaseRelation2D,
    "do_dbg_action":                  relationClass.finiteRelation2D,
    "do_ackermann_goal":              relationClass.AllBaseRelation2D,
    "do_ackermann_alignAttract":      relationClass.AllBaseRelation2D,
    "do_ackermann_phaseLag2D":        relationClass.AllBaseRelation2D,
    "do_ackermann_smallSigma":        relationClass.AllBaseRelation2D,
    }

ptcHandle_dict = {
    "do_FiniteDipole2D":              particleClass.finiteDipole2D,
    "do_LimFiniteDipole2D":           particleClass.limFiniteDipole2D,
    "do_behaviorParticle2D":          particleClass.particle2D,
    "dbg_behaviorParticle2D":         particleClass.particle2D,
    "do_behaviorWienerParticle2D":    particleClass.particle2D,
    "do_dbgBokaiZhang":               particleClass.particle2D,
    "do_actLimFiniteDipole2D":        particleClass.limFiniteDipole2D,
    "do_actLight2D":                  particleClass.limFiniteDipole2D,
    "do_phaseLag2D":                  particleClass.particle2D,
    "do_phaseLag2D_dbg":              particleClass.particle2D,
    "do_phaseLagWiener2D":            particleClass.particle2D,
    "do_phaseLagPeriodic2D":          particleClass.particle2D,
    "do_phaseLagPeriodic2D_LJ":       particleClass.particle2D,
    "do_phaseLag2D_AR":               particleClass.particle2D,
    "do_phaseLag2D_AR_Voronoi":       particleClass.particle2D,
    "do_phaseLagWiener2D_AR":         particleClass.particle2D,
    "do_phaseLagWiener2D_AR_Voronoi": particleClass.particle2D,
    "do_phaseLag2D_Wiener":           particleClass.particle2D,
    "do_dbg_action":                  particleClass.finiteDipole2D,
    "do_ackermann_goal":              particleClass.ackermann2D_goal,
    "do_ackermann_alignAttract":      particleClass.ackermann2D,
    "do_ackermann_phaseLag2D":        particleClass.ackermann2D,
    "do_ackermann_smallSigma":        particleClass.ackermann2D,
    }


# get kwargs
def get_problem_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    
    ini_t = np.float64(OptDB.getReal("ini_t", 0))
    ign_t = np.float64(OptDB.getReal("ign_t", ini_t))
    max_t = np.float64(OptDB.getReal("max_t", 1))
    update_fun = OptDB.getString("update_fun", "1fe")
    rtol = np.float64(OptDB.getReal("rtol", 1e-3))
    atol = np.float64(OptDB.getReal("atol", rtol * 1e-3))
    eval_dt = np.float64(OptDB.getReal("eval_dt", 0.01))
    calculate_fun = OptDB.getString("calculate_fun", "do_behaviorParticle2D")
    fileHandle = OptDB.getString("f", "dbg")
    save_every = np.float64(OptDB.getReal("save_every", 1))
    
    nptc = np.int64(OptDB.getInt("nptc", 5))
    overlap_epsilon = np.float64(OptDB.getReal("overlap_epsilon", 0))
    un = np.float64(OptDB.getReal("un", 1))
    ln = np.float64(OptDB.getReal("ln", 1))
    Xlim = np.float64(OptDB.getReal("Xlim", 1))
    attract = np.float64(OptDB.getReal("attract", 0))
    align = np.float64(OptDB.getReal("align", 0))
    viewRange = np.float64(OptDB.getReal("viewRange", 1)) * np.pi
    rot_noise = np.float64(OptDB.getReal("rot_noise", 0))
    trs_noise = np.float64(OptDB.getReal("trs_noise", 0))
    seed0 = OptDB.getInt("seed0", -1)
    
    # chimera
    phaseLag2D = np.float64(OptDB.getReal("phaseLag2D", 0)) * np.pi
    phaseLag_rdm_fct = np.float64(OptDB.getReal("phaseLag_rdm_fct", 0)) * np.pi
    localRange = np.float64(OptDB.getReal("localRange", 0))
    
    # Lennard Jone Potential
    LJ_A = np.float64(OptDB.getReal("LJ_UA", 0))
    LJ_B = np.float64(OptDB.getReal("LJ_UB", 0))
    LJ_a = np.float64(OptDB.getReal("LJ_la", 0))
    LJ_b = np.float64(OptDB.getReal("LJ_lb", 0))
    
    # periodic boundary condition
    Xrange = np.float64(OptDB.getReal("Xrange", Xlim))
    
    # ackermann parameters
    l_steer = np.float64(OptDB.getReal("l_steer", 1))
    w_steer = np.float64(OptDB.getReal("w_steer", 1))
    radian_tolerance = np.float64(OptDB.getReal("radian_tolerance", 0))
    
    err_msg = "wrong parameter eval_dt, eval_dt>0. "
    assert eval_dt > 0, err_msg
    # err_msg = 'wrong parameter nptc, at least 5 particles (nptc > 4).  '
    # assert nptc > 4, err_msg
    seed = seed0 if seed0 >= 0 else None
    np.random.seed(seed)
    
    problem_kwargs = {
        "ini_t":            ini_t,
        "ign_t":            ign_t,
        "max_t":            max_t,
        "update_fun":       update_fun,
        "update_order":     (rtol, atol),
        "eval_dt":          eval_dt,
        "calculate_fun":    calculate_fun_dict[calculate_fun],
        "prbHandle":        prbHandle_dict[calculate_fun],
        "rltHandle":        rltHandle_dict[calculate_fun],
        "ptcHandle":        ptcHandle_dict[calculate_fun],
        "fileHandle":       fileHandle,
        "save_every":       save_every,
        "nptc":             nptc,
        "overlap_epsilon":  overlap_epsilon,
        "un":               un,
        "ln":               ln,
        "Xlim":             Xlim,
        "Xrange":           Xrange,
        "attract":          attract,
        "align":            align,
        "viewRange":        viewRange,
        "phaseLag2D":       phaseLag2D,
        "localRange":       localRange,
        "LJ_A":             LJ_A,
        "LJ_B":             LJ_B,
        "LJ_a":             LJ_a,
        "LJ_b":             LJ_b,
        "rot_noise":        rot_noise,
        "trs_noise":        trs_noise,
        "l_steer":          l_steer,
        "w_steer":          w_steer,
        'radian_tolerance': radian_tolerance,
        "phaseLag_rdm_fct": phaseLag_rdm_fct,
        "seed":             seed,
        "tqdm_fun":         tqdm,
        }
    
    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    max_t = problem_kwargs["max_t"]
    ini_t = problem_kwargs["ini_t"]
    eval_dt = problem_kwargs["eval_dt"]
    # fileHandle = problem_kwargs['fileHandle']
    # PWD = os.getcwd()
    
    # spf.petscInfo(self.father.logger, problem_kwargs)
    doPrb1 = problem_kwargs["calculate_fun"](**problem_kwargs)
    prb1 = doPrb1.do_calculate(ini_t=ini_t, max_t=max_t, eval_dt=eval_dt, )  # type: problemClass._baseProblem
    # for obji in prb1.obj_list:
    #     if obji.rank0:
    #         print(obji.W_steer_hist)
    # prb1.dbg_t_hist(np.linspace(0, 1, 10 ** 8))
    # do_pickle(prb1, **problem_kwargs)
    do_hdf5(prb1, **problem_kwargs)
    prb1.hdf5_load()
    # import time
    # time.sleep(2)
    export_trajectory2D(prb1)
    export_avrPhaseVelocity(prb1, tavr=np.min((10, max_t / 10)))
    # export_20220629(prb1, tavr=np.min((10, max_t / 10)))
    export_20220629(prb1, tavr=eval_dt)
    export_ackermann(prb1, tavr=None)
    # export_avrPhaseVelocity(prb1, tavr=1)
    
    return True


def main_ignFirst(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    ini_t = problem_kwargs["ini_t"]
    ign_t = problem_kwargs["ign_t"]
    max_t = problem_kwargs["max_t"]
    eval_dt = problem_kwargs["eval_dt"]
    
    doPrb1 = problem_kwargs["calculate_fun"](**problem_kwargs)
    prb1 = doPrb1.ini_calculate()
    if ign_t > ini_t:
        prb1.do_save = False
        prb1.update_self(t0=ini_t, t1=ign_t, eval_dt=eval_dt, pick_prepare=False)
        prb1.do_save = True
    prb1.update_self(t0=ign_t, t1=max_t, eval_dt=eval_dt)
    do_hdf5(prb1, **problem_kwargs)
    prb1.hdf5_load()
    export_trajectory2D(prb1)
    export_avrPhaseVelocity(prb1, tavr=np.min((10, max_t / 10)))
    export_20220629(prb1, tavr=eval_dt)
    export_ackermann(prb1, tavr=None)
    return True


def main_dbg(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    ini_t = problem_kwargs["ini_t"]
    ign_t = problem_kwargs["ign_t"]
    max_t = problem_kwargs["max_t"]
    eval_dt = problem_kwargs["eval_dt"]
    
    doPrb1 = problem_kwargs["calculate_fun"](**problem_kwargs)
    prb1 = doPrb1.ini_calculate()
    if ign_t > ini_t:
        prb1.do_save = False
        prb1.update_self(t0=ini_t, t1=ign_t, eval_dt=eval_dt, pick_prepare=False)
        prb1.do_save = True
    prb1.update_self(t0=ign_t, t1=max_t, eval_dt=eval_dt)
    do_hdf5(prb1, **problem_kwargs)
    prb1.hdf5_load()
    export_trajectory2D(prb1)
    export_avrPhaseVelocity(prb1, tavr=np.min((10, max_t / 10)))
    export_20220629(prb1, tavr=eval_dt)
    export_ackermann(prb1, tavr=None)
    return True


if __name__ == "__main__":
    OptDB = PETSc.Options()
    if OptDB.getBool("main_profile", False):
        OptDB.setValue("main_fun", False)
        main_profile()
    
    if OptDB.getBool("main_ignFirst", False):
        OptDB.setValue("main_fun", False)
        main_ignFirst()
    
    if OptDB.getBool("main_dbg", False):
        OptDB.setValue("main_fun", False)
        main_dbg()
    
    if OptDB.getBool("main_fun", True):
        main_fun()
