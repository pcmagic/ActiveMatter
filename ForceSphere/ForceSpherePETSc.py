# import matplotlib
# matplotlib.use('Agg', force=True)
import sys

import numpy as np
import petsc4py

petsc4py.init(sys.argv)

# from scipy.io import savemat, loadmat
# from src.ref_solution import *
# import warnings
# from memory_profiler import profile
# from time import time
from src.myio import *
from src.geo import *
from src.stokes_flow import *
from src.StokesFlowMethod import *
from act_src import problemClass, relationClass, particleClass, interactionClass
from act_codeStore import support_fun as spf


# initial all parameters
def get_prb_loop_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    
    ini_t = np.float64(OptDB.getReal("ini_t", 0))
    ign_t = np.float64(OptDB.getReal("ign_t", ini_t))
    max_t = np.float64(OptDB.getReal("max_t", 10))
    update_fun = OptDB.getString("update_fun", "1fe")
    rtol = np.float64(OptDB.getReal("rtol", 1e-2))
    atol = np.float64(OptDB.getReal("atol", rtol * 1e-3))
    eval_dt = np.float64(OptDB.getReal("eval_dt", 1))
    calculate_fun = OptDB.getString("calculate_fun", "do_behaviorParticle2D")
    fileHandle = OptDB.getString("f", "dbg")
    save_every = np.float64(OptDB.getReal("save_every", 1))
    
    rot_noise = np.float64(OptDB.getReal("rot_noise", 0))
    trs_noise = np.float64(OptDB.getReal("trs_noise", 0))
    seed0 = OptDB.getInt("seed0", -1)
    
    err_msg = "wrong parameter eval_dt, eval_dt>0. "
    assert eval_dt > 0, err_msg
    seed = seed0 if seed0 >= 0 else None
    np.random.seed(seed)
    
    problem_kwargs = {
        "ini_t":         ini_t,
        "ign_t":         ign_t,
        "max_t":         max_t,
        "update_fun":    update_fun,
        "update_order":  (rtol, atol),
        "eval_dt":       eval_dt,
        "calculate_fun": calculate_fun,
        "fileHandle":    fileHandle,
        "save_every":    save_every,
        "rot_noise":     rot_noise,
        "trs_noise":     trs_noise,
        "seed":          seed,
        "tqdm_fun":      tqdm,
        }
    
    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


def get_problem_kwargs(**main_kwargs):
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    #
    kT1 = OptDB.getReal('kT1', 4.141947e-06)
    # kT1 = 4.1419470e-6;
    # kT2 = 5.5226020e-6;
    # kT = 300*1.380649e-8;
    # mu = 1.0e-6  # Fluid viscosity(血浆的粘度)(g/um/s)
    mu = OptDB.getReal('mu', 1.0e-6)  # Fluid viscosity(血浆的粘度)(g/um/s)
    radius = OptDB.getReal('radius', 1)  # 所有球体的半径
    # 数据处理参数
    # rs2 = 6.0  # todo: varify, rs2 \in (6, 12, and other values).
    rs2 = OptDB.getReal('rs2', 12)  # todo: varify, rs2 \in (6, 12, and other values).
    sdis = OptDB.getReal('sdis', 1.0e-4)  # 最小表面间距
    For = OptDB.getReal('For', 0.1)  # 给定活性粒子的推进力
    Tor = OptDB.getReal('Tor', 0)  # 给定活性粒子的力矩
    
    # 生成随机球体
    length = OptDB.getReal('length', 10)  # 图像长   改
    width = OptDB.getReal('width', 10)  # 图像宽   改
    density = OptDB.getReal('density', 0.3)  # 密排度（区间：0-1，图像上的散斑密度）  改
    variation = OptDB.getReal('variation', 0.3)  # 偏移度（区间：0-1，图像上的散斑随机排布程度，0时即为圆点整列）
    
    # rng = np.random.seed(5)  # 生成一组随机数种子，使之后Section中产生的散斑伪随机, 这里将种子设置为0.可以调整
    diag_err = OptDB.getReal('diag_err', 1e-16)  # Avoiding errors introduced by nan values. (避免nan值引入的误差)
    
    # problem_kwargs['kT1'] = kT1
    problem_kwargs['mu'] = mu
    problem_kwargs['radius'] = radius
    problem_kwargs['rs2'] = rs2
    problem_kwargs['sdis'] = sdis
    problem_kwargs['For'] = For
    problem_kwargs['Tor'] = Tor
    #
    problem_kwargs['length'] = length
    problem_kwargs['width'] = width
    problem_kwargs['density'] = density
    problem_kwargs['variation'] = variation
    problem_kwargs['diag_err'] = diag_err
    problem_kwargs['matrix_method'] = 'forceSphere2d'
    # todo....
    
    kwargs_list = (get_prb_loop_kwargs(), main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


# todo ...
def print_ForceSphere_info(**problem_kwargs):
    print(111)
    return True


def print_case_info(**problem_kwargs):
    fileHandle = problem_kwargs['fileHandle']
    print_solver_info(**problem_kwargs)
    return True


def partical_generator_random(**problem_kwargs):
    np.random.seed(1)
    NS = 3
    dof = 2
    sphere_R = np.random.sample(NS)
    sphere_X = np.random.sample((NS, dof))
    
    sphere_phi = np.zeros((NS, 3))  # z 方向（球心处）
    # todo: modify to speedup
    for i in range(NS):
        ax = np.random.rand(1)
        sphere_phi[i, :] = np.hstack([np.cos(2 * ax * np.pi), np.sin(2 * ax * np.pi), 0])
    return sphere_R, sphere_X, sphere_phi


def partical_generator_dbg(**problem_kwargs):
    # test code from LBP, A000-Test_RK4 case (Liu Baopi version).
    sphere_X = np.array(((2.7, 2.9),
                         (0.1, 2.1),
                         (0.1, 0.1),
                         (1.9, 0.1)))
    sphere_R = np.ones(sphere_X.shape[0])
    
    sphere_phi = np.zeros_like(sphere_R)
    return sphere_R, sphere_X, sphere_phi


def partical_generator_LBP(**problem_kwargs):
    # 生成随机球体
    radius = problem_kwargs['radius']  # 所有球体的半径
    length = problem_kwargs['length']  # 图像长   改
    width = problem_kwargs['width']  # 图像宽   改
    density = problem_kwargs['density']  # 密排度（区间：0-1，图像上的散斑密度）  改
    variation = problem_kwargs['variation']  # 偏移度（区间：0-1，图像上的散斑随机排布程度，0时即为圆点整列）
    
    diameter = 2 * radius  # 散斑直径（单位：像素）
    
    # 生成随机散斑在图像上的位置
    spacing = diameter / density ** (1 / 2)  # 2D散斑个数
    colu = int(length // spacing)  # x轴
    cols = int(width // spacing)  # y轴
    xmin = 0.5 * (length - colu * spacing)  # x轴散散斑边界位置
    ymin = 0.5 * (width - cols * spacing)  # y轴散斑边界位置
    x = np.tile((xmin + (np.arange(colu) + 1) * spacing), (cols, 1)).T
    y = np.tile((ymin + (np.arange(cols) + 1) * spacing), (colu, 1))
    # 增加随机移动量
    limit = 0.5 * variation * spacing
    x = x + limit * (np.random.random((colu, cols)) - 2)
    y = y + limit * (np.random.random((colu, cols)) - 2)
    NS = x.size  # 总的粒子数
    #
    sphere_X = np.array([x.ravel(), y.ravel()]).T
    sphere_R = radius * np.ones(NS)  # spherical radius
    sphere_phi = spf.warpToPi(2 * np.random.rand(NS) * np.pi)  # z 方向（球心处）
    return sphere_R, sphere_X, sphere_phi


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    problem_kwargs['matrix_method'] = 'forceSphere2d'
    
    # test damo
    # sphere_R, sphere_X, D3 = partical_generator(**problem_kwargs)
    # sphere_R, sphere_X, D3 = partical_generator_random(**problem_kwargs)
    sphere_R, sphere_X, sphere_phi = partical_generator_dbg(**problem_kwargs)
    dof = sphere_X.shape[1]
    
    sphere_geo0 = sphere_particle_2d()
    sphere_geo0.set_dof(dof)
    sphere_geo0.set_nodes(sphere_X, -1)
    sphere_geo0.set_sphere_R(sphere_R)
    sphere_geo0.set_velocity(np.zeros_like(sphere_X).flatten())
    sphere_geo0.set_phi(sphere_phi)
    #
    sphere_obj0 = ForceSphereObj()
    sphere_obj0.set_data(f_geo=sphere_geo0, u_geo=sphere_geo0, name=fileHandle)
    #
    prb_MR = sf.ForceSphere2DProblem(**problem_kwargs)
    prb_MR.add_obj(sphere_obj0)
    prb_MR.initial_lub()
    prb_MR.create_matrix()
    prb_MR.solve_resistance()
    
    # loop
    prb_loop = problemClass.ForceSphere2DProblem(name=fileHandle, **problem_kwargs)
    spf.petscInfo(prb_loop.logger, "#" * 72)
    spf.petscInfo(prb_loop.logger, "Generate Problem. ")
    # self._un = np.zeros_like(...)
    # self._ln = np.zeros_like(...)
    
    prb_loop.update_fun = problem_kwargs["update_fun"]
    prb_loop.update_order = problem_kwargs["update_order"]
    prb_loop.save_every = problem_kwargs["save_every"]
    prb_loop.tqdm_fun = problem_kwargs["tqdm_fun"]
    prb_loop.do_save = True
    
    prb_loop.relationHandle = relationClass.singleRelation2D()
    sphere_ptc = particleClass.ForceSphere2D(name="ForceSphere2D")
    sphere_ptc.X = sphere_geo0.get_nodes()
    sphere_ptc.phi = sphere_phi
    sphere_ptc.u = np.zeros_like(sphere_geo0.get_nodes())
    sphere_ptc.U = np.zeros_like(sphere_geo0.get_nodes())
    sphere_ptc.w = np.zeros(sphere_geo0.get_n_nodes())
    sphere_ptc.W = np.zeros(sphere_geo0.get_n_nodes())
    sphere_ptc.prb_MR = prb_MR
    prb_loop.add_obj(sphere_ptc)
    # spf.petscInfo(sphere_ptc.logger, "  All the particles have a unified %s=%f, " % ("spin", 0), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate %d particles with random seed %s" % (self.un.size, self.seed), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate method: random_sample. ")
    
    act1 = interactionClass.ForceSphere2D(name="ForceSphere2D")
    prb_loop.add_act(act1)
    ini_t, max_t, eval_dt = 0, 100, 1
    prb_loop.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    
    return True


def main_fun_v2(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    problem_kwargs['matrix_method'] = 'forceSphere2d_simp'
    problem_kwargs['length'] = 3
    problem_kwargs['width'] = 3
    problem_kwargs['sdis'] = 1.0e-4  # 最小表面间距
    problem_kwargs['update_fun'] = '4'
    problem_kwargs['update_order'] = (0, 0)
    
    # test damo
    # sphere_R, sphere_X, D3 = partical_generator(**problem_kwargs)
    # sphere_R, sphere_X, D3 = partical_generator_random(**problem_kwargs)
    sphere_R, sphere_X, sphere_phi = partical_generator_dbg(**problem_kwargs)
    dof = sphere_X.shape[1]
    
    sphere_geo0 = sphere_particle_2d()
    sphere_geo0.set_dof(dof)
    sphere_geo0.set_nodes(sphere_X, -1)
    sphere_geo0.set_sphere_R(sphere_R)
    sphere_geo0.set_velocity(np.zeros_like(sphere_X).flatten())
    sphere_geo0.set_phi(sphere_phi)
    #
    sphere_obj0 = ForceSphereObj()
    sphere_obj0.set_data(f_geo=sphere_geo0, u_geo=sphere_geo0, name=fileHandle)
    #
    prb_MR = sf.ForceSphere2DProblem(**problem_kwargs)
    prb_MR.add_obj(sphere_obj0)
    prb_MR.create_matrix()
    prb_MR.solve_resistance()
    PETSc.Sys.Print(' ')
    # loop
    prb_loop = problemClass.ForceSphere2DProblem(name=fileHandle, **problem_kwargs)
    spf.petscInfo(prb_loop.logger, "#" * 72)
    # spf.petscInfo(prb_loop.logger, "Generate Problem. ")
    
    prb_loop.update_fun = problem_kwargs["update_fun"]
    prb_loop.update_order = problem_kwargs["update_order"]
    prb_loop.save_every = problem_kwargs["save_every"]
    prb_loop.tqdm_fun = problem_kwargs["tqdm_fun"]
    prb_loop.do_save = True
    
    prb_loop.relationHandle = relationClass.nothingRelation2D()
    sphere_ptc = particleClass.ForceSphere2D(name="ForceSphere2D")
    sphere_ptc.X = sphere_geo0.get_nodes()
    sphere_ptc.phi = sphere_phi
    sphere_ptc.u = np.zeros_like(sphere_geo0.get_nodes())
    sphere_ptc.U = np.zeros_like(sphere_geo0.get_nodes())
    sphere_ptc.w = np.zeros(sphere_geo0.get_n_nodes())
    sphere_ptc.W = np.zeros(sphere_geo0.get_n_nodes())
    sphere_ptc.prb_MR = prb_MR
    prb_loop.add_obj(sphere_ptc)
    # spf.petscInfo(sphere_ptc.logger, "  All the particles have a unified %s=%f, " % ("spin", 0), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate %d particles with random seed %s" % (self.un.size, self.seed), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate method: random_sample. ")
    
    act1 = interactionClass.ForceSphere2D(name="ForceSphere2D")
    prb_loop.add_act(act1)
    ini_t, max_t, eval_dt = 0, 10, 1
    spf.petscInfo(prb_loop.logger, "Generate Problem finish. ")
    prb_loop.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    
    return True


def main_fun_v3(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    problem_kwargs['matrix_method'] = 'forceSphere2d_simp'
    problem_kwargs['length'] = 3
    problem_kwargs['width'] = 3
    problem_kwargs['sdis'] = 1.0e-4  # 最小表面间距
    problem_kwargs['update_fun'] = '4'
    problem_kwargs['update_order'] = (0, 0)
    
    # test damo
    # sphere_R, sphere_X, D3 = partical_generator(**problem_kwargs)
    # sphere_R, sphere_X, D3 = partical_generator_random(**problem_kwargs)
    sphere_R, sphere_X, sphere_phi = partical_generator_dbg(**problem_kwargs)
    dof = sphere_X.shape[1]
    
    sphere_geo0 = sphere_particle_2d()
    sphere_geo0.set_dof(dof)
    sphere_geo0.set_nodes(sphere_X, -1)
    sphere_geo0.set_sphere_R(sphere_R)
    sphere_geo0.set_velocity(np.zeros_like(sphere_X).flatten())
    sphere_geo0.set_phi(sphere_phi)
    #
    sphere_obj0 = ForceSphereObj()
    sphere_obj0.set_data(f_geo=sphere_geo0, u_geo=sphere_geo0, name=fileHandle)
    #
    prb_MR = sf.ForceSphere2DProblem(**problem_kwargs)
    prb_MR.add_obj(sphere_obj0)
    prb_MR.create_matrix()
    prb_MR.solve_resistance()
    PETSc.Sys.Print(' ')
    
    # loop
    prb_loop = problemClass.singleForceSphere2DProblem(name=fileHandle, **problem_kwargs)
    spf.petscInfo(prb_loop.logger, "#" * 72)
    # spf.petscInfo(prb_loop.logger, "Generate Problem. ")
    
    prb_loop.update_fun = problem_kwargs["update_fun"]
    prb_loop.update_order = problem_kwargs["update_order"]
    prb_loop.save_every = problem_kwargs["save_every"]
    prb_loop.tqdm_fun = problem_kwargs["tqdm_fun"]
    prb_loop.do_save = True
    
    prb_loop.relationHandle = relationClass.AllBaseRelation2D()
    for tX, tphi in zip(sphere_X, sphere_phi):
        sphere_ptc = particleClass.singleForceSphere2D(name="ForceSphere2D")
        sphere_ptc.X = tX
        sphere_ptc.phi = tphi
        sphere_ptc.u = np.zeros(1)
        sphere_ptc.U = np.zeros(2)
        sphere_ptc.w = np.zeros(1)
        sphere_ptc.W = np.zeros(1)
        sphere_ptc.prb_MR = prb_MR
        prb_loop.add_obj(sphere_ptc)
    # spf.petscInfo(sphere_ptc.logger, "  All the particles have a unified %s=%f, " % ("spin", 0), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate %d particles with random seed %s" % (self.un.size, self.seed), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate method: random_sample. ")
    
    act1 = interactionClass.ForceSphere2D(name="ForceSphere2D")
    prb_loop.add_act(act1)
    ini_t, max_t, eval_dt = 0, 10, 1
    spf.petscInfo(prb_loop.logger, "Generate Problem finish. ")
    prb_loop.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    
    return True


def main_fun_v4(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    # problem_kwargs['matrix_method'] = 'forceSphere2d_simp'
    # problem_kwargs['length'] = 100
    # problem_kwargs['width'] = 100
    # problem_kwargs['update_fun'] = '4'
    # problem_kwargs['update_order'] = (0, 0)
    # problem_kwargs['update_fun'] = '1fe'
    # problem_kwargs['update_order'] = (0, 0)
    # problem_kwargs['update_fun'] = '5bs'
    # problem_kwargs['update_order'] = (1e-9, 1e-12)
    ini_t = problem_kwargs['ini_t']
    max_t = problem_kwargs['max_t']
    eval_dt = problem_kwargs['eval_dt']
    
    # test damo
    # sphere_R, sphere_X, D3 = partical_generator(**problem_kwargs)
    # sphere_R, sphere_X, D3 = partical_generator_random(**problem_kwargs)
    # sphere_R, sphere_X, sphere_phi = partical_generator_dbg(**problem_kwargs)
    sphere_R, sphere_X, sphere_phi = partical_generator_LBP(**problem_kwargs)
    n_sphere, dof = sphere_X.shape
    
    # loop
    prb_loop = problemClass.ForceSphere2D_matrixPro(name=fileHandle, **problem_kwargs)
    spf.petscInfo(prb_loop.logger, "#" * 72)
    # spf.petscInfo(prb_loop.logger, "Generate Problem. ")
    
    prb_loop.update_fun = problem_kwargs["update_fun"]
    prb_loop.update_order = problem_kwargs["update_order"]
    prb_loop.save_every = problem_kwargs["save_every"]
    prb_loop.tqdm_fun = problem_kwargs["tqdm_fun"]
    prb_loop.do_save = True
    
    prb_loop.relationHandle = relationClass.nothingRelation2D(name=fileHandle)
    sphere_ptc = particleClass.ForceSphere2D_matrixObj(name=fileHandle)
    sphere_ptc.X = sphere_X
    sphere_ptc.phi = sphere_phi
    sphere_ptc.r = sphere_R
    sphere_ptc.u = np.zeros(n_sphere)
    sphere_ptc.U = np.zeros_like(sphere_X)
    sphere_ptc.w = np.zeros(n_sphere)
    sphere_ptc.W = np.zeros(n_sphere)
    # sphere_ptc.prb_MR = prb_MR
    prb_loop.add_obj(sphere_ptc)
    # spf.petscInfo(sphere_ptc.logger, "  All the particles have a unified %s=%f, " % ("spin", 0), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate %d particles with random seed %s" % (self.un.size, self.seed), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate method: random_sample. ")
    
    act1 = interactionClass.ForceSphere2D_matrixAct(name=fileHandle)
    prb_loop.add_act(act1)
    spf.petscInfo(prb_loop.logger, "Generate Problem finish. ")
    prb_loop.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    
    return True

# without matmult precondition
def main_fun_v5(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    # problem_kwargs['matrix_method'] = 'forceSphere2d_simp'
    problem_kwargs['length'] = 100
    problem_kwargs['width'] = 100
    # problem_kwargs['update_fun'] = '4'
    # problem_kwargs['update_order'] = (0, 0)
    # problem_kwargs['update_fun'] = '1fe'
    # problem_kwargs['update_order'] = (0, 0)
    # problem_kwargs['update_fun'] = '5bs'
    # problem_kwargs['update_order'] = (1e-9, 1e-12)
    ini_t = problem_kwargs['ini_t']
    max_t = problem_kwargs['max_t']
    eval_dt = problem_kwargs['eval_dt']
    
    # test damo
    # sphere_R, sphere_X, D3 = partical_generator(**problem_kwargs)
    # sphere_R, sphere_X, D3 = partical_generator_random(**problem_kwargs)
    # sphere_R, sphere_X, sphere_phi = partical_generator_dbg(**problem_kwargs)
    sphere_R, sphere_X, sphere_phi = partical_generator_LBP(**problem_kwargs)
    n_sphere, dof = sphere_X.shape
    
    # loop
    prb_loop = problemClass.ForceSphere2D_matrixPro(name=fileHandle, **problem_kwargs)
    spf.petscInfo(prb_loop.logger, "#" * 72)
    # spf.petscInfo(prb_loop.logger, "Generate Problem. ")
    
    prb_loop.update_fun = problem_kwargs["update_fun"]
    prb_loop.update_order = problem_kwargs["update_order"]
    prb_loop.save_every = problem_kwargs["save_every"]
    prb_loop.tqdm_fun = problem_kwargs["tqdm_fun"]
    prb_loop.do_save = True
    
    prb_loop.relationHandle = relationClass.nothingRelation2D(name=fileHandle)
    sphere_ptc = particleClass.ForceSphere2D_matrixObj(name=fileHandle)
    sphere_ptc.X = sphere_X
    sphere_ptc.phi = sphere_phi
    sphere_ptc.r = sphere_R
    sphere_ptc.u = np.zeros(n_sphere)
    sphere_ptc.U = np.zeros_like(sphere_X)
    sphere_ptc.w = np.zeros(n_sphere)
    sphere_ptc.W = np.zeros(n_sphere)
    # sphere_ptc.prb_MR = prb_MR
    prb_loop.add_obj(sphere_ptc)
    # spf.petscInfo(sphere_ptc.logger, "  All the particles have a unified %s=%f, " % ("spin", 0), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate %d particles with random seed %s" % (self.un.size, self.seed), )
    # spf.petscInfo(sphere_ptc.logger, "  Generate method: random_sample. ")
    
    act1 = interactionClass.ForceSphere2D_matrixAct(name=fileHandle)
    prb_loop.add_act(act1)
    spf.petscInfo(prb_loop.logger, "Generate Problem finish. ")
    prb_loop.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    
    return True


if __name__ == '__main__':
    main_fun_v4()
