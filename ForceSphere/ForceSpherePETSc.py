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


def get_prb_loop_kwargs(**main_kwargs):
    OptDB = PETSc.Options()
    
    ini_t = np.float64(OptDB.getReal("ini_t", 0))
    ign_t = np.float64(OptDB.getReal("ign_t", ini_t))
    max_t = np.float64(OptDB.getReal("max_t", 1))
    update_fun = OptDB.getString("update_fun", "1fe")
    rtol = np.float64(OptDB.getReal("rtol", 1e-2))
    atol = np.float64(OptDB.getReal("atol", rtol * 1e-3))
    eval_dt = np.float64(OptDB.getReal("eval_dt", 0.01))
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
        "ini_t":        ini_t,
        "ign_t":        ign_t,
        "max_t":        max_t,
        "update_fun":   update_fun,
        "update_order": (rtol, atol),
        "eval_dt":      eval_dt,
        "fileHandle":   fileHandle,
        "save_every":   save_every,
        "rot_noise":    rot_noise,
        "trs_noise":    trs_noise,
        "seed":         seed,
        "tqdm_fun":     tqdm,
        }
    
    kwargs_list = (main_kwargs,)
    for t_kwargs in kwargs_list:
        for key in t_kwargs:
            problem_kwargs[key] = t_kwargs[key]
    return problem_kwargs


# initial all parameters
def get_problem_kwargs(**main_kwargs):
    # format long
    # ============================================================================================================================================
    fid_conf = open('Configuration.txt', 'wt')
    fid_conf.write('Dimensionless Time & i & X & Y:\n')  # 构象
    
    fid_matFrm = open('Material_Frame.txt', 'wt')
    fid_matFrm.write('Dimensionless Time & i & D3x & D3y:\n')  # Material Frame
    
    fid_vt = open('Translation_Velocity.txt', 'wt')
    fid_vt.write('Dimensionless Time & i & UX & UY:\n')  # 平移速度
    
    fid_vr = open('Rotation_Velocity.txt', 'wt')
    fid_vr.write('Dimensionless Time & i & WZ:\n')  # 旋转速度
    
    # parameter setup
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # todo: modify
    problem_kwargs = get_solver_kwargs()
    OptDB = PETSc.Options()
    fileHandle = OptDB.getString('f', 'dbg_ForceSpherePETSc')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle
    kT1 = OptDB.getReal('kT1', 4.141947e-06)
    # 主体部分的计算
    kT1 = 4.141947e-06  # 室温(300K)时的热涨落的能量(g um2/s2)
    kT2 = 5.522602e-06  # 400K时的热涨落的能量(g um2/s2)
    kT = 300 * 1.380649e-08  # 300K时的热涨落的能量(g um2/s2)
    # mu = 1.0e-6  # Fluid viscosity(血浆的粘度)(g/um/s)
    mu = 1  # Fluid viscosity(血浆的粘度)(g/um/s)
    frac = 1 / (np.pi * mu)  # 前置系数
    a = 1.0  # 所有球体的半径
    dt0 = 6 * np.pi * mu * a ** 3 / kT  # 时间单位
    # 数据处理参数
    # rs2 = 6.0  # todo: varify, rs2 \in (6, 12, and other values).
    rs2 = 12  # todo: varify, rs2 \in (6, 12, and other values).
    sdis = 1.0e-10  # 最小表面间距
    dt = 2.50e-5 * dt0  # 时间步长
    time_h = 4 * dt  # 龙格库塔法时间步长
    For = 0.1  # 给定活性粒子的推进力
    Tor = 0.0 * kT  # 给定活性粒子的力矩
    N_dat = 200.0  # 数据输出（4的倍数）
    N_fig = 1000.0  # 画图（4的倍数）
    N_save = 20000.0  # 保存图片（N_fig的倍数）
    
    # 生成随机球体
    length = 500.0  # 图像长   改
    width = 500.0  # 图像宽   改
    
    radius = a  # spherical radius
    diameter = 2 * radius  # 散斑直径（单位：像素）
    density = 0.9  # 密排度（区间：0-1，图像上的散斑密度）  改
    variation = 0.9  # 偏移度（区间：0-1，图像上的散斑随机排布程度，0时即为圆点整列）
    # 可选设置
    eccentricity = 0.0  # 椭圆的偏心率（区间：[0,1]，为0时即为圆点）
    
    rng = np.random.seed(5)  # 生成一组随机数种子，使之后Section中产生的散斑伪随机, 这里将种子设置为0.可以调整
    diag_err = 1e-16  # Avoiding errors introduced by nan values. (避免nan值引入的误差)
    
    problem_kwargs['kT1'] = kT1
    problem_kwargs['rs2'] = rs2
    problem_kwargs['sdis'] = sdis
    problem_kwargs['length'] = length
    problem_kwargs['width'] = width
    problem_kwargs['mu'] = mu
    problem_kwargs['For'] = For
    problem_kwargs['Tor'] = Tor
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


def partical_generator(**problem_kwargs):
    ## partical generator
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # 画球体 * matlab语句为： XSC,YSC,ZSC = sphere(100) 这个是python的等价代码，可能因为两个软件的计算方式的不同导致生成的坐标结果存在差异
    theta0, phi0 = np.linspace(0, np.pi, 101), np.linspace(0, 2 * np.pi, 101)
    THETA, PHI = np.meshgrid(theta0, phi0)
    XSC = (np.sin(THETA) * np.cos(PHI)).T
    YSC = (np.sin(THETA) * np.sin(PHI)).T
    ZSC = (np.cos(THETA)).T
    
    # Section 2：生成随机散斑在图像上的位置
    major_radius = radius / np.sqrt(np.sqrt(1 - eccentricity * eccentricity))  # 长轴
    minor_radius = radius * np.sqrt(np.sqrt(1 - eccentricity * eccentricity))  # 短轴
    # 散斑个数
    spacing = diameter / density ** (1 / 2)  # 2D
    colu = int(length // spacing)  # x轴
    cols = int(width // spacing)  # y轴
    
    # 散斑位置
    x = np.zeros((colu, cols))
    y = np.zeros((colu, cols))
    # 散斑边界位置
    xmin = 0.5 * (length - colu * spacing)
    ymin = 0.5 * (width - cols * spacing)
    
    for i1 in range(colu):
        for i2 in range(cols):
            x[i1, i2] = xmin + (i1 + 1) * spacing
            y[i1, i2] = ymin + (i2 + 1) * spacing
    
    # # 加载随机种子
    S = rng
    # # 增加随机移动量
    limit = 0.5 * variation * spacing
    D = np.random.random((colu, cols)).T  # 为了使两边结果相同
    E = np.random.random((colu, cols)).T
    # x = x + limit * (D - 2)
    # y = y + limit * (E - 2)
    # x = x + limit * (np.random.random((colu, cols)).T - 2)
    # y = y + limit * (np.random.random((colu, cols)).T - 2)
    # NX = x.shape[0]
    # NY = x.shape[1]  # 各个轴向方向的粒子数
    # NX = 20
    # NY = 50
    NX = 2
    NY = 2
    
    NS = NX * NY  # 总的粒子数
    print(NS)
    radii = np.zeros((NS, 1))
    X0 = np.zeros((NS, 3))  # 各个粒子的坐标
    
    for i1 in range(NX):
        for i2 in range(NY):
            radii[i1 * NY + i2, 0] = radius  # 粒子的半径
            X0[i1 * NY + i2, 0] = x[i1, i2]
            X0[i1 * NY + i2, 1] = y[i1, i2]
            X0[i1 * NY + i2, 2] = 0  # 顺序：z > y > x
    
    R = radii
    X = X0
    D1 = np.zeros((NS, 3))  # x 方向（球心处）
    D2 = np.zeros((NS, 3))  # y 方向（球心处）
    D3 = np.zeros((NS, 3))  # z 方向（球心处）
    
    # todo: modify speedup
    for i in range(NS):
        # ax = np.random.random(1)
        # np.random.seed(22)
        ax = np.random.rand(1)
        # ax = np.sqrt(2) * erfinv(2 * ax - 1)  # 为了使python和matlab生成相同的随机数
        D3[i, :] = np.hstack([np.cos(2 * ax * np.pi), np.sin(2 * ax * np.pi), 0])
    
    # todo: check if return more parameters.
    return R, X


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


def partical_generator_LBP(**problem_kwargs):
    # test code from LBP, A000-Test_RK4 case (Liu Baopi version).
    sphere_X = np.array(((2.7, 2.9),
                         (0.1, 2.1),
                         (0.1, 0.1),
                         (1.9, 0.1)))
    sphere_R = np.ones(sphere_X.shape[0])
    
    sphere_phi = np.zeros_like(sphere_R)
    return sphere_R, sphere_X, sphere_phi


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    problem_kwargs['matrix_method'] = 'forceSphere2d'
    
    # test damo
    # sphere_R, sphere_X, D3 = partical_generator(**problem_kwargs)
    # sphere_R, sphere_X, D3 = partical_generator_random(**problem_kwargs)
    sphere_R, sphere_X, sphere_phi = partical_generator_LBP(**problem_kwargs)
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
    sphere_R, sphere_X, sphere_phi = partical_generator_LBP(**problem_kwargs)
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
    sphere_R, sphere_X, sphere_phi = partical_generator_LBP(**problem_kwargs)
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
    problem_kwargs['matrix_method'] = 'forceSphere2d_simp'
    problem_kwargs['length'] = 3
    problem_kwargs['width'] = 3
    problem_kwargs['sdis'] = 1.0e-4  # 最小表面间距
    problem_kwargs['update_fun'] = '4'
    problem_kwargs['update_order'] = (0, 0)
    
    # test damo
    # sphere_R, sphere_X, D3 = partical_generator(**problem_kwargs)
    # sphere_R, sphere_X, D3 = partical_generator_random(**problem_kwargs)
    sphere_R, sphere_X, sphere_phi = partical_generator_LBP(**problem_kwargs)
    n_sphere, dof = sphere_X.shape
    
    # loop
    prb_loop = problemClass.ForceSphere2D_matrix(name=fileHandle, **problem_kwargs)
    spf.petscInfo(prb_loop.logger, "#" * 72)
    # spf.petscInfo(prb_loop.logger, "Generate Problem. ")
    
    prb_loop.update_fun = problem_kwargs["update_fun"]
    prb_loop.update_order = problem_kwargs["update_order"]
    prb_loop.save_every = problem_kwargs["save_every"]
    prb_loop.tqdm_fun = problem_kwargs["tqdm_fun"]
    prb_loop.do_save = True
    
    prb_loop.relationHandle = relationClass.nothingRelation2D()
    sphere_ptc = particleClass.ForceSphere2D_matrix(name="ForceSphere2D")
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
    
    act1 = interactionClass.ForceSphere2D_matrix(name="ForceSphere2D")
    prb_loop.add_act(act1)
    ini_t, max_t, eval_dt = 0, 10, 1
    spf.petscInfo(prb_loop.logger, "Generate Problem finish. ")
    prb_loop.update_self(t0=ini_t, t1=max_t, eval_dt=eval_dt)
    
    return True


if __name__ == '__main__':
    main_fun_v4()
