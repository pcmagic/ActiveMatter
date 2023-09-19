# import matplotlib
# matplotlib.use('Agg', force=True)
import sys
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
    fileHandle = OptDB.getString('f', 'ForceSpherePETSc')
    OptDB.setValue('f', fileHandle)
    problem_kwargs['fileHandle'] = fileHandle
    kT1 = OptDB.getReal('kT1', 4.141947e-06)
    # 主体部分的计算
    kT1 = 4.141947e-06  # 室温(300K)时的热涨落的能量(g um2/s2)
    kT2 = 5.522602e-06  # 400K时的热涨落的能量(g um2/s2)
    kT = 300 * 1.380649e-08  # 300K时的热涨落的能量(g um2/s2)
    mu = 1.0e-6  # Fluid viscosity(血浆的粘度)(g/um/s)
    frac = 1 / (np.pi * mu)  # 前置系数
    a = 1.0  # 所有球体的半径
    dt0 = 6 * np.pi * mu * a ** 3 / kT  # 时间单位
    # 数据处理参数
    rs2 = 6.0
    sdis = 1.0e-10  # 最小表面间距
    dt = 2.50e-5 * dt0  # 时间步长
    time_h = 4 * dt  # 龙格库塔法时间步长
    For0 = np.sqrt(2 * kT * 6 * np.pi * mu * a / dt)  # 白噪声的力
    Tor0 = np.sqrt(2 * kT * 8 * np.pi * mu * a ** 3 / dt)  # 白噪声的力矩
    For = 0.0 * kT / a  # 给定活性粒子的推进力
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
    
    rng = np.random.seed(
            5)  # 生成一组随机数种子，使之后Section中产生的散斑伪随机, 这里将种子设置为0.可以调整 #########################################################################################################################
    diag_err = 1e-16  # Avoiding errors introduced by nan values. (避免nan值引入的误差)
    
    problem_kwargs['kT1'] = kT1
    problem_kwargs['rs2'] = rs2
    problem_kwargs['sdis'] = sdis
    problem_kwargs['length'] = length
    problem_kwargs['width'] = width
    problem_kwargs['mu'] = mu
    problem_kwargs['For'] = For
    problem_kwargs['For0'] = For0
    problem_kwargs['Tor'] = Tor
    problem_kwargs['Tor0'] = Tor0
    problem_kwargs['diag_err'] = diag_err
    # todo....
    
    for key in main_kwargs:
        problem_kwargs[key] = main_kwargs[key]
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
    NX = 20  #########################################################################################################################
    NY = 50  #########################################################################################################################
    
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


def main_fun(**main_kwargs):
    problem_kwargs = get_problem_kwargs(**main_kwargs)
    print_case_info(**problem_kwargs)
    fileHandle = problem_kwargs['fileHandle']
    
    # test damo
    # R, X = partical_generator(**problem_kwargs)
    np.random.seed(1)
    NS = 2
    dof = 2
    sphere_R = np.random.sample(NS)
    sphere_X = np.random.sample((NS, dof))
    sphere_V = np.random.sample((NS, dof))
    problem_kwargs['matrix_method'] = 'forceSphere2d'
    
    # numpy version
    from src import forceSphere2d as fs2
    diag_err = problem_kwargs['diag_err']  # Avoiding errors introduced by nan values. (避免nan值引入的误差)
    rs2 = problem_kwargs['rs2']
    sdis = problem_kwargs['sdis']
    length = problem_kwargs['length']
    width = problem_kwargs['width']
    mu = problem_kwargs['mu']
    For = problem_kwargs['For']
    For0 = problem_kwargs['For0']
    Tor = problem_kwargs['Tor']
    Tor0 = problem_kwargs['Tor0']
    lamb_inter_list, ptc_lub_list = fs2.MMD_lub(sphere_R, rs2)
    M_RPY, R_lub, Rtol = fs2.M_R_fun(sphere_R, sphere_X, rs2, sdis, length, width, ptc_lub_list, lamb_inter_list, mu=mu)
    F1 = fs2.F_fun(NS, For, For0, Tor, Tor0)
    # print()
    # print('ptc_lub_list')
    # print(ptc_lub_list)
    # print()
    # print('M_RPY')
    # print(M_RPY)
    # print()
    # print('R_lub')
    # print(R_lub)
    print()
    print('Rtol')
    print(Rtol)
    
    return True


if __name__ == '__main__':
    main_fun()
