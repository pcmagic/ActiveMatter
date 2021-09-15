
import numpy as np
from tqdm import tqdm

from act_src import particleClass
from act_src import interactionClass
from act_src import problemClass
from act_src import relationClass


def ini_particle(name='...'):
    tptc = particleClass.limFiniteDipole2D(length=length, name=name)
    tptc.u = u
    prb1.add_obj(tptc)
    return tptc

update_fun, update_order = '5bs', (1e-12, 1e-15)
overlap_epsilon = 1e-5
u, length = 1, 1
attract, align = 0, 0
tmax = 50
# x, y, phi1, phi2 = 0, 4,    np.pi / 2, np.pi / 2   # Liebergen2015, Fig 4.1, case b
# x, y, phi1, phi2 = 2, 4,    np.pi / 2, np.pi / 2   # Liebergen2015, Fig 4.1, case c
# x, y, phi1, phi2 = 4, 2.25, np.pi / 2, np.pi / 2   # Liebergen2015, Fig 4.1, case d
# x, y, phi1, phi2 = 4, 1,    np.pi / 2, np.pi / 2   # Liebergen2015, Fig 4.1, case e
x, y, phi1, phi2 = 4, 0,    np.pi / 2, np.pi / 2   # Liebergen2015, Fig 4.1, case f

# y, x, phi1, phi2 = 0.5,        -8, np.pi, 0   # Liebergen2015, Fig 4.2, case b
# y, x, phi1, phi2 = 0.874,      -8, np.pi, 0   # Liebergen2015, Fig 4.2, case c
# y, x, phi1, phi2 = 0.875,      -8, np.pi, 0   # Liebergen2015, Fig 4.2, case d
# y, x, phi1, phi2 = 1,          -8, np.pi, 0   # Liebergen2015, Fig 4.2, case e
# y, x, phi1, phi2 = 0.87465, -8, np.pi, 0   # Liebergen2015, Fig 4.2, case f
# y, x, phi1, phi2 = 0.87465151, -8, np.pi, 0   # Liebergen2015, Fig 4.2, case f

# x, y, phi1, phi2 = 0.5,  1,    0, np.pi   # Liebergen2015, Fig 4.5, case a
# x, y, phi1, phi2 = 0,    1.15, 0, np.pi   # Liebergen2015, Fig 4.5, case b
# x, y, phi1, phi2 = 0.649, 0.7,  0, np.pi   # Liebergen2015, Fig 4.5, case c

# x, y, phi1, phi2 = -0.85,  0,    0, np.pi / 3       # Liebergen2015, Fig 4.8, case b
# x, y, phi1, phi2 =  0,    -0.9,  0, np.pi / 3       # Liebergen2015, Fig 4.8, case c
# x, y, phi1, phi2 = -0.3,  -1.15, 0, np.pi * 2 / 3   # Liebergen2015, Fig 4.8, case d
# x, y, phi1, phi2 = -0.75,  0,    0, np.pi * 2 / 3   # Liebergen2015, Fig 4.8, case e
# x, y, phi1, phi2 = -0.77,  0.4,  0, np.pi * 2 / 3   # Liebergen2015, Fig 4.8, case f

# prb1 = problemClass.finiteDipole2DProblem(name='testFiniteDipole2D')
prb1 = problemClass.finiteDipole2DProblem(name='testFiniteDipole2D')
prb1.attract = attract
prb1.align = align
prb1.update_fun = update_fun
prb1.update_order = update_order
prb1.tqdm_fun = tqdm

tptc1 = ini_particle(name='ptc2D')
tptc1.X = (0, 0)
tptc1.phi = phi1
tptc2 = ini_particle(name='ptc2D')
# print(tptc1.phi, tptc1.P1)
# print(tptc2.phi, tptc2.P1)
tptc2.X = (x, y)
tptc2.phi = phi2
# print(tptc2.phi, tptc2.P1)
# tptc1.length = 0.1

rlt1 = relationClass._baseRelation2D(name='relation1')
rlt1.overlap_epsilon = overlap_epsilon
prb1.relationHandle = rlt1

act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
prb1.add_act(act1)
act2 = interactionClass.limFiniteDipole2D(name='limFiniteDipole2D')
prb1.add_act(act2)

prb1.update_step()
prb1.update_self(t1=tmax)