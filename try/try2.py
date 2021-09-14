import numpy as np
# from act_act_src import baseClass
from act_src import particleClass
from act_src import interactionClass
from act_src import problemClass
from act_src import relationClass


def ini_particle(name='...'):
    tptc = particleClass.finiteDipole2D(length=length, name='ptc2D')
    tptc.phi = np.pi / 2
    tptc.X = tptc.P1
    tptc.u = u
    prb1.add_obj(tptc)
    return tptc


test_n = 2
overlap_epsilon = 1e-5
u, length = 1, 1
attract, align = 0, 0

prb1 = problemClass.behavior2DProblem(name='testFiniteDipole2D')
prb1.attract = attract
prb1.align = align

tptc1 = ini_particle(name='ptc2D')
tptc1.X = (0, 0)
tptc2 = ini_particle(name='ptc2D')
tptc2.X = (0, 1)

rlt1 = relationClass.finiteRelation2D(name='relation1')
rlt1.overlap_epsilon = overlap_epsilon
prb1.relationHandle = rlt1

act1 = interactionClass.selfPropelled2D(name='selfPropelled2D')
prb1.add_act(act1)
act2 = interactionClass.FiniteDipole2D(name='FiniteDipole2D')
prb1.add_act(act2)
act3 = interactionClass.Attract2D(name='Attract2D')
prb1.add_act(act3)
act4 = interactionClass.Align2D(name='Align2D')
prb1.add_act(act4)

prb1.update_prepare()
prb1.update_self(t1=1.5)
print(111)
