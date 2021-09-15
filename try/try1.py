import numpy as np
# from act_act_src import baseClass
from act_src import particleClass
from act_src import interactionClass
from act_src import problemClass
from act_src import relationClass

prb1 = problemClass.behavior2DProblem(name='tryPrb')

# for i0 in range(3):
#     tptc = particleClass.particle2D(name='ptc2D')
#     tptc.P1 = np.random.sample((2,))
#     tptc.X = np.random.uniform(-10, 10, (2,))
#     tptc.u = np.random.sample(1)
#     prb1.add_obj(tptc)
tptc = particleClass.particle2D(name='ptc2D')
tptc.P1 = np.array((1, 0))
tptc.X = np.array((0, 0))
tptc.u = np.array((1, ))
prb1.add_obj(tptc)

rlt1 = relationClass._baseRelation2D(name='relation1')
prb1.relationHandle = rlt1
# rlt1.cal_theta_rho()

act1 = interactionClass.selfPropelled2D(name='action1')
prb1.add_act(act1)
prb1.update_step()
prb1.update_self(t1=0.1)
print(11111)