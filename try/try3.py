import numpy as np
from tqdm import tqdm
# from act_act_src import baseClass
from act_src import particleClass
from act_src import interactionClass
from act_src import problemClass
from act_src import relationClass

test_n = 5
overlap_epsilon = 1e-5
u, length = 1, 1
attract, align = 1, 1
tmax = 15

prb1 = problemClass.finiteDipole2DProblem(name='testFiniteDipole2D')
prb1.attract = attract
prb1.align = align
prb1.tqdm_fun = tqdm

np.random.seed(0)
for _ in range(test_n):
    tptc = particleClass.finiteDipole2D(length=length, name='ptc2D')
    tptc.P1 = np.random.sample((2,))
    tptc.X = np.random.uniform(-10, 10, (2,))
    tptc.u = u
    prb1.add_obj(tptc)
    # print(tptc.P1, tptc.phi / np.pi, tptc.X)

rlt1 = relationClass.VoronoiRelation2D(name='relation1')
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
# prb1.relationHandle.dbg_showVoronoi()
prb1.update_self(t1=tmax)
#
# from act_codeStore import support_fun_animation as spanm
# from IPython import display
# anim = spanm.make2D_X_video(prb1.t_hist, prb1.obj_list)
# video = anim.to_html5_video()
# html = display.HTML(video)
# display.display(html)