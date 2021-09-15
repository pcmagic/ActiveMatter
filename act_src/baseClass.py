# coding=utf-8
"""
20210810
Zhang Ji
general features of the classes.
"""

# import numpy as np
from petsc4py import PETSc
from act_codeStore import support_fun as spf


# import pickle
# from time import time
# from tqdm import tqdm
# from tqdm.notebook import tqdm as tqdm_notebook


class baseObj:
    def __init__(self, name='...', **kwargs):
        self._name = name
        self._kwargs = kwargs
        # self._type = '...'
        self._type = type(self).__name__
        self._father = None

    def __repr__(self):
        return self._type

    def __str__(self):
        return self.name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def type(self):
        return self._type

    @property
    def father(self):
        return self._father

    @father.setter
    def father(self, father):
        self._father = father

    def print_info(self):
        # OptDB = PETSc.Options()
        spf.petscInfo(self.father.logger, ' ')
        spf.petscInfo(self.father.logger, 'Information about %s (%s): ' % (str(self), self.type,))
        return True

    @staticmethod
    def vec_scatter(vec_petsc, destroy=True):
        scatter, temp = PETSc.Scatter().toAll(vec_petsc)
        scatter.scatterBegin(vec_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
        scatter.scatterEnd(vec_petsc, temp, False, PETSc.Scatter.Mode.FORWARD)
        vec = temp.getArray()
        if destroy:
            vec_petsc.destroy()
        return vec

    def destroy_self(self, **kwargs):
        pass
