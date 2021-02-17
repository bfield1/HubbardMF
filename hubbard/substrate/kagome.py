#!/usr/bin/python3

from inspect import isclass
from math import sqrt, pi

import numpy as np
from scipy.linalg import block_diag

from hubbard.substrate.base import HubbardSubstrate
from hubbard.kpoints.kagome import KagomeHubbardKPoints

class KagomeSubstrate(HubbardSubstrate, KagomeHubbardKPoints):
    # MRO: KagomeSubstrate -> HubbardSubstrate -> KagomeHubbardKPoints -> HubbardKPoints
    """Single kagome unit cell on a periodic substrate"""
    def __init__(self, u=0, t=1, offset=0, nup=0, ndown=0, allow_fractions=False, **kwargs):
        super().__init__(u=u,t=t,nup=nup,ndown=ndown,allow_fractions=allow_fractions,nrows=1,ncols=1,**kwargs)
        self.set_kinetic(t, offset) # To consider: Putting offset in KagomeHubbardKPoints
        self.positions = self.get_coordinates()
    #
    def set_kinetic(self, t=None, offset=None):
        """Sets hopping constant for kagome lattice to t"""
        if t is not None:
            self.t = t
        if offset is not None:
            self.offset = offset
    #
    def get_kinetic_no_substrate(self, k):
        c0 = -self.t * (1 + np.exp(2j*pi*k[0]))
        c1 = -self.t * (1 + np.exp(2j*pi*k[1]))
        c2 = -self.t * (1 + np.exp(2j*pi*(k[1]-k[0])))
        return np.array([[self.offset, c0.conjugate(), c1.conjugate()],
                         [c0, self.offset, c2.conjugate()],
                         [c1, c2, self.offset]])
