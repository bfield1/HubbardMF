#!/usr/bin/python3

"""
Mean field Hubbard model of the Kagome lattice, in a single unit cell.

Has periodic boundary conditions and explores full Brillouin zone.

Created: 2020-09-17
Last Modified: 2020-09-22
Author: Bernard Field
"""

import json
from math import cos, pi

import numpy as np

from hubbard.kagome import kagome_coordinates
from hubbard.kpoints.base import HubbardKPoints

class KagomeHubbardKPoints(HubbardKPoints):
    """Mean field Hubbard model of the Kagome lattice, in a single unit cell."""
    #
    ## IO
    #
    def __init__(self,u=0,t=1,nup=0,ndown=0,allow_fractions=False,**kwargs):
        """
        Create a kagome lattice and initialises things.

        Last Modified: 2020-09-17
        """
        self.nrows = 1
        self.ncols = 1
        super().__init__(3, u=u, nup=nup, ndown=ndown,
                         allow_fractions=allow_fractions, **kwargs)
        self.set_kinetic(t)
    #
    def copy(self):
        """
        Return a copy of this kagome lattice object.

        Output: KagomeHubbardKPoints object.

        Last Modified: 2020-09-17
        """
        kagome = KagomeHubbardKPoints(u=self.u, t=self.t, nup=self.nup.copy(),
                                      ndown=self.ndown.copy(),
                                      allow_fractions=self.allow_fractions)
        kagome.set_kmesh(self.kmesh)
        kagome.set_mag(self.mag)
        return kagome
    #
    @classmethod
    def load(cls,f):
        """
        Load a KagomeHubbardKPoints object from a JSON file f.

        Inputs: f - string, filename.
        Output: KagomeHubbardKPoints object.
        Last Modified: 2020-09-17
        """
        # Load the file
        with open(f) as file:
            di = json.load(file) # di for dictionary
        # Create a KagomeHubbardKPoints object
        kagome = cls(u=di['u'], t=di['t'], nup=np.asarray(di['nup']),
                     ndown=np.asarray(di['ndown']),
                     allow_fractions=di['allow_fractions'])
        kagome.set_kmesh(np.asarray(di['kmesh']))
        kagome.set_mag(di['mag'])
        return kagome
    #
    def save(self,f):
        """
        Save a JSON representation of the object's data to file f.

        Inputs: f - string, filename.
        Writes a test file f.
        Last Modified: 2020-09-17
        """
        with open(f,mode='w') as file:
            json.dump({
                'u' : self.u,
                't' : self.t,
                'mag' : self.mag,
                'nup' : self.nup.tolist(),
                'ndown' : self.ndown.tolist(),
                'allow_fractions' : self.allow_fractions,
                'kmesh' : self.kmesh.tolist(),
                }, file)
    #
    ## ELECTRON DENSITY MODIFIERS
    #
    #
    ## GETTERS
    #
    def get_coordinates(self):
        """
        Return the coordinates for plotting electron density.

        Output: a (self.nsites,2) ndarray of floats.
        Last Modified: 2020-09-17
        """
        return kagome_coordinates(self.nrows,self.ncols)
    #
    def get_kinetic(self,k):
        """
        Return the kinetic energy matrix for a given momentum.

        I use the real form of the matrix to avoid unnecessary complex numbers.
        This results in a gauge transformation on the eigenvectors, which
        only matters if you care about the phase (which I don't).
        
        Input: k, a length 2 list-like of numbers, representing the momentum
                in fractional coordinates.
        Output: a (3,3) symmetric real ndarray.
        Last Modified: 2020-09-22
        """
        c0 = -2 * self.t * cos(pi*k[0])
        c1 = -2 * self.t * cos(pi*k[1])
        c2 = -2 * self.t * cos(pi*(k[0]-k[1]))
        return np.array([[0,c0,c1],[c0,0,c2],[c1,c2,0]])
    #
    ## PLOTTERS
    #
    #
    ## SETTERS
    #
    def set_kinetic(self,t):
        """
        Set the hopping constant for the Hamiltonian.

        Inputs: t - real number. Hopping constant.
        Effect: sets self.t to t
        Last Modified: 2020-09-17
        """
        self.t = t
        
    
