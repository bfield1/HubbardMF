#!/usr/bin/python3

"""
Mean field Hubbard model of the Kagome lattice, in a single unit cell.

Has periodic boundary conditions and explores full Brillouin zone.

Created: 2020-09-17
Last Modified: 2021-05-18
Author: Bernard Field
    Copyright (C) 2021 Bernard Field, GNU GPL v3+
"""

import json
from math import cos, pi

import numpy as np

from hubbard.kagome import kagome_coordinates, kagome_adjacency_tensor
from hubbard.kpoints.base import HubbardKPoints


class KagomeHubbardKPoints(HubbardKPoints):
    """Mean field Hubbard model of the Kagome lattice, in a single unit cell."""
    #
    ## IO
    #
    def __init__(self,u=0,t=1,nup=0,ndown=0,allow_fractions=False,
                 nrows=1,ncols=1,**kwargs):
        """
        Create a kagome lattice and initialises things.

        Last Modified: 2020-11-06
        """
        self.nrows = nrows
        self.ncols = ncols
        self.reclat = np.array([[np.sqrt(3)/2, 1/2], [-np.sqrt(3)/2, 1/2]])
        super().__init__(3*nrows*ncols, u=u, nup=nup, ndown=ndown,
                         allow_fractions=allow_fractions, **kwargs)
        self.set_kinetic(t)
    #
    def copy(self):
        """
        Return a copy of this kagome lattice object.

        Output: KagomeHubbardKPoints object.

        Last Modified: 2020-11-13
        """
        kagome = KagomeHubbardKPoints(u=self.u, t=self.t, nup=self.nup.copy(),
                                      ndown=self.ndown.copy(), nrows=self.nrows,
                                      ncols=self.ncols,
                                      allow_fractions=self.allow_fractions)
        kagome.set_kmesh(self.kmesh)
        kagome.set_mag(self.mag)
        if self.modified_kin:
            kagome.set_kinetic(self.t, force_generic=True)
            kagome.kin = self.kin.copy()
        return kagome
    #
    @classmethod
    def load(cls,f):
        """
        Load a KagomeHubbardKPoints object from a JSON file f.

        Inputs: f - string, filename.
        Output: KagomeHubbardKPoints object.
        Last Modified: 2020-11-11
        """
        # Load the file
        with open(f) as file:
            di = json.load(file) # di for dictionary
        # Create a KagomeHubbardKPoints object
        kagome = cls(u=di['u'], t=di['t'], nrows=di.get('nrows',1),
                     ncols=di.get('ncols',1), nup=np.asarray(di['nup']),
                     ndown=np.asarray(di['ndown']),
                     allow_fractions=di['allow_fractions'])
        kagome.set_kmesh(np.asarray(di['kmesh']))
        kagome.set_mag(di['mag'])
        if 'kin' in di:
            kagome.set_kinetic(di['t'], force_generic=True)
            kagome.kin = np.asarray(di['kin'], dtype='complex')
            kagome.modified_kin = True
        return kagome
    #
    def save(self,f):
        """
        Save a JSON representation of the object's data to file f.

        Inputs: f - string, filename.
        Writes a test file f.
        Last Modified: 2021-05-18
        """
        di = {
                'u' : float(self.u),
                't' : float(self.t),
                'nrows' : int(self.nrows),
                'ncols' : int(self.ncols),
                'mag' : float(self.mag),
                'nup' : self.nup.tolist(),
                'ndown' : self.ndown.tolist(),
                'allow_fractions' : bool(self.allow_fractions),
                'kmesh' : self.kmesh.tolist(),
                }
        if self.modified_kin:
            di['kin'] = self.kin.real.tolist()
        with open(f,mode='w') as file:
            json.dump(di, file)
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

        For the single-unit-cell case,
        I use the real form of the matrix to avoid unnecessary complex numbers.
        This results in a gauge transformation on the eigenvectors, which
        only matters if you care about the phase (which I don't).
        
        For the supercell, I need to use the complex formulation.
        
        Input: k, a length 2 list-like of numbers, representing the momentum
                in fractional coordinates.
        Output: a (nsites,nsites) symmetric real ndarray.
        Last Modified: 2020-11-11
        """
        if self.nsites == 3 and not self.modified_kin:
            # The simple case of a single unit cell.
            c0 = -2 * self.t * cos(pi*k[0])
            c1 = -2 * self.t * cos(pi*k[1])
            c2 = -2 * self.t * cos(pi*(k[0]-k[1]))
            return np.array([[0,c0,c1],[c0,0,c2],[c1,c2,0]])
        else:
            # Multiple unit cells. Need to include phase factors at boundaries.
            kinetic = self.kin.copy()
            # e^ikb1
            kinetic[self.masks[0]] *= np.exp(complex(0,2*np.pi*k[0]))
            # e^ikb2
            kinetic[self.masks[1]] *= np.exp(complex(0,2*np.pi*k[1]))
            # e^ik(b2-b1)
            kinetic[self.masks[2]] *= np.exp(complex(0,2*np.pi*(k[1]-k[0])))
            # e^-ik(b2-b1)
            kinetic[self.masks[3]] *= np.exp(complex(0,-2*np.pi*(k[1]-k[0])))
            # e^-ikb2
            kinetic[self.masks[4]] *= np.exp(complex(0,-2*np.pi*k[1]))
            # e^-ikb1
            kinetic[self.masks[5]] *= np.exp(complex(0,-2*np.pi*k[0]))
            # Add back in the previously removed double-counted part.
            kinetic[self.two_mask] += self.kin[self.two_mask]
            return kinetic
    #
    ## PLOTTERS
    #
    #
    ## SETTERS
    #
    def set_kinetic(self, t, force_generic=False):
        """
        Set the hopping constant for the Hamiltonian.
        
        If nrows or ncols > 1, also pre-calculates the connectivity.

        Inputs: t - real number. Hopping constant.
        Effect: sets self.t to t
        Last Modified: 2020-11-11
        """
        self.t = t
        self.modified_kin = force_generic
        # Pre-compute as much as possible if we have a non-simple case.
        if self.nsites > 3 or force_generic:
            kinshape = (self.nsites,self.nsites)
            # Get the Gamma-point matrix.
            tensor = kagome_adjacency_tensor(self.nrows,self.ncols)
            # The case when nrows or ncols == 1 has some extra problems,
            # because you will get two different processes (one within the
            # supercell, and one from without) occupying the same matrix
            # element. These have a value of 2 in tensor.
            # We shall separate them out so we can treat them separately.
            self.two_mask = np.reshape((tensor == 2),kinshape,order='F')
            tensor[tensor==2] = 1
            self.kin = np.reshape(-t*tensor,kinshape,order='F').astype('complex')
            # Also have to cast to complex dtype, to allow complex numbers.
            # The k-dependence can be done as an elementwise multiplication.
            # Find where we cross over the boundaries.
            self.masks = []
            nrows = self.nrows
            ncols = self.ncols
            idrow = np.identity(nrows,dtype='bool').reshape(1,nrows,1,1,nrows,1)
            idcol = np.identity(ncols,dtype='bool').reshape(1,1,ncols,1,1,ncols)
            # (0,i,j)->(1,i-1,j), (2,i,j)->(1,i-1,j+1), needs e^ikb1
            m = np.zeros((3,nrows,1,3,nrows,1), dtype='bool')
            m[0,0,0,1,nrows-1,0] = True
            m = m*idcol
            if ncols > 1:
                m[(2,0,np.arange(ncols-1),1,nrows-1,np.arange(1,ncols))] = True
            self.masks.append(np.reshape(m, kinshape, order='F').transpose())
            # (0,i,j)->(2,i,j-1) (1,i,j)->(2,i+1,j-1), needs e^ikb2
            m = np.zeros((3,1,ncols,3,1,ncols), dtype='bool')
            m[0,0,0,2,0,ncols-1] = True
            m = m*idrow
            if nrows > 1:
                m[(1,np.arange(nrows-1),0,2,np.arange(1,nrows),ncols-1)] = True
            self.masks.append(np.reshape(m, kinshape, order='F').transpose())
            # (1,i,j)->(2,i+1,j-1) corner, needs e^ik(b2-b1)
            m = np.zeros((3,nrows,ncols,3,nrows,ncols), dtype='bool')
            m[1,nrows-1,0,2,0,ncols-1] = True
            self.masks.append(np.reshape(m, kinshape, order='F').transpose())
            # (2,i,j)->(1,i-1,j+1) corner, needs e^-ik(b2-b1)
            m = np.zeros((3,nrows,ncols,3,nrows,ncols), dtype='bool')
            m[2,0,ncols-1,1,nrows-1,0] = True
            self.masks.append(np.reshape(m, kinshape, order='F').transpose())
            # (2,i,j)->(0,i,j+1), (2,i,j)->(1,i-1,j+1), needs e^-ikb2
            m = np.zeros((3,1,ncols,3,1,ncols), dtype='bool')
            m[2,0,ncols-1,0,0,0] = True
            m = m*idrow
            if nrows > 1:
                m[(2,np.arange(1,nrows),ncols-1,1,np.arange(nrows-1),0)] = True
            self.masks.append(np.reshape(m, kinshape, order='F').transpose())
            # (1,i,j)->(0,i+1,j), (1,i,j)->(2,i+1,j-1), needs e^-ikb1
            m = np.zeros((3,nrows,1,3,nrows,1), dtype='bool')
            m[1,nrows-1,0,0,0,0] = True
            m = m*idcol
            if ncols > 1:
                m[(1,nrows-1,np.arange(1,ncols),2,0,np.arange(ncols-1))] = True
            self.masks.append(np.reshape(m, kinshape, order='F').transpose())
            # Sanity check
            # The masks should not overlap
            assert (sum([a.sum() for a in self.masks]) ==
                    np.sum(self.masks[0] | self.masks[1] | self.masks[2] |
                           self.masks[3] | self.masks[4] | self.masks[5]))
    #
    def set_kinetic_random(self,t,wt=0,we=0):
        """
        Creates a kinetic energy matrix with some random noise in it.

        Random noise is uniform. Noise ofhopping constants centred on
        t and noise of on-site energy centred on 0.

        Inputs: t - number. Hopping constant.
            wt - number. Width of the random noise of hopping.
            we - number. Width of the random noise of on-site energy.
        Effects: sets self.kin

        Last Modified: 2020-11-13
        """
        rng = np.random.default_rng()
        # Get the regular kinetic energy set
        self.set_kinetic(t, force_generic=True)
        self.modified_kin = True
        # Now apply random noise.
        # Get some random noise for the whole matrix.
        noise = rng.random(self.kin.shape)*wt - wt/2
        # Zero out elements which should be zero.
        noise[self.kin == 0] = 0
        # Make the noise matrix Hermitian.
        # Zero out the lower triangle, otherwise next step will be sum of two
        # random variates which is not a uniform distribution.
        noise = np.triu(noise) 
        noise = noise + noise.transpose()
        # Apply noise to the kinetic energy.
        self.kin += noise
        # Now get the noisy diagonal for on-site energy.
        self.kin += np.diag(rng.random(len(self.kin))*we - we/2)
        # Done.
