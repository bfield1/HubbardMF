#!/usr/bin/python3

"""
Container for substrate parameters and returner of dispersion relations.
For use with tight-binding model of stuff on substrates.
The substrates are structureless in real space and have a known dispersion
in momentum space.
Each substrate holds a single orbital only (for now, at least).

Author: Bernard Field
"""

from math import sqrt, pi

import numpy as np

class BaseSubstrate:
    def __init__(self, nx=None, ny=None, *args, **kwargs):
        self.set_params(nx, ny, *args, **kwargs)

    def set_params(self, nx=None, ny=None):
        """
        Sets parameters for the substrate.
        
        Inputs: nx - positive integer, number of substrate cells along first
                lattice vector in the supercell. Default 1.
            ny - positive integer, number of substrate cells along 2nd vector.
                Default 1.
        """
        # Set nx and ny if provided. Keep current value if not.
        # Initialise to 1 if no current value.
        if nx is not None:
            self.nx = nx
        elif not hasattr(self, 'nx'):
            self.nx = 1
        if ny is not None:
            self.ny = ny
        elif not hasattr(self, 'ny'):
            self.ny = 1
        # Total number of cells. A useful constant
        self.ncells = self.nx * self.ny
        # Number of reciprocal lattice points.
        self.glist = np.array([[i,j] for i in range(self.nx) for j in range(self.ny)])
    #
    def dispersion(self, k):
        """Takes wavevector k and returns the energy. Supports lists/arrays"""
        raise NotImplementedError
    #
    def get_matrix(self, k):
        """
        Returns the Hamiltonian of the substrate at wavevector k

        Inputs: k - array/list of numbers, a wavevector
        Output: ncells*ncells diagonal ndarray.
        """
        ks = self.glist + np.asarray(k) # Momenta
        es = self.dispersion(ks.transpose()) # Energies
        return np.diag(es)
    #
    def get_coupling(self, k, positions, coupling=1):
        """
        Returns the coupling matrix
        
        Coupling terms are of the form -coupling/sqrt(N) e^(2pi i r.(k+G))
        
        Inputs: k - shape (2,) numeric array, wavevector
            positions - shape (x,2) numeric array, positions in fractional
                coords of atoms in system coupling to 
            coupling - scalar or shape (x,) array, coupling constant.
                If array, is coupling constant for each atom.
        Output: shape (x,len(self.glist)) ndarray, complex.
        """
        ks = self.glist + np.asarray(k) # Momenta
        return -np.asarray(coupling).reshape((-1,1))/sqrt(self.ncells) *\
                np.exp(2j*pi*np.dot(positions, ks.transpose()))
    #
    def _get_gs_in_window(self, emin, emax, sampling=2):
        """
        Finds which G vectors contribute to bands within the specified energy window

        Inputs: emin - number, minimum energy.
            emax - number, maximum energy.
            sampling - positive integer. How many k-points in each direction
                to sample for each G vector.
        Outputs: glist - (x,2) ndarray, list of G vectors.
        """
        # Generate a fresh glist
        glist_full = np.array([[i,j] for i in range(self.nx) for j in range(self.ny)])
        # Find the sub-unit-cell displacements we need
        shifts = np.array([[i/sampling,j/sampling] for i in range(-sampling,sampling)
                            for j in range(-sampling,sampling)])
        # Initialise tracker of valid G's.
        validg = np.zeros(len(glist_full), dtype=bool)
        # Go through the k-mesh, and mark the G's which fall in the range.
        for s in shifts:
            en = self.dispersion((glist_full + s).transpose())
            validg += (en > emin) & (en < emax)
        return glist_full[validg]
    #
    def truncate_glist(self, emin, emax, sampling=2):
        """
        Truncates the list of G-vectors to an energy window.
        Bands entirely outside the window are dropped.

        Inputs: emin - number, minimum energy.
            emax - number, maximum energy.
            sampling - positive integer. How many k-points in each direction
                to sample for each G vector.
        """
        self.glist = self._get_gs_in_window(emin, emax, sampling)
    #
    def restore_glist(self):
        """Resets the glist to the default"""
        self.glist = np.array([[i,j] for i in range(self.nx) for j in range(self.ny)])


class SquareSubstrate(BaseSubstrate):
    def set_params(self, nx=None, ny=None, t=None, offset=None):
        """
        Sets parameters for the substrate.
        
        Inputs: nx - positive integer, number of substrate cells along first
                lattice vector in the supercell. Default 1.
            ny - positive integer, number of substrate cells along 2nd vector.
                Default 1
            t - number, hopping constant. Default 1.
            offset - number, vertical energy offset. Default 0.
        """
        # Set t if provided. Keep current value if not.
        if t is not None:
            self.t = t
        # Initialise to 1 if no current value.
        elif not hasattr(self, 't'):
            self.t = 1
        # Set offset if provided. Keep current value if not.
        if offset is not None:
            self.offset = offset
        # Initialise to 0 if no current value.
        elif not hasattr(self, 'offset'):
            self.offset = 0
        super().set_params(nx,ny)
    #
    def dispersion(self, k):
        return -2*self.t * (np.cos(2*pi*k[0]/self.nx) +
                            np.cos(2*pi*k[1]/self.ny)) + self.offset

class TriangleSubstrate(SquareSubstrate):
    def dispersion(self, k):
        return -2*self.t * (np.cos(2*pi*k[0]/self.nx) +
                            np.cos(2*pi*k[1]/self.ny) +
                            np.cos(2*pi*(k[1]/self.ny - k[0]/self.nx))) + self.offset

