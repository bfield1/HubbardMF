#!/usr/bin/python3

from inspect import isclass
from math import sqrt, pi

import numpy as np
from scipy.linalg import block_diag

from hubbard.kpoints.base import HubbardKPoints
import substratetb.substrate as sub
from substratetb.dftsubstrate import DFTSubstrate

class HubbardSubstrate(HubbardKPoints):
    """Base class for coupling Hubbard model to substrates"""
    # Map strings to substrate classes
    submap = {"square" : sub.SquareSubstrate,
              "triangle" : sub.TriangleSubstrate,
              "dft" : DFTSubstrate
              }
    #
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.substrate_list = []
        self.couplings = []
        self.base_nsites = self.nsites
        self.substrate_sites = []
        # Things child class must initialise:
        # self.reclat, if different from the identity
        # self.positions, a list of the coordinates (in fractional coords) of
        #   each atom in the TB system.
    #
    def get_kinetic(self, k):
        """
        Retrieves the full kinetic energy matrix

        Input: k - list-like of two numbers, wavevector (fractional coords)
        Output: (self.nsites,self.nsites) ndarray, Hermitian, complex.
        """
        # Get the Hamiltonian of the system on top
        htop = self.get_kinetic_no_substrate(k)
        # Catch the case of no substrate
        if len(self.substrate_list) == 0:
            return htop
        # Get the substrate Hamiltonians
        hsub = block_diag(*[s.get_matrix(k) for s in self.substrate_list])
        # Make the coupling
        hcouple = np.hstack([s.get_coupling(k, self.positions, c) for (s,c)
                             in zip(self.substrate_list, self.couplings)])
        # Combine everything into a nice big block matrix
        return np.block([[htop, hcouple], [hcouple.conjugate().transpose(), hsub]])
    #
    def get_kinetic_no_substrate(self, k):
        """
        Gets Hamiltonian for the non-substrate part of the system

        Input: wavevector k
        Output: Hermitian ndarray
        """
        raise NotImplementedError
    #
    def add_substrate(self, subtype, coupling, **kwargs):
        """
        Adds and initialises a new substrate
        
        Inputs: subtype - class or string. Substrate type.
            coupling - number. Coupling with substrate.
            **kwargs - substrate params (see substrate class).
        """
        if isclass(subtype):
            self.substrate_list.append(subtype(**kwargs))
        else:
            self.substrate_list.append(self.submap[subtype](**kwargs))
        self.couplings.append(coupling)
        # Add more electron sites.
        new_sites = len(self.substrate_list[-1].glist)
        self.substrate_sites.append(new_sites)
        self.nsites += new_sites
        self.nup = np.hstack((self.nup, np.zeros(new_sites)))
        self.ndown = np.hstack((self.ndown, np.zeros(new_sites)))
    #
    def change_substrate(self, index, subtype=None, coupling=None, **kwargs):
        """
        Replaces/modifies substrate and/or coupling at index
        
        Inputs: index - integer, list index
            subtype - optional string or class, substrate type
                (unchanged if unspecified)
            coupling - optional number. Coupling with substrate
                (unchanged if unspecified)
            **kwargs - substrate params (must be specified)
        """
        # Update substrate parameters
        if subtype is not None:
            if isclass(subtype):
                self.substrate_list[index] = subtype(**kwargs)
            else:
                self.substrate_list[index] = self.submap[subtype](**kwargs)
        else:
            self.substrate_list[index].set_params(**kwargs)
        # Update the coupling strength.
        if coupling is not None:
            self.couplings[index] = coupling
        # Update the number of electron sites (if needed)
        new_sites = len(self.substrate_list[index].glist)
        if new_sites != self.substrate_sites[index]:
            # Indices at start and after segment to be changed.
            i1 = self.base_sites + int(np.sum(self.substrate_sites[0:index]))
            i2 = i1 + self.substrate_sites[index] + 1
            # Replace old segement with zeros of new length
            self.nup = np.hstack((self.nup[0:i1], np.zeros(new_sites),
                                  self.nup[i2:]))
            self.nelectup = np.sum(self.nup)
            self.ndown = np.hstack((self.ndown[0:i1], np.zeros(new_sites),
                                    self.ndown[i2:]))
            self.nelectdown = np.sum(self.ndown)
            # Update number of sites
            self.substrate_sites[index] = new_sites
            self.nsites = self.base_nsites + np.sum(self.substrate_sites)
    #
    def remove_substrate(self, index):
        """Removes the substrate with index 'index'"""
        del self.substrate_list[index]
        del self.couplings[index]
        # Indices at start and after segment to be changed.
        i1 = self.base_nsites + int(np.sum(self.substrate_sites[0:index]))
        i2 = i1 + self.substrate_sites[index] + 1
        # Remove the segment
        self.nup = np.hstack((self.nup[0:i1], self.nup[i2:]))
        self.nelectup = np.sum(self.nup)
        self.ndown = np.hstack((self.ndown[0:i1], self.ndown[i2:]))
        self.nelectdown = np.sum(self.ndown)
        # Update number of segments
        del self.substrate_sites[index]
        self.nsites = self.base_nsites + np.sum(self.substrate_sites)
    #
    #def plot_bands(self, *args, **kwargs):
    #    if 'atoms' not in kwargs:
    #        super().plot_bands(*args, **kwargs, atoms=np.arange(self.base_nsites)
    #    else:
    #        super().plot_bands(*args, **kwargs)

