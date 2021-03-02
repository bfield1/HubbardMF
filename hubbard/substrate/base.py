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
        super().__init__(*args, allow_fractions=True, **kwargs)
        self.substrate_list = []
        self.couplings = []
        self.base_nsites = self.nsites
        self.substrate_sites = []
        # A dummy object for when I want to do things with just the overlayer.
        self.prototype = HubbardKPoints(self.nsites, allow_fractions=True)
        # u is set directly in the init of HubbardKPoints.
        # Can't use set_u there because base_nsites hasn't been set at that point.
        # So I use it here.
        self.set_u(self.u)
        # Things child class must initialise:
        # self.reclat, if different from the identity
        # self.positions, a list of the coordinates (in fractional coords) of
        #   each atom in the TB system.
        # self.prototype
    #
    def copy(self):
        """Returns a deep copy of itself."""
        # It's so complicated that it just isn't worth doing it manually.
        return copy.deepcopy(self)
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
    def set_electrons(self, nup=None, ndown=None, mu=None, T=None, separate_substrate=False,
                      **kwargs):
        """
        Sets the electron densities.

        The substrate systems have two parts which behave very differently.
        There is the lattice, similar to all the other HubbardMF classes.
        And there is the substrate, which is in a plane wave basis rather
        than a position basis. And I haven't k-resolved the density in the
        substrate either.
        The idea here is I want the substrate to be filled uniformly to set
        the chemical potential at a specific point.
        """
        # During initialisation and during density mixing, we want separate_substrate=False
        if not separate_substrate:
            super().set_electrons(nup=nup, ndown=ndown, **kwargs)
            return
        # Get the electron configuration for the lattice.
        self.prototype.set_electrons(self.nup[0:self.base_nsites], self.ndown[0:self.base_nsites], **kwargs)
        self.prototype.set_electrons(nup, ndown, **kwargs)
        nup_lat = self.prototype.nup
        ndown_lat = self.prototype.ndown
        # Get the electron configuration for the substrate
        if (mu is not None or T is not None) and len(self.substrate_list) > 0:
            if T is None or mu is None:
                raise TypeError("Provide both mu or T, or neither.")
            # Figure out how many electrons we need in the substrate
            N = self.nelect_from_chemical_potential(mu, T)
            N -= nup_lat.sum() + ndown_lat.sum()
            nsub = self.nsites - self.base_nsites # Number of substrate sites
            # Number of electrons per substrate 'site'
            N = min(max(N/nsub, 0), 2)
            nup_sub = np.ones(nsub) * N/2
            ndown_sub = np.ones(nsub) * N/2
        else:
            # Take the existing density
            nup_sub = self.nup[self.base_nsites:]
            ndown_sub = self.ndown[self.base_nsites:]
        # Set the electron density
        self.nup = np.hstack((nup_lat, nup_sub))
        self.nelectup = self.nup.sum()
        self.ndown = np.hstack((ndown_lat, ndown_sub))
        self.nelectdown = self.ndown.sum()
    #
    def set_u(self, u):
        # Sets u only on the main lattice, not the substrate
        try:
            len(u)
        except TypeError:
            # u is a scalar
            self.u = np.zeros(self.nsites)
            self.u[0:self.base_nsites] = u
            self.prototype.set_u(u)
        else:
            # u is a list
            if len(u) == self.nsites:
                self.u = np.asarray(u, dtype=float)
                self.prototype.set_u(u[0:self.base_nsites])
            elif len(u) == self.base_nsites:
                self.u = np.zeros(self.nsites)
                self.u[0:self.base_nsites] = u
                self.prototype.set_u(u)
            else:
                raise ValueError("u provided as a list, but length did not match nsites or base_nsites.")
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
        self.u = np.hstack((self.u, np.zeros(new_sites)))
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
            i1 = self.base_nsites + int(np.sum(self.substrate_sites[0:index]))
            i2 = i1 + self.substrate_sites[index]
            # Replace old segement with zeros of new length
            self.nup = np.hstack((self.nup[0:i1], np.zeros(new_sites),
                                  self.nup[i2:]))
            self.nelectup = np.sum(self.nup)
            self.ndown = np.hstack((self.ndown[0:i1], np.zeros(new_sites),
                                    self.ndown[i2:]))
            self.nelectdown = np.sum(self.ndown)
            self.u = np.hstack((self.u[0:i1], np.zeros(new_sites), self.u[i2:]))
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
        i2 = i1 + self.substrate_sites[index]
        # Remove the segment
        self.nup = np.hstack((self.nup[0:i1], self.nup[i2:]))
        self.nelectup = np.sum(self.nup)
        self.ndown = np.hstack((self.ndown[0:i1], self.ndown[i2:]))
        self.nelectdown = np.sum(self.ndown)
        self.u = np.hstack((self.u[0:i1], self.u[i2:]))
        # Update number of segments
        del self.substrate_sites[index]
        self.nsites = self.base_nsites + np.sum(self.substrate_sites)
    #
    #def plot_bands(self, *args, **kwargs):
    #    if 'atoms' not in kwargs:
    #        super().plot_bands(*args, **kwargs, atoms=np.arange(self.base_nsites)
    #    else:
    #        super().plot_bands(*args, **kwargs)

