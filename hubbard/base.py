#!/usr/bin/python3

"""
Base class for the Mean field Hubbard model, at the Gamma point only.

Has the mixing routines and other things which aren't dependent on the
specifics of the lattice.

There are a few main differences from the parent class HubbardKPoints.
 - While HubbardKPoints dynamically generates the kinetic energy matrix
    for a given k, here the kinetic energy has no variables which change
    mid-calculation, so the kinetic energy matrix is cached as self.kin.
    Many methods here access self.kin directly. set_kinetic is expected to
    set self.kin, rather than merely record parameters.
 - _eigensystem is streamlined to exploit the single k-point calculation.
 - set_kmesh won't do anything other than make the Gamma point.

A key advantage of the Gamma point only routine is that it simplifies some
algorithms. Building kinetic energy matrices is easier without having to figure
out where momentum comes into it. The move_electrons method currently is only
implemented for Gamma point only (although that's mainly a matter of it not being
the most useful method so not worth development time). Having a cached kinetic energy
matrix rather than needing to remember a bunch of parameters can make complex
Hamiltonians easier to deal with.
There might be a small performance improvement as well, although I haven't tested it.

Several things must be defined in the subclasses.
 - The kinetic energy matrix, self.kin. This needs to be set in the 
     initialisation step. Ideally you'd also allow it to be
     changed later. THIS ONE IS IMPORTANT.
 - The copy method. find_magnetic_minimum won't work if this isn't defined. And
    I don't know which data you need to feed into new instances.
 - get_coordinates - returns a (nsites, 2) ndarray which indicates
     the Cartesian coordinates of each site. Used for plotting.
     (I assume a 2D lattice for plotting. 3D plotting will require re-working
     the plotting routines to do 3D plots. That's a job for a subclass.)
 - save and load methods (you don't have to define them, but they aren't
     defined here). (I may find a way to abstract the saving/loading later.)
 - any bespoke electron density setting methods, in _electron_density_single_methods

Created: 2020-08-10
Last Modified: 2020-09-17
Author: Bernard Field
"""

from warnings import warn

import numpy as np

from hubbard.kpoints.base import HubbardKPoints

class Hubbard(HubbardKPoints):
    """Base class for the Mean field Hubbard model, at the Gamma point only."""
    #
    ## IO
    #
    # Everything of interest for IO is in the parent class.
    # The only thing to look out for is k-points, but that defaults to initialising
    # as Gamma point only so that's okay.
    #
    ## ELECTRON DENSITY MODIFIERS
    #
    def find_magnetic_minimum(self,max_iter=500,ediff=1e-6,mix=0.5,verbose=False):
        """
        Flips the spin of electrons (self-consistently) until we find an
        energy minimum.
        I do not recommend this method any more, although it is useable.
        A better method is linear_mixing with finite T.
        I note that for large U the energy landscape is difficult to
        traverse. This method (and linear_mixing) will only find a
        local minimum.
        
        Inputs: max_iter, ediff, mix - as for linear_mixing.
            verbose - Boolean, default False. Prints progress.
        
        Last Modified: 2020-08-10
        """
        # To save on typing, stick the keyword parameters into a dictionary
        # for linear_mixing.
        linmixparams = {
            'max_iter':max_iter,
            'ediff':ediff,
            'mix':mix,
            'print_residual':False
            }
        # First step: ensure current state is self-consistent
        en = self.linear_mixing(**linmixparams)
        # Decide if we want to go up or down, or stay put.
        # Energy of flipping one from down to up.
        nsites = self.nsites
        if self.nelectdown >= 1 and self.nelectup <= nsites-1:
            # Can we do such a flip?
            if verbose: print("Checking first up spin flip.")
            kagup = self.copy()
            kagup.move_electrons(1,-1)
            enup = kagup.linear_mixing(**linmixparams)
        else:
            # If not, set energy to very large value so we don't cross it.
            enup = 1e15
        # Energy of flipping from up to down
        if self.nelectup >= 1 and self.nelectdown <= nsites-1:
            # Can we do such a flip?
            if verbose: print("Checking first spin down flip.")
            kagdown = self.copy()
            kagdown.move_electrons(-1,1)
            endown = kagdown.linear_mixing(**linmixparams)
        else:
            # If not, set energy to very large value so we don't cross it.
            endown = 1e15
        # Check what we want to do.
        # If current energy is minimum, stay put
        if en <= enup and en <= endown:
            if verbose: print("Already at energy minimum.")
            return
        # Else, take the path of most energy drop.
        if enup < endown:
            go_up = True
        elif enup > endown:
            go_up = False
        else: # We have a tie.
            # Go towards zero magnetization,
            # going up if at zero.
            go_up = self.get_magnetization() <= 0
        # Adjust electrons in that direction.
        if go_up:
            if verbose: print("Searching upwards.")
            self.set_electrons(nup=kagup.nup,ndown=kagup.ndown)
            en = enup
        else:
            if verbose: print("Searching downwards.")
            self.set_electrons(nup=kagdown.nup,ndown=kagdown.ndown)
            en = endown
        #Iterate in that direction until a minimum is found.
        at_edge = lambda : ((go_up and (self.nelectup > nsites-1 and
                                       self.nelectdown < 1))
                            or ((not go_up) and (self.nelectdown > nsites-1 and
                                                 self.nelectup < 1)))
        while not at_edge():
            # Create the lattice with the next spin flip.
            kagnext = self.copy()
            if go_up:
                kagnext.move_electrons(1,-1)
            else:
                kagnext.move_electrons(-1,1)
            if verbose: print("Checking magnetization "+str(self.get_magnetization()))
            ennext = kagnext.linear_mixing(**linmixparams)
            # Check if the energy is higher.
            if ennext >= en:
                # If so, we're done
                break
            # Otherwise, set the current to the trial step
            self.set_electrons(nup=kagnext.nup,ndown=kagnext.ndown)
            en = ennext
        # Done
        if verbose: print("Found energy minimum.")
        return
    #
    def move_electrons(self,up=0,down=0):
        """
        Add or subtract spin up and down electrons, based on eigenstates.

        Because doing this changes the density and thus the Hamiltonian,
        repeated calls to this function will give a different result than
        a single call to this function.

        Inputs: up - integer, default 0. Number of spin up electrons to add.
                If negative, subtracts.
            down - integer, default 0. Number of spin down electrons to add.
                If negative, subtracts.
        Last Modified: 2020-09-16
        """
        # Check that up and down are valid.
        if up+self.nelectup > self.nsites or up+self.nelectup < 0:
            raise ValueError("Trying to put number of up electrons out of bounds.")
        if down+self.nelectdown > self.nsites or down+self.nelectdown < 0:
            raise ValueError("Trying to put number of down electrons out of bounds.")
        # Get the eigenvectors
        _, _, vup, vdown = self._eigensystem()
        # We want the amplitude squared of the eigenvectors
        vup = vup ** 2
        vdown = vdown ** 2
        if not self.allow_fractions:
            # Adjust the spin up density.
            if up > 0:
                self.nup += vup[:,self.nelectup:self.nelectup+up].sum(1)
            elif up < 0:
                self.nup -= vup[:,self.nelectup+up:self.nelectup].sum(1)
            # Adjust the spin down density.
            if down > 0:
                self.ndown += vdown[:,self.nelectdown:
                                    self.nelectdown+down].sum(1)
            elif down < 0:
                self.ndown -= vdown[:,self.nelectdown+down:
                                    self.nelectdown].sum(1)
        else:
            # Adjust the spin up densoity.
            fracN = self.nelectup - int(self.nelectup) # fractional occupation of current
            frac = self.nelectup+up - int(self.nelectup+up)# fractional of final.
            if up > 0:
                # Fractional occupation of current state
                self.nup += vup[:,int(self.nelectup)]*min(1-fracN,up)
                if 1-fracN < up:
                    # Still have some up left over.
                    # Add whole states.
                    self.nup += vup[:,int(self.nelectup)+1:int(self.nelectup+up)].sum(1)
                    # Add the remaining fractional part
                    self.nup += vup[:,int(self.nelectup+up)]*frac
            elif up < 0:
                self.nup -= vup[:,int(self.nelectup)]*min(fracN,-up)
                if fracN < -up:
                    self.nup -= vup[:,int(self.nelectup+up)+1:int(self.nelectup)].sum(1)
                    self.nup -= vup[:,int(self.nelectup+up)] * (1-frac)
            # Adjust the spin down density.
            fracN = self.nelectdown - int(self.nelectdown) # fractional occupation of current
            frac = self.nelectdown+down - int(self.nelectdown+down)# fractional of final.
            if down > 0:
                self.ndown += vdown[:,int(self.nelectdown)]*min(1-fracN,down)
                if 1-fracN < down:
                    self.ndown += vdown[:,int(self.nelectdown)+1:
                                        int(self.nelectdown+down)].sum(1)
                    self.ndown += vdown[:,int(self.nelectdown+down)]*frac
            elif down < 0:
                self.ndown -= vdown[:,int(self.nelectdown)]*min(fracN,-down)
                if fracN < -down:
                    self.ndown -= vdown[:,int(self.nelectdown+down)+1:
                                        int(self.nelectdown)].sum(1)
                    self.ndown -= vdown[:,int(self.nelectdown+down)] * (1-frac)
        # Adjust the numbers of electrons.
        self.nelectup += up
        self.nelectdown += down
    #
    ## GETTERS
    #
    def get_kinetic(self,k=None):
        """
        Returns the kinetic energy matrix.

        Inputs: None (k is a dummy variable for compatibility with parent class.)
        Output: (nsites,nsites) ndarray
        Last Modified: 2020-09-17
        """
        return self.kin
    #
    def _eigensystem(self):
        """
        Solves the eigensystem and returns the eigenenergies and states.

        Simplified due to only having a single k-point.
        
        Outputs: eup, edown, vup, vdown
            eup, edown - (nsites,) ndarray of numbers, sorted. Eigenenergies.
            vup, vdown - (nsites,nsites) ndarray of numbers. Eigenvectors.
                v[:,i] is the normalised eigenvector corresponding to the
                eigenvalue e[i], as output by np.linalg.eigh.

        Last Modified: 2020-09-16
        """
        # Get the potential matrices
        potup, potdown, _ = self._potential()
        # Solve for the single-electron energy levels.
        eup, vup = np.linalg.eigh(self.kin + potup)
        edown, vdown = np.linalg.eigh(self.kin + potdown)
        return eup, edown, vup, vdown
    #
    ## PLOTTERS
    #
    #
    ## SETTERS
    #
    def set_kinetic(self):
        """
        Use this method to set self.kin.
        self.kin is a Hermitian (nsites,nsites) ndarray.
        Implementation is lattice-dependent.
        """
        raise NotImplementedError
    #
    def set_kinetic_from_matrix(self,m):
        """
        Primarily for use when you have saved a matrix for later use.

        Inputs: m - array-like, Hermitian matrix, the Kinetic Energy
            and also all other single-particle terms in the Hamiltonian.
        Effect: sets self.kin

        Last Modified; 2020-08-10
        """
        # Check that m is valid.
        m = np.asarray(m)
        # Check that is it the right shape.
        if m.shape != (self.nsites, self.nsites):
            raise ValueError("m is not of the right shape.")
        # Check that it is Hermitian
        if not np.all(np.abs(m - m.conjugate().transpose()) < 1e-14):
            raise ValueError("m is not Hermitian.")
        # Set the matrix
        self.kin = m
    #
    def set_kmesh(self,*args,**kwargs):
        """
        As a Gamma-point only code, this is does nothing much.

        Sets kpoints to 1 and kmesh to be just the Gamma point.
        Complains if you try to make it do anything else.
        Last Modified: 2020-09-17
        """
        # If setting k-mesh to Gamma point only (as in initialisation), be silent.
        # Also be silent if no arguments provided.
        if not ((not args and not kwargs) or np.array_equal(args,[1]*self.dims) or
                np.array_equal(args,[[0]*self.dims])):
            warn("Attempted to change kmesh in Gamma point only implementation. "
                 "Ignoring.")
        self.kpoints = 1
        self.kmesh = np.array([[0]*self.dims])
