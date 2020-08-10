#!/usr/bin/python3

"""
Mean field Hubbard model of Kagome lattice.
Periodic boundary conditions are assumed.

Created: 2020-07-03
Last Modified: 2020-08-04
Author: Bernard Field
"""

import numpy as np
import matplotlib.pyplot as plt
from kagome_ising import kagome_adjacency_tensor
from kagome_ising import kagome_coordinates as kagome_coordinate_tensor
from warnings import warn
from numpy.random import default_rng
import json


# Initialise random number generator.
rng = default_rng()

class ConvergenceWarning(UserWarning):
    """
    Warning to raise when a density mixing algorithm
    does not achieve its convergence target.
    """
    pass
class MixingError(ValueError):
    """
    Exception to raise when a density mixing algorithm
    experiences a fatal error, such as when the algorithm
    (which may have been originally designed for unbounded
    parameters) wants to take the electron density out of
    bounds.
    """
    pass

class KagomeHubbard():
    # Table of Contents:
    #   - IO (ln 48)
    #   - ELECTRON DENSITY (ln 140)
    #   - GETTERS (ln 940)
    #   - PLOTTERS (ln 1325)
    #   - SETTERS (ln 1422)
    #
    ## IO
    #
    def __init__(self,nrows,ncols,u=0,t=1,nup=0,ndown=0,
                 allow_fractions=False,**kwargs):
        """
        Creates a kagome lattice and initialises
        things. nrows and ncols cannot be changed
        later without building a new instance.

        Inputs: nrows - positive integer, number of rows
                in the supercell.
            ncols - positive integer, number of columns in
                the supercell.
            t - number, optional (default 1). Hopping constant.
            u - number, optional (default 0). Hubbard U constant.
            nup,ndown,kwargs - arguments for set_electrons
                Defaults to no electrons. (Do not set backup)
        Last Modified: 2020-07-29
        """
        # Check that nrows and ncols are positive
        if nrows <= 0:
            raise ValueError("Number of rows must be positive.")
        if ncols <= 0:
            raise ValueError("Number of columns must be positive.")
        # Initialise the important constants.
        self.nrows = nrows
        self.ncols = ncols
        self.u = u
        self.allow_fractions = allow_fractions
        self.mag = 0 # Magnetic field
        # Initialise the kinetic energy.
        self.set_kinetic(t)
        # Initialise the electron densities.
        self.set_electrons(nup,ndown,**kwargs,backup=False)
    #
    def copy(self):
        """
        Returns a copy of this kagome lattice object.

        Outputs: KagomeHubbard object.

        Last Modified: 2020-08-04
        """
        # Copy charge density and U.
        kagome = KagomeHubbard(self.nrows,self.ncols,u=self.u,
                               nup=self.nup.copy(),ndown=self.ndown.copy(),
                               allow_fractions=self.allow_fractions)
        # Copy the kinetic energy matrix.
        kagome.set_kinetic_from_matrix(self.kin.copy())
        # Copy magnetic field
        kagome.set_mag(self.mag)
        return kagome
    #
    @classmethod
    def load(cls,f):
        """
        Loads a KagomeHubbard object from a JSON file f.
        Inputs: f - string, filename.
        Outputs: KagomeHubbard object.
        Last Modified: 2020-08-04
        """
        # Load the file
        with open(f) as file:
            di = json.load(file)
        # Create a KagomeHubbard object
        kagome = cls(nrows=di['nrows'],ncols=di['ncols'],
                               u=di['u'],nup=np.asarray(di['nup']),
                               ndown=np.asarray(di['ndown']),
                               allow_fractions=di['allow_fractions'])
        kagome.set_kinetic_from_matrix(np.asarray(di['kin']))
        kagome.set_mag(di['mag'])
        return kagome
    #
    def save(self,f):
        """
        Save a JSON representation of the object's data to file f.
        Inputs: f - string, filename.
        Writes a text file f.
        Last Modified: 2020-08-04
        """
        with open(f,mode='w') as file:
            json.dump({'nrows':self.nrows,
                       'ncols':self.ncols,
                       'u':self.u,
                       'mag':self.mag,
                       'nup':self.nup.tolist(),
                       'ndown':self.ndown.tolist(),
                       'allow_fractions':self.allow_fractions,
                       'kin':self.kin.tolist()},
                      file)
            # Have to use tolist because numpy arrays aren't JSON-able.
    #
    ## ELECTRON DENSITY MODIFIERS
    #
    def anderson_mixing(self,max_iter=100,rdiff=1e-6,mix=0.5,T=None):
        """
        A more elaborate scheme than linear mixing. It adds a sort of
        extra internal linear mixing step with an automatically chosen
        ratio and including information from the previous step.
        Unlike linear mixing, here I minimise the residual instead of
        the energy (although there is nothing stopping me from doing that
        in linear_mixing too).
        See Sec IIB of D.D.Johnson 1988 PRB v38 n 18 and
        Ref's 3 and 4 therein for the algorithm.

        Inputs: max_iter - positive integer, maximum number of iterations.
            rdiff - postivie number, value of the residual at which we should
                stop the cycle.
            mix - number between 0 and 1. Mixing parameter.
            T - optional, non-negative number. Temperature for determining
                occupation of states. If provided, allows fractional occupation
                and exchange between spin up and down channels.
        Last Modified: 2020-08-03
        """
        # Check if we have specified a temperature
        if T is not None:
            fermi = True
            if not self.allow_fractions:
                warn("allow_fractions is currently False. Mixing with "
                     "finite T requires this flag to be enabled. Setting "
                     "allow_fractions to True.")
                self.toggle_allow_fractions(True)
        else:
            fermi = False
        # We have to run an initial step by normal linear mixing.
        if fermi:
            _, nupoutold, ndownoutold = self._eigenstep_finite_T(T)
        else:
            _, nupoutold, ndownoutold = self._eigenstep()
        # Residues
        resold = np.concatenate((nupoutold - self.nup,ndownoutold - self.ndown))
        # If the residue is sufficiently small, we're done.
        if np.linalg.norm(resold) < rdiff:
            return
        # Record old input densities.
        nupinold = self.nup
        ndowninold = self.ndown
        # Do mixing
        self.set_electrons(nup=self.nup*(1-mix)+ nupoutold*mix,
                           ndown=self.ndown*(1-mix) + ndownoutold*mix)
        for i in range(max_iter):
            # Put in electron density, get electron density out.
            if fermi:
                _, nupout, ndownout = self._eigenstep_finite_T(T)
            else:
                _, nupout, ndownout = self._eigenstep()
            # Get residue
            res = np.concatenate((nupout-self.nup,ndownout-self.ndown))
            # If the residue is sufficiently small, we're done.
            if np.linalg.norm(res) < rdiff:
                return
            # Get Anderson mixing parameter
            beta = np.dot(res,res-resold)/(np.dot(res-resold,res-resold))
            # Do the Anderson pre-mixing mixing
            nupinnew = (1-beta)*self.nup + beta*nupinold
            nupoutnew = (1-beta)*nupout + beta*nupoutold
            ndowninnew = (1-beta)*self.ndown + beta*ndowninold
            ndownoutnew = (1-beta)*ndownout + beta*ndownoutold
            #Record input densities.
            nupinold = self.nup
            ndowninold = self.ndown
            # Do main mixing
            nup = nupinnew*(1-mix) + nupoutnew*mix
            ndown = ndowninnew*(1-mix) + ndownoutnew*mix
            # Check validity.
            if nup.max()>1 or nup.min()<0 or ndown.max()>1 or ndown.min()<0:
                # Electron density is out of bounds.
                # Fall back on simple linear mixing.
                nup = self.nup*(1-mix) + nupout*mix
                ndown = self.ndown*(1-mix) + ndownout*mix
            # Apply mixing
            self.set_electrons(nup=nup,ndown=ndown)
            # Record old output densities
            nupoutold = nupout
            ndownoutold = ndownout
            # Record old residue
            resold = res
        # Done, but due to exceeding the maximum iterations.
        warn("Self-consistency not reached after "+str(max_iter)+" steps. "
             "Residual in last step is "+str(np.linalg.norm(res))+".")
        return
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
        
        Last Modified: 2020-07-22
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
        nsites = 3*self.nrows*self.ncols
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
    def linear_mixing(self,max_iter=100,ediff=1e-6,mix=0.5,rdiff=5e-4,T=None,print_residual=False):
        """
        Iterates the Hubbard Hamiltonian towards self-consistency.
        Uses linear density mixing. pnew = (1-mix)*pin + mix*pout

        Inputs: mix - number between 0 and 1. Mixing factor. Default 0.5
            ediff - number. When energy step is less than this, stop.
                Default 1e-6.
            max_iter - integer. Max number of iterations. Default 100.
            rdiff - number. When residual is less than this, stop.
                Both ediff and rdiff must be satisfied.
            T - optional positive number. If set, we fill states using the
                Fermi-Dirac distribution rather than just filling the N
                lowest eigenstates. If set, the number of up and down
                spins is likely to vary, although the total number of
                spins should be constant. Requires allow_fractions=True.
            print_residual - Boolean, default True. Prints the residual
                of the density. You want this to be small.
                Set False if you are looping through stuff to avoid flooding
                stdout.
        Outputs: energy.
        Effects: sets nup and ndown to new values.

        Last Modified: 2020-08-03
        """
        # Check if we have specified a temperature
        if T is not None:
            fermi = True
            if not self.allow_fractions:
                warn("allow_fractions is currently False. Mixing with "
                     "finite T requires this flag to be enabled. Setting "
                     "allow_fractions to True.")
                self.toggle_allow_fractions(True)
        else:
            fermi = False
        # Initialise first energy as a very large number.
        en = 1e12
        for i in range(max_iter):
            # Do a step
            if fermi:
                ennew,nupnew,ndownnew = self._eigenstep_finite_T(T)
            else:
                ennew,nupnew,ndownnew = self._eigenstep()
            # Do mixing
            residual_up = nupnew - self.nup
            residual_down = ndownnew - self.ndown
            self.set_electrons(nup=self.nup*(1-mix) + nupnew*mix,
                               ndown=self.ndown*(1-mix) + ndownnew*mix)
            # Check for convergence
            res = np.linalg.norm(np.concatenate((residual_up,residual_down)))
            de = abs(ennew - en)
            if de < ediff and res < rdiff:
                break
            en = ennew
        # Raise warning if needed
        if de >= ediff or res >= rdiff:
            warn("Self-consistency not reached after "+str(max_iter)+" steps. "
                 "Residual = "+str(res)+", Ediff = "+str(de)+".",
                 ConvergenceWarning)
        if print_residual:
            print("Residual = "+str(res)+", Ediff = "+str(de))
        return en
    #
    def metropolis_mixing(self,beta,max_iter,dn=None,fraction=1):
        """
        Loop for _metropolis_step
        DON'T USE THIS METHOD. IT TURNS OUT TO CONVERGE TOWARDS
        UNPHYSICAL CONFIGURATIONS. This is due to bad assumptions
        on the part of the numerics. Self-consistency turns out to
        be important and I can't rely on the variational principle alone.
        Last Modified: 2020-07-31
        """
        warn("Don't use metropolis_mixing. "
             "It converges to unphysical configurations.",DeprecationWarning)
        if not self.allow_fractions:
            warn("allow_fractions is currently False. metropolis_mixing"
                 " requires this flag to be enabled. Setting allow_fractions"
                 " to True.")
            self.toggle_allow_fractions(True)
        if dn is None:
            dn = 1/(3*self.nrows*self.ncols)
        for _ in range(int(max_iter/2)):
            # Alternate between spin up and spin down.
            nupnew, ndownnew = self._metropolis_step(True,beta,dn,fraction)
            self.set_electrons(nup=nupnew,ndown=ndownnew)
            nupnew, ndownnew = self._metropolis_step(False,beta,dn,fraction)
            self.set_electrons(nup=nupnew,ndown=ndownnew)
    #
    def move_electrons(self,up=0,down=0):
        """
        Adds or subtracts spin up and down electrons, using the heuristic of
        adding/subtracting single energy eigenvalues.
        Because doing this changes the density and thus the Hamiltonian,
        repeated calls to this function will give a different result than
        a single call to this function.

        Inputs: up - integer, default 0. Number of spin up electrons to add.
                If negative, subtracts.
            down - integer, default 0. Number of spin down electrons to add.
                If negative, subtracts.
        Last Modified: 2020-07-22
        """
        # Check that up and down are valid.
        if up+self.nelectup > 3*self.nrows*self.ncols or up+self.nelectup < 0:
            raise ValueError("Trying to put number of up electrons out of bounds.")
        if down+self.nelectdown > 3*self.nrows*self.ncols or down+self.nelectdown < 0:
            raise ValueError("Trying to put number of down electrons out of bounds.")
        # Generate the potential energy.
        potup, potdown, _ = self._potential()
        # Solve for the single-electron energy levels.
        _, vup = np.linalg.eigh(self.kin + potup)
        _, vdown = np.linalg.eigh(self.kin + potdown)
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
    def pulay_mixing(self,max_iter=100,rdiff=1e-3,T=None):
        """
        Performs Direct Inversion of the Iterative Subspace (aka Pulay mixing)
        to find the self-consistent electron density. Linear mixing is
        used for the simple relaxation steps (needed to expand the subspace).
        Should only be used when relatively close to convergence.
        Pulay 1980 Chemical Physics Letters vol 73 is 2 p 393-398
        
        Inputs: max_iter - positive integer, maximum number of iterations.
            rdiff - postivie number, value of the residual at which we should
                stop the cycle.
            T - optional, non-negative number. Temperature for determining
                occupation of states. If provided, allows fractional occupation
                and exchange between spin up and down channels.
        Last Modified: 2020-08-04
        """
        # Check if T is specified.
        if T is not None:
            fermi = True
            if not self.allow_fractions:
                warn("allow_fractions is currently False. Mixing with "
                     "finite T requires this flag to be enabled. Setting "
                     "allow_fractions to True.")
                self.toggle_allow_fractions(True)
        else:
            fermi = False
        condmax = 1e4 # Maximum permissible value of the condition number.
        nsites = 3*self.nrows*self.ncols
        # Initialise our records.
        densities_up = np.empty((0,nsites))
        densities_down = np.empty((0,nsites))
        residues = np.empty((0,2*nsites))
        # Iterate
        for i in range(max_iter):
            # An eigenstep
            if fermi:
                _, nup, ndown = self._eigenstep_finite_T(T)
            else:
                _, nup, ndown = self._eigenstep()
            # Residue
            res = np.concatenate((nup - self.nup, ndown - self.ndown))
            # If the residue is sufficiently small, we're done.
            if np.linalg.norm(res) < rdiff:
                return
            # Record
            densities_up = np.vstack((densities_up,nup))
            densities_down = np.vstack((densities_down,ndown))
            residues = np.vstack((residues,res))
            # Construct the error matrix
            if i==0:
                matrix = np.array([[np.dot(res,res)]])
            else:
                vect = np.dot(residues[0:-1],res)
                ee = np.dot(res,res)
                matrix = np.c_[np.vstack((matrix,vect)),np.r_[vect,ee]]
            # Set up the DIIS equation to solve, in Lagrange multiplier form.
            # Pulay 1980 Eq 6.
            diismat = -np.ones((len(matrix)+1,len(matrix)+1))
            diismat[-1,-1] = 0
            diismat[0:-1,0:-1] = matrix
            # Check the condition number. If too high, we'll need to omit
            # old entries to get back to linear independence.
            if np.linalg.cond(diismat) > condmax:
                diismat = diismat[1:,1:]
                densities_up = densities_up[1:]
                densities_down = densities_down[1:]
                residues = residues[1:]
                matrix = matrix[1:,1:]
            # The vector to equal.
            b = np.zeros(len(diismat))
            b[-1] = -1
            # Solve.
            coef = np.linalg.solve(diismat,b)
            # This vector holds the coefficients, and also at the end the
            # lagrange multiplier which should equal the norm of the residium
            # vector squared.
            # Write new densities.
            nupnew = np.dot(coef[0:-1],densities_up)
            ndownnew = np.dot(coef[0:-1],densities_down)
            # Check validity
            if (nupnew.max()>1 or nupnew.min()<0
                or ndownnew.max()>1 or ndownnew.min()<0):
                raise MixingError("Electron density out of bounds.")
            # Apply mixing.
            self.set_electrons(nupnew,ndownnew)
        # Done, but due to exceeding the maximum iterations.
        warn("Self-consistency not reached after "+str(max_iter)+" steps. "
             "Residual in last step is "+str(np.linalg.norm(res))+".",
             ConvergenceWarning)
        return
    #
    def set_electrons(self,nup=None,ndown=None,backup=True,**kwargs):
        """
        Sets the electron densities.

        Inputs: nup, ndown - either an integer for number of electrons
                or a list-like for the electron density.
                If integer, must be between 0 and number of sites.
                If list-like, must be of right length and have
                values between 0 and 1.
                May also be None, in which case it is not set.
            backup - Boolean, default True. Does backing up of the electron
                density in case of an error. Set to False if you are running
                as part of the initialisation.
        Keyword Arguments:
            method - string. {'uniform', 'random', 'star'}. Specifies
                how to generate the electron density if a number
                of electrons is provided. Ignored if n is a list.
            alpha - optional positive number. Used for random method.
                Dirichlet alpha parameter. Default is chosen automatically
                to be as small as practical.
            points - Boolean, defaalt True. For 'star' method, whether to put
                spin up in the points of the star.
            
        Effects: Sets nup, ndown, nelectup, nelectdown.
            In the event of an Exception, no changes are made.

        Last Modified: 2020-07-15
        """
        # Process keyword arguments. kwargs.get(key,default)
        method = kwargs.pop('method','uniform')
        alpha = kwargs.pop('alpha',None)
        points = kwargs.pop('points',True)
        if len(kwargs) > 0:
            raise TypeError("Got an unexpected keyword argument '"+str(list(kwargs.keys())[0])+"'")
        if backup:
            # Write backups
            nup_backup = self.nup
            nelectup_backup = self.nelectup
            ndown_backup = self.ndown
            nelectdown_backup = self.nelectdown
        try:
            # Set spin up if applicable.
            if nup is not None:
                self.nup,self.nelectup = self._electron_density_single(
                    nup,method,alpha=alpha,points=points)
            # Set spin down if applicable.
            if ndown is not None:
                self.ndown,self.nelectdown = self._electron_density_single(
                    ndown,method,alpha=alpha,points=(not points))
        except:
            if backup:
                # In the event of an error, undo all the changes.
                self.nup = nup_backup
                self.nelectup = nelectup_backup
                self.ndown = ndown_backup
                self.nelectdown = nelectdown_backup
            raise
    #
    def _density_from_vectors(self,evect,nelect):
        """
        Calculates the electron density from the eigenvectors and
        number of electrons. Do this separately for spin up and down.

        Inputs: evect - array, eigenvector output of np.linalg.eigh
            nelect - integer. Number of electrons occupying these states.
        Output: a 1D array of length of each eigenvector, of real non-negative
            numbers, representing the local electron density.

        Last Modified: 2020-07-22
        """
        # Take the absolute value squared of each eigenvector
        # then sum over the bottom nelect eigenvectors.
        if not self.allow_fractions:
            density = np.sum(abs(evect[:,0:nelect])**2,1)
        else:
            density = np.sum(abs(evect[:,0:int(nelect)])**2,1)
            # Get the fractional part of the occupations.
            frac = nelect - int(nelect)
            # Add energy from fractionally occupied states.
            density += abs(evect[:,int(nelect)])**2 * frac
        return density
    #
    def _eigenstep(self):
        """
        Takes the electron densities and the parameters needed to
        make the Hamiltonian, then returns the energy and the
        electron densities predicted by the eigenvectors.
        Assumes the total number of up and down electrons stays
        constant.
        The idea is that you'll use the output densities to mix with
        the initial densities then use as new input densities in a
        self-consistency loop.
        Let M = 3*nrows*ncols, the number of sites.

        Presets: nup, ndown, u, kin, nelectup, nelectdown
        Outputs: energy (real number), nup, ndown

        Last Modified: 2020-07-22
        """
        # Generate the potential energy.
        potup, potdown, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, vup = np.linalg.eigh(self.kin + potup)
        edown, vdown = np.linalg.eigh(self.kin + potdown)
        # Calculate the energy by filling electrons.
        energy = self._energy_from_states(eup,edown,offset)
        # Calculate the new densities.
        nupnew = self._density_from_vectors(vup,self.nelectup)
        ndownnew = self._density_from_vectors(vdown,self.nelectdown)
        # Return
        return energy, nupnew, ndownnew
    #
    def _eigenstep_finite_T(self,T):
        """
        Takes the electron densities and the parameters needed to
        make the Hamiltonian, then returns the energy and the
        electron densities predicted by the eigenvectors.
        Fills the eigenstates by assuming a Fermi-Dirac distribution
        with temperature T.

        Inputs: T - positive number, temperature for Fermi Dirac distribution.
        Outputs: energy, nup, ndown

        Last Modified: 2020-07-29
        """
        # Generate the potential energy.
        potup, potdown, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, vup = np.linalg.eigh(self.kin + potup)
        edown, vdown = np.linalg.eigh(self.kin + potdown)
        # Get chemical potential
        energies = np.sort(np.concatenate((eup,edown)))
        mu = self._chemical_potential_from_states(T,self.nelectup+self.nelectdown,energies)
        # Calculate energy
        en = self._energy_from_states(eup,edown,offset,T,mu)
        # Get new electron densities.
        weightup = fermi_distribution(eup,T,mu)
        weightdown = fermi_distribution(edown,T,mu)
        nup = np.sum(weightup*abs(vup)**2,1)
        ndown = np.sum(weightdown*abs(vdown)**2,1)
        # Return
        return en, nup, ndown
    #
    def _electron_density_single(self,n,method,**kwargs):
        """
        Helper method for set_electrons, so I don't have to
        duplicate my code for the spin up and down cases.

        Inputs: n - either an integer for number of electrons
                or a list-like for the electron density.
                If integer, must be between 0 and number of sites.
                If list-like, must be of right length and have
                values between 0 and 1.
            method - string. {'uniform', 'random', 'star'}. Specifies
                how to generate the electron density if a number
                of electrons is provided. Ignored if n is a list.
        Keyword Arguments:
            alpha - optional positive number. Used for random method.
                Dirichlet alpha parameter. Default is chosen automatically.
            points - Boolean. Used for star method. Determines whether
                to put the density in the points of the stars first.
        Outputs: density - ndarray of shape (3*nrows*ncols,)
            nelect - integer.
        Last Modified; 2020-07-22
        """
        # Process kwargs
        alpha = kwargs.pop('alpha',None)
        points = kwargs.pop('points',True)
        if len(kwargs) > 0:
            raise TypeError("Got an unexpected keyword argument '"+
                            str(list(kwargs.keys())[0])+"'")
        # A useful constant
        nsites = 3*self.nrows*self.ncols # Number of sites.
        try:
            # First, we need to determine if n is the number of
            # electrons or a list representing the electron density.
            len(n)
        except TypeError:
            # n is the number of electrons.
            if not self.allow_fractions:
                nelect = int(round(n))
                # Check that n was at least close to an integer.
                if abs(nelect - n) > 1e-10:
                    warn("Number of electrons "+str(sum(n))+
                         " is not an integer. Rounding to "+str(nelect)+".")
            else:
                nelect = n
            # Generate an electron density by the appropriate method.
            if method == 'uniform':
                density = nelect/nsites * np.ones(nsites)
            elif method == 'random':
                density = random_density(nsites,nelect,alpha)
            elif method == 'star':
                density = self._electron_density_star(nelect,points)
            else:
                raise ValueError("Method "+str(method)+" does not exist.")
        else:
            # n is the electron density.
            density = np.asarray(n)
            if len(density) != nsites:
                raise ValueError("n has the wrong length.")
            if not self.allow_fractions:
                nelect = int(round(sum(n)))
                if abs(nelect - sum(n)) > 1e-10:
                    warn("Number of electrons "+str(sum(n))+
                         " is not an integer. Rounding to "+str(nelect)+".")
            else:
                nelect = sum(n)
        # Check that values are within bounds.
        if nelect < 0 or nelect > nsites:
            raise ValueError("The number of electrons is out of bounds.")
        if density.min() < 0 or density.max() > 1:
            raise ValueError("The electron density is out of bounds.")
        return density, nelect
    #
    def _electron_density_star(self,n,points):
        """
        Produces a star-like configuration of electron density.
        I found this to be the ground state of the Kagome Hubbard
        lattice for a good part of the parameter space.

        Inputs: n - integer, number of electrons.
            points - Boolean, whether to put most of the density in
                the points of the stars (True) or the inner bits (False).
        Output: 1D ndarray representing electron density.
        Last Modified: 2020-07-15
        """
        # Get the coordinates of the points of the star.
        rowcolmgrid = np.mgrid[0:self.nrows,0:self.ncols]
        # In the rows and columns, which sublattice site has the star point?
        ncoord = np.mod(rowcolmgrid[0] + 2*rowcolmgrid[1],3)
        # Convert to numerical coordinates.
        pcoords = (ncoord + 3*rowcolmgrid[0] + 3*self.nrows*rowcolmgrid[1]).flatten()
        # A useful constant: number of unit cells
        ncells = self.nrows*self.ncols
        # Four cases, bases on whether to fill points first, and the filling.
        if points:
            if n <= self.nrows*self.ncols: # One third filling or less
                density = np.zeros(3*ncells)
                density[pcoords] = n/ncells
                # Just fill the points of the stars.
            else:
                # Put overflow from star points into other points.
                density = np.ones(3*ncells) * (n/(2*ncells)-1/2)
                density[pcoords] = 1
        else:
            if n <= 2*self.nrows*self.ncols: # Two thirds filling or less
                # Leave the star points empty
                density = np.ones(3*ncells) * (n/(2*ncells))
                density[pcoords] = 0
            else:
                # Put overflow into the star points
                density = np.ones(3*ncells)
                density[pcoords] = n/ncells - 2
        assert abs(density.sum() - n) < 1e-14
        return density
    #
    def _metropolis_step(self,do_up,beta,dn,fraction=1,print_pert=False):
        """
        Performs a single step of the Metropolis algorithm,
        for searching for a new density.
        Sets allow_fractions to True.
        Does only one spin sector at a time, because to first order
        same spins don't interact with each other.
        
        Inputs: do_up - boolean, whether to do spin up or down sector.
            beta - positive number, the 'inverse temperature'. Smaller
                numbers increase the chance of stepping to a higher energy
                state.
            dn - number, preferably small (i.e. dn << 1). Maximum amount
                the density at a single site is allowed to change.
                In practice, this can be no greater than 1, because that
                is the maximum amount of density available to change.
            fraction - number, (0,1]. Fraction of sites to test density
                shifts on (determined randomly). 1 means all states. 0 means
                no states.
        Outputs: spin_up_density, spin_down_density

        Last Modified: 2020-07-24
        """
        if not self.allow_fractions:
            raise ValueError("allow_fractions flag is False. Cannot proceed.")
        # We will employ the Metropolis Algorithm and some perturbation theory.
        nsites = 3*self.nrows*self.ncols
        # Get a random distribution of shifts of up to +/- dn on each site.
        shift = 2*dn*rng.random(nsites)-dn
        # Clamp the values such that the result will be between 0 and 1.
        if do_up:
            shift = np.fmin(np.fmax(shift,-self.nup),1-self.nup)
        else:
            shift = np.fmin(np.fmax(shift,-self.ndown),1-self.ndown)
        # Zero out 1-fraction of the shifts.
        shift[rng.random(nsites) > fraction] = 0
        # Because, to first order, density perturbations are local only,
        # we can treat the shift at each site independently.
        # Calculate the energy shift at each site by 1st order
        # perturbation theory.
        # We'll need the eigenspectrum.
        # Generate the potential energy.
        potup, potdown, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, vup = np.linalg.eigh(self.kin + potup)
        edown, vdown = np.linalg.eigh(self.kin + potdown)
        # Sources of energy change:
        # 1) Band energy shifts of the opposite spin.
        # 2) Total energy shift from interaction between
        #    shift and opposite spin density.
        # 3) Change in occupation of same spin bands.
        # 1) Band shift
        if do_up:
            evect = vdown
            nelect = self.nelectdown
            density = self.ndown
        else:
            evect = vup
            nelect = self.nelectup
            density = self.nup
        # Get the 1st order energy shift at each lattice site
        # for each eigenvector.
        vshift = self.u*(shift[:,np.newaxis])*abs(evect)**2
        # Sum over the occupied eigenvectors
        pertE = np.sum(vshift[:,0:int(nelect)],1)
        # Get the fractional part of the occupations.
        frac = nelect - int(nelect)
        # Add energy from fractionally occupied states.
        pertE += vshift[:,int(nelect)] * frac
        # 2) Total energy shift
        pertE -= self.u * shift * density
        # 3) Change in occupation
        if do_up:
            eigs = eup
            nelect = self.nelectup
        else:
            eigs = edown
            nelect = self.nelectdown
        # fractional part of the current occupation
        frac = nelect - int(nelect)
        # Change in occupation of current band
        pertE += np.fmin(np.fmax(shift,-frac),1-frac)*eigs[int(nelect)]
        # Change in occupation of the band above
        if int(nelect)+1 < nsites:
            pertE += np.fmax(shift+frac-1,0) * eigs[int(nelect)+1]
        # Change in occupation of the band below
        if int(nelect)-1 >= 0:
            pertE += np.fmin(shift+frac,0) * eigs[int(nelect)-1]
        # We now have the perturbation to energy from shifts in density
        # at each site.
        # Convert it into Metropolis probabilities for transitions
        prob = np.exp(-pertE * beta)
        # Now we use those probabilities to randomly exclude some shifts.
        r = rng.random(nsites)
        shift[r > prob] = 0 #i.e. keep all r < prob
        pertE[r > prob] = 0
        if print_pert: print("Perturbation energy shift: "+str(pertE.sum()))
        # Apply the shift.
        if do_up:
            return self.nup+shift, self.ndown
        else:
            return self.nup, self.ndown+shift
    #
    ## GETTERS
    #
    def chemical_potential(self,T,N=None):
        """
        Determines what the chemical potential should be
        to yield the desired number of electrons at a
        given temperature.

        Inputs: T - nonnegative number, temperature.
                If T==0, simply returns the energy of the
                highest occupied state.
            N - number of electrons. Defaults to however
                many the system currently has.
        Output: the chemical potential, a number
        Last Modified: 2020-07-29
        """
        if T < 0:
            raise ValueError("T cannot be negative.")
        if N is None:
            N = self.nelectup + self.nelectdown
        # Get the eigenenergy spectrum.
        # Generate the potential energy.
        potup, potdown, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, _ = np.linalg.eigh(self.kin + potup)
        edown, _ = np.linalg.eigh(self.kin + potdown)
        # Consolidate and sort the eigenenergies
        energies = np.sort(np.concatenate((eup,edown)))
        return self._chemical_potential_from_states(T,N,energies)
    #
    def density_of_states(self,sigma,de,emin=None,emax=None):
        """
        Compute density of states with Gaussian smearing.

        Inputs: sigma - positive number, Gaussian smearing width.
            de - positive number, step-size for energy plotting.
            emin - number, optional. Minimum energy to plot.
                Defaults to minimum eigenvalue - 3 sigma.
            emax - number, optional. Maximum energy to plot.
                Defaults to maximum eigenvalue + 3 sigma.
        Outputs: Three numerical 1D ndarrays of equal length.
            energy - the energy axis.
            dosup - the DOS of the spin up states.
            dosdown - the DOS of the spin down states.

        Last Modified: 2020-07-14
        """
        # Get the eigenenergy spectrum.
        # Generate the potential energy.
        potup, potdown, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, _ = np.linalg.eigh(self.kin + potup)
        edown, _ = np.linalg.eigh(self.kin + potdown)
        # Determine energy bounds if not specified.
        if emin is None:
            emin = min(eup.min(),edown.min()) - 3*sigma
        if emax is None:
            emax = max(eup.max(),edown.max()) + 3*sigma
        # Get the energy axis.
        energy = np.arange(emin,emax+1e-15,de) # +epsilon to include endpoint.
        # Gaussian smearing on energy levels gives DOS.
        # Get coordinate grids.
        energymesh, eupmesh = np.meshgrid(energy, eup, sparse=True)
        _, edownmesh = np.meshgrid(energy, edown, sparse=True)
        # Do Gaussian smearing.
        dosup = (np.exp(-(energymesh - eupmesh)**2 / (2*sigma))/
                 (sigma*np.sqrt(2*np.pi))).sum(axis = 0)
        dosdown = (np.exp(-(energymesh - edownmesh)**2 / (2*sigma))/
                 (sigma*np.sqrt(2*np.pi))).sum(axis = 0)
        # Return
        return energy, dosup, dosdown
    #
    def eigenstates(self,mode='states',emin=None,emax=None):
        """
        Returns a selection of band-decomposed electron densities.
        Defaults to returning all of them.

        Inputs: mode - string. How to interpret emin and emax.
                'states': as 1-based indices.
                'energy': as energies.
                'fermi': as energies relative to the Fermi level.
            emin - number. Minimum band to plot.
            emax - number. Maximum band to plot.
        Outputs:
            List of eigenenergies of spin up states.
            List of electron densities of spin up states.
            List of eigenenergies of spin down states.
            List of electron densities of spin down states.

        Last Modified: 2020-07-22
        """
        # Get eigenstates
        # Generate the potential energy.
        potup, potdown, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, vup = np.linalg.eigh(self.kin + potup)
        edown, vdown = np.linalg.eigh(self.kin + potdown)
        # Convert emin and emax to indices
        if mode=='states': # emin and emax are indices, 1-based indexing.
            if emin is None:
                emin = 1
            if emax is None:
                emax = 3*self.nrows*self.ncols
            iminup = int(emin-1)
            imaxup = int(emax-1)
            imindown = int(emin-1)
            imaxdown = int(emax-1)
        elif mode=='energy': # emin and emax are energies.
            if emin is None:
                emin = min(eup.min(),edown.min())
            if emax is None:
                emax = max(eup.max(),edown.max())
            iminup = (eup >= emin).nonzero()[0][0]
            imaxup = (eup <= emax).nonzero()[0][-1]
            imindown = (edown >= emin).nonzero()[0][0]
            imaxdown = (edown <= emax).nonzero()[0][-1]
        elif mode=='fermi': # they are energies relative to the Fermi energy.
            fermiup, fermidown = self.fermi()
            if emin is None:
                emin = min(eup.min()-fermiup,edown.min()-fermidown)
            if emax is None:
                emax = max(eup.max()-fermiup,edown.max()-fermidown)
            iminup = (eup >= (emin+fermiup)).nonzero()[0][0]
            imaxup = (eup <= (emax+fermiup)).nonzero()[0][-1]
            imindown = (edown >= (emin+fermidown)).nonzero()[0][0]
            imaxdown = (edown <= (emax+fermidown)).nonzero()[0][-1]
        else:
            raise ValueError('Mode '+str(mode)+' is not recognised.')
        return (eup[iminup:imaxup+1],
                abs(vup.transpose()[iminup:imaxup+1])**2,
                edown[imindown:imaxdown+1],
                abs(vdown.transpose()[imindown:imaxdown+1])**2)
    #
    def energy(self,T=None):
        """
        Calculates the energy from the current electron density.
        Does not go through self-consistency.

        Input: T - optional positive number. If provided,
            energy is calculated with a Fermi distribution
            for the occupancies.
        Output: energy, a number.

        Last Modified: 2020-07-21
        """
        # Generate the potential energy.
        potup, potdown, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, _ = np.linalg.eigh(self.kin + potup)
        edown, _ = np.linalg.eigh(self.kin + potdown)
        if T is not None:
            mu = self.chemical_potential(T)
        else:
            mu = None
        return self._energy_from_states(eup,edown,offset,T=T,mu=mu)
    #
    def fermi(self, midpoint=False):
        """
        Calculates the Fermi level for the single particle states.
        Really, it just returns the energy of the highest occupied states.
        If midpoint=True, it returns the energy in between the highest occupied
        and lowest unoccupied states.

        Outputs: two numbers, the Fermi level for the spin up and spin down states.

        Last Modified: 2020-07-29
        """
        # Generate the potential energy.
        potup, potdown, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, _ = np.linalg.eigh(self.kin + potup)
        edown, _ = np.linalg.eigh(self.kin + potdown)
        if self.allow_fractions:
            Nup = int(round(self.nelectup))
            Ndown = int(round(self.nelectdown))
        else:
            Nup = self.nelectup
            Ndown = self.nelectdown
        # Check the highest occupied states for the Fermi.
        # Get spin up
        if Nup == 0:
            fermi_up = eup[0]
        elif midpoint:
            fermi_up = (eup[Nup] + eup[Nup-1])/2
        else:
            fermi_up = eup[Nup-1]
        # get spin down
        if Ndown == 0:
            fermi_down = edown[0]
        elif midpoint:
            fermi_down = (edown[Ndown] + edown[Ndown-1])/2
        else:
            fermi_down = edown[Ndown-1]
        # Return fermi
        return fermi_up, fermi_down
    #
    def get_charge_density(self):
        """
        Returns the charge density.
        Output: (3*nrows*ncols,) ndarray.
        Last Modified: 2020-07-13
        """
        return self.nup + self.ndown
    #
    def get_electron_number(self):
        """
        Returns the number of electrons.
        Output: Number
        Last Modified: 2020-07-29
        """
        return self.nelectup + self.nelectdown
    #
    def get_kinetic(self):
        """
        Returns the kinetic energy matrix.
        Output: (3*nrows*ncols,3*nrows*ncols) ndarray
        Last Modified: 2020-07-15
        """
        return self.kin
    #
    def get_magnetization(self):
        """
        Returns the net spin
        Output: Number, the net spin.
        Last Modified: 2020-07-20
        """
        return self.nelectup - self.nelectdown
    #
    def get_spin_density(self):
        """
        Returns the spin density.
        Output: (3*nrows*ncols,) ndarray.
        Last Modified: 2020-07-13
        """
        return self.nup - self.ndown
    #
    def local_magnetic_moment(self):
        """
        Returns the average value of the local magnetic moment squared.
        Output: Number
        Last Modified: 2020-07-03
        """
        return (self.get_spin_density()**2).mean()
    #
    def residual(self,T=None):
        """
        Returns the residual of the electron density,
        which is a key convergence figure.
        Self-consistency has been reached if this number
        is 'small', ideally 0.

        Input: T - optional number. Temperature for eigenstep.
        Outputs: residual , a non-negative number

        Last Modified: 2020-08-06
        """
        # Get densities after one iteration.
        if T is None:
            _, nupnew, ndownnew = self._eigenstep()
        else:
            _, nupnew, ndownnew = self._eigenstep_finite_T(T)
        # Calculate residual squared.
        residual_up = nupnew - self.nup
        residual_down = ndownnew - self.ndown
        res = np.linalg.norm(np.concatenate((residual_up,residual_down)))
        return res
    #
    def _chemical_potential_from_states(self,T,N,energies):
        """
        Determines what the chemical potential should be
        to yield the desired number of electrons at a
        given temperature.

        Inputs: T - nonnegative number, temperature.
                If T==0, simply returns the energy of the
                highest occupied state.
            N - number of electrons.
            energies - sorted list/ndarray of all eigenenergies.
        Output: the chemical potential, a number
        Last Modified: 2020-07-29
        """
        if T < 0:
            raise ValueError("T cannot be negative.")
        # Check impossible cases
        if N > len(energies) or N < 0:
            raise ValueError("N is out of bounds.")
        # Special case: T=0
        if T==0:
            # At zero temperature, we cannot have fractional
            # occupation. N must be an integer.
            N = int(round(N))
            # Get the N'th eigenenergy.
            mu = energies[N-1]
        else:
            # Get a first guess of the chemical potential.
            mu = energies[int(round(N))-1]
            nguess = fermi_distribution(energies,T,mu).sum()
            # Get a good initial step-size
            # Should either be comparable to the energy gap or
            # the temperature.
            ediff = np.diff(energies)
            i = int(round(N)) - 1 # index
            if i >= len(ediff):
                step = max(T,ediff[-1])
            elif i <= 0:
                step = max(T,ediff[0])
            else:
                step = max(T,ediff[i],ediff[i-1])
            # Perform a binary search to find mu.
            first_sweep = True
            start_low = (N < nguess)
            while abs(N-nguess) > 1e-12:
                # Check if we have found a cross-over point.
                if first_sweep:
                    if (start_low and N > nguess) or (not start_low and N < nguess):
                        first_sweep = False
                if not first_sweep:
                    step /= 2 # Halve step-size for binary search.
                # Increment chemical potential in correct direction.
                if N > nguess:
                    mu += step
                else:
                    mu -= step
                # Recalculate our guess at the chemical potential
                nguess = fermi_distribution(energies,T,mu).sum()
        return mu
    #
    def _energy_from_states(self,eup,edown,offset,T=None,mu=None):
        """
        Inputs:
            eup - ndarray of spin up eigenstates.
            edown - ndarray of spin down eigenstates
            offset - number, energy offset from _potential
            T - number, optional. Temperature for Fermi distribution
                If not provided, just fill lowest energy states
                for each spin sector.
            mu - number, optional. Chemical potential. Must be provided
                if T is provided.
        Last Modified: 2020-07-22
        """
        if T is None:
            # Calculate the energy by filling electrons.
            if not self.allow_fractions:
                # Occupation is strictly integers.
                en = (sum(eup[0:self.nelectup]) + sum(edown[0:self.nelectdown])
                      + offset)
            else:
                # Occupation may be fractional.
                en = (sum(eup[0:int(self.nelectup)])
                      + sum(edown[0:int(self.nelectdown)]) + offset)
                # Get the fractional part of the occupations.
                fracup = self.nelectup - int(self.nelectup)
                fracdown = self.nelectdown - int(self.nelectdown)
                # Add energy from fractionally occupied states.
                if fracup > 0: en += eup[int(self.nelectup)] * fracup
                if fracdown > 0: en += edown[int(self.nelectdown)] * fracdown
        else:
            # Calculate energies by weighting states by Fermi distribution
            if mu is None:
                raise TypeError("If T is provided, mu must also be provided.")
            energies = np.concatenate((eup,edown))
            en = np.sum(energies * fermi_distribution(energies, T, mu)) + offset
        return en
    #
    def _potential(self):
        """
        Creates the mean-field Hubbard potential part of
        the kagome Hamiltonian matrix.
        Give it the densities/occupations of up and down electrons,
        and it will give the Hubbard potential for up and down electrons.

        Presets: u, nup, ndown. mag
        Outputs: potup, potdown, offset
            two square diagonal ndarrays, of dimension equal to nup, and a number.
            potup is potential for spin up electrons.
            potdown is potential for spin down electrons.
            offset is a real number, to be added to the total energy.

        Last Modified: 2020-07-29
        """
        # V = U(nup<ndown> + ndown<nup> - <nup><ndown>)
        # Also, the magnetic field shifts the energy of up and down electrons.
        # We have <nup> and <ndown>
        nsites = 3 * self.nrows * self.ncols
        offset = -self.u * np.sum(self.nup*self.ndown) # -u <nup><ndown>
        potup = self.u*np.diag(self.ndown) - self.mag * np.eye(nsites)
        potdown = self.u*np.diag(self.nup) + self.mag * np.eye(nsites)
        return potup, potdown, offset
    #
    ## PLOTTERS
    #
    def plot_spin(self,marker_scale=1):
        """
        Plots the spin density.
        Size/area of marker is proportional to magnitude.
        Yellow is spin up. Blue is spin down.

        Inputs: marker_scale - number, optional. Factor to scale markersize by.
        Effects: Makes a plot.

        Last Modified: 2020-07-10
        """
        # Get the Cartesian coordinates of each point.
        coords = kagome_coordinates(self.nrows,self.ncols)
        x = coords[:,0]
        y = coords[:,1]
        # Get the spin moment.
        spin = self.nup - self.ndown
        # Convert sign of spin to colour.
        color_up = 'y'
        color_down = 'b'
        spin_col = [ color_down if x<0 else color_up for x in np.sign(spin) ]
        # Default marker size is 6. Default marker area is 6**2=36.
        plt.scatter(x,y,abs(spin)*36*(marker_scale**2),spin_col)
        plt.gca().set_aspect('equal')
        plt.show()
    #
    def plot_charge(self,marker_scale=1):
        """
        Plots the charge density.
        Marker size/area is proportional to the charge density.

        Inputs: marker_scale - number, optional. Factor to scale markersize by.
        Effect: Makes a plot.

        Last Modified: 2020-07-10
        """
        # Get the Cartesian coordinates of each point.
        coords = kagome_coordinates(self.nrows,self.ncols)
        x = coords[:,0]
        y = coords[:,1]
        # Get the charge density.
        chg = self.nup + self.ndown
        # Default marker size is 6. Default marker area is 6**2=36.
        plt.scatter(x,y,chg*36*(marker_scale**2))
        plt.gca().set_aspect('equal')
        plt.show()
    #
    def plot_DOS(self,sigma,de,emin=None,emax=None,midpoint=True):
        """
        Plots the density of states. See density_of_states for arguments.
        Also shows the Fermi levels, although this Fermi level does
        not fully account for the broadening.

        Last Modified: 2020-07-10
        """
        # Calculate the DOS.
        energy, dosup, dosdown = self.density_of_states(sigma,de,emin,emax)
        # Calculate the Fermi level.
        fermi_up, fermi_down = self.fermi(midpoint)
        # Plot the DOS.
        plt.plot(energy, dosup, 'k', energy, -dosdown, 'k') # Black lines
        plt.xlabel('Energy')
        plt.ylabel('DOS')
        # Plot the Fermi
        plt.vlines(fermi_up,0,dosup.max(),colors='b')
        plt.vlines(fermi_down,-dosdown.max(),0,colors='b')
        # Show the plot.
        plt.show()
    #
    def plot_spincharge(self,marker_scale=1):
        """
        Plots the spin and charge density.
        Marker size/area is proportional to the charge density.
        Marker colour is proportional to the spin density (yellow up, blue down).

        Inputs: marker_scale - number, optional. Factor to scale markersize by.
        Effect: Makes a plot

        Last Modified: 2020-07-10
        """
        # Get the Cartesian coordinates of each point.
        coords = kagome_coordinates(self.nrows,self.ncols)
        x = coords[:,0]
        y = coords[:,1]
        # Get the charge density.
        chg = self.nup + self.ndown
        # Get the spin moment.
        spin = self.nup - self.ndown
        # Plot
        plt.scatter(x,y,chg*36*(marker_scale**2),spin,cmap="BrBG_r",vmin=-1,vmax=1)
        plt.gca().set_aspect('equal')
        cb = plt.colorbar()
        cb.ax.set_title('Spin')
        plt.show()
    #
    ## SETTERS
    #
    def set_kinetic(self,t):
        """
        Creates the kinetic energy part of the kagome
        Hamiltonian matrix.
        Coordinates are sublattice+row*3+col*nrows.

        Inputs: t - real number. Hopping constant.
        Effect: sets self.kin to a 3*nrows*ncols dimensional square ndarray.

        Last Modified: 2020-07-10
        """
        # Have the kinetic energy term in tensor form,
        # where sublattice site, row and column are separate indices.
        tensor = -t*kagome_adjacency_tensor(self.nrows,self.ncols)
        # Reshape to a square matrix.
        length = 3*self.nrows*self.ncols
        self.kin = np.reshape(tensor,(length,length),order='F')
    #
    def set_kinetic_from_matrix(self,m):
        """
        Primarily for use when you have saved a matrix for later use.

        Inputs: m - array-like, Hermitian matrix, the Kinetic Energy
            and also all other single-particle terms in the Hamiltonian.
        Effect: sets self.kin

        Last Modified; 2020-07-15
        """
        # Check that m is valid.
        m = np.asarray(m)
        # Check that is it the right shape.
        if m.shape != (3*self.nrows*self.ncols, 3*self.nrows*self.ncols):
            raise ValueError("m is not of the right shape.")
        # Check that it is Hermitian
        if not np.all(np.abs(m - m.conjugate().transpose()) < 1e-14):
            raise ValueError("m is not Hermitian.")
        # Set the matrix
        self.kin = m
    #
    def set_kinetic_random(self,t,wt=0,we=0):
        """
        Creates a kinetic energy matrix with some random noise in it.
        Random noise is uniform. Noise of hopping constants centred on
        t and noise of on-site energy centred on 0.

        Inputs: t - number. Hopping constant.
            wt - number. Width of the random noise of hopping.
            we - number. Width of the random noise of on-site energy.
        Effects: sets self.kin

        Last Modified: 2020-07-15
        """
        # Get the regular kinetic energy set.
        self.set_kinetic(t)
        # Get some random noise for the whole matrix.
        noise = rng.random(self.kin.shape)*wt - wt/2
        # Zero out elements which should be zero.
        noise[self.kin == 0] = 0
        # Make the noise matrix Hermitian.
        # Zero out the lower triangle, otherwise next step will be sum of two random
        # variates which is not a uniform distribution.
        noise = np.triu(noise) 
        noise = noise + noise.transpose()
        # Apply noise to the kinetic energy.
        self.kin += noise
        # Now get the noisy diagonal for on-site energy.
        self.kin += np.diag(rng.random(len(self.kin))*we - we/2)
        # Done.
    #
    def set_mag(self,mag):
        """
        Sets the magnetic field. +ve favours spin up.
        Input: mag - real number
        Effect: sets self.mag to mag
        Last Modified: 2020-07-29
        """
        self.mag = mag
    #
    def set_u(self,u):
        """
        Sets the Hubbard U parameter.

        Input: u - real number.
        Effect: Sets self.u to u.

        Last Modified: 2020-07-10
        """
        self.u = u
    #
    def toggle_allow_fractions(self,val=None):
        """
        Toggles the flag which allows the number of electrons
        to be fractional. Handles converting the number of
        electrons back to an integer if necessary.

        Inputs: val - optional Boolean. Sets allow_fractions
            to this value if provided. Toggles allow_fractions otherwise.
        Output: Boolean, new value of allow_fractions.

        Last Modified: 2020-07-24
        """
        if val is None:
            # If val not set, we toggle.
            val = not allow_fractions
        if val:
            # Setting to True is easy. Nothing to be done.
            self.allow_fractions = True
        else:
            # Setting to False is harder. Got to make sure
            # the electrons are all proper integers.
            nelectup_old = self.nelectup
            nelectdown_old = self.nelectdown
            # Set the number of electrons to integers.
            self.nelectup = int(round(self.nelectup))
            self.nelectdown = int(round(self.nelectdown))
            # Rescale the electron densities to match.
            if nelectup_old > 0:
                self.nup *= self.nelectup/nelectup_old
            if nelectdown_old > 0:
                self.ndown *= self.nelectdown/nelectdown_old
            # Ensure no values exceed 1.
            while np.any(self.nup > 1):
                warn("New electron occupation exceeds 1. Reducing "
                     "number of spin up electrons.")
                self.nelectup -= 1
                self.nup *= self.nelectup/(self.nelectup+1)
            while np.any(self.ndown > 1):
                warn("New electron occupation exceeds 1. Reducing "
                     "number of spin down electrons.")
                self.nelectdown -= 1
                self.ndown *= self.nelectdown/(self.nelectdown+1)
            # Now toggle the flag.
            self.allow_fractions = False
        return self.allow_fractions
    #




def fermi_distribution(e,T,mu):
    """
    The Fermi distribution.
    Allows numpy array input for e and mu.
    Handles T=0, assuming e=mu should return 1.
    Inputs: e - energy
        T - temperature
        mu - chemical potential
    Output: scalar between 0 and 1 if inputs are scalar.
        ndarray of scalars matching input size if otherwise.
    Last Modified: 2020-07-23
    """
    if T==0:
        # T==0 case is a step function.
        comp = e<=mu # E below mu is 1, above is 0.
        if isinstance(comp,bool):
            # Inputs were all scalar.
            return int(comp)
        else:
            # Inputs are not scalar; probably a list.
            return np.asarray(comp).astype(int)
    else:
        # Generic T case.
        return 1/(np.exp((e-mu)/T)+1)


def simulate(nrows,ncols,nelectup,nelectdown,u,**kwargs):
    """
    Sets up and does a Hubbard calculation on a kagome lattice.

    Inputs: nrows - positive integer. Number of rows in the supercell.
        ncols - positive integer. Number of columns in the supercell.
        nelectup - non-negative integer. Number of spin up electrons.
            Must be no greater than 3*nrows*ncols.
        nelectdown - non-negative integer. Number of spin down electrons.
            Must be no greater than 3*nrows*ncols.
        u - real number. Hubbard U parameter.
    Keyword Arguments:
        t - real number, default 1. Hopping parameter.
        scheme - string, default 'linear'. Mixing scheme.
            Currently, only linear mixing is implemented.
        mix - number between 0 and 1, default 0.5. Linear mixing parameter.
        ediff - positive number, default 1e-6. Energy difference below which
            we assume the self-consistency cycle has converged.
        max_iter - positive integer, default 500. Maximum number of iterations
            of self-consistency to perform.
        initial - string, default 'random'. Determines how the electron density
            is initialised. Options are 'random' and 'uniform'.
        alpha - optional positive number, default None. For 'random' initial.
            Dirichlet parameter for random generation. If not set, is
            automatically chosen for optimal spread.

    Outputs: en, nup, ndown
        en - real number. The energy of the system.
        nup - (3*nrows*ncols,) ndarray of floats between 0 and 1.
            Site-occupation (or density) of spin up electrons.
            Should sum to nelectup.
        ndown - as nup, but for spin down electrons.
    
    Last Modified: 2020-07-10
    """
    # Process default values. kwargs.get(key,default)
    # t, hopping parameter.
    t = kwargs.get('t',1)
    # Mixing scheme.
    scheme = kwargs.get('scheme','linear')
    # Parameters for mixing scheme.
    if scheme == 'linear':
        mix = kwargs.get('mix',0.5)
        ediff = kwargs.get('ediff',1e-6)
        max_iter = kwargs.get('max_iter',500)
    else:
        raise ValueError("Mixing scheme "+str(scheme)+" is not implemented.")
    # Density initialisation scheme
    initial = kwargs.get('initial','random')
    if initial == 'random':
        alpha = kwargs.get('alpha',None)
    # Input processing complete.
    # Create the object
    kagome = KagomeHubbard(nrows,ncols,t=t,u=u,nup=nelectup,ndown=nelectdown,
                           method=initial,alpha=alpha)
    # Run the simulation
    if scheme == 'linear':
        kagome.linear_mixing(max_iter,ediff,mix)
    else:
        raise ValueError("Mixing scheme "+str(scheme)+" is not implemented.")
    return kagome

def random_density(n,total,alpha=None):
    """
    Produces a random electron density, using the Dirichlet
    distribution and an empirically chosen alpha parameter.
    You can specify alpha if you like.

    Inputs: n - positive integer, number of sites/length of list.
        total - non-negative number, number of electrons.
        alpha - optional positive number. If set, this value
            of alpha is used rather than the default.
    Output: numpy array of length n of random numbers between 0 and 1
        which sum to total.

    Last Modified: 2020-07-09
    """
    # Impossible cases
    if n < 1 or total < 0 or total > n:
        raise ValueError("n or total is out of bounds.")
    # Trivial cases
    if n==1:
        return np.array([total])
    if total == 0:
        return np.zeros(n)
    if total == n:
        return np.ones(n)
    # If n/total > 0.5, we solve for hole density instead.
    if total/n > 1/2:
        holes = True
        total = n - total
    else:
        holes = False
    # Alpha, the Dirichlet parameter.
    if alpha is None:
        alpha = choose_alpha(n,total)
    # Get the random density.
    density = bounded_random_numbers_with_sum_dirichlet(n,total,alpha)
    if holes:
        # Convert from hole density to electron density.
        density = 1 - density
    return density

def choose_alpha(n,total):
    """
    Picks an alpha which, when used with
    bounded_random_numbers_with_sum_dirichlet,
    will result in 1/10 samples being valid. I consider
    this to be a reasonable compromise for low alpha for
    more spread while maintaining okay performance.
    Alpha is chosen by an empirical formula obtained by
    numerical fitting. It is tested for N<=1000 and
    total/N <= 0.6. Does not apply for total/N close to
    1, where alpha should asymptote to infinity (but my
    formula does not).
    I give alpha a lower bound of 1.

    Inputs: n - positive integer, length of list to return.
        total - non-negative number. Value list should sum to.
    Output: alpha - positive number.

    Last Modified: 2020-07-09
    """
    return max(1, 0.028 * n**0.38 * np.exp(7*total/n))

def bounded_random_numbers_with_sum_dirichlet(n,total,alpha):
    """
    Returns a list of n random numbers which sum to total and
    are each between 0 and 1.
    Uses a Dirichlet distribution, which generates numbers between
    0 and 1 which sum to 1.
    Larger alpha is more likely to converge but has a smaller spread.
    Specifically, small alpha has an exponential-like distribution,
    with many values close to zero but a long tail.
    Large alpha has a bell-curve distribution around total/n.
    alpha should be at least 1. 5 or 10 is okay for half filling.
    If more than half filling, I recommend doing 1-total instead,
    although alpha of 50 or 100 can also be used.

    Inputs: n - positive integer, length of list to return.
        total - non-negative number. Value list should sum to.
        alpha - positive number, parameter for Dirichlet. Larger is less spread.
    Output: (n,) ndarray of numbers between 0 and 1 which sum to total.

    Last Modified: 2020-07-09
    """
    # Catch impossible cases.
    if n<1 or total<0 or total>n:
        raise ValueError("n or total are out of bounds.")
    # Catch trivial cases.
    if n==1:
        return np.array([total])
    if total==0:
        return np.zeros(n)
    if total==n:
        return np.ones(n)
    # Repeatedly try to generate a valid distribution.
    while True:
        # We multiply by total to get it to sum to total.
        vals = rng.dirichlet(alpha*np.ones(n))*total
        # However, this may make some of the numbers larger
        # than 1. Only accept the trial if all values are
        # less than or equal to 1.
        if vals.max() <= 1:
            return vals

def kagome_coordinates(nrows,ncols):
    """
    For each point in the kagome lattice, gives its Cartesian coordinates.

    Inputs: nrows and ncols, integers, number of rows and columns in supercell.
    Output: a (3*nrows*ncols, 2) array

    Last Modified: 2020-07-03
    """
    return np.reshape(kagome_coordinate_tensor(nrows,ncols),
                      (3*nrows*ncols,2),order='F')

def plot_spincharge(nrows,ncols,nup,ndown,marker_scale=1):
    """
    Plots the spin and charge density.
    Marker size/area is proportional to the charge density.
    Marker colour is proportional to the spin density (yellow up, blue down).

    Inputs: marker_scale - number, optional. Factor to scale markersize by.
    Effect: Makes a plot

    Last Modified: 2020-07-17
    """
    # Get the Cartesian coordinates of each point.
    coords = kagome_coordinates(nrows,ncols)
    x = coords[:,0]
    y = coords[:,1]
    # Get the charge density.
    chg = nup + ndown
    # Get the spin moment.
    spin = nup - ndown
    # Plot
    plt.scatter(x,y,chg*36*(marker_scale**2),spin,cmap="BrBG_r",vmin=-1,vmax=1)
    plt.gca().set_aspect('equal')
    cb = plt.colorbar()
    cb.ax.set_title('Spin')
    plt.show()

def plot_charge(nrows,ncols,chg,marker_scale=1):
    """
    Plots the charge density.
    Marker size/area is proportional to the charge density.

    Inputs: marker_scale - number, optional. Factor to scale markersize by.
    Effect: Makes a plot.

    Last Modified: 2020-07-17
    """
    # Get the Cartesian coordinates of each point.
    coords = kagome_coordinates(nrows,ncols)
    x = coords[:,0]
    y = coords[:,1]
    # Default marker size is 6. Default marker area is 6**2=36.
    plt.scatter(x,y,chg*36*(marker_scale**2))
    plt.gca().set_aspect('equal')
    plt.show()
