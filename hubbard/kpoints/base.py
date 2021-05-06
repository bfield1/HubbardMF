#!/usr/bin/python3

"""
Base class for Mean Field Hubbard model with Brillouin zone sampling.
Has the mixing routines and other things which aren't dependent on the
specifics of the lattice.

The primary change from the Gamma point only code is in the eigenstep
method, which has been rebuilt to allow k-point sampling.
The kinetic energy matrix is also no longer stored but must be
dynamically constructed for each k-point.

Several things must be defined in the subclasses.
 - get_kinetic and set_kinetic. get_kinetic needs to be callable after
    initialisation. THIS ONE IS IMPORTANT.
 - The copy method. I don't know which data you need to feed into new instances.
 - get_coordinates - returns a (nsites, 2) ndarray which indicates
     the Cartesian coordinates of each site. Used for plotting.
     (I assume a 2D lattice for plotting. 3D plotting will require re-working
     the plotting routines to do 3D plots. That's a job for a subclass.)
 - save and load methods (you don't have to define them, but they aren't
     defined here). (I may find a way to abstract the saving/loading later.)
 - any bespoke electron density setting methods, in
     _electron_density_single_methods.
There is also the variable self.dims. It tells you how many dimensions the
k-space vectors exist in. It defaults to 2, but you can make it any positive
integer. Please don't change it after initialisation.

Created: 2020-09-16
Last Modified: 2021-05-05
Author: Bernard Field
    Copyright (C) 2021 Bernard Field, GNU GPL v3+
"""

from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

from hubbard.misc import ConvergenceWarning, MixingError, fermi_distribution,\
     random_density, choose_alpha, bounded_random_numbers_with_sum_dirichlet,\
     random_points_density

class HubbardKPoints():
    """Base class for Mean Field Hubbard model with Brillouin zone sampling."""
    #
    ## IO
    #
    def __init__(self,nsites,u=0,nup=0,ndown=0,allow_fractions=False,**kwargs):
        """
        Initialises things.
        Crucially, this initialiser does NOT set the kinetic energy
        part of the Hamiltonian. That is lattice specific.

        Inputs: nsites - positive integer. Number of sites. Cannot be
                changed later.
            u - number. Hubbard U parameter.
            nup,ndown,kwargs - arguments for set_electrons.
        Last Modified: 2020-10-07
        """
        if nsites <= 0:
            raise ValueError("Number of sites must be positive.")
        self.nsites = nsites
        """Number of sites."""
        self.u = u
        """Hubbard U parameter."""
        self.allow_fractions = allow_fractions
        """Flag for allow_fractions (toggle with toggle_allow_fractions)."""
        self.mag = 0
        """Magnetic field."""
        self.set_electrons(nup,ndown,**kwargs,backup=False)
        # If number of dimensions has not already been set, set it.
        if not hasattr(self, 'dims'):
            self.dims = 2
            """Number of dimensions in momentum-space."""
        if not hasattr(self, 'reclat'):
            self.reclat = np.eye(self.dims)
            """Reciprocal lattice vectors. Used for plotting."""
        # Initialise the k-mesh as a single k-point.
        self.set_kmesh(*([1]*self.dims))
    #
    def copy(self):
        """
        MUST BE OVERRIDDEN IN CHILD CLASS.
        This method is meant to return a copy of the current instance.

        Expected Output: Hubbard-derived object.
        """
        raise NotImplementedError
    #
    @classmethod
    def load(cls,f):
        """
        MUST BE OVERRIDDEN IN CHILD CLASS.
        Loads a Hubbard-derived object from file f.
        Inputs: f - filename.
        Outputs: Hubbard-derived object
        """
        raise NotImplementedError
    #
    def save(self,f):
        """
        MUST BE OVERRIDDEN IN CHILD CLASS.
        Save this object to file f.
        Inputs: f - filename
        Writes a file to f.
        """
        raise NotImplementedError
    #
    ## ELECTRON DENSITY MODIFIERS
    #
    def anderson_mixing(self,max_iter=100,rdiff=1e-6,mix=0.5,T=None,mu=None):
        """
        A more elaborate scheme than linear mixing. It adds a sort of
        extra internal linear mixing step with an automatically chosen
        ratio and including information from the previous step.
        Here I minimise the residual.
        See Sec IIB of D.D.Johnson 1988 PRB v38 n 18 and
        Ref's 3 and 4 therein for the algorithm.

        Inputs: max_iter - positive integer, maximum number of iterations.
            rdiff - postivie number, value of the residual at which we should
                stop the cycle.
            mix - number between 0 and 1. Mixing parameter.
            T - optional, non-negative number. Temperature for determining
                occupation of states. If provided, allows fractional occupation
                and exchange between spin up and down channels.
            mu - optional, number. Chemical potential for Grand Canonical
                Ensemble. Number of electrons becomes non-constant. Requires
                fractional electrons and setting T.
        Output: number, the residual
        Last Modified: 2021-04-16
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
        if mu is not None:
            if not fermi:
                raise TypeError("If mu is provided, T must be provided.")
        # Set Grand Canonical Ensemble flag
        gce = mu is not None
        # We have to run an initial step by normal linear mixing.
        if fermi:
            _, nupoutold, ndownoutold = self._eigenstep_finite_T(T)
        else:
            _, nupoutold, ndownoutold = self._eigenstep()
        # Residues
        resold = np.concatenate((nupoutold - self.nup,ndownoutold - self.ndown))
        # If the residue is sufficiently small, we're done.
        if np.linalg.norm(resold) < rdiff:
            return np.linalg.norm(resold)
        # Record old input densities.
        nupinold = self.nup
        ndowninold = self.ndown
        # Do mixing
        self.set_electrons(nup=self.nup*(1-mix)+ nupoutold*mix,
                           ndown=self.ndown*(1-mix) + ndownoutold*mix)
        for i in range(max_iter):
            # Put in electron density, get electron density out.
            if fermi:
                if gce:
                    _, nupout, ndownout = self._eigenstep_GCE(T,mu)
                else:
                    _, nupout, ndownout = self._eigenstep_finite_T(T)
            else:
                _, nupout, ndownout = self._eigenstep()
            # Get residue
            res = np.concatenate((nupout-self.nup,ndownout-self.ndown))
            # If the residue is sufficiently small, we're done.
            if np.linalg.norm(res) < rdiff:
                return np.linalg.norm(res)
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
        return np.linalg.norm(res)
    #
    def linear_mixing(self,max_iter=100,ediff=1e-6,mix=0.5,rdiff=5e-4,T=None,print_residual=False,mu=None,return_energy=True,debug=False):
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
            print_residual - Boolean, default False. Prints the residual
                of the density. You want this to be small.
                Set False if you are looping through stuff to avoid flooding
                stdout.
            mu - optional number. Chemical potential. If set, calculates mixing
                in the grand canonical ensemble. Requires T and 
                allow_fractions=True. Number of electrons will vary.
            return_energy - Boolean. If true, return the energy. If false,
                return the residual.
            debug - Boolean. If True, records the density and other variables
                at each step and returns it in a dictionary.
        Outputs: if not debug: a number- energy or residual
            if debug: a dictionary of lists. Each list entry is the value at
            the end of each step. (Except 'energy', which is the value at
            the start of the step.) Dictionary has nup, ndown, residual, and
            energy.
        Effects: sets nup and ndown to new values.

        Last Modified: 2021-04-16
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
        if mu is not None:
            if not fermi:
                raise TypeError("If mu is provided, T must be provided.")
        # Set Grand Canonical Ensemble flag
        gce = mu is not None
        # Initialise debug data
        if debug:
            debugdata = {'nup':[], 'ndown':[], 'residual':[], 'energy':[]}
        # Initialise first energy as a very large number.
        en = 1e12
        for i in range(max_iter):
            # Do a step
            if fermi:
                if gce:
                    ennew,nupnew,ndownnew = self._eigenstep_GCE(T,mu)
                else:
                    ennew,nupnew,ndownnew = self._eigenstep_finite_T(T)
            else:
                ennew,nupnew,ndownnew = self._eigenstep()
            # Do mixing
            residual_up = nupnew - self.nup
            residual_down = ndownnew - self.ndown
            # While analytically it should be impossible for the values to fall
            # outside the allowed range, floating point errors can cause it
            # to be outside by a tiniest smigeon. Catch these.
            nupnext = self.nup*(1-mix) + nupnew*mix
            ndownnext = self.ndown*(1-mix) + ndownnew*mix
            nupnext[nupnext > 1] = 1
            nupnext[nupnext < 0] = 0
            ndownnext[ndownnext > 1] = 1
            ndownnext[ndownnext < 0] = 0
            self.set_electrons(nup=nupnext, ndown=ndownnext)
            # Check for convergence
            res = np.linalg.norm(np.concatenate((residual_up,residual_down)))
            de = abs(ennew - en)
            # Before breaking, record debug data.
            if debug:
                debugdata['nup'].append(nupnext)
                debugdata['ndown'].append(ndownnext)
                debugdata['residual'].append(res)
                debugdata['energy'].append(ennew)
            # Check for convergence.
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
        if debug:
            return debugdata
        elif return_energy:
            return en
        else:
            return res
    #
    def pulay_mixing(self,max_iter=100,rdiff=1e-3,T=None, mu=None, debug=False):
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
            mu - optional number. Chemical potential. If set, calculates mixing
                in the grand canonical ensemble. Requires T and 
                allow_fractions=True. Number of electrons will vary.
                NOTE: Pulay mixing is very slow at converging the total number
                of electrons. If using mu, either stick with linear mixing or 
                use a much stricter initial convergence threshold.
            debug - Boolean. If True, records the density and other variables
                at each step and returns it in a dictionary.
        Outputs: if not debug: the residual
            if debug: a dictionary of lists. Each list entry is the value at
            the end of each step. Dictionary has nup, ndown, and residual.
        Output: number, the residual
        Last Modified: 2021-05-05
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
        if mu is not None:
            if not fermi:
                raise TypeError("If mu is set, T must be set.")
        # Set Grand Canonical Ensemble flag
        gce = mu is not None
        # Initialise debug record
        if debug:
            debugdata = {'nup':[], 'ndown':[], 'residual':[]}
        condmax = 1e4 # Maximum permissible value of the condition number.
        # (I haven't actually tested what a good value for this is,
        # but it seems to work okay most of the time.)
        nsites = self.nsites
        # Initialise our records.
        densities_up = np.empty((0,nsites))
        densities_down = np.empty((0,nsites))
        residues = np.empty((0,2*nsites))
        # Iterate
        for i in range(max_iter):
            # An eigenstep
            if fermi:
                if gce:
                    _, nup, ndown = self._eigenstep_GCE(T,mu)
                else:
                    _, nup, ndown = self._eigenstep_finite_T(T)
            else:
                _, nup, ndown = self._eigenstep()
            # Residue
            res = np.concatenate((nup - self.nup, ndown - self.ndown))
            # If the residue is sufficiently small, we're done.
            if np.linalg.norm(res) < rdiff:
                if debug:
                    debugdata['nup'].append(self.nup.copy())
                    debugdata['ndown'].append(self.ndown.copy())
                    debugdata['residual'].append(np.linalg.norm(res))
                    return debugdata
                else:
                    return np.linalg.norm(res)
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
            # Floating point errors can take densities out of bounds in
            # limiting cases. Catch these cases.
            if self.nelectup == self.nsites:
                nupnew = np.ones(self.nsites)
            elif self.nelectup == 0:
                nupnew = np.zeros(self.nsites)
            if self.nelectdown == self.nsites:
                ndownnew = np.ones(self.nsites)
            elif self.nelectdown == 0:
                ndownnew = np.zeros(self.nsites)
            # Record debug data
            if debug:
                debugdata['nup'].append(nupnew)
                debugdata['ndown'].append(ndownnew)
                debugdata['residual'].append(np.linalg.norm(res))
            # Check validity
            if (nupnew.max()>1 or nupnew.min()<0
                or ndownnew.max()>1 or ndownnew.min()<0):
                if debug:
                    raise MixingError("Electron density out of bounds. Try linear_mixing.", debugdata)
                else:
                    raise MixingError("Electron density out of bounds. Try linear_mixing.")
            # Apply mixing.
            self.set_electrons(nupnew,ndownnew)
        # Done, but due to exceeding the maximum iterations.
        warn("Self-consistency not reached after "+str(max_iter)+" steps. "
             "Residual in last step is "+str(np.linalg.norm(res))+".",
             ConvergenceWarning)
        if debug:
            return debugdata
        else:
            return np.linalg.norm(res)
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
            method - string. {'uniform', 'random'}. Specifies
                how to generate the electron density if a number
                of electrons is provided. Ignored if n is a list.
            alpha - optional positive number. Used for random method.
                Dirichlet alpha parameter. Default is chosen automatically
                to be as small as practical.
            
        Effects: Sets nup, ndown, nelectup, nelectdown.
            In the event of an Exception, no changes are made.

        Last Modified: 2020-08-11
        """
        # Process keyword arguments. kwargs.get(key,default)
        method = kwargs.pop('method','uniform')
        alpha = kwargs.pop('alpha',None)
        # Any remaining kwargs are class specific.
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
                    nup,method,up=True,alpha=alpha,**kwargs)
            # Set spin down if applicable.
            if ndown is not None:
                self.ndown,self.nelectdown = self._electron_density_single(
                    ndown,method,up=False,alpha=alpha,**kwargs)
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
        Get electron density from eigenvectors and number of electrons.
        
        Do this separately for spin up and down.

        Inputs: evect - array, eigenvector output of np.linalg.eigh
            nelect - integer. Number of electrons occupying these states.
        Output: a 1D array of length of each eigenvector, of real non-negative
            numbers, representing the local electron density.

        Last Modified: 2020-09-16
        """
        nstates = nelect * self.kpoints
        # Take the absolute value squared of each eigenvector
        # then sum over the bottom nelect eigenvectors.
        if not self.allow_fractions:
            density = np.sum(abs(evect[:,0:nstates])**2,1)
        else:
            density = np.sum(abs(evect[:,0:int(nstates)])**2,1)
            # Get the fractional part of the occupations.
            frac = nstates - int(nstates)
            # Add energy from fractionally occupied states.
            density += abs(evect[:,int(nstates)])**2 * frac
        # Divide by kpoints to normalise.
        return density/self.kpoints
    #
    def _eigenstep(self):
        """
        Evaluate the Hamiltonian, get new energy and density.

        This is the core part of the self-consistency loop.
        Takes the electron densities and the parameters needed to
        make the Hamiltonian, then returns the energy and the
        electron densities predicted by the eigenvectors.
        Assumes the total number of up and down electrons stays
        constant.
        The idea is that you'll use the output densities to mix with
        the initial densities then use as new input densities in a
        self-consistency loop.

        Presets: nup, ndown, u, set_kinetic, nelectup, nelectdown
        Outputs: energy (real number), nup, ndown

        Last Modified: 2020-09-16
        """
        # Generate the potential energy.
        _, _, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, edown, vup, vdown = self._eigensystem()
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
        Evaluate the Hamiltonian, get new energy and density. Use finite T.

        Takes the electron densities and the parameters needed to
        make the Hamiltonian, then returns the energy and the
        electron densities predicted by the eigenvectors.
        Fills the eigenstates by assuming a Fermi-Dirac distribution
        with temperature T.

        Inputs: T - positive number, temperature for Fermi Dirac distribution.
        Outputs: energy, nup, ndown

        Last Modified: 2020-09-16
        """
        # Generate the potential energy.
        _, _, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, edown, vup, vdown = self._eigensystem()
        # Get chemical potential
        energies = np.sort(np.concatenate((eup,edown)))
        mu = self._chemical_potential_from_states(
            T,(self.nelectup+self.nelectdown)*self.kpoints,energies)
        # Calculate energy
        en = self._energy_from_states(eup,edown,offset,T,mu)
        # Get new electron densities.
        weightup = fermi_distribution(eup,T,mu)
        weightdown = fermi_distribution(edown,T,mu)
        nup = np.sum(weightup*abs(vup)**2,1)/self.kpoints
        ndown = np.sum(weightdown*abs(vdown)**2,1)/self.kpoints
        # Return
        return en, nup, ndown
    #
    def _eigenstep_GCE(self,T,mu):
        """
        Evaluate the Hamiltonian, get new energy and density, in grand canonical ensemble

        Takes the electron densities and the parameters needed to
        make the Hamiltonian, then returns the energy and the
        electron densities predicted by the eigenvectors.
        Fills the eigenstates by assuming a Fermi-Dirac distribution
        with temperature T and chemical potential mu.

        Inputs: T - positive number, temperature for Fermi Dirac distribution.
            mu - number, chemical potential
        Outputs: energy, nup, ndown

        Last Modified: 2021-03-23
        """
        # Generate the potential energy.
        _, _, offset = self._potential()
        # Solve for the single-electron energy levels.
        eup, edown, vup, vdown = self._eigensystem()
        # Calculate energy
        en = self._energy_from_states(eup,edown,offset,T,mu)
        # Get new electron densities.
        weightup = fermi_distribution(eup,T,mu)
        weightdown = fermi_distribution(edown,T,mu)
        nup = np.sum(weightup*abs(vup)**2,1)/self.kpoints
        ndown = np.sum(weightdown*abs(vdown)**2,1)/self.kpoints
        # Apply GCE correction to energy (-mu*N)
        en -= mu * (nup.sum() + ndown.sum())
        # Return
        return en, nup, ndown
    #
    def _electron_density_single(self,n,method,up,alpha=None,**kwargs):
        """
        Helper method for set_electrons, so I don't have to
        duplicate my code for the spin up and down cases.

        Inputs: n - either a number of electrons
                or a list-like for the electron density.
                If number, must be between 0 and number of sites.
                If list-like, must be of right length and have
                values between 0 and 1.
            method - string. {'uniform', 'random'}. Specifies
                how to generate the electron density if a number
                of electrons is provided. Ignored if n is a list.
            up - Boolean, whether or not this is for the spin-up electrons.
                Doesn't do anything on its own, but may be passed to
                other methods.
        Keyword Arguments:
            alpha - optional positive number. Used for random method.
                Dirichlet alpha parameter. Default is chosen automatically.
        Outputs: density - ndarray of shape (3*nrows*ncols,)
            nelect - integer.
        Last Modified: 2021-02-19
        """
        # A useful constant
        nsites = self.nsites
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
            if nelect < 0 or nelect > nsites:
                raise ValueError("The number of electrons is out of bounds.")
            # Generate an electron density by the appropriate method.
            density = self._electron_density_single_methods(nelect,method,up,
                                                            alpha=alpha,**kwargs)
        else:
            # n is the electron density.
            density = np.asarray(n, dtype=float)
            if density.min() < 0 or density.max() > 1:
                raise ValueError("The electron density is out of bounds.")
            if len(density) != nsites:
                raise ValueError("n has the wrong length.")
            if not self.allow_fractions:
                nelect = int(round(sum(n)))
                if abs(nelect - sum(n)) > 1e-10:
                    warn("Number of electrons "+str(sum(n))+
                         " is not an integer. Rounding to "+str(nelect)+".")
                    # Re-scale the density
                    sca = nelect/sum(n)
                    if sca < 1:
                        # Scale the electrons
                        density *= sca
                    else:
                        # Scale the holes
                        sca = (nsites - nelect)/(nsites - sum(n))
                        density = 1 - sca + sca*density
            else:
                nelect = sum(n)
        # Check that values are within bounds.
        # We had checks earlier checking the inputs, but here we double check.
        if nelect < 0 or nelect > nsites:
            raise ValueError("The number of electrons is out of bounds.")
        if density.min() < 0 or density.max() > 1:
            raise ValueError("The electron density is out of bounds.")
        return density, nelect
    #
    def _electron_density_single_methods(self,nelect,method,up,**kwargs):
        """
        Helper method for _electron_density_single.
        Handler for methods for specifying the electron density from
        the number of electrons.
        Should a child class wish to implement further methods, they should
        put `density=super()._electron_density_single_method(*args,**kwargs)'
        after checking for their own custom methods (in the else statement),
        to pass the execution back up the stack.
        
        Inputs: nelect - number of electrons. This has been tested for being
                a fraction or integer.
            method - a string. Specifies the method.
            up - boolean. Specifies whether this is a spin up or down electron.
        Keyword Arguments:
            alpha - positive number. Dirichlet alpha parameter for 'random' method.
            points - boolean. Parameter for 'impurity' method. Are up spins
                the localised ones?
        Output: electron density - (self.nsites,) ndarray, values between 0 and 1.
        Last Modified: 2021-03-02
        """
        # Process kwargs
        alpha = kwargs.get('alpha',None)
        points = kwargs.get('points',True)
        # Switch-case the method.
        if method == 'uniform':
            # Uniform electron density.
            density = nelect/self.nsites * np.ones(self.nsites)
        elif method == 'random':
            # Fully random electron density.
            density = random_density(self.nsites,nelect,alpha)
        elif method == 'impurity':
            # One spin channel is fully localised in random
            # sites while the other channel is uniform.
            if (points and not up) or (not points and up):
                # Uniform
                density = nelect/self.nsites * np.ones(self.nsites)
            else:
                # Random points
                density = random_points_density(self.nsites,nelect)
        elif method == 'scale':
            # Scale the existing electron density
            # Get the existing electron density.
            if up:
                nelect_old = self.nelectup
                density_old = self.nup
            else:
                nelect_old = self.nelectdown
                density_old = self.ndown
            # Scale the density.
            if nelect_old > nelect:
                # Reducing number of electrons, scale electrons
                density = density_old * nelect/nelect_old
            elif nelect_old < nelect:
                # Increasing number of electrons. Scale holes.
                # If scaled electrons, we'd risk going out of bounds.
                sca = (self.nsites - nelect)/(self.nsites - nelect_old)
                density = 1 - sca + sca * density_old
        else:
            # Cannot find the method. Raise error.
            raise ValueError("Method "+str(method)+" does not exist.")
        return density
    #
    ## GETTERS
    #
    def band_structure(self, klist, nsec):
        """
        Get the band structure - eigenenergies in paths between some k-points.

        Currently the path in the Brillouin zone needs to be continuous.
        I might eventually add support for discontinuous band structures.

        Inputs: klist - list-like containing length self.dim list-likes of
                numbers, i.e. k-points. k-points are in fractional coordinates.
                We compute the band structure in straight lines between
                consecutive points in klist.
            nsec - positive integer. Number of k-points to evaluate along each
                path (not including the starting point).
        Outputs: a shape (nsec*(len(klist)-1)+1,self.dim) ndarray, which is the
                list of all the k-points, in fractional coordinates.
            A shape (nsec*(len(klist)-1)+1,2,self.nsites) ndarray, which is the
                spin polarised band structure. 1st axis maps to the list of
                kpoints. 2nd axis is spin (up then down).

        Last Modified: 2020-10-02
        """
        # Convert to numpy array
        klist = np.asarray(klist)
        # Generate full list of k-points from the provided list of key points.
        # Piecewise parametric lines. The parameter within each segment,
        # (i/nsec - int(i/nsec)), has range [0,1).
        # It is essentially k[n+1]*t + k[n]*(1-t).
        klist_full = [ (klist[int(np.ceil(i/nsec))] - klist[int(i/nsec)])
                       * (i/nsec - int(i/nsec)) + klist[int(i/nsec)]
                       for i in range(nsec*(len(klist)-1)+1) ]
        # Return the band structure
        return klist_full, self.eigenvalues_at_kpoints(klist_full)
    #
    def band_structure_projected(self, klist, nsec, atoms):
        """
        Get the band structure with projections onto some atoms.

        Currently the path in the Brillouin zone needs to be continuous.
        I might eventually add support for discontinuous band structures.

        Inputs: klist - list-like containing length self.dim list-likes of
                numbers, i.e. k-points. k-points are in fractional coordinates.
                We compute the band structure in straight lines between
                consecutive points in klist.
            nsec - positive integer. Number of k-points to evaluate along each
                path (not including the starting point).
            atoms - list of integers, indices of sites/orbitals we want to see
                the projection onto.
        Outputs: klist_full - a shape (nsec*(len(klist)-1)+1,self.dim) ndarray,
                which is the list of all the k-points, in fractional coordinates.
            bands - A shape (nsec*(len(klist)-1)+1,2,self.nsites) ndarray, which
                is the spin polarised band structure. 1st axis maps to the list
                of kpoints. 2nd axis is spin (up then down).
            projections - ndarray, same shape as bands. The projections of each
                eigenstate in bands onto atoms.

        Last Modified: 2021-03-22
        """
        # Make the full list of k-points.
        klist = np.asarray(klist)
        klist_full = [ (klist[int(np.ceil(i/nsec))] - klist[int(i/nsec)])
                       * (i/nsec - int(i/nsec)) + klist[int(i/nsec)]
                       for i in range(nsec*(len(klist)-1)+1) ]
        bands = np.empty((len(klist_full), 2, self.nsites))
        projections = np.empty(bands.shape)
        # Got to do eigensysteming manually
        potup, potdown, _ = self._potential()
        for n, k in enumerate(klist_full):
            kin = self.get_kinetic(k)
            # Spin up
            bands[n,0], v = np.linalg.eigh(kin + potup)
            projections[n,0] = (abs(v[atoms,:])**2).sum(axis=0)
            # Spin down
            bands[n,1], v = np.linalg.eigh(kin + potdown)
            projections[n,1] = (abs(v[atoms,:])**2).sum(axis=0)
        return klist_full, bands, projections
    #
    def chemical_potential(self,T,N=None):
        """
        Determine chemical potential for a given temperature.
        
        Determines what the chemical potential should be
        to yield the desired number of electrons at a
        given temperature.

        Inputs: T - nonnegative number, temperature.
                If T==0, simply returns the energy of the
                highest occupied state.
            N - number of electrons. Defaults to however
                many the system currently has, times the number
                of k-points.
        Output: the chemical potential, a number
        Last Modified: 2020-09-16
        """
        if T < 0:
            raise ValueError("T cannot be negative.")
        if N is None:
            N = (self.nelectup + self.nelectdown)*self.kpoints
        # Get the eigenenergy spectrum.
        eup, edown, _, _ = self._eigensystem()
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

        Last Modified: 2020-09-16
        """
        # Get the eigenenergy spectrum.
        eup, edown, _, _ = self._eigensystem()
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
        return energy, dosup/self.kpoints, dosdown/self.kpoints
    #
    def eigenvalues_at_kpoints(self, klist=None):
        """
        Returns eigenenergies at the specified k-points.

        Input: klist, list-like containing length self.dim list-likes
                of numbers. i.e. a list of k-points. Defaults to self.kmesh.
                k-points are in fractional coordinates.
        Output: a shape (len(klist),2,self.nsites) ndarray. First axis is
            the k-points. 2nd axis is the spin (up or down). Third are the
            eigenergies.

        Last Modified: 2020-10-02
        """
        # Default value
        if klist is None:
            klist = self.kmesh
        # Initialise
        bands = np.empty((len(klist),2,self.nsites))
        potup, potdown, _ = self._potential()
        # Iterate through the k-points.
        for n, k in enumerate(klist):
            kin = self.get_kinetic(k)
            # Spin up eigenvalues
            bands[n,0] = np.linalg.eigvalsh(kin + potup)
            # Spin down eigenvalues
            bands[n,1] = np.linalg.eigvalsh(kin + potdown)
        return bands
    #
    def eigenstates(self,mode='states',emin=None,emax=None):
        """
        Returns a selection of band-decomposed electron densities.
        
        Defaults to returning all of them.
        Currently lacks the means to distinguish between states
        from different k-points.

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

        Last Modified: 2020-09-16
        """
        # Get eigenstates
        eup, edown, vup, vdown = self._eigensystem()
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
    def energy(self,T=None, mu=None):
        """
        Calculates the energy from the current electron density.
        Does not go through self-consistency.

        Input: T - optional positive number. If provided,
            energy is calculated with a Fermi distribution
            for the occupancies.
            mu - optional number. If provided, T must be provided.
            Chemical potential. Calculates energy within grand canonical
            ensemble.
        Output: energy, a number.

        Last Modified: 2021-03-23
        """
        # Generate the potential energy.
        _, _, offset = self._potential()
        # Extra offset if grand canonical ensemble
        if mu is not None:
            offset -= mu * self.get_electron_number()
        # Solve for the single-electron energy levels.
        eup, edown, _, _ = self._eigensystem()
        if T is not None:
            if mu is None:
                mu = self.chemical_potential(T)
        else:
            if mu is not None:
                raise TypeError("If mu is provided, T must be provided.")
        return self._energy_from_states(eup,edown,offset,T=T,mu=mu)
    #
    def fermi(self, midpoint=False):
        """
        Calculates the Fermi level for the single particle states.
        
        Really, it just returns the energy of the highest occupied states.
        If midpoint=True, it returns the energy in between the highest occupied
        and lowest unoccupied states.
        It's probably a better idea to use chemical_potential for most uses.

        Outputs: two numbers, the Fermi level for spin up and spin down states.

        Last Modified: 2021-02-17
        """
        # Solve for the single-electron energy levels.
        eup, edown, _, _ = self._eigensystem()
        if self.allow_fractions:
            Nup = int(round(self.nelectup*self.kpoints))
            Ndown = int(round(self.nelectdown*self.kpoints))
        else:
            Nup = self.nelectup*self.kpoints
            Ndown = self.nelectdown*self.kpoints
        # Check the highest occupied states for the Fermi.
        # Get spin up
        if Nup == 0:
            fermi_up = eup[0]
        elif midpoint and len(eup) < Nup:
            fermi_up = (eup[Nup] + eup[Nup-1])/2
        else:
            fermi_up = eup[Nup-1]
        # get spin down
        if Ndown == 0:
            fermi_down = edown[0]
        elif midpoint and len(edown) < Ndown:
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
    def get_coordinates(self):
        """
        MUST BE OVERRIDDEN IN SUBCLASS.
        Returns the coordinates for plotting electron density.
        Output: a (self.nsites,2) ndarray of floats.
        """
        raise NotImplementedError
    #
    def get_electron_number(self):
        """
        Return the number of electrons.
        
        Output: Number
        Last Modified: 2020-07-29
        """
        return self.nelectup + self.nelectdown
    #
    def get_kinetic(self,k):
        """
        MUST BE OVERRIDDEN IN SUBCLASS.
        
        Returns the kinetic energy matrix for a given momentum
        Expected Input: k, a length self.dims vector.
        Expected Output: (nsites,nsites) ndarray
        Last Modified: 2020-09-16
        """
        raise NotImplementedError
    #
    def get_magnetization(self):
        """
        Return the net spin.
        
        Output: Number, the net spin.
        Last Modified: 2020-07-20
        """
        return self.nelectup - self.nelectdown
    #
    def get_spin_density(self):
        """
        Return the spin density.
        
        Output: (3*nrows*ncols,) ndarray.
        Last Modified: 2020-07-13
        """
        return self.nup - self.ndown
    #
    def local_magnetic_moment(self):
        """
        Return the average value of the local magnetic moment squared.
        
        Output: Number
        Last Modified: 2020-07-03
        """
        return (self.get_spin_density()**2).mean()
    #
    def nelect_from_chemical_potential(self, mu, T):
        """
        Solves for number of electrons to get a chemical potential mu.

        Not self-consistent. You've got to figure that one out yourself.

        Inputs: mu - number, chemical potential
            T - non-negative number, temperature for Fermi-Dirac distribution
        Output: N - non-negative number, number of electrons.

        Last Modified: 2021-03-02
        """
        # Get the eigenstates
        eup, edown, _, _ = self._eigensystem()
        # Concatenate
        energies = np.sort(np.hstack((eup, edown)))
        # Solve
        N = fermi_distribution(energies,T,mu).sum()/self.kpoints
        # Put within valid range
        N = min(max(N, 0), 2*self.nsites)
        return N
    #
    def residual(self,T=None,mu=None):
        """
        Return the residual of the electron density.
        
        A key convergence figure.
        Self-consistency has been reached if this number
        is 'small', ideally 0.

        Input: T - optional number. Temperature for eigenstep.
            mu - optional number. Chemical potential for GCE eigenstep.
        Outputs: residual , a non-negative number

        Last Modified: 2021-03-23
        """
        # Get densities after one iteration.
        if T is None:
            _, nupnew, ndownnew = self._eigenstep()
        else:
            if mu is None:
                _, nupnew, ndownnew = self._eigenstep_finite_T(T)
            else:
                _, nupnew, ndownnew = self._eigenstep_GCE(T,mu)
        # Calculate residual squared.
        residual_up = nupnew - self.nup
        residual_down = ndownnew - self.ndown
        res = np.linalg.norm(np.concatenate((residual_up,residual_down)))
        return res
    #
    def _chemical_potential_from_states(self,T,N,energies):
        """
        Determine chemical potential for a given temperature and set of states.

        I note that my T=0 algorithm sometimes gives different results to the
        limit T->0.
        
        Inputs: T - nonnegative number, temperature.
                If T==0, simply returns the energy of the
                highest occupied state.
            N - number of occupied states (electrons * kpoints).
            energies - sorted list/ndarray of all eigenenergies.
        Output: the chemical potential, a number
        Last Modified: 2021-02-18
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
            if N == 0:
                mu = energies[0]
            else:
                mu = energies[N-1]
        else:
            # Convert to array so it works in fermi_distribution
            energies = np.asarray(energies)
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
            while abs(N-nguess) > 1e-12 and step > np.spacing(mu):
                # Stop either when we have equality or our step size has
                # shrunk into oblivion and we cannot get any closer at
                # machine precision.
                # Check if we have found a cross-over point.
                if first_sweep:
                    if ((start_low and N > nguess) or
                        (not start_low and N < nguess)):
                        first_sweep = False
                if not first_sweep:
                    step /= 2 # Halve step-size for binary search.
                # Increment chemical potential in correct direction.
                if N > nguess:
                    mu += step
                else:
                    mu -= step
                # Recalculate our guess at the number of electrons.
                nguess = fermi_distribution(energies,T,mu).sum()
        return mu
    #
    def _eigensystem(self):
        """
        Solve the eigensystem and return the eigenenergies and states.

        Outputs: eup, edown, vup, vdown
            eup, edown - (nsites*kpoints,) ndarray of numbers, sorted.
                    Eigenenergies.
            vup, vdown - (nsites,nsites*kpoints) ndarray of numbers.
                Eigenvectors.
                v[:,i] is the normalised eigenvector corresponding to the
                eigenvalue e[i], as output by np.linalg.eigh.

        Last Modified: 2020-09-16
        """
        # Grab constants for conciseness
        nsites = self.nsites
        # Get the potential matrices
        potup, potdown, _ = self._potential()
        # Initialise
        eup = np.empty(nsites * self.kpoints)
        edown = np.empty(nsites * self.kpoints)
        vup = np.empty((nsites, nsites * self.kpoints),dtype='complex')
        vdown = np.empty((nsites, nsites * self.kpoints),dtype='complex')
        # Solve for the single-electron energy levels at each k-point
        for n, k in enumerate(self.kmesh):
            kin = self.get_kinetic(k)
            eup[n*nsites:(n+1)*nsites], vup[:,n*nsites:(n+1)*nsites] =\
                                        np.linalg.eigh(kin + potup)
            edown[n*nsites:(n+1)*nsites], vdown[:,n*nsites:(n+1)*nsites] =\
                                        np.linalg.eigh(kin + potdown)
        # Sort eigenstates by energy.
        # Skip this step if only one k-point due to redundancy.
        if self.kpoints > 1:
            inds_up = np.argsort(eup)
            inds_down = np.argsort(edown)
            eup = eup[inds_up]
            edown = edown[inds_down]
            vup = vup[:,inds_up]
            vdown = vdown[:,inds_down]
        return eup, edown, vup, vdown
    #
    def _energy_from_states(self,eup,edown,offset,T=None,mu=None):
        """
        Calculate energy from pre-calculated eigenstates.
        
        Inputs:
            eup - ndarray of spin up eigenstates.
            edown - ndarray of spin down eigenstates
            offset - number, energy offset from _potential
            T - number, optional. Temperature for Fermi distribution
                If not provided, just fill lowest energy states
                for each spin sector.
            mu - number, optional. Chemical potential. Must be provided
                if T is provided.
        Last Modified: 2020-09-16
        """
        if T is None:
            # Number of states.
            Nup = self.nelectup * self.kpoints
            Ndown = self.nelectdown * self.kpoints
            # Calculate the energy by filling electrons.
            if not self.allow_fractions:
                # Occupation is strictly integers.
                en = (sum(eup[0:Nup]) + sum(edown[0:Ndown])) / self.kpoints \
                     + offset
            else:
                # Occupation may be fractional.
                en = sum(eup[0:int(Nup)]) + sum(edown[0:int(Ndown)])
                # Get the fractional part of the occupations.
                fracup = Nup - int(Nup)
                fracdown = Ndown - int(Ndown)
                # Add energy from fractionally occupied states.
                if fracup > 0: en += eup[int(Nup)] * fracup
                if fracdown > 0: en += edown[int(Ndown)] * fracdown
                en = en / self.kpoints + offset
        else:
            # Calculate energies by weighting states by Fermi distribution
            if mu is None:
                raise TypeError("If T is provided, mu must also be provided.")
            energies = np.concatenate((eup,edown))
            en = np.sum(energies *
                        fermi_distribution(energies, T, mu))/self.kpoints \
                        + offset
        return en
    #
    def _potential(self):
        """
        Create the mean-field Hubbard potential part of the Hamiltonian.
        
        Give it the densities/occupations of up and down electrons,
        and it will give the Hubbard potential for up and down electrons.

        Presets: u, nup, ndown, mag
        Outputs: potup, potdown, offset
            two square diagonal ndarrays, of dimension equal to nup, and a number.
            potup is potential for spin up electrons.
            potdown is potential for spin down electrons.
            offset is a real number, to be added to the total energy.

        Last Modified: 2021-02-17
        """
        # V = U(nup<ndown> + ndown<nup> - <nup><ndown>)
        # Also, the magnetic field shifts the energy of up and down electrons.
        # We have <nup> and <ndown>
        offset = -np.sum(self.u*self.nup*self.ndown) # -u <nup><ndown>
        potup = self.u*np.diag(self.ndown) - self.mag * np.eye(self.nsites)
        potdown = self.u*np.diag(self.nup) + self.mag * np.eye(self.nsites)
        return potup, potdown, offset
    #
    ## PLOTTERS
    #
    def plot_bands(self, klist, nsec, T=None, emin=None, emax=None,
                   klabels=None, atoms=None):
        """
        Plot the band structure, for some given k-points.

        Inputs: klist - list-like containing length self.dim list-likes of
                numbers, i.e. k-points. k-points are in fractional coordinates.
                We compute the band structure in straight lines between
                consecutive points in klist.
            nsec - positive integer. Number of k-points to evaluate along each
                path (not including the starting point).
            T - non-negative number. Temperature with which to calculate
                the chemical potential. If omitted, no Fermi energy is used.
                If provided, energies are shifted to be relative to Fermi.
            emin - number. Minimum energy to plot.
            emax - number. Maximum energy to plot.
            klabels - length len(klist) list of strings, marking the tick
                labels.
            atoms - optional. List of integers. Indices of sites/orbitals to
                project onto. Shows as circles on the band structure.
        Effects: Makes a plot.

        Last Modified: 2021-03-22
        """
        # Get the band structure.
        if atoms is None:
            kp, bands = self.band_structure(klist, nsec)
        else:
            kp, bands, projections = self.band_structure_projected(klist, nsec, atoms)
        # Shift the bands by the chemical potential
        if T is not None:
            bands -= self.chemical_potential(T)
        # Convert the k-points from fractional to cartesian coords,
        # then turn it into a 1D distance.
        kp = np.matmul(kp, self.reclat)
        klen = np.cumsum(np.linalg.norm(np.diff(kp, axis=0, prepend=kp[0:1]),
                                        axis=1))
        # Identify the tic locations
        tics = klen[np.arange(0, nsec*(len(klist)-1)+1, nsec)]
        # Plot spin up bands
        for b in bands[:,0,:].transpose():
            plt.plot(klen,b,color='r')
        # Plot spin down bands
        for b in bands[:,1,:].transpose():
            plt.plot(klen,b,color='b')
        # Plot projection onto atoms.
        if atoms is not None:
            for (b,p) in zip(bands[:,0,:].transpose(), projections[:,0,:].transpose()):
                plt.scatter(klen,b,s=p*36, edgecolor='r', facecolor='none', zorder=2.1, marker='o')
            for (b,p) in zip(bands[:,1,:].transpose(), projections[:,1,:].transpose()):
                plt.scatter(klen,b,s=p*36, edgecolor='b', facecolor='none', zorder=2.1, marker='o')
        plt.ylim(emin, emax)
        plt.xticks(tics, klabels)
        plt.grid(axis='x')
        plt.axhline(y=0, color='0.5', linewidth=0.5, zorder=1.5)
        plt.show()
    #
    def plot_spin(self,marker_scale=1):
        """
        Plots the spin density.
        Size/area of marker is proportional to magnitude.
        Yellow is spin up. Blue is spin down.

        Inputs: marker_scale - number, optional. Factor to scale markersize by.
        Effects: Makes a plot.

        Last Modified: 2021-02-17
        """
        # Get the Cartesian coordinates of each point.
        coords = self.get_coordinates()
        x = coords[:,0]
        y = coords[:,1]
        # Get the spin moment.
        spin = (self.nup - self.ndown)[:len(coords)]
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

        Last Modified: 2021-02-17
        """
        # Get the Cartesian coordinates of each point.
        coords = self.get_coordinates()
        x = coords[:,0]
        y = coords[:,1]
        # Get the charge density.
        chg = (self.nup + self.ndown)[:len(coords)]
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
    def plot_spincharge(self,marker_scale=1,scale=1):
        """
        Plots the spin and charge density.
        Marker size/area is proportional to the charge density.
        Marker colour is proportional to the spin density (yellow up, blue down).

        Inputs: marker_scale - number, optional. Factor to scale markersize by.
            scale - number (0,1], optional. Spin density is plotted from 
                -scale to scale.
        Effect: Makes a plot

        Last Modified: 2020-11-30
        """
        # Get the Cartesian coordinates of each point.
        coords = self.get_coordinates()
        x = coords[:,0]
        y = coords[:,1]
        # Get the charge density.
        chg = (self.nup + self.ndown)[:len(coords)]
        # Get the spin moment.
        spin = (self.nup - self.ndown)[:len(coords)]
        # Plot
        plt.scatter(x,y,chg*36*(marker_scale**2),spin,cmap="BrBG_r",vmin=-scale,vmax=scale)
        plt.gca().set_aspect('equal')
        cb = plt.colorbar()
        cb.ax.set_title('Spin')
        plt.show()
    #
    ## SETTERS
    #
    def set_kinetic(self):
        """
        Use this method to set self.kin (for Gamma-point only) or
        the parameters to generate the kinetic energy from.
        Implementation is lattice-dependent.
        """
        raise NotImplementedError
    #
    def set_kmesh(self,*args,method='monkhorst'):
        """
        Set the k-mesh used for Brillouin zone sampling.

        Uses fractional coordinates.
        There are two ways to pass arguments.
        The first way is to specify self.dims integers, which produces a
        Monkhorst pack grid.
        The second way is to provide an explicit list of k-points.

        Inputs (1): self.dims positive integers.
        Inputs (2): list of lists of length self.dims numbers.

        Keyword arguments: method - string, {'monkhorst', 'gamma'}
                'monkhorst' makes a regular Monkhorst Pack grid.
                'gamma' makes a Gamma-centred Monkhorst Pack grid.

        Sets self.kmesh (array of k-points, shape (self.kpoints,self.dims))
         and self.kpoints (number of k-points).

        Last Modified: 2020-09-16
        """
        # Identify the nature of our input arguments.
        # If argument is integer, make Monkhorst Pack grid.
        if isinstance(args[0],int):
            if len(args) != self.dims:
                raise TypeError("set_kmesh expected "+str(self.dims)+
                                " integer arguments, got "+str(len(args)))
            # Check that args are all positive integers.
            for i in args:
                if not isinstance(i,int) or i <= 0:
                    raise ValueError("Expected arguments"
                                     " to be positive integers.")
            # Create a list of evenly spaced k-points.
            coords = [np.arange(0,1,1/n) for n in args]
            self.kpoints = np.prod(args)
            self.kmesh = np.array(np.meshgrid(*coords))\
                         .reshape((2,self.kpoints)).transpose()
            if method == 'monkhorst':
                # In original Monkhorst-Pack scheme, an even number
                # of subdivisions leads to shifting away from k-point
                # by half a subdivision
                shift = np.mod(np.array(args)+1,2)/2/np.array(args)
                self.kmesh += shift
            elif method == 'gamma':
                pass
            else:
                raise TypeError("set_kmesh doesn't recognise method "
                                +str(method))
        else:
            # Assume argument to be an explicit list of k-points.
            self.kmesh = args[0]
            self.kpoints = len(self.kmesh)
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
        Toggles flag which allows fractional electrons, handling conversion.
        
        Handles converting the number of electrons back to an integer if
        necessary.

        Inputs: val - optional Boolean. Sets allow_fractions
            to this value if provided. Toggles allow_fractions otherwise.
        Output: Boolean, new value of allow_fractions.

        Last Modified: 2021-02-18
        """
        if val is None:
            # If val not set, we toggle.
            val = not self.allow_fractions
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
                sca = self.nelectup/nelectup_old
                if sca < 1:
                    self.nup *= sca
                elif sca > 1:
                    # Do scaling of hole density instead.
                    sca = (self.nsites - self.nelectup)/(self.nsites - nelectup_old)
                    self.nup = 1 - sca + sca * self.nup
            if nelectdown_old > 0:
                sca = self.nelectdown/nelectdown_old
                if sca < 1:
                    self.ndown *= sca
                elif sca > 1:
                    # Do scaling of hole density instead.
                    sca = (self.nsites - self.nelectdown)/(self.nsites - nelectdown_old)
                    self.ndown = 1 - sca + sca * self.ndown
            # Now toggle the flag.
            self.allow_fractions = False
        return self.allow_fractions

