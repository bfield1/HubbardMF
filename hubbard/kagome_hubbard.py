#!/usr/bin/python3

"""
Mean field Hubbard model of Kagome lattice.
Periodic boundary conditions are assumed.

Created: 2020-07-03
Last Modified: 2020-08-04
Author: Bernard Field
"""

import numpy as np
from hubbard.kagome_ising import kagome_adjacency_tensor
from hubbard.kagome_ising import kagome_coordinates as kagome_coordinate_tensor
import json
from hubbard.base import Hubbard

class KagomeHubbard(Hubbard):
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
        Last Modified: 2020-08-10
        """
        # Check that nrows and ncols are positive
        if nrows <= 0:
            raise ValueError("Number of rows must be positive.")
        if ncols <= 0:
            raise ValueError("Number of columns must be positive.")
        # Initialise the important constants.
        self.nrows = nrows
        self.ncols = ncols
        # Call the parent initialiser.
        super().__init__(3*nrows*ncols,u=u,nup=nup,ndown=ndown,
                         allow_fractions=allow_fractions,**kwargs)
        # Initialise the kinetic energy.
        self.set_kinetic(t)
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
    ## GETTERS
    #
    def get_coordinates(self):
        """
        Returns the coordinates for plotting electron density.
        Output: a (self.nsites,2) ndarray of floats.
        Last Modified: 2020-08-10
        """
        return kagome_coordinates(self.nrows,self.ncols)
    #
    ## PLOTTERS
    #
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

        Last Modified: 2020-08-10
        """
        # Have the kinetic energy term in tensor form,
        # where sublattice site, row and column are separate indices.
        tensor = -t*kagome_adjacency_tensor(self.nrows,self.ncols)
        # Reshape to a square matrix.
        length = self.nsites
        self.kin = np.reshape(tensor,(length,length),order='F')
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

def kagome_coordinates(nrows,ncols):
    """
    For each point in the kagome lattice, gives its Cartesian coordinates.

    Inputs: nrows and ncols, integers, number of rows and columns in supercell.
    Output: a (3*nrows*ncols, 2) array

    Last Modified: 2020-07-03
    """
    return np.reshape(kagome_coordinate_tensor(nrows,ncols),
                      (3*nrows*ncols,2),order='F')
