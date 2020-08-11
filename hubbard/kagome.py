#!/usr/bin/python3

"""
Mean field Hubbard model of Kagome lattice.
Periodic boundary conditions are assumed.

Created: 2020-07-03
Last Modified: 2020-08-11
Author: Bernard Field
"""

import numpy as np
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
    def _electron_density_single_methods(self,nelect,method,up,**kwargs):
        """
        Handler for setting the electron density of the Kagome lattice.
        Accepts the 'star' method and 'points' keyword.
        See also the parent method.
        
        Inputs: nelect - number of electrons.
            method - string, {'star'}, specifies the method.
            up - Boolean. Whether this is spin up electrons.
        Keyword Arguments:
            points - Boolean. Default True. For star method, whether to
                place the spin up electrons at the points of the star.
        Output: electron density - (self.nsites,) ndarray.
        Last Modified: 2020-08-11
        """
        # Process kwargs
        points = kwargs.get('points',True)
        # Check the method.
        if method == 'star':
            # Star method. We alternate based on spin up or down.
            if up:
                density = self._electron_density_star(nelect,points)
            else:
                density = self._electron_density_star(nelect,(not points))
        else:
            # The method is not here. Pass execution up the MRO.
            density = super()._electron_density_single_methods(nelect,
                                                    method,up,**kwargs)
        return density
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
    Last Modified: 2020-08-10
    """
    # Lattice vectors.
    a1 = np.array([1/2,np.sqrt(3)/2])
    a2 = np.array([-1/2,np.sqrt(3)/2])
    # Sub-lattice vectors
    sub = np.array([[0,0],a1/2,a2/2])
    # There's probably a vectorised way to do this, but I don't know it.
    # I generally don't need to run this too frequently, and its
    # efficiency isn't atrocious.
    coords = np.empty((3,nrows,ncols,2))
    for i in range(3):
        for j in range(nrows):
            for k in range(ncols):
                coords[i,j,k] = sub[i]+j*a1+k*a2
    return np.reshape(coords,(3*nrows*ncols,2),order='F')

def kagome_adjacency_tensor(nrows, ncols):
    """
    Creates the adjacency tensor for the kagome lattice.
    First three coords is input. Second 3 is output.
    Returns a 3*nrows*ncols*3*nrows*ncols ndarray.
    Elements are 1 or 0. Can also have 2's if nrows or ncols is 1.
    Last Modified: 2020-06-03
    """
    # Initialise the adjacency tensor.
    adj = np.zeros((3,nrows,ncols,3,nrows,ncols))
    # We have 12 adjacencies (counting each direction separately)
    # Use multiplication and broadcasting.
    #(0,i,j)->(1,i,j)
    # Generate matrices mapping each of coordinates.
    m0to1 = np.zeros((3,1,1,3,1,1))
    m0to1[0,0,0,1,0,0] = 1
    idrow = np.identity(nrows).reshape(1,nrows,1,1,nrows,1)
    idcol = np.identity(ncols).reshape(1,1,ncols,1,1,ncols)
    # Multiply together to get the full operation.
    op = m0to1*idrow*idcol
    # Add to the adjacency matrix.
    adj += op
    # (0,i,j)->(2,i,j)
    m0to2 = np.zeros((3,1,1,3,1,1))
    m0to2[0,0,0,2,0,0] = 1
    op = m0to2*idrow*idcol
    adj += op
    # (0,i,j)->(1,i-1,j)
    rowm1 = np.zeros((nrows,nrows)) # Row minus 1
    rowm1[(np.arange(nrows),np.mod(np.arange(nrows)-1,nrows))] = 1
    # mod implements periodic boundary conditions.
    rowm1 = rowm1.reshape((1,nrows,1,1,nrows,1))
    op = m0to1*rowm1*idcol
    adj += op
    # (0,i,j)->(2,i,j-1)
    colm1 = np.zeros((ncols,ncols))
    colm1[(np.arange(ncols),np.mod(np.arange(ncols)-1,ncols))] = 1
    colm1 = colm1.reshape((1,1,ncols,1,1,ncols))
    op = m0to2*idrow*colm1
    adj += op
    # (1,i,j)->(0,i,j)
    m1to0 = np.zeros((3,1,1,3,1,1))
    m1to0[1,0,0,0,0,0] = 1
    op = m1to0*idrow*idcol
    adj += op
    # (1,i,j)->(2,i,j)
    m1to2 = np.zeros((3,1,1,3,1,1))
    m1to2[1,0,0,2,0,0] = 1
    op = m1to2*idrow*idcol
    adj += op
    # (1,i,j)->(0,i+1,j)
    rowp1 = np.zeros((nrows,nrows)) # Row plus 1
    rowp1[(np.arange(nrows),np.mod(np.arange(nrows)+1,nrows))] = 1
    rowp1 = rowp1.reshape((1,nrows,1,1,nrows,1))
    op = m1to0*rowp1*idcol
    adj += op
    # (1,i,j)->(2,i+1,j-1)
    op = m1to2*rowp1*colm1
    adj += op
    # (2,i,j)->(0,i,j)
    m2to0 = np.zeros((3,1,1,3,1,1))
    m2to0[(2,0,0,0,0,0)] = 1
    op = m2to0*idrow*idcol
    adj += op
    # (2,i,j)->(1,i,j)
    m2to1 = np.zeros((3,1,1,3,1,1))
    m2to1[(2,0,0,1,0,0)] = 1
    op = m2to1*idrow*idcol
    adj += op
    # (2,i,j)->(0,i,j+1)
    colp1 = np.zeros((ncols,ncols))
    colp1[(np.arange(ncols),np.mod(np.arange(ncols)+1,ncols))] = 1
    colp1 = colp1.reshape((1,1,ncols,1,1,ncols))
    op = m2to0*idrow*colp1
    adj += op
    # (2,i,j)->(1,i-1,j+1)
    op = m2to1*rowm1*colp1
    adj += op
    # Sanity checking
    # We have the right number of elements
    assert adj.sum()==12*nrows*ncols
    # The tensor is symmetric
    assert (adj == adj.transpose((3,4,5,0,1,2))).all()
    # Return
    return adj
