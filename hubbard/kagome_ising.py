#!/usr/bin/python3

"""Metropolis algorithm for Ising model of Kagome lattice.
Each atomic site is given by three coordinates, (n,i,j).
n is 0,1,2, and represents sublattice sites A, B, C.
i and j are translations by lattice vectors from the home unit cell.
i goes up to nrows. j goes up to ncols.
Periodic boundary conditions are assumed.
"""

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

# Initialise random number generator
rng = default_rng()

def random_spin(nrows,ncols):
    """Creates a random spin state.
    Returns a 3*nrows*ncols array of random -1 or 1."""
    # Make a random array of 0's and 1's of the correct size.
    # Then raise (-1) to the power of that array, elementwise.
    return (-1)**rng.integers(0,1,size=(3,nrows,ncols),endpoint=True)

def calculate_energy(spin,adj):
    """Takes a spin state and adjacency matrix.
    Calculates the energy."""
    return np.einsum('abc,abcdef,def',spin,adj,spin)/2

def spin_flip_energy(spin,adj):
    """Takes a spin state and adjacency matrix.
    Returns an array telling you the energy to flip each individual spin."""
    # We do a tensor multiplication to get the spins of neighbours,
    # then we do an elementwise multiplication to multiply by its own spin.
    return -2 * spin * np.einsum('abcdef,def',adj,spin)

def metropolis_step(spin,adj,T,ratio):
    """
    Performs a single step of the Metropolis algorithm.
    Takes an initial spin state and an adjacency matrix.
    Also takes a temperature T, in units of J/k_B.
    And take the ratio of spins to test for flipping.
    Returns another spin state.
    """
    # Uses the Boltzmann distribution to determine probabilities of spin flips,
    # capped at 1. Then we multiply by ratio.
    flip_prob = np.minimum(1,np.exp(-spin_flip_energy(spin,adj)/T)) * ratio
    # Random numbers
    rand = rng.random(spin.shape)
    # Use them to get yes/no on the spin flips.
    action = (-1)**(rand < flip_prob)
    # Perform the action of spin flipping and return
    return spin*action

def search_for_minimum(spin,adj,T,ratio,steps):
    min_energy = calculate_energy(spin,adj)
    min_state = spin
    for _ in range(steps):
        spin = metropolis_step(spin,adj,T,ratio)
        if calculate_energy(spin,adj) < min_energy:
            min_energy = calculate_energy(spin,adj)
            min_state = spin
    return min_state, min_energy

def kagome_adjacency_tensor(nrows, ncols):
    """Creates the adjacency tensor for the kagome lattice.
    First three coords is input. Second 3 is output.
    Returns a 3*nrows*ncols*3*nrows*ncols ndarray.
    Elements are 1 or 0. Can also have 2's if nrows or ncols is 1."""
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

def kagome_coordinates(nrows,ncols):
    """For each point in the kagome lattice,
    gives its cartesian coordinates.
    Returns a 3*nrows*ncols*2 array."""
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
    return coords

def plot_spin(spin):
    """Takes a spin configuration, and plots it in realspace."""
    # Determine nrows and ncols
    _,nrows,ncols = spin.shape
    # Get the coordinates.
    coords = kagome_coordinates(nrows,ncols)
    # Flatten
    spin = spin.flatten()
    coords = coords.reshape(3*nrows*ncols,2)
    # Separate spin up and spin down
    spin_up = coords[spin==1]
    spin_down = coords[spin==-1]
    # Plot
    plt.plot(spin_up[:,0],spin_up[:,1],'y.')
    plt.plot(spin_down[:,0],spin_down[:,1],'b.')
    plt.gca().set_aspect('equal')
    plt.show()
