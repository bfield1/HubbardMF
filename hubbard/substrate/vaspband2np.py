#!/usr/bin/python3
"""
Turns VASP PROCAR and EIGENVAl files into something compatible with dftsubstrate

Takes a pyrprocar ProcarParser object or EigenvalParser object, made in VASP,
with its irreducible Brillouin zone, and expands it to the full Brillouin
zone and writes a single band to file.

All indexing is zero-based (Pythonic). This is different to the indexing
native to EIGENVAl and PROCAR which is one-based.
"""

from math import cos, sin, pi

import numpy as np
from pyprocar.procarparser import ProcarParser
from pyprocar.utilsprocar import UtilsProcar
from pyprocar.procarselect import ProcarSelect
import matplotlib.pyplot as plt

from hubbard.substrate.eigenvalparser import EigenvalParser

# Input methods

def repair_procar(infile, outfile=None):
    """Repairs PROCAR infile, save to outfile or in place"""
    if outfile is None:
        outfile = infile
    utils = UtilsProcar()
    utils.ProcarRepair(infile, outfile)

def load_procar(infile, outcar=None, fermi=None):
    """Loads a PROCAR file, and adjusts for Fermi energy"""
    procar = ProcarParser()
    try:
        # Load the file
        procar.readFile(procar=infile)
    except ValueError as e:
        # If PROCAR is badly formatted, advise proper course of action
        if str(e) == "Badly formated Kpoints headers, try `--permissive`":
            raise ValueError("Badly formatted Kpoints headers. Try `repair_procar(infile)`.")
        else:
            raise
    # Adjust the bands by the Fermi level if applicable.
    if outcar is not None:
        utils = UtilsProcar()
        fermi = utils.FermiOutcar(outcar)
    if fermi is not None:
        procar.bands -= fermi
    # Done.
    return procar

def load_eigenval(infile, outcar=None, fermi=None):
    """Loads an EIGENVAL file, and adjusts for Fermi energy"""
    eigen = EigenvalParser()
    eigen.readEigenval(infile, outcar, fermi)
    return eigen

# Output methods

def save_band(filename, procar, iband, outcar):
    """
    Saves a band (on a grid) as a numpy file

    Inputs: filename - file to save to
        procar - ProcarParser or EigenvalParser object (has kpoints and bands)
        iband - integer, index of band to save
        outcar - OUTCAR file
    Calls np.save on filename. Per np.save, data is appended to filename.
    """
    map_grid = _map_ibz_to_grid(procar.kpoints, outcar)
    np.save(filename, procar.bands[map_grid, iband])

def savetxt_band(filename, procar, iband, outcar):
    """
    Takes an unprocessed PROCAR, and saves a band as a grid in a text file


    Inputs: filename - file to save to
        procar - ProcarParser or EigenvalParser object (has kpoints and bands)
        iband - integer, index of band to save
        outcar - OUTCAR file
    """
    map_grid = _map_ibz_to_grid(procar.kpoints, outcar)
    data = procar.bands[map_grid, iband]
    # The trick is saving a 3D array to a 2D format.
    # https://stackoverflow.com/a/3685339/10778387
    with open(filename, 'w') as outfile:
        # Write a header to record the shape and whether we are Gamma.
        if _is_gamma_centered(procar.kpoints):
            gletter = 'Gamma'
        else:
            gletter = 'Monkhorst' # i.e. original Monkhorst
        outfile.write('# {0} {1} {2} {3}\n'.format(*map_grid.shape, gletter))
        # Iterate over slices in z/the third axis, writing them separately
        for i in range(map_grid.shape[2]):
            # PROCAR has energy data with 8 decimal places.
            np.savetxt(outfile, data[:,:,i], fmt='%9.8f')
            # Add a comment line separating the slices
            # np.loadtxt automatically ignores comment lines.
            outfile.write('#\n')


# Functions for analysing the bands and picking which ones you want.

def bands_in_energy_window(bands, emin, emax):
    """
    Takes the bands array, and determines which bands are between emin and emax

    Inputs: bands - (nk, nb) ndarray
        emin - number, lower bound on energy
        emax - number, upper bound on energy
    Output: list of integers between 0 and nb-1, indices of bands which have
        any points within the energy window.
    """
    points_in_range = (bands >= emin) & (bands <= emax)
    bands_in_range = np.any(points_in_range, axis=0)
    return list(np.arange(len(bands_in_range))[bands_in_range])

def find_crossings(procar):
    """
    Locates potential band crossings.

    Input: ProcarParser or EigenvalParser with loaded file
    Output: (nkpoints,nbands) Boolean ndarray
    """
    derivative = _max_difference(procar)
    # Where the energy difference between two bands is less than the energy
    # by which one of those bands has changed from adjacent k-points (times
    # a fudge factor), we suspect these two bands might cross here.
    # Energy difference between two bands
    ediff = procar.bands[:,1:] - procar.bands[:,:-1]
    # The greater of the derivative of those two bands
    derivmax = np.maximum(derivative[:,1:], derivative[:,:-1])
    # Compare sizes
    crossings =  ediff < (derivmax * 1.5)
    # Pad to be of full size
    crossings_full = np.zeros(derivative.shape, dtype=bool)
    crossings_full[:,:-1] = crossings
    return crossings_full

def plot_band_character(procar, iband, outcar, spins=[0], atoms=[-1], orbitals=[-1],
                        levels=None, axis=2, vmin=None, vmax=None, show=True):
    """
    Plots the band character in the full grid, with band being contours and colour being character.
    
    Inputs: procar - ProcarParser-like object with PROCAR data loaded.
        iband - integer, index of band to plot
        outcar - OUTCAR filename
        spins - list of integers
        atoms - list of integers
        orbitals - list of integers
        levels - optional, list of numbers (for energy contour plotting)
        axis - integer, axis in k-grid to sum/project along. Default 2.
        vmin - number, minimum projection value to plot
        vmax - number, maximum projection value to plot
        show - Boolean, whether to plt.show()
    """
    # Get filtered spd data.
    procar_filtered = ProcarSelect(procar)
    procar_filtered.selectIspin(spins)
    procar_filtered.selectAtoms(atoms)
    procar_filtered.selectOrbital(orbitals)
    # Get mapping of k-points to full grid
    map_grid = _map_ibz_to_grid(procar.kpoints, outcar)
    # Get x and y axes
    if axis==0:
        nx,ny = map_grid.shape[1:]
    elif axis==1:
        nx = map_grid.shape[0]
        ny = map_grid.shape[2]
    else:
        nx,ny = map_grid.shape[0:2]
    # If BZ is 2D, we can plot energy contours.
    if map_grid.shape[axis] == 1:
        plt.contour(np.arange(nx)/nx, np.arange(ny)/ny,
                procar.bands[map_grid,iband].squeeze(axis=axis), levels=levels, colors='k')
    # Plot the map
    plt.imshow(procar_filtered.spd[map_grid,iband].sum(axis=axis), vmin=vmin, vmax=vmax,
            origin='lower', extent=(-0.5/nx, 1-0.5/nx, -0.5/ny, 1-0.5/ny))
    plt.colorbar()
    if show:
        plt.show()

# Helper functions for the above.

def _max_difference(procar):
    """
    Gets the maximum change in energy between a point and its neighbour within a band

    Input: ProcarParser or EigenvalParser with loaded file
    Output: (nkpoints,nbands) ndarray
    """
    adj_list = kpoint_adjacency_list(procar.kpoints)
    out = np.empty(procar.bands.shape)
    for ik in range(len(procar.kpoints)):
        # There's got to be a way to vectorise over the bands.
        for ib in range(out.shape[1]):
            dmax = 0
            for ik2 in adj_list[ik]:
                d = abs(procar.bands[ik,ib] - procar.bands[ik2,ib])
                if d > dmax:
                    dmax = d
            out[ik,ib] = dmax
    return out

def _rotation_matrix(angle, vect):
    """3D rotation matrix of an angle around a vector."""
    (ux, uy, uz) = np.asarray(vect)/np.linalg.norm(vect) # Normalise
    c = cos(angle)
    mc = 1 - c
    s = sin(angle)
    return np.array( [[c + ux**2 * mc, ux*uy*mc - uz*s, ux*uz*mc + uy*s],
                      [uy*ux*mc + uz*s, c + uy**2 * mc, uy*uz*mc - ux*s],
                      [uz*ux*mc - uy*s, uz*uy*mc + ux*s, c + uz**2 * mc]] )

def get_reclat(outcar):
    """Get the reciprocal lattice from OUTCAR"""
    utils = UtilsProcar()
    return utils.RecLatOutcar(outcar)

def _fractional_rotation_matrix(angle, vect, reclat):
    """3D rotation matrix of an angle around a vector in Cartesian, acting on fractional coords"""
    return np.dot(np.linalg.inv(reclat.transpose()),
            np.dot(_rotation_matrix(angle, vect), reclat.transpose()))

def _get_symmetry_operations(outcar):
    """
    Returns the symmetry operations on the k-mesh from OUTCAR

    Does not handle non-zero translation vectors, because I don't know what
    normalisation they would use.
    Because I use a single-precision table, matrices are only good to single
    precision.

    Input: outcar - filename of OUTCAR file
    Output: list of (3,3) ndarrays, rotation matrices, which act on fractional
            coordinates.
    """
    # Get reciprocal lattice vectors (we'll need them later)
    reclat = get_reclat(outcar)
    # Read OUTCAR to find the space group operations table
    with open(outcar, 'r') as f:
        # Look for the table title
        match = 'Space group operators:'
        found = False
        for line in f:
            if len(line) >= len(match) and line[:len(match)] == match:
                found = True
                break
        if not found:
            raise ValueError("Could not find '" + match + "'.")
        # The next line are the column headers. Skip those.
        f.readline()
        # Put the operators in a list
        op_table = []
        for line in f:
            # Check if the line is empty (i.e. end of table)
            if len(line) == 0 or line.isspace():
                break
            # Convert row to numbers and put in table
            op_table.append([float(x) for x in line.split()])
        # We have our table so are done with the file
    # Table column headers
    # irot  det(A)  alpha  n_x  n_y  n_z  tau_x  tau_y  tau_z
    # Do sanity checks on the table elements
    if not all([r[1]==1 or r[1]==-1 for r in op_table]):
        raise ValueError("Determinants should be 1 or -1")
    if not all([r[6]==0 or r[7]==0 or r[8]==0 for r in op_table]):
        raise ValueError("Non-zero translation detected. Translations not supported.")
    # Now turn the table into matrix operations.
    mat_list = []
    for row in op_table:
        # det(A) * rotation_matrix(angle, (nx, ny, nz))
        mat = row[1] * _fractional_rotation_matrix(row[2]/180*pi, row[3:6], reclat)
        mat_list.append(mat)
    return mat_list

def _grid_shape_from_kpoints(kpoints):
    """Takes a set of kpoints, and determines what shape Monkhorst-Pack grid they belong to"""
    kshape = [] # Will become a 3-tuple
    symprec = 1e-5 # Constant, for effectively zero
    # For each dimension, determine grid spacing, and thus number of k-points
    for i in range(3):
        # Get the spacing between different coordinates
        spacing = np.diff(np.unique(kpoints[:,i]))
        # Exclude any spacing that should be zero
        spacing = spacing[spacing > symprec]
        # Extrapolate to number of points in unit length
        if spacing.size == 0:
            kshape.append(1)
        else:
            n = int(np.round(1/spacing.min()))
            kshape.append(n)
    # Return the 3-tuple
    return tuple(kshape)

def _map_ibz_to_grid(kpoints, outcar):
    """
    Takes irreducible Brillouin zone kpoints, and maps to full Brillouin zone

    Assumes a Monkhorst-Pack k-mesh.
    Maps each index in kpoints to 2D indices of points in a mesh
    with domain 0<=k<1.
    Inputs: kpoints - (n,3) array, list of k-points.
        outcar - OUTCAR file
    Output: grid_map - (N1,N2,N3) integer array, where each integer
        is an index in kpoints. The grid corresponds to a Monkhorst-Pack
        k-mesh, where k-points are k_i = (n_i+s_i)/N_i, where n_i is the index
        in grid_map, and s_i is 0 or 1/2 if the grid is not Gamma-centered and
        has an even N_i.
    """
    # Get the full k-grid size. We'll need it later
    grid_shape = np.asarray(_grid_shape_from_kpoints(kpoints))
    # Check if we're a gamma-centered k-mesh, by checking if Gamma is present
    gamma = _is_gamma_centered(kpoints)
    # Determine the offset of the grid origin
    if gamma:
        offset = np.array([0,0,0])
    else:
        # If not Gamma-centered and have even dimension, offset by half a gridpoint
        offset = (1 - np.mod(grid_shape,2))/2
    # Get the symmetry operations
    sym_ops = _get_symmetry_operations(outcar)
    # Apply the symmetry operations to get full k-points.
    ks = [np.dot(op, kpoints.transpose()).transpose() for op in sym_ops]
    full_kpoints = np.vstack(ks)
    # k -> -k is another symmetry operation
    full_kpoints = np.vstack((full_kpoints, -full_kpoints))
    # Wrap all points to be in the domain
    full_kpoints[full_kpoints < 0] += 1
    full_kpoints[full_kpoints >= 1] -= 1
    # The index of full_kpoints modulo len(kpoints) maps to index of kpoints
    # The index of a k-point in the k-mesh is k*N - offset, from the formula
    # for generating the MP mesh k = (n + offset)/N
    indices = np.round(full_kpoints * grid_shape - offset).astype(int)
    # Due to rounding, we may also need to wrap the indices
    indices -= (grid_shape * np.floor(indices/grid_shape)).astype(int)
    assert np.all(indices >= 0), "Found negative indices. Check the offset."
    # Now we stick kpoints indices into a shape grid_shape array
    # We also want to check to be sure that duplicates are consistent,
    # and that all points are filled.
    # Intialise values to -1, which we take to be the empty value.
    map_grid = -np.ones(grid_shape, dtype=int)
    for (igrid, ik) in zip(indices, range(len(indices))):
        # Index of original k-point
        ik0 = np.mod(ik, len(kpoints))
        # Check the map grid
        if map_grid[tuple(igrid)] == -1:
            # If empty, set to k-point index
            map_grid[tuple(igrid)] = ik0
        elif map_grid[tuple(igrid)] != ik0:
            # Otherwise, check that point is the same
            raise ValueError("K-points "+str(map_grid[tuple(igrid)])+" and "+str(ik0)+" both map to grid point "+str(igrid))
    # Check that all values are set
    if np.any(map_grid == -1):
        raise ValueError("K-points do not fill full Brillouin zone.")
    return map_grid

def _is_gamma_centered(kpoints):
    """Tests for having the Gamma point."""
    # gamma being three 0's.
    return np.any(np.all(kpoints == 0, axis=1))

