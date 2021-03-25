#!/usr/bin/python3

"""
Analysis scripts and batch processing for a single unit cell of Kagome Hubbard.

Created: 2020-09-18
Last Modified: 2020-09-21
Author: Bernard Field
"""

import math

import numpy as np

from hubbard.kpoints.kagome import KagomeHubbardKPoints
import hubbard.utils as ut
import progress

def spin_configurations(nelect):
    """
    Gets the 6 symmetry-distinct permutations of spin up, down, and no spin

    Those configurations are 000, 100, 110, 1-10, 11-1, 111.
    Returns it as a list of dicts for argument unpacking.

    Inputs: nelect - a number between 0 and 6. Number of electrons.
    Output: length 6 list of dictionaries with keys 'nup' and 'ndown' each
        containing a shape (3,) ndarray of floats.
    Last Modified: 2020-09-18
    """
    if nelect > 6 or nelect < 0:
        raise ValueError("nelect must be between 0 and 6.")
    # Charge density, we assume to be uniform
    chg = np.ones(3)*nelect/3
    # The spin density will need a scale factor, to keep the electron densities
    # within the right bounds.
    sca = min(nelect/3,2-nelect/3)
    data = []
    for config in [[0,0,0], [1,0,0], [1,1,0], [1,-1,0], [1,1,-1], [1,1,1]]:
        # sca*config is the spin density.
        nup = (chg + sca*np.array(config))/2
        ndown = (chg - sca*np.array(config))/2
        data.append({'nup':nup, 'ndown':ndown})
    return data

def converge_spin_configurations(hubbard, nelect, rdiff, rdiff_initial=0.01,
                                 T=None, debug=False, interval=500):
    """
    Gets 6 converged kagome Hubbards from the 6 initial spin configurations.

    Inputs: hubbard - template KagomeHubbardKPoints object, with everything
            except electron densities set.
        nelect - number of electrons.
        rdiff - number, target residual
        rdiff_initial - number, target residual for the initial linear mixing.
        T - number, optional. Temperature for Fermi Dirac distribution.
        debug - Boolean, wheather to print progress.
        interval - integer. How often to print progress.
    Output: a list of 6 KagomeHubbardKPoints objects.
    Last Modified: 2020-09-18
    """
    spin_configs = spin_configurations(nelect)
    kag_list = []
    for s in spin_configs:
        kag = hubbard.copy()
        kag.set_electrons(**s)
        ut.converge(kag, rdiff, rdiff_initial, T, debug, interval)
        kag_list.append(kag)
    return kag_list


def charge_u_spectrum(nlist, ulist, kpoints, rdiff, rdiff_initial=0.01, T=None):
    """
    Runs converge_spin_configurations for all n and u in nlist and ulist.

    Returns a single list with all the kagomes, ready for ut.write_list.
    Last Modified: 2020-09-18
    """
    kag_list = []
    progress.new(60, len(nlist)*len(ulist))
    for i,n in enumerate(nlist):
        for j,u in enumerate(ulist):
            hubbard = KagomeHubbardKPoints(u=u, allow_fractions=True)
            hubbard.set_kmesh(kpoints, kpoints)
            kag_list += converge_spin_configurations(hubbard, n, rdiff,
                                                     rdiff_initial, T)
            progress.update(j+i*len(ulist))
    progress.end()
    return kag_list

def write_spin_summary(kag_list, filename):
    """
    Writes a text summary of the spins, U, and charge of kagomes.
    """
    with open(filename,'w') as f:
        f.write("# U | Charge | Energy | Net Spin | Site 1 | Site 2 | Site 3 |"
                " sqrt<m^2>\n")
        for k in kag_list:
            spin = k.get_spin_density()
            m2 = np.sqrt((spin**2).mean())
            f.write(' '.join([str(k.u), str(k.get_electron_number()),
                              str(k.energy()), str(k.get_magnetization()),
                              str(spin[0]), str(spin[1]), str(spin[2]),
                              str(m2)])+'\n')

def find_u_from_spin(spin, nelect, kpoints, tol=1e-5, umin=0, umax=20,
                    rdiff=1e-4, rdiff_initial=1e-2, T=None):
    """
    Finds the U which gives the closest match to the given spin and electrons.

    Inputs: spin - list-like of three numbers. Target spin density.
        nelect - number between 0 and 6, number of electrons.
        kpoints - positive integer, size of k-mesh.
        tol - positive number, tolerance for golden section search.
    Output: u, a number, and the difference between it and the target.
    Last Modified: 2020-09-21
    """
    if nelect > 6 or nelect < 0:
        raise ValueError("nelect must be between 0 and 6.")
    if len(spin) != 3:
        raise ValueError("spin must be of length 3.")
    # Get our initial spin up and down densities
    nup = (nelect/3 + np.asarray(spin))/2
    ndown = (nelect/3 - np.asarray(spin))/2
    # Get a template kagome hubbard.
    kagome = KagomeHubbardKPoints(nup=nup, ndown=ndown, allow_fractions=True)
    kagome.set_kmesh(kpoints,kpoints)
    # Make the function which gets the difference between the actual and target
    # spin densities.
    def diff(u):
        kag = kagome.copy()
        kag.set_u(u)
        ut.converge(kag, rdiff, rdiff_initial, T)
        actual_spin = kag.get_spin_density()
        return np.linalg.norm(np.asarray(spin) - actual_spin)
    # Search for the minimum U. This assumes unimodality.
    (u1, u2) = gss(diff, umin, umax, tol)
    final_u = (u1+u2)/2
    return final_u, diff(final_u)






"""Python program for golden section search.  This implementation
   reuses function evaluations, saving 1/2 of the evaluations per
   iteration, and returns a bounding interval.
   https://en.wikipedia.org/wiki/Golden-section_search"""

def gss(f, a, b, tol=1e-5):
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    Example:
    >>> f = lambda x: (x-2)**2
    >>> a = 1
    >>> b = 5
    >>> tol = 1e-5
    >>> (c,d) = gss(f, a, b, tol)
    >>> print(c, d)
    1.9999959837979107 2.0000050911830893
    """

    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2
    
    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    if yc < yd:
        return (a, d)
    else:
        return (c, b)
