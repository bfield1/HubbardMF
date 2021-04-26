#!/usr/bin/python3

"""
Helper scripts for analysing my Mean field Hubbard model of the Kagome lattice.

Created: 2020-08-04
Last Modified: 2021-04-16
Author: Bernard Field
    Copyright (C) 2021 Bernard Field, GNU GPL v3+
"""

import warnings
import os.path
from time import sleep
from glob import glob

import numpy as np

from hubbard.misc import ConvergenceWarning, MixingError, choose_alpha
from hubbard.kagome import KagomeHubbard
import hubbard.progress as progress


def converge(hub,rdiff,rdiff_initial=1e-2,T=None,debug=False,interval=500,mu=None):
    """
    Brings a Hubbard model to self-consistency,
    regardless of how long it takes.
    Inputs: hub - Hubbard object.
        rdiff - number, target residual
        rdiff_initial - number, target residual for the initial linear mixing step.
        T - number, optional. Temperature for Fermi Dirac distribution.
        debug - Boolean, whether to print progress.
        interval - integer. How often to print progress.
        mu - number, optional. Chemical potential for GCE mixing.
    Last Modified: 2021-04-16
    """
    # Suppress ConvergenceWarnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",ConvergenceWarning)
        # We need to perform an initial linear mixing.
        res = hub.residual(T,mu) # Initial determination of the residual.
        while res > rdiff_initial:
            last_residual = res
            res = hub.linear_mixing(max_iter=interval,ediff=0.1,
                                 rdiff=rdiff_initial,T=T,print_residual=False,mu=mu,return_energy=False)
            if debug: print("Linear mixing. Residual={0}".format(res))
            # Catch a pathological case where residual doesn't decrease.
            if np.isclose(last_residual,res):
                if debug: print("Residual didn't decrease. Trying smaller mixing.")
                res = hub.linear_mixing(max_iter=interval,ediff=0.1,mix=0.1,
                                 rdiff=rdiff_initial,T=T,print_residual=False,mu=mu,return_energy=False)
                if debug: print("Linear mixing (mix=0.1). Residual={0}".format(res))
        # Now do main mixing.
        while res > rdiff:
            try:
                res = hub.pulay_mixing(max_iter=interval,rdiff=rdiff,T=T,mu=mu)
                if debug: print("Pulay mixing. Residual={0}".format(res))
            except MixingError:
                # Pulay mixing has a chance to take the electron density out of
                # bounds. It can happen when Pulay mixing diverges.
                # Since there was an error, the residual was not returned.
                if debug:
                    print("Failure in Pulay mixing. Residual={0}. "
                          "Doing some linear mixing.".format(hub.residual(T,mu)))
                res = hub.linear_mixing(max_iter=10,rdiff=rdiff,ediff=0.1,T=T,mu=mu, return_energy=False)
                if debug: print("Linear mixing. Residual={0}".format(res))

def sweep_spin(template,n,nsteps,rdiff,rdiff_initial=1e-2,
               T=None,interval=500,positive_only=True,repeats=1,verbose=False,
               mmin=None,mmax=None,quiet=False,mu=None):
    """
    For a given set of parameters, steps through different nup and ndown ratios.
    Relaxes the Hubbard systems, and returns all the results as a list.
    Has special handling of KeyboardInterrupt. Interrupting once will skip the
    current data point. Interrupting twice within 0.5 seconds will abort the sweep,
    but still return the data collected so far.
    
    Inputs: template - template Hubbard object (with right size,
            U, kinetic energy, etc., but electrons are ignored)
        n - number of electrons.
        nsteps - how many steps (as in linspace).
        rdiff, rdiff_initial, T, interval - as for converge
        positive_only - Boolean. Only check positive magnetization.
            Recommend True when magnetic field is zero, becasue symmetry.
            Ignored if mmin is set.
        repeats - positive integer. How many times to repeat each measurement.
        verbose - Boolean.
        mmin - optional number. Minimum magnetization to look at.
            Defaults to 0 if positive_only else -mmax.
        mmax - optional number. Maximum magnetization to look at.
            Defaults to the smaller of n or nsites.
        quiet - Boolean. Verbose overrules.
        mu - optional number. Chemical potential for GCE mixing.
    
    Output: list of Hubbard objects.

    Last Modified: 2021-04-26
    """
    # Initialise
    hub_list = [] # Results list
    nsites = template.nsites
    # Bounds
    if mmax is None:
        mmax = min(n,nsites)
    if mmin is None:
        if positive_only:
            mmin = 0
        else:
            mmin = -mmax
    nupmin = max(0,n-nsites,(n+mmin)/2)
    nupmax = min((mmax+n)/2,n,nsites)
    # Check an error case
    if nupmin != nupmax and nsteps == 1:
        raise ValueError("Cannot have nsteps=1 when have unequal mmin and mmax.")
    # Pick an alpha
    alpha = choose_alpha(nsites,min(nupmax,nsites/2))*10
    # Progress bar
    show_progress = (not verbose) and (not quiet)
    if show_progress:
        if nupmin == nupmax:
            progress.new(60,repeats,0)
        else:
            progress.new(60,nupmax,nupmin)
    elif verbose:
        print("Nup from "+str(nupmin)+" to "+str(nupmax))
    # Iterate
    samples,stepsize = np.linspace(nupmin,nupmax,nsteps,retstep=True)
    try:
        for nup in samples:
            for i in range(repeats):
                try:
                    if verbose: print("Nup="+str(nup)+", repeat "+str(i+1))
                    # Get a copy I can work with.
                    hub = template.copy()
                    # Set the electron density
                    hub.set_electrons(nup=nup,ndown=n-nup,method='random',alpha=alpha)
                    # Converge
                    converge(hub,rdiff,rdiff_initial,T=T,interval=interval,debug=verbose,mu=mu)
                    # Record
                    hub_list.append(hub)
                except KeyboardInterrupt:
                    # If interrupt, give a half second to wait for a second interrupt.
                    if verbose: print("Skipping...")
                    sleep(0.5)
                if show_progress:
                    if nupmin==nupmax: progress.update(i+1)
                    else: progress.update(nup+i/repeats*stepsize)
    except KeyboardInterrupt:
        # If we interrupt, still return cleanly.
        if show_progress:
            print('X',end='')
        elif verbose:
            print("Aborting.")
    if show_progress:
        progress.end()
    elif verbose:
        print("Finished.")
    # Return
    return hub_list


def write(kagome,directory='.',ext=''):
    """
    Write kagome to a file in directory.
    Automatically handles naming conventions.
    Filename includes U, supercell, and number of electrons.
    Other details should be specified in ext.
    Inputs: kagome - KagomeHubbard object.
        directory - string to a directory.
        ext - string. Extension to append to automatically
            generated name (but before the number)
    Last Modified: 2020-08-04
    """
    # Check existence of directory.
    if not os.path.isdir(directory):
        raise FileNotFoundError(str(directory)+" not found.")
    # Create the base name
    base_name = ('U'+str(kagome.u)+'cell'+str(kagome.nrows)+'x'+str(kagome.ncols)
            +'n'+str(int(round(kagome.get_electron_number())))+ext)
    # Append number.
    i = 0
    true_name = base_name + '_' + str(i).zfill(4)
    # Check existence.
    while os.path.isfile(directory+'/'+true_name):
        # If exists, try the next number.
        i += 1
        true_name = base_name + '_' + str(i).zfill(4)
    # Write
    kagome.save(directory+'/'+true_name)

def write_list(kagome_list,directory='.',ext=''):
    """
    Iterates over kagome_list and applies write() to each element.
    Last Modified: 2020-08-06
    """
    for kagome in kagome_list:
        write(kagome,directory,ext)

def read_list(directory='.', pattern='U*cell*x*n*', cls=KagomeHubbard):
    """
    Reads a directory filled with Hubbard files.

    Uses glob matching for the pattern.

    Inputs: directory - string to a directory
        pattern - string, glob pattern for the files to load.
        cls - class. Class to interpret the files as.
    Output: list of cls objects.
    Last Modified: 2020-09-18
    """
    true_pattern = directory + '/' + pattern
    files = glob(true_pattern)
    kag_list = [ cls.load(f) for f in files ]
    return kag_list

def boltzmann_average(hub_list,T,func,Ten=None):
    """
    Takes the mean of some value of Hubbard objects,
    weighted by a Boltzmann factor.
    Can also take the minimum energy value with T=0.

    Inputs: hub_list - list of Hubbard objects
        T - number, temperature for the Boltzmann factor.
        func - callable. Takes a KagomeHubbard object and returns a number.
        Ten - number, optional. Temperature for calculating the energy.
    Output: a number
    Last Modified: 2020-08-11
    """
    # Energies
    en = [h.energy(Ten) for h in hub_list]
    if T == 0:
        # Filter for minimum energy states.
        boltz = np.zeros_like(en)
        boltz[en == min(en)] = 1
        boltz /= boltz.sum() # Normalise
    else:
        # Weight states by a Boltzmann factor.
        boltz = np.exp(-(np.asarray(en)-min(en))/T)
        # Subtract minimum energy to avoid overflow.
        # Does not change the outcome.
        boltz /= boltz.sum() # Normalise
    # Evaluate the function
    vals = np.array([ func(h) for h in hub_list ])
    # Return, taking the weighted average.
    return np.sum(vals*boltz)

def cull_list(hub_list,e,T=None, mu=None):
    """
    Returns a hub_list with only states with energy less than e above
    the minimum energy.
    Inputs: hub_list - list of Hubbard objects
        e - number
        T - number, optional, temperature for calculating energy.
        mu - number, optional, chemical potential for energy.
    Output: list of Hubbard objects.
    Last Modified: 2021-04-14
    """
    en = np.array([h.energy(T,mu) for h in hub_list])
    return list(np.asarray(hub_list)[en <= (en.min()+e)])
