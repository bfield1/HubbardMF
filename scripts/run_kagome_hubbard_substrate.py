#!/usr/bin/python3

from math import sqrt
import pickle
import argparse
import os.path

import numpy as np
import pandas as pd

from hubbard.substrate.kagome import KagomeSubstrate
from hubbard.kpoints.kagome import KagomeHubbardKPoints
import hubbard.utils as ut
import hubbard.progress as progress

"""
Script which I used to generate the results of the MFH+Substrate model
in the paper "Correlation-induced magnetism in substrate-supported 2D
metal-organic frameworks", Bernard Field et al 2022.
"""

# When I used this script, I had the substrate band files in the same directory
# as I was running the script.
# They are currently in the folder substrate_bands
# I've configured this script to work when it runs from the folder you find it
# in. If you want it to work differently, you'll need to modify the line below.
path_to_substrates = 'substrate_bands/'
# path_to_substrates = ''
# In an ideal world I would have made it generalisable. But you get what you're
# given.

def create_ag(t, shift=0, include_16=False, nk=25):
    """
    Creates a KagomeSubstrate mimicking DCA-Cu/Ag(111).

    Inputs: t - number, kagome hopping constant (eV). Should be around 0.05.
        shift - number, shift of kagome Fermi level away from default (eV).
        include_16 - Boolean, whether to include band 16, which does not couple
            to the kagome lattice on account of being on the opposite face.
        nk - positive integer. Number of k-points in each direction in the MP grid.
    Output: KagomeSubstrate object.

    Last Modified: 2021-05-03
    """
    kagome = KagomeSubstrate(t=t, offset=shift)
    kagome.set_kmesh(nk,nk)
    # Load the substrate bands
    if include_16:
        kagome.add_substrate(coupling=0, subtype='dft', filename=path_to_substrates+'Ag_band16.dat', nx=7, ny=7)
    for i in (17,18,19):
        kagome.add_substrate(coupling=0, subtype='dft', filename=path_to_substrates+'Ag_band{0}.dat'.format(i), nx=7, ny=7)
    # Set a sensible electron density (so we have somewhere to start)
    n = kagome.nelect_from_chemical_potential(mu=0, T=0.005)
    kagome.set_electrons(nup=n/2, ndown=n/2, method='uniform')
    return kagome

def create_ag4(t, shift=0, include_H=False, nk=25):
    """
    Creates a KagomeSubstrate mimicking DCA-Cu/Ag(111).

    Inputs: t - number, kagome hopping constant (eV). Should be around 0.05.
        shift - number, shift of kagome Fermi level away from default (eV).
        include_H - Boolean, whether to include band 21, which does not couple
            to the kagome lattice on account of being on the opposite face.
        nk - positive integer. Number of k-points in each direction in the MP grid.
    Output: KagomeSubstrate object.

    Last Modified: 2021-08-04
    """
    kagome = KagomeSubstrate(t=t, offset=shift)
    kagome.set_kmesh(nk,nk)
    # Load the substrate bands
    if include_H:
        kagome.add_substrate(coupling=0, subtype='dft', filename=path_to_substrates+'Ag4_band21.dat', nx=7, ny=7)
    for i in (22,23,24,25):
        kagome.add_substrate(coupling=0, subtype='dft', filename=path_to_substrates+'Ag4_band{0}.dat'.format(i), nx=7, ny=7)
    # Set a sensible electron density (so we have somewhere to start)
    n = kagome.nelect_from_chemical_potential(mu=0, T=0.005)
    kagome.set_electrons(nup=n/2, ndown=n/2, method='uniform')
    return kagome
    
def create_graphite(t, shift=0, nk=25):
    """
    Creates a KagomeSubstrate mimicking DCA-Cu/Graphite (three-layer).

    Inputs: t - number, kagome hopping constant (eV). Should be around 0.05.
        shift - number, shift of kagome Fermi level away from default (eV).
        nk - positive integer. Number of k-points in each direction in the MP grid.
    Output: KagomeSubstrate object.

    Last Modified: 2021-08-10
    """
    kagome = KagomeSubstrate(t=t, offset=shift)
    kagome.set_kmesh(nk,nk)
    # Load the substrate bands
    for i in (11,12,13,14):
        kagome.add_substrate(coupling=0, subtype='dft', filename=path_to_substrates+'Graphite_band{0}.dat'.format(i), nx=8, ny=8)
    # Set a sensible electron density (so we have somewhere to start)
    n = kagome.nelect_from_chemical_potential(mu=0, T=0.005)
    kagome.set_electrons(nup=n/2, ndown=n/2, method='uniform')
    return kagome


def create_cu(t, shift=0, include_16=False, nk=25):
    """
    Creates a KagomeSubstrate mimicking DCA-Cu/Cu(111).

    Inputs: t - number, kagome hopping constant (eV). Should be around 0.05.
        shift - number, shift of kagome Fermi level from default (eV).
        include_16 - Boolean, whether to include band 16, which does not couple
            to the kagome lattice on account of being on the opposite face.
        nk - positive integer. Number of k-points in each direction in the MP grid.
    Output: KagomeSubstrate object.

    Last Modified: 2021-04-27
    """
    kagome = KagomeSubstrate(t=t, offset=shift)
    kagome.set_kmesh(nk,nk)
    # Load the substrate bands
    if include_16:
        kagome.add_substrate(coupling=0, subtype='dft', filename=path_to_substrates+'Cu_Band16.dat', nx=8, ny=8)
    for i in (17,18,19):
        kagome.add_substrate(coupling=0, subtype='dft', filename=path_to_substrates+'Cu_Band{0}.dat'.format(i), nx=8, ny=8)
    # Set a sensible electron density (so we have somewhere to start)
    n = kagome.nelect_from_chemical_potential(mu=0, T=0.005)
    kagome.set_electrons(nup=n/2, ndown=n/2, method='uniform')
    return kagome

def create_triangle(t, shift=0, nk=25, bandwidth=2, nrows=1, nbands=1, bandshift=0.1):
    """
    Creates a KagomeSubstrate with a triangular substrate.

    Inputs: t - number, kagome hopping constant (eV). Should be around 0.05.
        shift - number, shift of kagome Fermi level from default (eV).
        nk - positive integer. Number of k-points in each direction in the MP grid.
        bandwidth - number, bandwidth (eV) of a single substrate band. (9*hopping)
        nrows - positive integer. Size of substrate supercell.
        nbands - positive integer. Number of substrate bands.
        bandshift - number. Energy offset (eV) of the substrate bands from each other.
    Output: KagomeSubstrate object.

    Last Modified: 2021-04-30
    """
    kagome = KagomeSubstrate(t=t, offset=shift)
    kagome.set_kmesh(nk,nk)
    # Determine the offsets for each of the substrate bands.
    offsets = np.linspace(-(nbands-1)/2, (nbands-1)/2, nbands)*bandshift + bandwidth/9*1.5
    # The bands are all centered around zero energy.
    # Create the bands
    for o in offsets:
        kagome.add_substrate(coupling=0, subtype='triangle', nx=nrows, ny=nrows, t=bandwidth/9, offset=o)
    n = kagome.nelect_from_chemical_potential(mu=0, T=t*0.1)
    kagome.set_electrons(nup=n/2, ndown=n/2, method='uniform')
    return kagome


def set_coupling(kagome, coupling):
    """Sets all the substrates in kagome to have coupling."""
    for i in range(len(kagome.substrate_list)):
        kagome.change_substrate(i, coupling=coupling)

def sweep_coupling(kagome, cstart, cstep, mmin=0.0015, cmax=np.inf, mu=0, T=0.005, rdiff=1e-4, interval=10, nmin=0, nmax=6, mustep=None, mustepmin=1e-6, pmcompare=True):
    """
    Using kagome as a template, increase coupling until magnetisation vanishes.

    KeyboardInterrupt will return the progress so far, including the current
    partially converged step in case you wanted to continue converging it.

    Inputs: kagome - HubbardSubstrate object, with everything set up.
            This object will not be altered.
        cstart - number. Value of coupling to start at.
        cstep - positive number. Value to increment coupling by each step.
        mmin - number. When local magnetic moments drop below this value, stop.
        cmax - number, optional. Maximum value of coupling to check.
        mu - number, chemical potential.
        T - positive number, temperature.
        rdiff - positive number, target residual.
        interval - positive integer, number of eigensteps between printing.
        nmin - number, optional. Minimum electron number to aim for.
        nmax - number, optional. Maximum electron number to aim for.
        mustep - positive number, optional. Maximum amount to alter offset by each step.
            mustep must be set for nmin and nmax to have an effect.
        mustepmin - positive number, optional. Minimum amount to alter offset by each step.
        pmcompare - Boolean. If True, compare to an analagous system with a paramagnetic
            spin configuration at each step. (Has a computational overhead.)
    Output: list of converged HubbardSubstrate objects (as kagome).
    """
    if nmin > nmax:
        raise ValueError("nmin must be less than nmax")
    if mustep is not None and mustepmin > mustep:
        raise ValueError("mustepmin must be less than mustep")
    try:
        hub_list = [] # Results list
        c = cstart # coupling
        kag = kagome.copy()
        mag = sqrt(kag.local_magnetic_moment())
        laststep = 0 # Used for binary search of offset
        mustep_original = mustep
        first = True
        # Set up PM control
        if pmcompare:
            pm = kagome.copy()
            pm.set_electrons(pm.get_electron_number()/2, pm.get_electron_number()/2, method='uniform')
        # Go through the coupling values
        print('Starting to sweep coupling...')
        while ((mag > mmin) or first) and c < cmax:
            first = False
            # We use the previous results as input to the next loop.
            set_coupling(kag, c)
            print('Converging coupling = {0}, offset = {1}.'.format(c,kag.offset))
            ut.converge(kag, rdiff=rdiff, rdiff_initial=rdiff, T=T, mu=mu, debug=True, interval=interval)
            mag = sqrt(kag.local_magnetic_moment())
            ne = kag.get_lattice_charge()
            print('Coupling: {0}, sqrt<m^2>: {1}, N_e: {2}, offset: {3}.'.format(c, mag, ne, kag.offset))
            if pmcompare:
                set_coupling(pm, c)
                print(f'PM Control: Converging coupling = {c}, offset = {kag.offset}.')
                ut.converge(pm, rdiff=rdiff, rdiff_initial=rdiff, T=T, mu=mu, debug=True, interval=interval)
                print(f'PM Control: Coupling: {c}, sqrt<m^2>: {sqrt(pm.local_magnetic_moment())}, N_e: {pm.get_lattice_charge()}, offset: {pm.offset}.')
                # Compare energy
                kagen = kag.energy(T=T, mu=mu)
                pmen = pm.energy(T=T, mu=mu)
                if kagen <= pmen:
                    hub_list.append(kag.copy())
                else:
                    hub_list.append(pm.copy())
                print(f'Kagome energy: {kagen}. PM Control energy: {pmen}.')
            else:
                hub_list.append(kag.copy())
            if mustep is not None:
                # Check if electron number is in bounds.
                if ne <= nmax and ne >= nmin:
                    # If so, proceed to the next coupling
                    c += cstep
                    # Reset the search parameters.
                    laststep = 0
                    mustep = mustep_original
                else:
                    # If not, start a binary search for optimal offset.
                    if ne > nmax:
                        if laststep < 0:
                            mustep /= 2
                        kag.set_kinetic(offset=kag.offset+mustep)
                        if pmcompare:
                            pm.set_kinetic(offset=pm.offset+mustep)
                        laststep = 1
                    else: # ne < nmin
                        if laststep > 0:
                            mustep /= 2
                        kag.set_kinetic(offset=kag.offset-mustep)
                        if pmcompare:
                            pm.set_kinetic(offset=pm.offset-mustep)
                        laststep = -1
                    # If mustep is too small, undo and break the loop
                    if mustep < mustepmin:
                        kag.set_kinetic(offset=kag.offset+mustep*laststep)
                        if pmcompare:
                            pm.set_kinetic(offset=pm.offset+mustep*laststep)
                        c += cstep
                        laststep = 0
                        mustep = mustep_original
            else:
                c += cstep
                # If our control is lower energy, record its local magnetic moment.
                if pmcompare:
                    if pmen < kagen:
                        mag = sqrt(pm.local_magnetic_moment())
        print('Finished sweeping coupling.')
        return hub_list
    except KeyboardInterrupt:
        print('Aborting...')
        hub_list.append(kag.copy())
        return hub_list

def save_list(kag_list, outdir, ext=''):
    """
    Saves a list of kagome's to pickles in outdir.
    """
    print("Saving files...")
    # Remove trailing slash.
    if outdir[-1] == '/':
        outdir = outdir[:-1]
    for k in kag_list:
        # Generate the desired filename.
        try:
            len(k.u)
        except TypeError:
            u = int(np.round(k.u*1000))
        else:
            u = int(np.round(k.u[0]*1000)) # U in meV.
        t = int(np.round(k.t*1000)) # t in meV.
        c = int(np.round(k.couplings[0]*1000)) # Coupling in meV
        mu = round(-k.offset/k.t,3)
        fname = 'U{0}mu{1}c{2}t{3}{4}.pickle'.format(u,mu,c,t,ext)
        # Check if it already exists.
        i=0
        # Append a number if it does exist to make it unique.
        while os.path.exists(outdir+'/'+fname):
            fname = 'U{0}mu{1}c{2}t{3}{4}_{5}.pickle'.format(u,mu,c,t,ext,i)
            i += 1
        # Write the data
        print("Saving file to {0}/{1}".format(outdir, fname))
        with open(outdir+'/'+fname, 'wb') as f:
            pickle.dump(k, f)

def single_point_fixed_n(kagome, nmin, nmax, step=0.1, stepmin=1e-6, mu=0, T=0.005, rdiff=1e-4, interval=10, adiabatic=True, pmcompare=True):
    """
    Adjust offset until electron number is in range. Returns a list.

    Using kagome as a template, adjusts offset self-consistently until number
    of electrons in the lattice is between nmin and nmax. Returns a list of
    KagomeSubstrate objects.
    
    Returns progress to date on KeyboardInterrupt.

    Inputs: kagome - KagomeSubstrate object, with everything set up.
            This object will not be altered.
        nmin - number. Lower bound of acceptable number of electrons.
        nmax - number. Upper bound of acceptable number of electrons.
        step - number. Increment for offset each step.
        stepmin - number. Minimum allowable increment for the offset.
        mu - number. Chemical potential for ut.converge.
        T - positive number. Temperature for ut.converge.
        rdiff - positive number. Target residual for convergence.
        interval - positive integer. Convergence steps between printing.
        adiabatic - Boolean. If True, the density from one value of the offset
            is used as the input density for the next value.
            If False, input density is always the original kagome density.
        pmcompare - Boolean. If True, compare to an analagous system with a paramagnetic
            spin configuration at each step. (Has a computational overhead.)
    Output: list of KagomeHubbard objects. The last one has the desired number
        of electrons in the lattice.
    """
    if nmin > nmax:
        raise ValueError("nmin must be less than nmax.")
    if stepmin > step:
        raise ValueError("stepmin must be less than step.")
    # Set up.
    hub_list = [] # Results list
    laststep = 0 # Used for binary search offset
    kag = kagome.copy()
    ne = kag.get_lattice_charge()
    if pmcompare:
        pm = kagome.copy()
        pm.set_electrons(pm.get_electron_number()/2, pm.get_electron_number()/2, method='uniform')
    first_time = True
    print('Starting the search offset...')
    try:
        while ne < nmin or ne > nmax or first_time or step < stepmin:
            first_time = False
            print(f'Converging offset = {kag.offset}')
            ut.converge(kag, rdiff=rdiff, rdiff_initial=rdiff, T=T, mu=mu, debug=True, interval=interval)
            ne = kag.get_lattice_charge()
            print(f'Offset: {kag.offset}, N_e: {ne}, sqrt<m^2>: {sqrt(kag.local_magnetic_moment())}')
            if pmcompare:
                print(f'PM control, offset = {kag.offset}')
                pm.set_kinetic(offset=kag.offset)
                ut.converge(pm, rdiff=rdiff, rdiff_initial=rdiff, T=T, mu=mu, debug=True, interval=interval)
                print(f'PM Control: Offset: {pm.offset}, N_e: {pm.get_lattice_charge()}')
                # Compare energy
                kagen = kag.energy(T=T, mu=mu)
                pmen = pm.energy(T=T, mu=mu)
                print(f'Kagome energy: {kagen}. PM Control energy: {pmen}.')
                if pmen < kagen:
                    print(f'PM control is lower energy than kagome. Replacing.')
                    kag = pm.copy()
            hub_list.append(kag.copy())
            if ne > nmax:
                if laststep < 0:
                    step /= 2
                kag.set_kinetic(offset=kag.offset+step)
                laststep = 1
            elif ne < nmin:
                if laststep > 0:
                    step /= 2
                kag.set_kinetic(offset=kag.offset-step)
            if not adiabatic:
                kag.set_electrons(nup=kagome.nup, ndown=kagome.ndown)
        print('Finished searching offset.')
        return hub_list
    except KeyboardInterrupt:
        print('Aborting...')
        hub_list.append(kag.copy())
        return hub_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MFH substrate stuff.")
    parser.add_argument('-t', type=float, help="Hopping constant t (eV)")
    parser.add_argument('-u', type=float, help="Coulomb repulsion U (eV)")
    parser.add_argument('-m','--mu', type=float, help="Chemical potential/offset (units of t)")
    parser.add_argument('-c','--cstart', type=float, default=0, help="Starting coupling (eV)")
    parser.add_argument('-s','--cstep', type=float, default=0.025, help="Coupling step (eV)")
    parser.add_argument('-o','--outdir', default='.', help="Output directory")
    parser.add_argument('-n','--nelect', type=float, nargs=2,
            help="Minimum and maximum allowable number of electrons in the kagome lattice. Optional.")
    parser.add_argument('--substrate', choices=['Ag','Ag4','Cu','triangle','graphite'], help="Which substrate to use.")
    parser.add_argument('-r','--nrows', type=int, default=1, help="Size of substrate supercell (triangle only).")
    parser.add_argument('-w','--bandwidth', type=float, default=2, help="Substrate bandwidth (eV) (triangle only).")
    parser.add_argument('--bandshift', type=float, default=0.1,
            help="Energy shift between substrate bands (eV) (triangle only).")
    parser.add_argument('-b','--nbands', type=int, default=1, help="Number of substrate bands (triangle only).")
    parser.add_argument('-i','--interval', type=int, default=10, help="Number of eigensteps between printing.")
    parser.add_argument('-f','--infile', type=str,
            help="Pickle containing a KagomeSubstrate object to initialise the system.")
    parser.add_argument('--single', action='store_true', help="Do only a single coupling value (requires nelect).")
    args = parser.parse_args()
    # Do argument checking before we commit to calculations.
    if args.cstep <= 0:
        raise ValueError("ctsep must be positive.")
    if not os.path.isdir(args.outdir):
        raise NotADirectoryError("outdir must be a valid directory.")
    if args.single and (args.nelect is None):
        raise TypeError("nelect must be provided for single mode.")
    if args.infile is None:
        if args.t is None:
            raise TypeError("t must be provided if no infile is provided.")
        if args.u is None:
            raise TypeError("u must be provided if no infile is provided.")
        if args.mu is None:
            raise TypeError("mu must be provided if no infile is provided.")
        if args.substrate is None:
            raise TypeError("substrate must be provided if no infile is provided.")
        # Create initial spin density (without substrate).
        kag1 = KagomeHubbardKPoints(nrows=1, ncols=1, u=args.u, t=args.t, allow_fractions=True)
        kag1.set_kmesh(25,25)
        print("Generating initial spin density.")
        if args.nelect is not None:
            n = sum(args.nelect)/2
        else:
            n = 2
        kag_list = ut.sweep_spin(kag1, n, 2, 1e-4, T=0.1*args.t, repeats=3, mmax=1, mu=args.mu*args.t)
        # Choose the minimum energy state
        kag1 = ut.cull_list(kag_list, 0, T=0.1*args.t, mu=args.mu*args.t)[0]
        # Generate the substrate system
        print("Creating substrate system.")
        if args.substrate == 'Ag':
            kagome = create_ag(args.t, -args.mu*args.t)
        elif args.substrate == 'Ag4':
            kagome = create_ag4(args.t, -args.mu*args.t)
        elif args.substrate == 'Cu':
            kagome = create_cu(args.t, -args.mu*args.t)
        elif args.substrate == 'graphite':
            kagome = create_graphite(args.t, -args.mu*args.t)
        elif args.substrate == 'triangle':
            kagome = create_triangle(args.t, -args.mu*args.t, bandwidth=args.bandwidth,
                        nrows=args.nrows, nbands=args.nbands, bandshift=args.bandshift)
        else:
            raise ValueError("{0} is not a known substrate.".format(args.substrate))
        kagome.set_u(args.u)
        kagome.set_electrons(nup=kag1.nup, ndown=kag1.ndown, separate_substrate=True)
    else:
        # An initial file has been provided. Use this as our template.
        with open(args.infile, 'rb') as f:
            kagome = pickle.load(f)
        # Override the template variables if they have been provided on the command line.
        if args.t is not None:
            kagome.set_kinetic(t=args.t)
        else:
            # Otherwise, record what the relevant variables are.
            args.t = kagome.t
        if args.mu is not None:
            kagome.set_kinetic(offset=-args.mu*args.t)
        else:
            args.mu = round(-kagome.offset/args.t,3)
        if args.u is not None:
            kagome.set_u(args.u)
        else:
            args.u = kagome.u[0]
    if args.single:
        # Single-coupling calculation.
        # Set the coupling.
        set_coupling(kagome, args.cstart)
        kag_list = single_point_fixed_n(kagome, args.nelect[0], args.nelect[1],
                T=0.1*args.t, interval=args.interval, adiabatic=False, step=0.1*args.t)
    else:
        # Sweep coupling
        if args.nelect is None:
            kag_list = sweep_coupling(kagome, args.cstart, args.cstep, T=0.1*args.t, interval=args.interval)
        else:
            kag_list = sweep_coupling(kagome, args.cstart, args.cstep, T=0.1*args.t,
                    nmin=args.nelect[0], nmax=args.nelect[1], interval=args.interval,
                    mustep=0.1*args.t, mustepmin=0.001*args.t)
    # Save the list.
    print("Saving files...")
    for k in kag_list:
        # Generate the desired filename.
        u = int(np.round(args.u*1000)) # U in meV.
        t = int(np.round(args.t*1000)) # t in meV.
        c = int(np.round(k.couplings[0]*1000)) # Coupling in meV
        if args.nelect is None:
            mu = args.mu # mu in units of t.
        else:
            mu = round(-k.offset/args.t,3)
        fname = 'U{0}mu{1}c{2}t{3}.pickle'.format(u,mu,c,t)
        # Check if it already exists.
        i=0
        # Append a number if it does exist to make it unique.
        while os.path.exists(args.outdir+'/'+fname):
            fname = 'U{0}mu{1}c{2}t{3}_{4}.pickle'.format(u,mu,c,t,i)
            i += 1
        # Write the data
        print("Saving file to {0}/{1}".format(args.outdir, fname))
        with open(args.outdir+'/'+fname, 'wb') as f:
            pickle.dump(k, f)
