#!/usr/local/bin/python3

"""
Calculate the density of states (DOS) of DFT-derived substrates

Script for calculating DOS and related properties from the DFT-derived band
structures (as produced by hubbard.substrate.vaspband2np).
We are given a grid of energy points which fill a Monkhorst-pack grid in
the Brillouin zone. We assume bilinear interpolation between these points.
We also assume periodic boundary conditions.
With those assumption, we can analytically calculate the DOS of the band.
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from hubbard.substrate.dftsubstrate import DFTSubstrate
import scripts.bilinearDOS as bilinearDOS

# We use DFTSubstrate to import the data array (rather than duplicating the logic)

def load_band(filename, text=None):
    """
    Reads and returns band structure array from file. Auto-detect file type

    Inputs: filename - string.
        text - optional Boolean. Specify if file is text-type or numpy array.
            Will auto-detect based on filename if None (.npy is numpy array,
            otherwise text).
    Output: numpy array.
    """
    if text is None:
        text = filename[-4:] != '.npy'
    if text:
        dftsub = DFTSubstrate(filename=filename)
    else:
        nparray = np.load(filename)
        # The position of the Gamma point isn't actually important.
        # I set gamma=True to suppress warnings for leaving unset.
        dftsub = DFTSubstrate(array=nparray, gamma=True)
    # Get the processed data grid from the substrate object.
    return dftsub.datagrid

# Now we can calculate the DOS and related properties.
def dos(e, array):
    """
    From a grid of energy points, calculates the DOS at energy e.

    Inputs: e - scalar, energy to measure at.
        array - 2D array of numbers, band dispersion sampled on a Monkhorst-
            Pack grid.
    Output: number - the density of states.
    """
    # Ensure it is np.array so can access the methods.
    array = np.asarray(array)
    # Get the dimensions
    nx, ny = array.shape
    # Iterate over the array, accumulating the DOS
    dossum = 0
    for x in range(nx):
        for y in range(ny):
            # Periodic boundary conditions
            x2 = (x+1) % nx
            y2 = (y+1) % ny
            dossum += bilinearDOS.full_dos(e, array[x,y], array[x,y2], array[x2,y], array[x2,y2])
    # Normalise by number of k-points.
    dossum /= nx*ny
    return dossum

def nstates(e, array):
    """
    From a grid of energy points, calculates the number of states below energy e
    
    Inputs: e - scalar, energy to measure at.
        array - 2D array of numbers, band dispersion sampled on a Monkhorst-
            Pack grid.
    Output: number - the number of states.
    """
    # Ensure it is np.array so can access the methods.
    array = np.asarray(array)
    # Get the dimensions
    nx, ny = array.shape
    # Iterate over the array, accumulating the DOS
    nstatessum = 0
    for x in range(nx):
        for y in range(ny):
            # Periodic boundary conditions
            x2 = (x+1) % nx
            y2 = (y+1) % ny
            nstatessum += bilinearDOS.nstates(e, array[x,y], array[x,y2], array[x2,y], array[x2,y2])
    # Normalise by number of k-points.
    nstatessum /= nx*ny
    # Times two for spin up and down
    nstatessum *= 2
    return nstatessum

def plot_dos(array, emin=None, emax=None, ymax=None, npoints=100):
    """
    Plots the DOS

    Inputs:
        array - 2D array of numbers, band dispersion sampled on a Monkhorst-
            Pack grid.
        emin - number. Lower bound to plot. Default band minimum.
        emax - number. Upper bound to plot. Default band maximum.
        ymax - number. Upper bound to plot. Default no limit.
    Output: fig, ax
    """
    array = np.asarray(array)
    # Get limits
    if emin is None:
        emin = array.min()
    if emax is None:
        emax = array.max()
    # Generate data.
    x = np.linspace(emin, emax, npoints)
    y = [ dos(e,array) for e in x ]
    # Plot
    fig, ax = plt.subplots()
    ax.plot(x,y)
    if ymax is not None:
        ax.set_ylim(0,ymax)
    # Annotate
    ax.set_xlabel('Energy')
    ax.set_ylabel('DOS')
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Calculate DOS or related quantity for a DFT substrate band.")
    parser.add_argument('energy', type=float, help="Energy to calculate at.")
    parser.add_argument('band', help="File containing band data.")
    parser.add_argument('-d','--dos', action='store_true', help="Calculate the DOS")
    parser.add_argument('-n','--nstates', action='store_true', help="Calculate number of states")
    parser.add_argument('-a','--average', type=float, help="Calculate the average DOS between this energy and the other energy.")
    parser.add_argument('-i','--info', action='store_true', help="Print information on the band.")
    parser.add_argument('-t','--type', choices=['txt','npy'], help="File type of band (auto-detects if not specified)")
    parser.add_argument('-p','--plot', action='store_true', help="Plots the DOS.")
    args = parser.parse_args()
    # If no flags are given, assume we want to calculate DOS.
    if not args.dos and not args.nstates and not args.info and (args.average is None) and not args.plot:
        args.dos = True
    # Load the array
    if args.type == 'txt':
        text = True
    elif args.type == 'npy':
        text = False
    else:
        text = None
    array = load_band(args.band, text=text)
    # Calculate info
    if args.info:
        print("Band minimum:", array.min())
        print("Band maximum:", array.max())
        print("K-mesh:", array.shape)
    # Calculate DOS
    if args.dos:
        print("DOS:", dos(args.energy, array))
    # Calculate nstates
    if args.nstates:
        print("Nstates:", nstates(args.energy, array))
    # Calculate average DOS
    if args.average is not None:
        n1 = nstates(args.average, array)
        n2 = nstates(args.energy, array)
        print("Average DOS:", (n2 - n1)/(args.energy - args.average))
    if args.plot:
        fig, ax = plot_dos(array)
        plt.show()
