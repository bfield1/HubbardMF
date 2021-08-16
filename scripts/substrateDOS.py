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

import numpy as np

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
