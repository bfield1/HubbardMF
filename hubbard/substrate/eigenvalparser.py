#!/usr/bin/python3

"""
This file loads a VASP EIGENVAL file.
"""

import numpy as np

class EigenvalParser():
    """Parser and container for EIGENVAL files"""
    def __init__(self):
        self.nelect = None
        self.nbands = None
        self.nkpt = None
        self.kpoints = None
        self.weights = None
        self.bands = None
        self.occupation = None

    def readEigenval(self, filename, outcar=None, fermi=None):
        """Reads and loads an EIGENVAL file. Spinless only at the moment"""
        with open(filename, 'r') as f:
            # The first 5 lines have nothing we want
            for _ in range(5):
                # Doiscard first 5 lines
                f.readline()
            # Next line is number of electrons, number of k-points, number of bands.
            self.nelect, self.nkpt, self.nbands = [int(x) for x in f.readline().split()]
            # Set up the arrays.
            self.kpoints = np.empty((self.nkpt, 3))
            self.weights = np.empty(self.nkpt)
            self.bands = np.empty((self.nkpt, self.nbands))
            self.occupation = np.empty((self.nkpt, self.nbands))
            # Iterate
            for ikpt in range(self.nkpt):
                # Skip blank line
                f.readline()
                # Read k-point
                tmpline = [float(x) for x in f.readline().split()]
                self.kpoints[ikpt] = tmpline[0:3]
                self.weights[ikpt] = tmpline[3]
                # Read band data
                for iband in range(self.nbands):
                    # Columns are band number, energy, occupation
                    tmpline = [float(x) for x in f.readline().split()]
                    self.bands[ikpt, iband] = tmpline[1]
                    self.occupation[ikpt, iband] = tmpline[2]
        # Finished with EIGENVAL
        # Correct energies by the Fermi energy if applicable.
        if outcar is not None:
            # Read OUTCAR and find Fermi
            with open(outcar, 'r') as f:
                for line in f:
                    if 'E-fermi' in line:
                        # Record the fermi energy
                        fermi = float(line.split()[2])
                        break
        # Finished with OUTCAR
        if fermi is not None:
            # Subtract off the Fermi energy
            self.bands -= fermi
