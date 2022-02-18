#!/usr/bin/python3

"""
Substrate class for bands made from ab initio data.
Copyright (C) 2022 Bernard Field, GNU GPL v3+
"""
from warnings import warn

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .substrate import BaseSubstrate

class DFTSubstrate(BaseSubstrate):
    """
    A substrate where the dispersion is given from file as a grid of energies
    """
    def set_params(self, nx=None, ny=None, offset=None, filename=None, array=None, gamma=None):
        """
        Sets numerical parameters for the substrate.

        Inputs: nx - positive integer, number of substrate cells along first
                lattice vector in the supercell.
            ny - positive integer, number of substrate cells along 2nd vector.
            offset - number, vertical energy offset
        """
        if filename is not None and array is not None:
            raise TypeError("set_params only accepts one of filename or array.")
        # Set offset if provided. Keep current value otherwise.
        # Initialise to 0 if no current value.
        if offset is not None:
            self.offset = offset
        elif not hasattr(self, 'offset'):
            self.offset = 0
        # If only gamma provided, set it directly.
        if filename is None and array is None and gamma is not None:
            self.gamma = gamma
        # If filename or array provided, load it.
        if filename is not None:
            self.loadtxt(filename, gamma)
        if array is not None:
            self.loadarray(array, gamma)
        super().set_params(nx,ny)
    #
    def loadtxt(self, filename, gamma=None):
        """
        Loads an array saved by numpy savetxt or readdft.vaspband2np.savetxt_band
        
        Checks for a header of the form '# 3 4 1 Gamma'
        First we have numbers indicating the shape of the array.
        The third dimension should be 1 if present.
        The array must be broadcastable to a 2D array.
        The next non-numeric word must start with G or g to indicate Gamma-centered k-mesh
        or with M or m to indicate an original Monkhorst-Pack scheme.
        If this is not specified here, set gamma manually in the function call.
        If this header is not present, no worries, just specify gamma.

        The first axis indicates coordinate in the first reciprocal lattice
        vector, and second axis along second vector. Monkhorst Pack mesh assumed.

        Inputs: filename - file to load
            gamma - optional Boolean, whether or not the array is Gamma-centered.
                If not specified, will attempt to read it from the file.
                If specified, will override the file.
        """
        # Load the data
        arr = np.loadtxt(filename)
        # Check the header line for if it contains useful information (as per readdft)
        with open(filename,'r') as f:
            line = f.readline()
        # First line is a comment line
        if line[0] == '#':
            words = line[1:].split()
            # Check reported shape
            shape = []
            for w in words:
                if w.isnumeric() and len(shape) < 3:
                    shape.append(int(w))
                else:
                    # This word is the Gamma flag.
                    # First word starts with g or G, is Gamma
                    # First word starts with m or M, is classical Monkhorst
                    # Unless explicitly override
                    if gamma is None:
                        if w[0]=='G' or w[0]=='g':
                            gamma = True
                        elif w[0]=='M' or w[0]=='m':
                            gamma = False
                    break
            # Check shapes
            if len(shape) == 3:
                if shape[2] != 1:
                    raise ValueError("Loaded array is a flattened 3D array, but Substrate only supports 2D.")
            if len(shape) > 0 and tuple(shape[0:2]) != arr.shape:
                # Only warn if a shape is specified
                warn("Text file says shape is "+str(tuple(shape[0:2]))+", but loaded array is shape "+str(arr.shape))
        # Process the array
        self.loadarray(arr, gamma)
    #
    def loadarray(self, arr, gamma=None):
        """
        Converts a 2D numpy array into a dispersion.
        
        Inputs: arr - np.ndarray, numeric. Must be broadcastable to 2D.
            gamma - boolean. Whether or not the array is Gamma centered.
                If not specified, will use existing value for gamma if
                present. If not set, will raise a warning and use a defualt
                value (currently True).
        """
        # First some checks
        if gamma is None:
            # If gamma is not specified...
            if hasattr(self, 'gamma'):
                # use existing value, if applicable
                gamma = self.gamma
            else:
                # If no existing value, warn and fallback.
                warn("gamma not specified. Using default value.")
                gamma = True
        # Check array shape. Should be 2D.
        arr = np.asarray(arr) # Duck typing
        # If 3D,
        if arr.ndim >= 3:
            # If too many non-singleton dimensions, we got problems
            if sum(np.array(arr.shape)>1) > 2:
                raise ValueError("Array has too many non-singleton dimensions.")
            # If exactly two dimensions, just squeeze the array
            elif sum(np.array(arr.shape)>1) == 2:
                arr = arr.squeeze()
            # If one dimension, want to do things while preserving order
            elif sum(np.array(arr.shape)>1) == 1:
                if arr.shape[0] == 1:
                    arr = arr.reshape((1,-1))
                else:
                    arr = arr.reshape((-1,1))
            else: # 0D
                arr = arr.reshape((1,1))
        # If 1D or 0D, extend
        if arr.ndim <= 1:
            arr = arr.reshape((-1, 1))
        # We have now converted arr to be 2D.
        # We shall now record the data
        self.datagrid = arr.copy()
        self.gamma = gamma
        # We now build the interpolation function
        self._initialise_dispersion()
    #
    def _initialise_dispersion(self):
        # Determine the k-mesh
        # Monkhorst-pack offsets
        if self.gamma:
            mp_offset = np.array([0,0,0])
        else:
            mp_offset = (1 - np.mod(self.datagrid.shape, 2))/2
        # Grid dimensions
        nx, ny = self.datagrid.shape
        # Get x and y coordinate axes, with padding.
        x = (np.linspace(-1, nx, num=nx+2, endpoint=True)+mp_offset[0])/nx
        y = (np.linspace(-1, ny, num=ny+2, endpoint=True)+mp_offset[1])/ny
        # Create padded data (padding is for periodic boundaries)
        paddata = np.hstack((self.datagrid[:,-1:], self.datagrid, self.datagrid[:,0:1]))
        paddata = np.vstack((paddata[-1], paddata, paddata[0]))
        # Create the linear interpolator.
        # Use bilinear interpolation
        self.interpolated = RectBivariateSpline(x, y, paddata, kx=1, ky=1)
    #
    def dispersion(self, k):
        # Mod wraps to periodic boundary conditions
        # nx and ny rescale the Brillouin zone
        # float converts the array output of scipy's interpolate
        try:
            self.interpolated
        except AttributeError:
            raise DispersionNotSetError("The dispersion has not been set. Set the "
                                        "dispersion using the loadtxt or loadarray methods "
                                        "of the substrate.")
        else:
            result = self.interpolated(np.mod(k[0]/self.nx, 1), np.mod(k[1]/self.ny, 1), grid=False) + self.offset
            if result.ndim == 0:
                return float(result)
            else:
                return result


class DispersionNotSetError(Exception):
    pass
