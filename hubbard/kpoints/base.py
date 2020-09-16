#!/usr/bin/python3

"""
Base class for Mean Field Hubbard model with Brillouin zone sampling.

The primary change from the Gamma point only code is in the eigenstep
method, which has been rebuilt to allow k-point sampling.
The kinetic energy matrix is also no longer stored but must be
dynamically constructed for each k-point.

Created: 2020-09-16
Last Modified: 2020-09-16
Author: Bernard Field
"""

import numpy as np

from hubbard.base import Hubbard

class HubbardKpoints(Hubbard):
    """Base class for Mean Field Hubbard model with Brillouin zone sampling."""

    def get_kinetic(self,k):
        """
        MUST BE OVERRIDDEN IN CHILD CLASS.
        This method is meant to be given a momentum k and return
        an nsites*nsites matrix representing the kinetic energy.

        Expected Output: (self.nsites,self.nsites) ndarray
        """
        raise NotImplementedError

    def move_electrons(self,up=0,down=0):
        """Inherited method without implementation here."""
        raise NotImplementedError

    
