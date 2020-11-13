#!/usr/bin/python3
"""Miscellaneous functions intended for use with the Hubbard package."""

import numpy as np


class ConvergenceWarning(UserWarning):
    """
    Warning to raise when a density mixing algorithm
    does not achieve its convergence target.
    """
    pass

class MixingError(ValueError):
    """
    Exception to raise when a density mixing algorithm
    experiences a fatal error, such as when the algorithm
    (which may have been originally designed for unbounded
    parameters) wants to take the electron density out of
    bounds.
    """
    pass

def fermi_distribution(e,T,mu):
    """
    The Fermi distribution.
    
    Allows numpy array input for e and mu.
    Handles T=0, assuming e=mu should return 1.
    Inputs: e - energy
        T - temperature
        mu - chemical potential
    Output: scalar between 0 and 1 if inputs are scalar.
        ndarray of scalars matching input size if otherwise.
    Last Modified: 2020-07-23
    """
    if T==0:
        # T==0 case is a step function.
        comp = e<=mu # E below mu is 1, above is 0.
        if isinstance(comp,bool):
            # Inputs were all scalar.
            return int(comp)
        else:
            # Inputs are not scalar; probably a list.
            return np.asarray(comp).astype(int)
    else:
        # Generic T case.
        return 1/(np.exp((e-mu)/T)+1)

def random_density(n,total,alpha=None):
    """
    Produce a random electron density from the Dirichlet distribution.
    
    Uses an empirically chosen alpha parameter.
    You can specify alpha if you like.

    Inputs: n - positive integer, number of sites/length of list.
        total - non-negative number, number of electrons.
        alpha - optional positive number. If set, this value
            of alpha is used rather than the default.
    Output: numpy array of length n of random numbers between 0 and 1
        which sum to total.

    Last Modified: 2020-11-13
    """
    rng = np.random.default_rng()
    # Impossible cases
    if n < 1 or total < 0 or total > n:
        raise ValueError("n or total is out of bounds.")
    # Trivial cases
    if n==1:
        return np.array([total])
    if total == 0:
        return np.zeros(n)
    if total == n:
        return np.ones(n)
    # If n/total > 0.5, we solve for hole density instead.
    if total/n > 1/2:
        holes = True
        total = n - total
    else:
        holes = False
    # Alpha, the Dirichlet parameter.
    if alpha is None:
        alpha = choose_alpha(n,total)
    # Get the random density.
    density = bounded_random_numbers_with_sum_dirichlet(n,total,alpha)
    if holes:
        # Convert from hole density to electron density.
        density = 1 - density
    return density

def choose_alpha(n,total):
    """
    An empirically chosen minimum practical alpha for Dirichlet.
    
    Picks an alpha which, when used with
    bounded_random_numbers_with_sum_dirichlet,
    will result in 1/10 samples being valid. I consider
    this to be a reasonable compromise for low alpha for
    more spread while maintaining okay performance.
    Alpha is chosen by an empirical formula obtained by
    numerical fitting. It is tested for N<=1000 and
    total/N <= 0.6. Does not apply for total/N close to
    1, where alpha should asymptote to infinity (but my
    formula does not).
    I give alpha a lower bound of 1.

    Further testing has found that, typically, the ground state electron
    density will have substantially less spread than what is given
    by this function. You may wish to use a larger value of alpha
    for faster convergence.

    Inputs: n - positive integer, length of list to return.
        total - non-negative number. Value list should sum to.
    Output: alpha - positive number.

    Last Modified: 2020-07-09
    """
    return max(1, 0.028 * n**0.38 * np.exp(7*total/n))

def bounded_random_numbers_with_sum_dirichlet(n,total,alpha):
    """
    Return a list of n random numbers between 0 and 1 which sum to total.
    
    Uses a Dirichlet distribution, which generates numbers between
    0 and 1 which sum to 1.
    Larger alpha is more likely to converge but has a smaller spread.
    Specifically, small alpha has an exponential-like distribution,
    with many values close to zero but a long tail.
    Large alpha has a bell-curve distribution around total/n.
    alpha should be at least 1. 5 or 10 is okay for half filling.
    If more than half filling, I recommend doing 1-total instead,
    although alpha of 50 or 100 can also be used.

    Inputs: n - positive integer, length of list to return.
        total - non-negative number. Value list should sum to.
        alpha - positive number, parameter for Dirichlet. Larger is less spread.
    Output: (n,) ndarray of numbers between 0 and 1 which sum to total.

    Last Modified: 2020-11-13
    """
    rng = np.random.default_rng()
    # Catch impossible cases.
    if n<1 or total<0 or total>n:
        raise ValueError("n or total are out of bounds.")
    # Catch trivial cases.
    if n==1:
        return np.array([total])
    if total==0:
        return np.zeros(n)
    if total==n:
        return np.ones(n)
    # Repeatedly try to generate a valid distribution.
    while True:
        # We multiply by total to get it to sum to total.
        vals = rng.dirichlet(alpha*np.ones(n))*total
        # However, this may make some of the numbers larger
        # than 1. Only accept the trial if all values are
        # less than or equal to 1.
        if vals.max() <= 1:
            return vals

def random_points_density(n,total):
    """
    Give a list of length n, with 'total' sites as 1, while others are zero.
    
    In the case of non-integer 'total', puts the fractional
    component in one of the sites.

    Inputs: n - positive integer, length of list to return.
        total - non-negative number. Value list should sum to.
    Output: (n,) ndarray of numbers between 0 and 1 which sum to total.

    Last Modified: 2020-11-13
    """
    rng = np.random.default_rng()
    # Initialise empty array.
    density = np.zeros(n)
    # Set some rando sites to 1.
    density[rng.choice(n,size=int(total),replace=False)] = 1
    # Handle the fractional component
    if total - int(total) > 0 and total < n:
        # Choose a random site not yet occupied.
        indices = np.where(density==0)[0]
        # Put the fractional component there.
        density[rng.choice(indices)] = nelect-int(nelect)
    return density
