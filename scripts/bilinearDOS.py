#!/usr/local/bin/python3

import sys
from math import pi
import math

import numpy as np
from scipy.integrate import quad

"""
Trial script for calculating the DOS from a bilinear interpolation

This can be done analytically, so I shall do it analytically.

First step is defining the constant-energy contours.
The challenges there are ensuring the limits of the contours are within the
domain, and catching all the edge cases.

Next step is integrating along the contours to get the DOS.
This can be done analytically.

We can also integrate the region bounded by the contours to get the number of states.
This can be done analytically.

We can numerically integrate the DOS to get the number of states as well,
allowing us to check the self-consistency of the method.

The functions you want as an external user are the following:
    full_dos(e,e11,e12,e21,e22) - calculate the DOS at energy e (with spin degeneracy).
    nstates(e,e11,e12,e21,e22) - calculate number of states below energy e (without spin degeneracy).
    integrate_dos(e,e11,e12,e21,e22) - numerically integrate the DOS up to energy e.
        I recommend nstates instead as it is analytic, but integrate_dos is useful for comparison.

Arguments from command line:
    e, e11, e12, e21, e22
    e - the energy to inspect
    e11, e12, e21, e22 - the energies at the left/right lower/upper corners of the Brillouin zone.
"""

def abcd(e11,e12,e21,e22) -> tuple:
    """Convert corner energies into polynomial coefficients"""
    return e11, e21-e11, e12-e11, e22-e12-e21+e11

def determinant(e,a,b,c,d) -> float:
    """The derivative of the contour dy/dx has the same sign as this."""
    return a*d - b*c - d*e

def check_in_range(e,e11,e12,e21,e22) -> bool:
    """Checks if e is inside the box."""
    return ((e>=e11) or (e>e12) or (e>e21) or (e>e22)) and ((e<=e11) or (e<e12) or (e<e21) or (e<e22))

def check_all_flat(e11,e12,e21,e22) -> bool:
    """Check if the dispersion is flat."""
    return ((e11==e12) and (e11==e21) and (e11==e22))

def check_vert_exists(e11,e12,e21,e22) -> bool:
    """Check if a vertical contour exists in the box."""
    a,b,c,d = abcd(e11,e12,e21,e22)
    # First case is all contours are vertical. Second is more generic.
    return ((d==0) and (c==0)) or ((d!=0) and (-c/d>=0) and (-c/d<1))

def check_horizontal_exists(e11,e12,e21,e22) -> bool:
    """Check if a horizontal contour exists in the box."""
    a,b,c,d = abcd(e11,e12,e21,e22)
    # First case is all contours are horizontal. Second is more generic.
    return ((d==0) and (b==0)) or ((d!=0) and (-b/d>=0) and (-b/d<1))

def check_contour_flat(e,e11,e12,e21,e22) -> bool:
    """Checks if the chosen contour is horizontal or vertical."""
    return determinant(e,*abcd(e11,e12,e21,e22)) == 0

def check_need_two_curves(e,e11,e12,e21,e22) -> bool:
    """Checks if we need two separate contours."""
    # Strictly speaking, this is two separate contours for some energy.
    # A full check would also determine if the second contour is not null.
    # But a null contour is okay to handle, so I can spare some logic.
    return check_horizontal_exists(e11,e12,e21,e22) and check_vert_exists(e11,e12,e21,e22)

def get_double_limits(e,e11,e12,e21,e22) -> tuple:
    """Use if check_need_two_curves and not check_contour_flat. Returns 2*2 tuple"""
    a,b,c,d = abcd(e11,e12,e21,e22)
    det = determinant(e,a,b,c,d)
    if b == 0:
        x0 = det/abs(det)*10
    else:
        x0 = (e-a)/b
    if b == -d:
        x1 = -det/abs(det)*10
    else:
        x1 = (e-a-c)/(b+d)
    if det > 0:
        return ((0, max(x1,0)), (min(x0,1), 1))
    else:
        return ((0, max(x0,0)), (min(x1,1),1))

def get_vertical_contour(e,e11,e12,e21,e22) -> float:
    """Use if check_contour_flat and check_vert_exists and not check_all_flat. Returns scalar"""
    a,b,c,d = abcd(e11,e12,e21,e22)
    if d == 0:
        return (e-a)/b
    else:
        return -c/d

def get_horizontal_contour(e,e11,e12,e21,e22) -> float:
    """Use if check_contour_flat and check_horizontal_exists and not check_all_flat. Returns scalar"""
    a,b,c,d = abcd(e11,e12,e21,e22)
    if d == 0:
        return (e-a)/c
    else:
        return -b/d

def get_single_limits(e,e11,e12,e21,e22) -> tuple:
    """Use if not check_contour_flat and not check_need_two_curves. Returns 2-tuple"""
    a,b,c,d = abcd(e11,e12,e21,e22)
    det = determinant(e,a,b,c,d)
    if b == 0:
        x0 = -det/abs(det)*10
    else:
        x0 = (e-a)/b
    if b == -d:
        x1 = det/abs(det)*10
    else:
        x1 = (e-a-c)/(b+d)
    if det > 0:
        xl,xh = (max(0,x0), min(1,x1))
    else:
        xl,xh = (max(0,x1), min(1,x0))
    # x low, x high
    # If a horizontal exists, the break will cause the limits to wrap around
    if xl > 1: xl = 0
    if xh < 0: xh = 1
    return xl, xh

def calc_dos_single(xL, xH, e11, e12, e21, e22) -> float:
    """Given limits, system: return DOS from a single contour"""
    a,b,c,d = abcd(e11,e12,e21,e22)
    if d == 0:
        if c == 0: return np.inf
        return 2 * (xH - xL)/abs(c)
    else:
        if (xL*d+c == 0) or (xH*d+c == 0): return np.inf
        return 2 * abs(math.log((xH*d+c)/(xL*d+c))/d)

def calc_dos_vertical(x, e11, e12, e21, e22) -> float:
    """Returns DOS for a vertical contour at given x"""
    a,b,c,d = abcd(e11,e12,e21,e22)
    if b == 0 or d+b == 0: return np.inf
    if d == 0:
        return 2/abs(b)
    else:
        return 2 * abs(math.log((d+b)/b)/d)

def full_dos(e, e11, e12, e21, e22) -> float:
    """Does all the logic for determining the DOS and returns it"""
    box = (e11,e12,e21,e22)
    if not check_in_range(e,*box):
        # Energy out of range, so DOS is zero.
        return 0.0
    if check_all_flat(*box):
        # Dispersion is entirely flat, so DOS is infinite
        return np.inf
    if check_contour_flat(e,*box):
        if check_need_two_curves(e,*box):
            # Two intersecting contours. Logarithmic singularity.
            return np.inf
        # Otherwise, we only need one of a horizontal or vertical contour
        if check_vert_exists(*box):
            return calc_dos_vertical(get_vertical_contour(e,*box), *box)
        if check_horizontal_exists(*box):
            return calc_dos_single(0, 1, *box)
        else:
            raise Exception("Contour is flat, but neither vert nor horizontal exists. I don't know how you got here.")
    if check_need_two_curves(e,*box):
        lim1, lim2 = get_double_limits(e,*box)
        return calc_dos_single(*lim1,*box) + calc_dos_single(*lim2,*box)
    return calc_dos_single(*get_single_limits(e,*box),*box)

def calc_area_under_curve(e,xL,xH,e11,e12,e21,e22) -> float:
    """Calculate the area under a curved contour"""
    a,b,c,d = abcd(e11,e12,e21,e22)
    if d == 0:
        return xH*(e-a-b*xH/2)/c - xL*(e-a-b*xL/2)/c
    else:
        return (d*e-a*d+b*c)/(d**2) * math.log((c+xH*d)/(c+xL*d)) - b/d * (xH - xL)

def nstates(e,e11,e12,e21,e22,debug=False) -> float:
    """Calculate the area bounded by the energy contour (i.e. number of states)"""
    box = (e11,e12,e21,e22)
    if not check_in_range(e,*box):
        if e >= e11:
            # Energy is above the band maximum
            if debug: print("Energy is above the band maximum")
            return 1.0
        else:
            # Energy is below the band minimum
            if debug: print("Energy is below the band minimum")
            return 0.0
    if check_all_flat(*box):
        # Dispersion is flat and energy is inside it
        # In truth, the dispersion is undefined
        # But we'll call it 1.
        if debug: print("Dispersion is flat and occupied.")
        return 1.0
    if check_contour_flat(e,*box):
        if check_need_two_curves(e,*box):
            x = get_vertical_contour(e,*box)
            y = get_horizontal_contour(e,*box)
            if debug: print(f"We have vertical and horizontal contours at x={x} and y={y}.")
            # Two rectangles.
            if e11 <= e:
                if debug: print("Lower left plus upper right.")
                return x*y + (1-x)*(1-y)
            else:
                if debug: print("Upper left plus lower right.")
                return x*(1-y) + (1-x)*y
        elif check_horizontal_exists(*box):
            y = get_horizontal_contour(e,*box)
            if debug: print(f"We have horizontal contour at y={y}.")
            if e11 <= e:
                return y
            else:
                return 1-y
        elif check_vert_exists(*box):
            x = get_vertical_contour(e,*box)
            if debug: print(f"We have vertical contour at x={x}.")
            if e11 <= e:
                return x
            else:
                return 1-x
        else:
            raise Exception("Contour is flat, but neither vert nor horizontal exists. I don't know how you got here.")
    lower_corners = [ e >= x for x in box] # Corners which are lower energy than E.
    if debug: print("lower_corners:",lower_corners)
    # Check some quick edge cases. These should in principle be caught elsewhere, but just in case...
    if sum(lower_corners) == 0:
        if debug: print("Energy below band minimum")
        return 0.0
    if sum(lower_corners) == 4:
        if debug: print("Energy above band maximum")
        return 1.0
    if check_need_two_curves(e,*box):
        lim1, lim2 = get_double_limits(e,*box)
        if debug: print("Double limits detected:",lim1,lim2)
        # Even if we have potential for two curves, some curves might be null.
        if sum(lower_corners) == 2:
            if debug: print("We really have two curves.")
            # We have two separate curves.
            # lim1 is on the x=0 side. lim2 is on the x=1 side.
            # First handle lim1
            if lower_corners[0]:
                if debug: print("Lower left")
                area = calc_area_under_curve(e,*lim1,*box)
            elif lower_corners[1]:
                if debug: print("Upper left")
                area = lim1[1] - lim1[0] - calc_area_under_curve(e,*lim1,*box)
            else:
                raise Exception("Expected e11 or e12 to be lower energy. Found neither. Bug in logic.")
            # Now handle lim2
            if lower_corners[2]:
                if debug: print("Lower right")
                area += calc_area_under_curve(e,*lim2,*box)
            elif lower_corners[3]:
                if debug: print("Upper right")
                area += lim2[1] - lim2[0] - calc_area_under_curve(e,*lim2,*box)
            else:
                raise Exception("Expected e21 or e22 to be lower energy. Found neither. Bug in logic.")
            a,b,c,d = abcd(*box)
            # If the energy is above the cross-over point,
            # we also need to include the space between the two curves.
            if e > a - b*c/d:
                if debug: print("Energy above cross-over point. Adding region between contours.")
                area += lim2[0] - lim1[1]
            return area
        else:
            if debug: print("There's only one non-null contour")
            # There can only be one curve.
            # Reduce it to a single limit and carry on to the single curve case.
            if lim1[0] == lim1[1]:
                lim = lim2
            elif lim2[0] == lim2[1]:
                lim = lim1
            else:
                raise Exception("check_need_two_curves is True. Corner counting indicates there is only one curve. However, neither lim1 or lim2 were null. Bug in logic.")
    # Invoke single curve logic.
    else:
        lim = get_single_limits(e,*box)
    if debug: print("Limits of single contour:",lim)
    if sum(lower_corners) == 1 or sum(lower_corners) == 3:
        # Our contour bounds a single corner.
        if debug: print("Single corner is bounded")
        # Corners with y=0
        if (((lower_corners[0] or lower_corners[2]) and (sum(lower_corners) == 1))
                or ((lower_corners[1] and lower_corners[3]) and (sum(lower_corners) == 3))):
            if debug: print("Corner is at y=0")
            area = calc_area_under_curve(e,*lim,*box)
        else:
            # Corner with y=1
            if debug: print("Corner is at y=1")
            area = lim[1] - lim[0] - calc_area_under_curve(e,*lim,*box)
        if sum(lower_corners) == 1:
            # Integrate inside the contour
            if debug: print("Integrate inside the contour")
            return area
        else:
            # Integrate outside the contour
            if debug: print("Integrate outside the contour")
            return 1 - area
    elif sum(lower_corners) == 2:
        # Our contour divides the square in half.
        if debug: print("Contour divides the square in half. Two corners bounded.")
        if lim[0] == 0 and lim[1] == 1:
            if debug: print("Contour cuts horizontally.")
            # Contour is horizontal
            if lower_corners[0]:
                if debug: print("Integrate lower half.")
                return calc_area_under_curve(e,*lim,*box)
            else:
                if debug: print("Integrate upper half.")
                return 1 - calc_area_under_curve(e,*lim,*box)
        else:
            # Contour is vertical
            if debug: print("Contour cuts vertically.")
            area = calc_area_under_curve(e,*lim,*box)
            if debug: print("Area under curve is", area)
            # Must determine if slope is positive or negative.
            det = determinant(e,*abcd(*box))
            if debug: print("Determinant is", det)
            if lower_corners[0]:
                # Integrate left side of box
                if debug: print("Integrate left half.")
                if det < 0:
                    # Area under curve is on left
                    if debug: print("Area under curve is on left.")
                    return area + lim[0]
                else:
                    # Area under curve is on right
                    if debug: print("Area under curve is on right.")
                    return lim[1] - area
            else:
                # Integrate right side of box
                if debug: print("Integrate right half.")
                if det < 0:
                    # Area under curve is on left
                    if debug: print("Area under curve is on left.")
                    return 1 - lim[0] - area
                else:
                    # Area under curve is on right
                    if debug: print("Area under curve is on right.")
                    return 1 - lim[1] + area
    else:
        raise Exception("I don't know how you got here. I should have covered all sum(lower_corners) cases.")
    raise Exception("I don't know how you got here. I should have returned something by now.")

def integrate_dos(e,e11,e12,e21,e22) -> float:
    """
    Numerically integrates the DOS up to energy e.
    I recommend nstates as it is exact, but this is for cross-checking.
    Do not use if check_all_flat is True.
    """
    box = (e11,e12,e21,e22)
    # If energy is below the band minimum, nothing to integrate.
    if e <= min(box):
        return 0.0
    # Determine if we need any break-points.
    # There is a potential discontinuity at the band edges.
    points = [max(box)]
    # If we have two intersecting contours, that presents a logarithmic singularity.
    if check_horizontal_exists(*box) and check_vert_exists(*box):
        a,b,c,d = abcd(*box)
        # We have the flat contours at this energy.
        points.append(a - b*c/d)
    # Integrate
    return quad(full_dos, min(box), e, args=box, points=points)


if __name__ == "__main__":
    e, e11, e12, e21, e22 = [ float(x) for x in sys.argv[1:] ]
    box = (e11,e12,e21,e22)
    print(f"E = {e}, e11 = {e11}, e12 = {e12}, e21 = {e21}, e22 = {e22}")
    print("a = {0}, b = {1}, c = {2}, d = {3}".format(*abcd(*box)))
    print("determinant =", determinant(e, *abcd(*box)))
    print("Is E in range?", check_in_range(e,*box))
    print("Is the dispersion flat?", check_all_flat(*box))
    print("Is there a vertical contour in the box?", check_vert_exists(*box))
    print("Is there a horizontal contour in the box?", check_horizontal_exists(*box))
    print("Is our contour flat?", check_contour_flat(e,*box))
    print("Do we need two curves?", check_need_two_curves(e,*box))
    if check_in_range(e,*box):
        if check_need_two_curves(e,*box) and not check_contour_flat(e,*box):
            lim1, lim2 = get_double_limits(e,*box)
            print("Contour x limits:", lim1, lim2)
            print("DOS of first contour:", calc_dos_single(*lim1,*box))
            print("DOS of second contour:", calc_dos_single(*lim2,*box))
        elif check_contour_flat(e,*box) and not check_all_flat(*box):
            if check_vert_exists(*box):
                x = get_vertical_contour(e,*box)
                print("Vertical contour at x =", x)
                if check_horizontal_exists(*box):
                    print("DOS of vertical contour:", np.inf)
                else:
                    print("DOS of vertical contour:", calc_dos_vertical(x,*box))
            if check_horizontal_exists(*box):
                y = get_horizontal_contour(e,*box)
                print("Horizontal contour at y =", y)
                if check_vert_exists(*box):
                    print("DOS of horizontal contour:", np.inf)
                else:
                    print("DOS of horizontal contour:", calc_dos_single(0,1,*box))
        elif not check_all_flat(*box): # Implied, if not check_need_two_curves and not check_contour_flat
            lim = get_single_limits(e,*box)
            print("Contour x limits:", lim)
            print("DOS of contour:", calc_dos_single(*lim,*box))
    print("DOS of system:", full_dos(e,*box))
    print("Number of states:", 2*nstates(e,*box, debug=True))
    if not check_all_flat(*box):
        print("Integrated DOS:", integrate_dos(e,*box))
