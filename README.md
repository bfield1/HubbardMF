    HubbardMF - Python package for simulations of a Mean Field Hubbard model.
    Copyright (C) 2021 Bernard Field

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

This is code for running simulations of a Mean Field Hubbard (MFH) model.

If you use this code in a published work, please cite the following references:
-To be published-

# Theory

The Hubbard model is a simple tight-binding model for interacting electrons.
The Hubbard Hamiltonian contains two terms. The first term is a hopping term between adjacent lattice sites, which is standard for tight-binding models. The second term is an on-site interaction with strength U between electrons of opposite spin on the same site.
However, the Hubbard Hamiltonian is rarely exactly solveable. Solving it requires some approximations. The simplest approximation is the mean-field approximation. In it we treat the effects of electron-electron interaction in a scalar manner, such that the system can be solved in the same manner as for a non-interacting system. It is a coarse approximation, but very simple to implement and solve.
The crucial variable in the mean-field approximation is the electron density. We solve the electron density self-consistently, such that the mean-field electron density matches the electron density one would calculate from summing the filled single-electron eigenstates.
Iterative methods for achieving a self-consistent density use mixing schemes. Two mixing schemes are implemented here: Simple linear mixing, and Pulay mixing.

# Code Examples

The first step is to create a Hubbard object, initialised with a suitable initial electron density.
I use a Kagome lattice in my examples.

```python3
from hubbard.kagome import KagomeHubbard
kagome = KagomeHubbard(nrows=6, ncols=6, nup=36, ndown=36, u=8, method='random')
```

Then you need to converge the electron density to self-consistency, which is measured through the residual, which is the difference the mean-field electron density and the electron density calculated from single-electron eigenstates.
This can be done by calling a mixing method directly, repeating the call until it is sufficiently converged.
```python3
kagome.linear_mixing(ediff=1e-2, rdiff=1e-4)
```
Or you can use the converge utility.
```python3
import hubbard.utils as ut
ut.converge(kagome, rdiff=1e-4)
```

Once you have a self-consistent electron density, you can calculate properties of interest, such as energy,
```python3
kagome.energy()
```
or density of states,
```python3
kagome.plot_DOS(0.05,0.02)
```
or showing the spin and charge density,
```python3
kagome.plot_spincharge()
```
or the local magnetic moment,
```python3
kagome.local_magnetic_moment()
```

You can run calculations with Fermi-Dirac smearing of the electron occupation, using an effective temperature T. I generally recommend using this, as it treats degeneracies better than not using it, and it allows the net magnetization to change such that the spin up and down chemical potentials are aligned.
To use Fermi-Dirac smearing, fractional electron occupancies need to be allowed for.
```python3
kagome.allow_fractional_electrons(True)
```
If you forget this, the flag will be activated automatically.
Then convergence, energy and other properties can be calculated using Fermi-Dirac smearing.
```python3
kagome.set_electrons(nup=48, ndown=24, method='random')
ut.converge(kagome, rdiff=1e-4, T=0.1)
# Calculate/display parameters
kagome.energy(T=0.1)
kagome.residual(T=0.1)
kagome.get_magnetization()
kagome.chemical_potential(0.1)
```

It can be useful to run calculations which samples the Brillouin zone (where we are considering periodic systems).
We use the hubbard.kpoints subpackage for this (which the Gamma-point only code is a child of).
```python3
from hubbard.kpoints.kagome import KagomeHubbardKPoints
kagome = KagomeHubbardKPoints(u=8, nup=[0.75,0.75,0], ndown=[0,0.75,0.75], allow_fractions=True)
```
When using k-points, you need to set the k-point mesh, which is used for calculating all important properties. While you can explicitly specify the k-points, the default scheme is to use a Monkhorst-Pack grid (which is a rectangular grid in fractional coordinates in k-space).
```python3
kagome.set_kmesh(25, 25)
ut.converge(kagome, rdiff=1e-4, T=0.1)
```
You can plot band structures, tracing paths through the Brillouin zone by specifying the corners of those paths in fractional coordinates.
The following example follows all the high symmetry points and directions of the kagome lattice.
```python3
klist = [[0,0], [0,1/2], [1/3,2/3], [0,0], [1/2,1/2], [2/3,1/3], [0,0], [1/2,0], [1/3,-1/3], [0,0]]
gamma = r'$\Gamma$'
klabels = [gamma, 'M', 'K', gamma, 'M', 'K', gamma, 'M', 'K', gamma]
kagome.plot_bands(klist, 10, T=0.1, klabels=klabels)
```

# Making your own model

You can create your own custom models by specifying the kinetic energy part of the Hamiltonian. See the set\_kinetic method, and either the get\_kinetic method or kin attribute for k-point and Gamma-point-only systems.
If you have something which depends on spin, or a total energy offset, then you will need to modify the potential/interaction part of the Hamiltonian too. See the \_potential method.
If you want to plot the results in real-space, you will need to specify the get\_coordinates method.
If you want to plot in reciprocal space (such as with plot\_bands), you should set the reciprocal lattice vectors. See the reclat attribute.
When working in kpoints, you also need to specify the number of dimensions you are working in (we default to 2) so that we know how many numbers are needed to represent each k-point. See the dims attribute.
