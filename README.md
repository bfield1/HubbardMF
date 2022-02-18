    HubbardMF - Python package for simulations of a Mean Field Hubbard model.
    Copyright (C) 2022 Bernard Field

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

Dhaneesh Kumar, Jack Hellerstedt, Bernard Field, Benjamin Lowe, Yuefeng Yin, Nikhil V. Medhekar, Agustin Schiffrin. 'Manifestation of Strongly Correlated Electrons in a 2D Kagome Metal-Organic Framework', *Adv. Funct. Mater.* 2021, 2106474. <https://doi.org/10.1002/adfm.202106474>

If you use the substrate module, also cite the following reference:

Bernard Field, Agustin Schiffrin, Nikhil V. Medhekar. 'Correlation-induced magnetism in substrate-supported 2D metal-organic frameworks', In preparation 2022

This code can also be cited directly:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5502306.svg)](https://doi.org/10.5281/zenodo.5502306)

# Theory

The Hubbard model is a simple tight-binding model for interacting electrons.

The Hubbard Hamiltonian contains two terms. The first term is a hopping term between adjacent lattice sites, which is standard for tight-binding models. The second term is an on-site interaction with strength U between electrons of opposite spin on the same site.

However, the Hubbard Hamiltonian is rarely exactly solveable. Solving it requires some approximations. The simplest approximation is the mean-field approximation. In it we treat the effects of electron-electron interaction in a scalar manner, such that the system can be solved in the same manner as for a non-interacting system. It is a coarse approximation, but very simple to implement and solve.

The crucial variable in the mean-field approximation is the electron density. We solve the electron density self-consistently, such that the mean-field electron density matches the electron density one would calculate from summing the filled single-electron eigenstates.

Iterative methods for achieving a self-consistent density use mixing schemes. Two mixing schemes are implemented here: Simple linear mixing, and Pulay mixing.

# Code Examples

Below, I present a basic overview of the code base with some core examples. If you require further details or examples, you can inspect the documentation within the code or read the scripts (in the scripts directory) which I used for real use-cases.

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

# Substrates

The substrate module adds a non-interacting substrate with explicitly described bands to the model. We assume point-like hopping between the lattice and the substrate. The substrate is expressed in a plane wave Bloch basis.

## Generating bands from DFT calculations

The hubbard.substrate.vaspband2np module allows converting PROCAR or EIGENVAL files output by VASP into a format readable by hubbard.substrate.dftsubstrate.

Reading PROCAR files is done using the pyprocar module, which must be installed separately: https://github.com/romerogroup/pyprocar

First, do a VASP calculation with your substrate. The calculation should be of a 2D slab, with a Monkhorst-Pack k-mesh that has only 1 point in the orthogonal direction.
Next, take the OUTCAR and either the PROCAR or EIGENVAL files from this calculation. The OUTCAR file is needed to identify the symmetry operations. You will also need an OUTCAR file to determine the Fermi level, although the Fermi level could be taken from an earlier self-consistent calculation.

To convert the fourth band of PROCAR into a text file, you can do
```python3
import hubbard.substrate.vaspband2np as vb
procar = vb.load_procar('PROCAR', outcar='OUTCAR')
vb.savetxt_band('band.dat', procar, 3, 'OUTCAR')
```
Analagous functions exist for EIGENVAL files and saving directly as numpy arrays.

There are several helper functions for analysing the bands. `bands_in_energy_window` is useful for identifying the indices of bands of interest. If you are using PROCAR, you can use `plot_band_character` to identify the spd-projected atomic character of the bands over the Brillouin zone. This is useful for seeing if a band has any surface contribution near Fermi.

## Initialising a Hubbard-Substrate model

Currently only a single kagome unit cell is implemented for the HubbardSubstrate system.

Create the system in a manner similar to this example:
```python3
from hubbard.substrate.kagome import KagomeSubstrate
# Initialise the kagome lattice
kagome = KagomeSubstrate(t=1, offset=0, u=6)
kagome.set_kmesh(25,25)
```
You then need to insert the substrates. There are a few pre-programmed dispersions, or you can specify it from file.
To load a substrate from 'band.dat' and put it in a 5-by-5 supercell with a coupling strength of 0.5, use:
```python3
kagome.add_substrate(coupling=0.5, subtype='dft', filename='band.dat', nx=5, ny=5)
```
You can modify substrate parameters after the fact by the `change_substrate` and `delete_substrate` methods.

Once you have created all the substrates you need, you then need to fill the system with electrons.
First, you probably want to fill the substrate with a uniform sea of electrons which takes it up to the chemical potential.
```python3
n = kagome.nelect_from_chemical_potential(mu=0, T=1/10)
kagome.set_electrons(nup=n/2, ndown=n/2, method='uniform')
```
You then need to specify the initial electron density for the kagome lattice. You can use any of the normal arguments for `set_electrons` with the additional flag `separate_substrate=True` to modify only the electron density in the kagome lattice and not the substrate.

Once you have the initial electron density set up, you can do a convergence like for any other model.
```python3
import hubbard.utils as ut
ut.converge(kagome, rdiff=1e-4, rdiff_initial=1e-4, T=1/10, mu=0)
```
There are a few things of note here. I prefer using the grand canonical ensemble for solving Substrate models because it better captures having a large sea of electrons in the substrate. To do that I specify a numerical value for `mu`, the chemical potential. You don't have to do this if you don't want to, but using just the canonical ensemble can result in the Fermi level of your substrate moving.

If you do use the grand canonical ensemble, Pulay mixing is very unreliable. Use linear mixing instead. This is specified in `ut.converge` by setting `rdiff_initial` (tolerance for the linear mixing phase) to be equal to `rdiff` (overall tolerance).

For analysing the results, the normal methods apply. Additionally, the `local_magnetic_moment` method measures just the Hubbard lattice's sites, and the `plot_bands` method defaults to highlighting the sites in the Hubbard lattice. There is also the `get_lattice_charge` method which returns the number of electrons in the lattice but not the substrate.

A JSON-based save-load method has not been implemented for HubbardSubstrate. Use Python pickles instead.

## Making your own model

### Substrates

You can easily specify new substrates as children of the BaseSubstrate class.
You will need to specify the `set_params` method if you have any free variables, making sure to also call the parent `set_params` to set the supercell.
You will also need to specify the `dispersion` method, which takes a wavevector and returns a dispersion in a numpy-friendly way.

### HubbardSubstrate

Creating a child class of HubbardSubstrate is slightly more involved (if I rebuilt it I would make it far easier to generalise, but I haven't needed to yet).

In __init__, you need to specify the following after calling the parent __init__:
 - `self.prototype`, a HubbardKPoints instance which mirrors the system without a substrate. It is used as a place to write electron densities, so it just needs the right number of sites.
 - `self.reclat`, the reciprocal lattice vectors, if different from the identity.
 - `self.positions`, the coordinates (in fractional coordinates) of each atom in the tight-binding lattice. This is required to calculate the phase of the coupling.

You need to specify the `get_kinetic_no_substrate` method, which returns the non-interacting Hamiltonian for the non-substrate part of the system.

You can exploit multiple inheritance to simplify some of the matters, although you need to overwrite some of the inherited methods that don't work.


