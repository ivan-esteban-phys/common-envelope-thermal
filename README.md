# common-envelope-thermal
This code computes the thermal neutrino spectrum produced in neutron-star common-envelope accretion. For simplicity, it only includes the dominant contribution from positron annihilation.

For information on the physics, see the companion paper [arXiv:2311.xxxxx](https://arxiv.org/abs/2311.xxxxx). All formulae used in the code are explicitly written and explained in the Supplemental Material of that paper. Please cite it if you use this code.

## Prerequisites
You will need
* `gcc`, or other C compiler
* `numpy`
* `scipy`
* `pyMesaUtils`, the Python interface to the MESA astrophysics library. Available in https://github.com/rjfarmer/pyMesa
* `pynucastro`, a nuclear astrophysics library. It can be installed running `pip install pynucastro`

## Basic usage
There is a C file to speed up computations, that can be compiled running `make`. The code contains two different Python modules (see their documentation for more information).

### superEddington ###
This module contains a class that computes the temperature, density, velocity, and electron fraction profile of steady-state spherically-symmetric super-Eddington accretion. As explained in the paper, this requires an accretion shock. There are two main functions in the code,
* `get_preshock_profile`: given boundary conditions at a large radius, it computes the pre-shock temperature, density, velocity and electron fraction.
* `get_postshock_profile`: given pre-shock conditions (computed using the function above) and the radius of the shock, it computes the _neutron star radius_ together with the post-shock temperature, density, velocity and electron fraction. The shock radius must be varied until the returned neutron star radius coincides with the physical one.
This code produces Fig. A1 in the paper.

### nuHotPlasma ###
This module contains a class that computes the neutrino (and antineutrino) spectra emitted by a spherically-symmetric hot plasma due to positron annihilation. There are two main functions in the code,
* `get_specific_nu_spectrum`: given the temperature and electron chemical potential, it returns the number of neutrinos emitted per unit time, volume, and neutrino energy.
* `get_nu_spectrum`: given arrays of temperature and electron chemical potential as a function of radius (that can be computed with the `superEddington` module), it returns the number of neutrinos emitted per unit time and neutrino energy. It includes gravitational reshift and neutrino absorption by the neutron star.
