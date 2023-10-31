import ctypes
import numpy as np
import scipy.integrate as integ

__author__ = "Ivan Esteban"
__email__ = "ivan.esteban@ehu.eus"

class nuHotPlasma:

    """
    Class that computes the neutrino spectrum emitted by a spherically-symmetric hot plasma due to positron annihilation.
    Neutrino and antineutrino spectra are identical.
    **Neutrino oscillations are not included**

    Optional parameters
    -------------------

    r_s           --- Schwarzschild radius [cm], to compute gravitational redshift. Default : 0 (i.e., no redshift)
    r_absorption  --- Radius of a neutrino-absorbing object (neutron star/black hole) at the center of the system [cm]. Default : 0, i.e., no absorption
    
    Methods
    -------

    get_specific_nu_spectrum             --- Single-flavor neutrino emission rate per unit time, volume, and neutrino energy [s^-1 cm^-3 MeV^-1]
    get_specific_nu_spectrum_all_flavors --- All-flavor Neutrino emission rate per unit time, volume, and neutrino energy [s^-1 cm^-3 MeV^-1]
    get_nu_spectrum                      --- Single-flavor neutrino emission rate per unit time and neutrino energy [s^-1 MeV^-1].
                                             A radial temperature and chemical potential profile must be provided.
    get_nu_spectrum_all_flavors          --- All-flavor neutrino emission rate per unit time and neutrino energy [s^-1 MeV^-1].
                                             A radial temperature and chemical potential profile must be provided.
    """
    
    ## Internal parameters ##
    
    # Import C library
    try:
        _lib_C = ctypes.CDLL('auxiliary_funcs.so')
    except OSError as e:
        raise OSError(str(e) + "\nYou must first compile the C libraries. You can do it by running make")
    
    _R = _lib_C.R
    _R.restype = ctypes.c_double
    _R.argtypes = 5*[ctypes.c_double]
    _R_fla = _lib_C.R_fla
    _R_fla.restype = ctypes.c_double
    _R_fla.argtypes = 5*[ctypes.c_double] + [ctypes.c_int]

    _m_e = ctypes.c_double.in_dll(_lib_C, "m_e").value # Electron mass [MeV]
    
    _MeV_4_to_s_m1_cm_m3 = ctypes.c_double.in_dll(_lib_C, "MeV_4_to_s_m1_cm_m3").value # MeV^4 [s^-1 cm^-3]
    
    def __init__(self, r_s=0, r_absorption=0):
        """
        Parameters
        ----------
        r_s : float, optional
            Schwarzschild radius [cm], used to compute gravitational redshift. Default : 0, i.e., no redshift
        r_absorption: float, optional
            Radius of a neutrino-absorbing object (neutron star/black hole) at the center of the system [cm]
            Default : 0, i.e., no absorption
        """

        self.r_s = r_s
        self.r_absorption = r_absorption

    def get_specific_nu_spectrum(self, E_nu, T, eta_e, fla='e'):
        """
        Returns the single-flavor neutrino emission rate due to positron annihilation per unit time, volume, and neutrino energy [s^-1 cm^-3 MeV^-1]

        Parameters
        ----------
        E_nu : float
            Neutrino energy [MeV]
        T : float
            Temperature [K]
        eta_e : float
            mu_e/T, with mu_e the electron chemical potential including electron mass
        fla : string, optional
            'e' for nu_e, 'x' for nu_mu or nu_tau. Default: 'e'
        """
        if T < 2e9: # For too low temperatures, the integral is exponentially suppressed and doesn't converge
            return 0
        if fla not in ['e', 'x']:
            raise ValueError("The flavor must be either 'e' for electron or 'x' for mu/tau")

        def integrand(costheta, E_nubar):
            res = self._R_fla(E_nu, E_nubar, costheta, T, eta_e, 0 if fla=='e' else 1)
            return res            
        
        flx_positron = self._MeV_4_to_s_m1_cm_m3 * integ.dblquad(integrand,
                                                                 self._m_e**2/E_nu, min(100, 15*T/1e10), # The first limit is kinematic, corresponding to cos(theta) > -1
                                                                 lambda E_nubar: -1, lambda E_nubar: 1 - 2*self._m_e**2/(E_nu*E_nubar),
                                                                 epsrel=1e-4)[0]
        return flx_positron


    def get_specific_nu_spectrum_all_flavors(self, E_nu, T, eta_e):
        """
        Returns the all-flavor neutrino emission rate due to positron annihilation per unit time, volume, and neutrino energy [s^-1 cm^-3 MeV^-1]

        Parameters
        ----------
        E_nu : float
            Neutrino energy [MeV]
        T : float
            Temperature [K]
        eta_e : float
            mu_e/T, with mu_e the electron chemical potential including electron mass
        """
        if T < 2e9: # For too low temperatures, the integral is exponentially suppressed and doesn't converge
            return 0

        def integrand(costheta, E_nubar):
            return self._R(E_nu, E_nubar, costheta, T, eta_e)
        
        flx_positron = self._MeV_4_to_s_m1_cm_m3 * integ.dblquad(integrand,
                                                                 self._m_e**2/E_nu, min(100, 15*T/1e10), # The first limit is kinematic, corresponding to cos(theta) > -1
                                                                 lambda E_nubar: -1, lambda E_nubar: 1 - 2*self._m_e**2/(E_nu*E_nubar),
                                                                 epsrel=1e-4)[0]
        return flx_positron


    def get_nu_spectrum(self, E_nu, r_array, T_array, eta_e_array, fla='e'):
        """
        Returns the single-flavor neutrino emission rate due to positron annihilation per unit time and neutrino energy [s^-1 MeV^-1]
        Assumes a spherically symmetric setup. For efficiency and precision, it's better to only include regions with high temperature > 2e9 K
        
        Parameters
        ----------
        E_nu : float
            Neutrino energy [MeV]
        r_array : np.array(float)
            *Sorted* array with the radii that emit neutrinos
        T_array : np.array(float)
            Array with the temperatures at the positions in r_array
        eta_e : np.array(float)
            Array with mu_e/T at the positions in r_array. mu_e is the electron chemical potential including electron mass
        fla : string, optional
            'e' for nu_e, 'x' for nu_mu or nu_tau. Default: 'e'        
        """

        if fla not in ['e', 'x']:
            raise ValueError("The flavor must be either 'e' for electron or 'x' for mu/tau")        

        if np.any(r_array[:-1] > r_array[1:]):
            raise ValueError("The array with radii must be sorted")        

        def integrand(r):
            T = np.interp(r, r_array, T_array)
            eta_e = np.interp(r, r_array, eta_e_array)

            res = self.get_specific_nu_spectrum(E_nu / np.sqrt(1-self.r_s/r), T, eta_e, fla)
            res *= 4*np.pi*r**2 # Volume
            res *= 1 - 0.5*(1-np.sqrt(1-(self.r_absorption/r)**2)) # Neutrino absorption by NS/BH
            
            return res

        return integ.quad(integrand, r_array[0], r_array[-1], epsrel=1e-3)[0]


    def get_nu_spectrum_all_flavors(self, E_nu, r_array, T_array, eta_e_array):
        """
        Returns the all-flavor neutrino emission rate due to positron annihilation per unit time and neutrino energy [s^-1 MeV^-1]
        Assumes a spherically symmetric setup. For efficiency and precision, it's better to only include regions with high temperature > 2e9 K        

        Parameters
        ----------
        E_nu : float
            Neutrino energy [MeV]
        r_array : np.array(float)
            *Sorted* array with the radii that emit neutrinos
        T_array : np.array(float)
            Array with the temperatures at the positions in r_array
        eta_e : np.array(float)
            Array with mu_e/T at the positions in r_array. mu_e is the electron chemical potential including electron mass
        """

        if np.any(r_array[:-1] > r_array[1:]):
            raise ValueError("The array with radii must be sorted")
            
        def integrand(r):
            T = np.interp(r, r_array, T_array)
            eta_e = np.interp(r, r_array, eta_e_array)

            res = self.get_specific_nu_spectrum_all_flavors(E_nu / np.sqrt(1-self.r_s/r), T, eta_e)
            res *= 4*np.pi*r**2 # Volume
            res *= 1 - 0.5*(1-np.sqrt(1-(self.r_absorption/r)**2)) # Neutrino absorption by NS/BH
            
            return res

        return integ.quad(integrand, r_array[0], r_array[-1], epsrel=1e-3)[0]
