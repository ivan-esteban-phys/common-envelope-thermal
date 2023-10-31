import pyMesa as pym
import scipy.integrate as integ
import numpy as np
import pynucastro as pyna
import itertools
import scipy.optimize as optim

__author__ = "Ivan Esteban"
__email__ = "ivan.esteban@ehu.eus"

class superEddington:

    """
    Class that solves the steady-state hydrodynamic equations for super-Eddington accretion: radiation diffusion is neglected, and neutrino cooling is included. This approximation is valid for fluid velocities much greater than the diffusion velocity.

    Mandatory parameters
    -------------------

    Mdot          --- Accretion rate [Solar masses/year]
    M_NS          --- Neutron star mass [Solar masses]
    
    Optional parameters
    -------------------

    Y_initial     --- Initial molar abundances of isotopes [Y_i = n_i / n_{nucleons}, with n number density]
                      Default: 75% in mass H1, 25% in mass He4
    
    Methods
    -------    

    get_preshock_profile  --- Returns the fluid properties before the accretion shock
    get_postshock_profile --- Returns the fluid properties after the accretion shock    
    """

    ## We initialize the MESA modules ##
    print("Setting up MESA...", end=' ', flush=True)
    _MESA_ierr = 0

    _crlibm_lib, _ = pym.loadMod("math")
    _crlibm_lib.math_init()

    _const_lib, _const_def = pym.loadMod("const")
    _const_lib.const_init(pym.MESA_DIR, _MESA_ierr)

    _eos_lib, _eos_def = pym.loadMod("eos")
    _eos_lib.eos_init(pym.EOSDT_CACHE, True, _MESA_ierr)
    _eos_handle = _eos_lib.alloc_eos_handle(_MESA_ierr)

    _chem_lib, _chem_def = pym.loadMod("chem")
    _chem_lib.chem_init('isotopes.data', _MESA_ierr)

    _neu_lib, _neu_def = pym.loadMod("neu")
    _neu_config_pars = {'log10_Tlim_neu' : 7.5, # For T < Tlim, turn off neutrino losses
                             'flags_neu': np.full(_neu_def.num_neu_types.get(), True)} # Which sources of neutrino losses should we include? [Pair production, plasmon decay, photoproduction, bremsstrahlung​, recombination]
    print("MESA set up")

    ## We set up nuclear physics, with a very generous list of nuclei ##
    print("Setting up pynucastro...", end=' ', flush=True)
    _nucs = 'n h1-3 he3-4 he6 li6-8 be7 be9-11 b10-12 o14-18 c11-14 n13-16 f16-20 ne18-23 na20-26 mg21-29 al23-32 si24-34 p27-36 s30-38 cl31-40 ar31-43 k35-44 ca36-48 sc40-49 ti40-52 v45-54 cr46-56 mn48-61 fe50-60'
    def _make_species_list(input_list):
        output_list = []
        for spec in input_list.split(' '):
                if spec=='n':
                        output_list.append('n')
                        continue
                f = [ ''.join(x) for _,x in itertools.groupby(spec,key=str.isdigit)]
                if len(f) == 2:
                        output_list.append(spec)
                if len(f) == 4:
                        a1, a2 = int(f[1]), int(f[3])
                        for i in range(a2-a1+1):
                                output_list.append(f[0]+str(a1+i))
        return output_list
    _reaclib_library = pyna.ReacLibLibrary()
    _nuc_list = _make_species_list(_nucs)
    _net = _reaclib_library.linking_nuclei(_nuc_list)
    _nuc_rc = pyna.RateCollection(libraries=_net)
    _nuc_Z = np.array([pyna.Nucleus(name).Z for name in _nuc_list])
    _nuc_A = np.array([pyna.Nucleus(name).A for name in _nuc_list])
    print("pynucastro set up")    
    
    def __init__(self, Mdot, M_NS, Y_initial = {'h1': 0.75/1, 'he4': 0.25/4}):
        """
        Parameters
        ----------
        Mdot : float
            Accretion rate [Solar masses/year]
        M_NS: float
            Neutron star mass [Solar masses]
        Y_initial: dict, optional
            Initial molar abundances of isotopes
            Default: 75% in mass H1, 25% in mass He4
        """
        
        ## We read the arguments and convert to cgs ##
        self.Mdot = Mdot * self._const_def.Msun.value / self._const_def.secyer.value
        self.M_NS = M_NS * self._const_def.Msun.value

        self.Y_initial_list = np.zeros(len(self._nuc_list))
        self.Ye_initial = 0
        for nuc_name in Y_initial:
            self.Y_initial_list[self._nuc_list.index(nuc_name)] = Y_initial[nuc_name]
            self.Ye_initial += pyna.Nucleus(nuc_name).Z  * Y_initial[nuc_name]
        
    def _neutrino_loss(self, T, rho, Z_list, A_list, N_fraction):
        """
        Returns the energy loss rate to neutrinos per unit mass [erg/g/s]
        
        Parameters
        ----------
        T : float
            Temperature [K]
        rho : float
            Density [g/cm^3]
        Z_list : np.array(float)
            Array with the atomic numbers of nuclei
        A_list : np.array(float)
            Array with the mass numbers of nuclei
        N_fraction : np.array(float)
            Array with the *number* fractions of nuclei
        """

        # Average quantities
        abar = np.sum(A_list * N_fraction)
        zbar = np.sum(Z_list * N_fraction)

        # Auxiliary variables
        log10T = np.log10(T)
        log10Rho = np.log10(rho)

        # Return variables
        loss = np.zeros(self._neu_def.num_neu_rvs.get())
        sources = np.zeros((self._neu_def.num_neu_types.get(), self._neu_def.num_neu_rvs.get()))
    
        neu_res = self._neu_lib.neu_get(T, log10T, rho, log10Rho,
                                        abar, zbar,
                                        self._neu_config_pars['log10_Tlim_neu'], self._neu_config_pars['flags_neu'],
                                        loss, sources, self._MESA_ierr)
        if self._MESA_ierr != 0:
            raise Exception("Error when computing neutrino losses")
    
        return neu_res["loss"][self._neu_def.ineu.get()-1]

    def _eos(self, T, rho, Z_list, A_list, N_fraction):
        """
        Returns a dictionary with the equation of state parameters

        Parameters
        ----------
        T : float
            Temperature [K]
        rho : float
            Density [g/cm^3]
        Z_list : np.array(float)
            Array with the atomic numbers of nuclei
        A_list : np.array(float)
            Array with the mass numbers of nuclei
        N_fraction : np.array(float)
            Array with the *number* fractions of nuclei
        """

        # Chemistry
        species = len(Z_list)
        chem_id = np.array([self._chem_lib.lookup_zn(Z_, N_) for (Z_, N_) in zip(Z_list, A_list-Z_list)]) # Indices in the chemical networks
        net_iso = chem_id # Indices in the nuclear networks

        atomic_weights = np.array(self._chem_def.element_atomic_weight)
        xa = N_fraction * atomic_weights[Z_list-1] # Mass fractions
        xa /= np.sum(xa)

        # Auxiliary variables
        log10T = np.log10(T)
        log10Rho = np.log10(rho)

        # Return variables
        res = np.zeros(self._eos_def.num_eos_basic_results.get())
        d_dlnRho_const_T = np.zeros(self._eos_def.num_eos_basic_results.get())
        d_dlnT_const_Rho = np.zeros(self._eos_def.num_eos_basic_results.get())
        d_dxa_const_TRho = np.zeros([self._eos_def.num_eos_basic_results.get(), species])

        eos_res = self._eos_lib.eosDT_get(self._eos_handle, species, chem_id, net_iso, xa, 
                                          rho, log10Rho, T, log10T,
                                          res, d_dlnRho_const_T, d_dlnT_const_Rho, d_dxa_const_TRho, self._MESA_ierr)
        if self._MESA_ierr != 0:
            raise Exception("Error when computing equation of state")        
        if eos_res["d_dxa"].shape == (1,):
            eos_res["d_dxa"] = np.zeros([self._eos_def.num_eos_basic_results.get(), species])

        result = {"P": np.exp(eos_res["res"][self._eos_def.i_lnpgas.get()-1]) + 1./3. * self._const_def.crad.value * T**4, # *Total* pressure [erg/cm^3]
                  "eps": np.exp(eos_res["res"][self._eos_def.i_lnE.get()-1]), # Internal specific energy [erg/g]
                  "c_V": eos_res["res"][self._eos_def.i_cv.get()-1], # Constant-volume specific heat [erg/g/K]
                  "gamma_1": eos_res["res"][self._eos_def.i_gamma1.get()-1], # Adiabat: dlogP/dlogrho|s
                  "gamma_3": eos_res["res"][self._eos_def.i_gamma3.get()-1], # Adiabat: dlogT/dlogrho|s + 1
                  "eta": eos_res["res"][self._eos_def.i_eta.get()-1], # electron chemical potential/(k_B T). It doesn't include the electron mass
                }
        return result

    def _eos_get_T(self, P, rho, Z_list, A_list, N_fraction):
        """
        Returns the temperature corresponding to the pressure P and density rho

        Parameters
        ----------
        P : float
            Pressure [erg/cm^3]
        rho : float
            Density [g/cm^3]
        Z_list : np.array(float)
            Array with the atomic numbers of nuclei
        A_list : np.array(float)
            Array with the mass numbers of nuclei
        N_fraction : np.array(float)
            Array with the *number* fractions of nuclei
        """

        # Chemistry
        species = len(Z_list)
        chem_id = np.array([self._chem_lib.lookup_zn(Z_, N_) for (Z_, N_) in zip(Z_list, A_list-Z_list)]) # Indices in the chemical networks
        net_iso = chem_id # Indices in the nuclear networks

        atomic_weights = np.array(self._chem_def.element_atomic_weight)
        xa = N_fraction * atomic_weights[Z_list-1] # Mass fractions
        xa /= np.sum(xa)

        # Auxiliary variables
        which_other = self._eos_def.i_logPtot
        other_value = np.log10(P)
        log10Rho = np.log10(rho)

        logT_tol = 1e-4
        other_tol = 1e-4
        max_iter = int(1e6)
        logT_guess = np.log10(3*P/float(self._const_def.crad))**(1./4.)

        # Return variables
        logT_result = 0
        res = np.zeros(self._eos_def.num_eos_basic_results.get())
        d_dlnRho_const_T = np.zeros(self._eos_def.num_eos_basic_results.get())
        d_dlnT_const_Rho = np.zeros(self._eos_def.num_eos_basic_results.get())
        d_dxa_const_TRho = np.zeros([self._eos_def.num_eos_basic_results.get(), species])
        eos_calls = 0

        return 10**self._eos_lib.eosDT_get_T(self._eos_handle, species, chem_id, net_iso, xa, 
                                             log10Rho, which_other, other_value,
                                             logT_tol, other_tol, max_iter, logT_guess,
                                             self._const_def.arg_not_provided, self._const_def.arg_not_provided, self._const_def.arg_not_provided, self._const_def.arg_not_provided,
                                             logT_result, res, d_dlnRho_const_T, d_dlnT_const_Rho, d_dxa_const_TRho, eos_calls, self._MESA_ierr)["logt_result"]

    def _get_NSE_Y(self, rho, T, Ye):
        """ Returns molar abundances in nuclear statistical equilibrium """
        Y = np.zeros(len(self._nuc_list))

        comp = self._nuc_rc.get_comp_nse(rho, T, Ye, use_coulomb_corr=True)
        for nucleus in comp.X.keys():
            Y[self._nuc_list.index(nucleus.short_spec_name)] = comp.X[nucleus] / nucleus.A
        return Y    

    def _get_EC_rate(self, T, eta_e):
        """ Returns the electron capture rate on free protons [s^-1].
        Eq. (3) in astro-ph/9807012
        
        Parameters
        ----------    
        T : float
            Temperature [K]
        eta_e : float
            mu_e/T, with mu_e the electron chemical potential including electron mass
        """

        Q = (self._const_def.mn.value - self._const_def.mp.value)/self._const_def.mev2gr.value # Neutron-proton mass difference [MeV]
        m_e = self._const_def.me.value/self._const_def.mev2gr.value # Electron mass [MeV]
        k_B = self._const_def.boltzm.value/self._const_def.mev_to_ergs.value # Boltzmann constant [MeV/K]
        
        if ((eta_e * k_B*T - m_e) < 0.1) and (k_B*T < 0.1): # If kinetic Fermi energy and temperature are smaller than 0.1 MeV, the rate is negligibly small
            return 0
        
        I = 1/m_e**5 * integ.quad(lambda E: E * np.sqrt(E**2-m_e**2) * (E-Q)**2 / (1 + np.exp(E/(k_B*T) - eta_e)),
                                  Q, max(Q, eta_e + 30*k_B*T), epsrel=1e-3)[0]

        return I * np.log(2) / 1065

    def _get_EC_energy_loss_rate(self, T, eta_e, n_p):
        """ Returns the energy loss rate due to electron capture on free protons [erg/cm^3/s].
        Eq. (1) in Egawa & Yokoi, 1977

        
        Parameters
        ----------    
        T : float
            Temperature [K]
        eta_e : float
            mu_e/T, with mu_e the electron chemical potential including electron mass
        n_p : float
            Free proton density [cm^-3]
        """

        Q = (self._const_def.mn.value - self._const_def.mp.value)/self._const_def.mev2gr.value # Neutron-proton mass difference [MeV]
        m_e = self._const_def.me.value/self._const_def.mev2gr.value # Electron mass [MeV]
        k_B = self._const_def.boltzm.value/self._const_def.mev_to_ergs.value # Boltzmann constant [MeV/K]        

        if ((eta_e * k_B*T - m_e) < 0.1) and (k_B*T < 0.1): # If Fermi energy and temperature are smaller than 0.1 MeV, the rate is negligibly small
            return 0        
        
        I = 1/m_e**5 * integ.quad(lambda E: E * np.sqrt(E**2-m_e**2) * (E-Q)**3 / (1 + np.exp(E/(k_B*T) - eta_e)),
                                  Q, max(Q, eta_e + 30*k_B*T), epsrel=1e-3)[0]

        Q_nu = I * np.log(2) / 1065 
        return (Q_nu + Q*self._get_EC_rate(T, eta_e)) * self._const_def.mev_to_ergs.value * n_p

    def _derivs_log_r(self, log_r, y):
        """
        Returns the derivatives of {ln([-v, T]), Ye} with respect to ln(r)

        Parameters
        ----------
        log_r : float
            ln(r/cm), with r radius
        y : np.array(float)
            y[0] = ln(-v/(cm/s)), with v velocity
            y[1] = ln(T/K), with T temperature
            y[2] = Ye, with Ye electron abundance (#electrons per nucleon)
        """

        # Constants
        G = self._const_def.standard_cgrav.value
        c = self._const_def.clight.value

        # And the variables: fundamental
        v = - np.exp(y[0])
        T = np.exp(y[1])
        Ye = y[2]

        # and derived
        r = np.exp(log_r)
        rho = - self.Mdot/(4*np.pi*r**2 * v)

        if T < 5e9: # No nuclear reaction has happened
            Y = self.Y_initial_list
        else:
            if Ye < 0.01: # We have neutronized
                Ye = 0
                Y = np.zeros(len(self._nuc_list))
                Y[self._nuc_list.index("n")] = 1
                Y[self._nuc_list.index("h1")] = 1e-10
            else:
                Y = self._get_NSE_Y(rho, T, Ye)

        Y = np.where(Y<0, 0, Y)
        N_fraction = Y / np.sum(Y)

        msa_eos = self._eos(T, rho, self._nuc_Z, self._nuc_A, N_fraction)
        
        P = msa_eos["P"]
        eps = msa_eos["eps"]
        c_V = msa_eos["c_V"]
        gamma_1 = msa_eos["gamma_1"]
        gamma_3 = msa_eos["gamma_3"]
        eta_e = msa_eos["eta"]
        m_e = self._const_def.me.value/self._const_def.mev2gr.value # Electron mass [MeV]
        k_B = self._const_def.boltzm.value/self._const_def.mev_to_ergs.value # Boltzmann constant [MeV/K]        
        eta_e += m_e / (k_B * T) # We add the electron mass
        if eta_e < 0: # Avoid underflow for large neutron density
            eta_e = 0

        c_s2 = c**2 * gamma_1 / (1 + rho/P*(eps+c**2)) # Sound speed squared [Eq. (38) in arXiv:2104.00691]

        w_over_csq = rho + (eps*rho)/c**2 + P/c**2
        GR_fctr = (v/c)**2 + 1 - 2*G*self.M_NS/(r*c**2)

        L_nu = rho * self._neutrino_loss(T, rho, self._nuc_Z, self._nuc_A, N_fraction)
        if Ye > 0.01:
            L_nu += self._get_EC_energy_loss_rate(T, eta_e, Y[self._nuc_list.index("h1")]*rho*self._const_def.avo.value)

        dv_dr = - G*self.M_NS/r**2 + 2*c_s2/r * GR_fctr + (gamma_3-1)/(v*w_over_csq) * L_nu * GR_fctr
        dv_dr /= (v**2 - c_s2*GR_fctr)/v

        dT_dr = - 1/c_V * L_nu/(rho*v) - T*(gamma_3 - 1) * (2/r + 1/v*dv_dr)

        dYe_dr = -self._get_EC_rate(T, eta_e) * Y[self._nuc_list.index("h1")] / v

        return np.array([r/v * dv_dr, r/T * dT_dr, r * dYe_dr])

    def _derivs_log_tau(self, log_tau, y):
        """
        Returns the derivatives of [ln(r), ln(-v), ln(T), Ye] with respect to ln(tau), where we define

        \tau \equiv 1 - \int_r^{r_shock} 1/v(r) dr

        Parameters
        ----------    
        log_tau : float
            ln(tau/s)
        y : np.array(float)
            y[0] = ln(r/cm), with r radius
            y[1] = ln(-v/(cm/s)), with v velocity
            y[2] = ln(T/K), with T temperature
            y[3] = Ye, with Ye electron abundance (#electrons per nucleon)
        """
        
        # We first read the variables
        tau = np.exp(log_tau)

        ln_r = y[0]
        ln_v = y[1]
        ln_T = y[2]
        Ye = y[3]

        # And obtain derived variables
        r = np.exp(ln_r)
        v = -np.exp(ln_v)

        # We compute derivatives
        dy_dlogr = self._derivs_log_r(ln_r, np.array([ln_v, ln_T, Ye]))
        # And we convert to d/dlogtau
        fctr = tau/r * v

        return np.concatenate([[fctr], dy_dlogr * fctr])

    def _shock_jump(self, ln_v, ln_T, Ye, r):
        """
        Returns the ln(-v_2/(cm/s)), ln(T_2/K), where v_2 and T_2 are the post-shock velocity and temperature, respectively

        Parameters
        ----------
        ln_v    --- ln(-v/(cm/s)), with v the pre-shock velocity
        ln_T    --- ln(T/K), with T the pre-shock temperature
        Ye      --- Electron fraction
        r       --- Radius [cm]
        """
        v_in = -np.exp(ln_v)
        T_in = np.exp(ln_T)
        rho_in = - self.Mdot/(4*np.pi*r**2 * v_in)
        c = self._const_def.clight.value

        if T_in < 5e9: # No nuclear reaction has happened
            Y = self.Y_initial_list
        else:
            if Ye < 0.01: # We have neutronized
                Ye = 0
                Y = np.zeros(len(self._nuc_list))
                Y[self._nuc_list.index("n")] = 1
                Y[self._nuc_list.index("h1")] = 1e-10
            else:
                Y = get_NSE_Y(rho, T_in, Ye)
        Y = np.where(Y<0, 0, Y)
        N_fraction = Y / np.sum(Y)

        def jump(x_2, x_1):
            """ Auxiliary function for solving the shock. The roots of this function solve the shock equations """
            # We read the arguments
            v_2 = x_2[0]
            rho_2 = np.exp(x_2[1])
            T_2 = np.exp(x_2[2])

            v_1 = x_1[0]
            rho_1 = np.exp(x_1[1])
            T_1 = np.exp(x_1[2])

            # We obtain thermodynamic variables
            eos_res_1 = self._eos(T_1, rho_1, self._nuc_Z, self._nuc_A, N_fraction)
            eos_res_2 = self._eos(T_2, rho_2, self._nuc_Z, self._nuc_A, N_fraction)

            P_1 = eos_res_1["P"]
            eps_1 = eos_res_1["eps"]
            P_2 = eos_res_2["P"]
            eps_2 = eos_res_2["eps"]

            w_1 = rho_1*c**2 + (eps_1*rho_1) + P_1 
            w_2 = rho_2*c**2 + (eps_2*rho_2) + P_2   
            v_t_1 = np.sqrt(c**2 + v_1**2)
            v_t_2 = np.sqrt(c**2 + v_2**2)    

            return [1 - (rho_2*v_2) / (rho_1*v_1),
                    1 - (c*v_t_2-c**2 + eps_2*v_t_2/c + P_2/rho_2*v_t_2/c) / (c*v_t_1-c**2 + eps_1*v_t_1/c + P_1/rho_1*v_t_1/c), # We divide the equation in Houck & Chevalier by rho*v and subtract c^2 to obtain a better numerical behavior in the non-relativistic regime.
                    1 - (w_2/c**2*v_2**2 + P_2) / (w_1/c**2*v_1**2 + P_1)]

        v0, ln_rho0, ln_T0 = optim.fsolve(jump,
                                          [v_in/7, np.log(rho_in*7), np.log(self._eos_get_T(6/7*rho_in*v_in**2, rho_in*7, self._nuc_Z, self._nuc_A, N_fraction))], # We use the relativistic strong shock as a guess
                                          args=[v_in, np.log(rho_in), np.log(T_in)])
        return np.log(-v0), ln_T0

    def get_preshock_profile(self, r_inf = 1e12, r0 = 1e6, T_inf = 5e3, v_inf = None, verbose=False):
        """
        Returns a dictionary with the fluid properties (radius, velocity, density, temperature & electron fraction) before the accretion shock

        Parameters
        ----------
        r_inf : float, optional
            Initial radius far from the neutron star where we start the integration [cm]
            Default: 1e12
        r0 : float, optional
            Final radius near to the neutron star where we stop the integration [cm]
            Default: 1e5
        T_inf : float, optional
            Temperature at r_inf [K]. The final results are not sensitive to this
            Default: 5000
        v_inf : float, optional
            Velocity at r_inf [cm/s]. The final results are not sensitive to this
            Default: None, which sets free-fall velocity = -sqrt(2 G M_NS / r_inf). This is an attractor solution
        verbose : bool, optional
            Whether to show the steps of the ODE solver. Default: false        
        """
        if v_inf is None: # Free-fall velocity
            G = self._const_def.standard_cgrav.value
            v_inf = - np.sqrt(2 * G * self.M_NS / r_inf)

        # Initial conditions
        y_inf = np.concatenate([[np.log(-v_inf), np.log(T_inf), self.Ye_initial]])

        # Arrays with the solutions
        r_list = [r_inf]
        v_list = [v_inf]
        T_list = [T_inf]
        Ye_list = [self.Ye_initial]
        
        # We set up the ODE solver
        solver = integ.ode(self._derivs_log_r).set_integrator("vode", method="BDF", rtol=1e-4, atol=1e-4)
        solver.set_initial_value(y_inf, np.log(r_inf))

        N_steps = 1000 # Hard-coded. This works pretty well
        d_log_r = (np.log(r_inf) - np.log(r0)) / N_steps

        while solver.successful() and solver.t > np.log(r0):
            r = np.exp(solver.t - d_log_r)
            y = solver.integrate(solver.t - d_log_r)

            r_list.append(r)
            v_list.append(-np.exp(y[0]))
            T_list.append(np.exp(y[1]))
            Ye_list.append(y[2])
            if verbose:
                print("r, v, T, Ye, rho: %.3e %.3e %.3e %.3e %.3e" % (r_list[-1], v_list[-1], T_list[-1], Ye_list[-1], - self.Mdot/(4*np.pi*r_list[-1]**2 * v_list[-1]) ))

        rho_list = - self.Mdot/(4*np.pi*np.array(r_list)**2 * np.array(v_list))
        
        return {"r": np.array(r_list[::-1]), "v": np.array(v_list[::-1]), "rho": np.array(rho_list[::-1]),
                "T": np.array(T_list[::-1]), "Ye": np.array(Ye_list[::-1])}

    def get_postshock_profile(self, preshock_profile, r_shock, verbose=False):
        """
        Returns a dictionary with the fluid properties (radius, velocity, density, temperature, electron fraction, neutron star radius) after the accretion shock. The neutron star radius is defined as the radius where the velocity sharply drops (that we take as v < 0.1 cm/s, but the result is not sensitive to this)
        To avoid numerical instabilities, we assume that matter has been neutronized for electron fraction < 1e-2

        Parameters
        ----------
        preshock_profile : dict
            Dictionary with the pre-shock fluid properties
        
            preshock_profile["r"] : radius [cm]
            preshock_profile["v"] : velocity [cm/s]
            preshock_profile["T"] : temperature [K]
            preshock_profile["Ye"] : electron fraction
        r_shock : float
            Shock radius [cm]. This must be varied until the returned neutron star radius is the true neutron star radius
        verbose : bool, optional
            Whether to show the steps of the ODE solver. Default: false
        """

        if np.any(preshock_profile["r"][:-1] > preshock_profile["r"][1:]):
            raise ValueError("The array with radii must be sorted")
        
        # We solve the shock jump conditions
        v_in = np.interp(r_shock, preshock_profile["r"], preshock_profile["v"])
        T_in = np.interp(r_shock, preshock_profile["r"], preshock_profile["T"])
        Ye_in = np.interp(r_shock, preshock_profile["r"], preshock_profile["Ye"])
        
        ln_v_out, ln_T_out = self._shock_jump(np.log(-v_in), np.log(T_in), Ye_in, r_shock)

        # Initial conditions
        y0 = np.concatenate([[np.log(r_shock), ln_v_out, ln_T_out, Ye_in]])

        # Arrays with the solutions
        r_list = [r_shock]
        v_list = [-np.exp(ln_v_out)]
        T_list = [np.exp(ln_T_out)]
        Ye_list = [self.Ye_initial]

        # We set up the ODE solver
        tau0 = 1
        solver = integ.ode(self._derivs_log_tau).set_integrator("vode", method="BDF", rtol=1e-4, atol=1e-4)
        solver.set_initial_value(y0, np.log(tau0))
        
        d_log_tau = 1e-3

        while solver.successful() and abs(v_list[-1]) > 0.1:
            y = solver.integrate(solver.t + d_log_tau)

            r_list.append(np.exp(y[0]))
            v_list.append(-np.exp(y[1]))
            T_list.append(np.exp(y[2]))
            Ye_list.append(y[3])
            
            if verbose:
                print("r, v, T, Ye, rho: %.3e %.3e %.3e %.3e %.3e" % (r_list[-1], v_list[-1], T_list[-1], Ye_list[-1], - self.Mdot/(4*np.pi*r_list[-1]**2 * v_list[-1]) ))

        rho_list = - self.Mdot/(4*np.pi*np.array(r_list)**2 * np.array(v_list))
        
        return {"r": np.array(r_list[::-1]), "v": np.array(v_list[::-1]), "rho": np.array(rho_list[::-1]),
                "T": np.array(T_list[::-1]), "Ye": np.array(Ye_list[::-1]), "r_NS": r_list[-1]}
