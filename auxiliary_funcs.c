/* ====================================================================
 * This file contains auxiliary functions to compute neutrino spectra
 *
 * Compile it as
 *
 * gcc -O3 -shared -o auxiliary_funcs.so -fPIC -lM auxiliary_funcs.c
 * 
 * The polylogarithms have been extracted from the Polylogarithm library
 *  (https://github.com/Expander/polylogarithm),
 * licensed under the MIT License.
 * ==================================================================== */

#include <math.h>
#define SQR(x) ((x)*(x))

/* Constants */
const double G_F = 1.1663787e-11; // G_F [MeV^{-2}]
const double m_e = 0.51099895; // Electron mass [MeV]
const double sinsqthetaW = 0.2224; // Weinberg angle
const double k_B = 8.6173333e-11; // Boltzmann constant [MeV/K]
const double MeV_4_to_s_m1_cm_m3 = 1.9773103e+53; // MeV^4 [s^-1 cm^-3]

/* Couplings. We follow the conventions in Dicus, Phys.Rev.D 6 (1972) 941-949 */
const double C_V_e = 0.5 + 2*sinsqthetaW;
const double C_A_e = 0.5;
const double C_V_x = C_V_e - 1;
const double C_A_x = C_A_e - 1;
const double beta_1 = SQR(C_V_e - C_A_e) + 2*SQR(C_V_x - C_A_x);
const double beta_2 = SQR(C_V_e + C_A_e) + 2*SQR(C_V_x + C_A_x);
const double beta_3 = (SQR(C_V_e) - SQR(C_A_e)) + 2*(SQR(C_V_x) - SQR(C_A_x));
const double beta_1_e = SQR(C_V_e - C_A_e);
const double beta_2_e = SQR(C_V_e + C_A_e);
const double beta_3_e = SQR(C_V_e) - SQR(C_A_e);
const double beta_1_x = SQR(C_V_x - C_A_x);
const double beta_2_x = SQR(C_V_x + C_A_x);
const double beta_3_x = SQR(C_V_x) - SQR(C_A_x);

/* Auxiliary polylogarithm functions */

double li2(double x){
  /**
   * @brief Real dilogarithm \f$\operatorname{Li}_2(x)\f$
   * @param x real argument
   * @return \f$\operatorname{Li}_2(x)\f$
   * @author Alexander Voigt
   *
   * Implemented as a rational function approximation with a maximum
   * error of 5e-17
   * [[arXiv:2201.01678](https://arxiv.org/abs/2201.01678)].
   */
  
   const double PI = 3.1415926535897932;
   const double P[] = {
      0.9999999999999999502e+0,
     -2.6883926818565423430e+0,
      2.6477222699473109692e+0,
     -1.1538559607887416355e+0,
      2.0886077795020607837e-1,
     -1.0859777134152463084e-2
   };
   const double Q[] = {
      1.0000000000000000000e+0,
     -2.9383926818565635485e+0,
      3.2712093293018635389e+0,
     -1.7076702173954289421e+0,
      4.1596017228400603836e-1,
     -3.9801343754084482956e-2,
      8.2743668974466659035e-4
   };

   double y = 0, r = 0, s = 1;

   /* transform to [0, 1/2] */
   if (x < -1) {
      const double l = log(1 - x);
      y = 1/(1 - x);
      r = -PI*PI/6 + l*(0.5*l - log(-x));
      s = 1;
   } else if (x == -1) {
      return -PI*PI/12;
   } else if (x < 0) {
      const double l = log1p(-x);
      y = x/(x - 1);
      r = -0.5*l*l;
      s = -1;
   } else if (x == 0) {
      return 0;
   } else if (x < 0.5) {
      y = x;
      r = 0;
      s = 1;
   } else if (x < 1) {
      y = 1 - x;
      r = PI*PI/6 - log(x)*log1p(-x);
      s = -1;
   } else if (x == 1) {
      return PI*PI/6;
   } else if (x < 2) {
      const double l = log(x);
      y = 1 - 1/x;
      r = PI*PI/6 - l*(log(y) + 0.5*l);
      s = 1;
   } else {
      const double l = log(x);
      y = 1/x;
      r = PI*PI/3 - 0.5*l*l;
      s = -1;
   }

   const double y2 = y*y;
   const double y4 = y2*y2;
   const double p = P[0] + y * P[1] + y2 * (P[2] + y * P[3]) +
                    y4 * (P[4] + y * P[5]);
   const double q = Q[0] + y * Q[1] + y2 * (Q[2] + y * Q[3]) +
                    y4 * (Q[4] + y * Q[5] + y2 * Q[6]);

   return r + s*y*p/q;
}

/// Li_3(x) for x in [-1,0]
static double li3_neg(double x){
   const double cp[] = {
      0.9999999999999999795e+0, -2.0281801754117129576e+0,
      1.4364029887561718540e+0, -4.2240680435713030268e-1,
      4.7296746450884096877e-2, -1.3453536579918419568e-3
   };
   const double cq[] = {
      1.0000000000000000000e+0, -2.1531801754117049035e+0,
      1.6685134736461140517e+0, -5.6684857464584544310e-1,
      8.1999463370623961084e-2, -4.0756048502924149389e-3,
      3.4316398489103212699e-5
   };

   const double x2 = x*x;
   const double x4 = x2*x2;
   const double p = cp[0] + x*cp[1] + x2*(cp[2] + x*cp[3]) +
      x4*(cp[4] + x*cp[5]);
   const double q = cq[0] + x*cq[1] + x2*(cq[2] + x*cq[3]) +
      x4*(cq[4] + x*cq[5] + x2*cq[6]);

   return x*p/q;
}

/// Li_3(x) for x in [0,1/2]
static double li3_pos(double x){
   const double cp[] = {
      0.9999999999999999893e+0, -2.5224717303769789628e+0,
      2.3204919140887894133e+0, -9.3980973288965037869e-1,
      1.5728950200990509052e-1, -7.5485193983677071129e-3
   };
   const double cq[] = {
      1.0000000000000000000e+0, -2.6474717303769836244e+0,
      2.6143888433492184741e+0, -1.1841788297857667038e+0,
      2.4184938524793651120e-1, -1.8220900115898156346e-2,
      2.4927971540017376759e-4
   };

   const double x2 = x*x;
   const double x4 = x2*x2;
   const double p = cp[0] + x*cp[1] + x2*(cp[2] + x*cp[3]) +
      x4*(cp[4] + x*cp[5]);
   const double q = cq[0] + x*cq[1] + x2*(cq[2] + x*cq[3]) +
      x4*(cq[4] + x*cq[5] + x2*cq[6]);

   return x*p/q;
}

double li3(double x){
  /**
   * @brief Real trilogarithm \f$\operatorname{Li}_3(x)\f$
   * @param x real argument
   * @return \f$\operatorname{Li}_3(x)\f$
   * @author Alexander Voigt
   */
  
   const double zeta2 = 1.6449340668482264;
   const double zeta3 = 1.2020569031595943;

   // transformation to [-1,0] and [0,1/2]
   if (x < -1) {
      const double l = log(-x);
      return li3_neg(1/x) - l*(zeta2 + 1.0/6*l*l);
   } else if (x == -1) {
      return -0.75*zeta3;
   } else if (x < 0) {
      return li3_neg(x);
   } else if (x == 0) {
      return 0;
   } else if (x < 0.5) {
      return li3_pos(x);
   } else if (x == 0.5) {
      return 0.53721319360804020;
   } else if (x < 1) {
      const double l = log(x);
      return -li3_neg(1 - 1/x) - li3_pos(1 - x)
         + zeta3 + l*(zeta2 + l*(-0.5*log(1 - x) + 1.0/6*l));
   } else if (x == 1) {
      return zeta3;
   } else if (x < 2) {
      const double l = log(x);
      return -li3_neg(1 - x) - li3_pos(1 - 1/x)
         + zeta3 + l*(zeta2 + l*(-0.5*log(x - 1) + 1.0/6*l));
   } else { // x >= 2.0
      const double l = log(x);
      return li3_pos(1/x) + l*(2*zeta2 - 1.0/6*l*l);
   }
}

double F0(double z){
  return log(1+exp(z));
}

double F1(double z){
  return -li2(-exp(z));
}

double F2(double z){
  return -2*li3(-exp(z));
}

double G0(double y, double eta, double eta_prime){
  return F0(eta_prime - y) - F0(eta - y);
}

double G1(double y, double eta, double eta_prime){
  return F1(eta_prime - y) - F1(eta - y);
}

double G2(double y, double eta, double eta_prime){
  return F2(eta_prime - y) - F2(eta - y);
}

/* Main functions for neutrino pair production*/

double I1(double E_nu, double E_nubar, double costheta, double T, double eta_e){
  /** 
    E_nu     --- Neutrino energy [MeV]
    E_nubar  --- Antineutrino energy [MeV]
    costheta --- cos(relative angle)
    T        --- Temperature [MeV]
    eta_e    --- mu_e/T, with mu_e the electron chemical potential including electron mass
  */

  double Delta_e = sqrt(SQR(E_nubar) + SQR(E_nu) + 2*E_nu*E_nubar*costheta);
  double eta_prime = eta_e + (E_nu + E_nubar)/T;
  double Emax = 0.5*(E_nu+E_nubar) + Delta_e/2. * sqrt(1 - 2*SQR(m_e)/(E_nu*E_nubar*(1-costheta)));
  double Emin = 0.5*(E_nu+E_nubar) - Delta_e/2. * sqrt(1 - 2*SQR(m_e)/(E_nu*E_nubar*(1-costheta)));
  double ymax = Emax/T;
  double ymin = Emin/T;
  double A = SQR(E_nu) + SQR(E_nubar) - E_nu*E_nubar*(3+costheta);
  double B = E_nu * (E_nu * E_nubar * (3-costheta) + SQR(E_nubar)*(1+3*costheta) - 2*SQR(E_nu));
  double C = SQR(E_nu) * (SQR(E_nu + E_nubar*costheta) - 0.5 * SQR(E_nubar) * (1-SQR(costheta)) - 0.5*SQR(m_e * Delta_e/E_nu) * (1+costheta) / (1-costheta));

  double prefactor = -2*M_PI*T * SQR(E_nu) * SQR(E_nubar) * SQR(1-costheta) / ( (exp((E_nu+E_nubar)/T) - 1) * pow(Delta_e, 5));  

  return prefactor * (A * SQR(T) * ( (G2(ymax, eta_e, eta_prime) - G2(ymin, eta_e, eta_prime))
				     + (2*ymax*G1(ymax, eta_e, eta_prime) - 2*ymin*G1(ymin, eta_e, eta_prime))
				     + (SQR(ymax) * G0(ymax, eta_e, eta_prime) - SQR(ymin) * G0(ymin, eta_e, eta_prime)) )
		      + B * T * ( (G1(ymax, eta_e, eta_prime) - G1(ymin, eta_e, eta_prime))
				  + (ymax * G0(ymax, eta_e, eta_prime) - ymin * G0(ymin, eta_e, eta_prime)) )
		      + C * (G0(ymax, eta_e, eta_prime) - G0(ymin, eta_e, eta_prime)) );
}

double I2(double E_nu, double E_nubar, double costheta, double T, double eta_e){
  /**
    E_nu     --- Neutrino energy [MeV]
    E_nubar  --- Antineutrino energy [MeV]
    costheta --- cos(relative angle)
    T        --- Temperature [MeV]
    eta_e    --- mu_e/T, with mu_e the electron chemical potential including electron mass
  */
    
  return I1(E_nubar, E_nu, costheta, T, eta_e);
}

double I3(double E_nu, double E_nubar, double costheta, double T, double eta_e){
  /**
    E_nu     --- Neutrino energy [MeV]
    E_nubar  --- Antineutrino energy [MeV]
    costheta --- cos(relative angle)
    T        --- Temperature [MeV]
    eta_e    --- mu_e/T, with mu_e the electron chemical potential including electron mass
  */
    
  double Delta_e = sqrt(SQR(E_nubar) + SQR(E_nu) + 2*E_nu*E_nubar*costheta);
  double eta_prime = eta_e + (E_nu + E_nubar)/T;
  double Emax = 0.5*(E_nu+E_nubar) + Delta_e/2. * sqrt(1 - 2*SQR(m_e)/(E_nu*E_nubar*(1-costheta)));
  double Emin = 0.5*(E_nu+E_nubar) - Delta_e/2. * sqrt(1 - 2*SQR(m_e)/(E_nu*E_nubar*(1-costheta)));
  double ymax = Emax/T;
  double ymin = Emin/T;
    
  double prefactor = - 2*M_PI*T * SQR(m_e) * E_nu * E_nubar * (1-costheta) / ( (exp((E_nu+E_nubar)/T) - 1) * Delta_e);

  return prefactor * (G0(ymax, eta_e, eta_prime) - G0(ymin, eta_e, eta_prime));
}

double R(double E_nu, double E_nubar, double costheta, double T, double eta_e){
  /**
    Number of neutrino-antineutrino pairs produced per:
       - Time
       - Volume
       - Neutrino energy
       - Antineutrino energy
       - cos(angle between neutrino and antineutrino)
    [MeV^2]   

    E_nu     --- Neutrino energy [MeV]
    E_nubar  --- Antineutrino energy [MeV]
    costheta --- cos(relative angle)
    T        --- Temperature [K]
    eta_e    --- mu_e/T, with mu_e the electron chemical potential including electron mass
  */
  T *= k_B; // Convert from K to MeV

  return 4 * E_nu * E_nubar / pow(2*M_PI, 6) * SQR(G_F) * (beta_1 * I1(E_nu, E_nubar, costheta, T, eta_e)
							   + beta_2 * I2(E_nu, E_nubar, costheta, T, eta_e)
							   + beta_3 * I3(E_nu, E_nubar, costheta, T, eta_e));
}

double R_fla(double E_nu, double E_nubar, double costheta, double T, double eta_e, int fla){
  /**
    Number of neutrino-antineutrino pairs with flavor fla produced per:
       - Time
       - Volume
       - Neutrino energy
       - Antineutrino energy
       - cos(angle between neutrino and antineutrino)
    [MeV^2]   

    E_nu     --- Neutrino energy [MeV]
    E_nubar  --- Antineutrino energy [MeV]
    costheta --- cos(relative angle)
    T        --- Temperature [K]
    eta_e    --- mu_e/T, with mu_e the electron chemical potential including electron mass
    fla      --- 0 for nu_e, 1 for nu_x (= nu_mu = nu_tau, *not* the sum)
  */
  T *= k_B; // Convert from K to MeV

  if (fla==0)
    return 4 * E_nu * E_nubar / pow(2*M_PI, 6) * SQR(G_F) * (beta_1_e * I1(E_nu, E_nubar, costheta, T, eta_e)
							     + beta_2_e * I2(E_nu, E_nubar, costheta, T, eta_e)
							     + beta_3_e * I3(E_nu, E_nubar, costheta, T, eta_e));
  else if (fla==1)
    return 4 * E_nu * E_nubar / pow(2*M_PI, 6) * SQR(G_F) * (beta_1_x * I1(E_nu, E_nubar, costheta, T, eta_e)
							     + beta_2_x * I2(E_nu, E_nubar, costheta, T, eta_e)
							     + beta_3_x * I3(E_nu, E_nubar, costheta, T, eta_e));
  else
    return 0;
}
