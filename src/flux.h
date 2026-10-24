// flux.h -- Numerical Euler fluxes; pure functions with no state.
//
// Conserved variables per unit volume: rho, rho*un, rho*ut, E, with E = p/(gamma-1) + rho*(un^2+ut^2)/2.
//   un is face-normal velocity; ut is tangential velocity, transported by advection.
// flux_hllc_t / flux_ausmpup_t accept left/right states (rho, un, ut, p)
// and return F[4] = (mass, normal momentum, tangential momentum, energy).
// The 1D flux_hllc / flux_ausmpup wrappers use ut = 0 and return three components.
// This keeps one flux implementation for 1D/2D; axis-aligned 2D cases must match 1D bit for bit.
// All quantities are nondimensional; both states must use the same reference scales.
#ifndef CFD_FLUX_H
#define CFD_FLUX_H

#include <algorithm>
#include <cmath>

namespace cfd {

// Exact Euler flux F(U), including tangential velocity, for the consistency check F(U,U) = F(U).
inline void flux_euler_exact_t(double rho, double un, double ut, double p,
                               double gamma, double F[4]) {
  const double E = p / (gamma - 1.0) + 0.5 * rho * un * un + 0.5 * rho * ut * ut;
  F[0] = rho * un;
  F[1] = rho * un * un + p;
  F[2] = rho * un * ut;
  F[3] = un * (E + p);
}

inline void flux_euler_exact(double rho, double u, double p, double gamma,
                             double F[3]) {
  double G[4];
  flux_euler_exact_t(rho, u, 0.0, p, gamma, G);
  F[0] = G[0];
  F[1] = G[1];
  F[2] = G[3];
}

// ---------------------------------------------------------------------------
// HLLC (Toro, Spruce, Speares 1994), with Einfeldt wave speeds from Roe averages.
// Resolves contacts, including stationary contacts; more conservative near strong rarefactions than AUSM-type fluxes.
// Tangential velocity is advected: each star region retains the tangential velocity of its side.
// ---------------------------------------------------------------------------
inline void flux_hllc_t(double rhoL, double uL, double vL, double pL,
                        double rhoR, double uR, double vR, double pR,
                        double gamma, double F[4]) {
  const double aL = std::sqrt(gamma * pL / rhoL);
  const double aR = std::sqrt(gamma * pR / rhoR);
  const double EL = pL / (gamma - 1.0) + 0.5 * rhoL * uL * uL + 0.5 * rhoL * vL * vL;
  const double ER = pR / (gamma - 1.0) + 0.5 * rhoR * uR * uR + 0.5 * rhoR * vR * vR;

  // Roe averages.
  const double sL = std::sqrt(rhoL), sR = std::sqrt(rhoR);
  const double HL = (EL + pL) / rhoL, HR = (ER + pR) / rhoR;
  const double uT = (sL * uL + sR * uR) / (sL + sR);
  const double vT = (sL * vL + sR * vR) / (sL + sR);
  const double HT = (sL * HL + sR * HR) / (sL + sR);
  const double aT = std::sqrt(
      std::max((gamma - 1.0) * (HT - 0.5 * uT * uT - 0.5 * vT * vT), 1e-300));

  const double SL = std::min(uL - aL, uT - aT);
  const double SR = std::max(uR + aR, uT + aT);

  double FL[4], FR[4];
  flux_euler_exact_t(rhoL, uL, vL, pL, gamma, FL);
  flux_euler_exact_t(rhoR, uR, vR, pR, gamma, FR);

  if (SL >= 0.0) {
    for (int k = 0; k < 4; ++k) F[k] = FL[k];
    return;
  }
  if (SR <= 0.0) {
    for (int k = 0; k < 4; ++k) F[k] = FR[k];
    return;
  }

  // Contact-wave speed S*.
  const double Sstar =
      (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) /
      (rhoL * (SL - uL) - rhoR * (SR - uR));

  if (Sstar >= 0.0) {
    const double c = rhoL * (SL - uL) / (SL - Sstar);
    const double Us0 = c;
    const double Us1 = c * Sstar;
    const double Us2 = c * vL;
    const double Us3 =
        c * (EL / rhoL + (Sstar - uL) * (Sstar + pL / (rhoL * (SL - uL))));
    F[0] = FL[0] + SL * (Us0 - rhoL);
    F[1] = FL[1] + SL * (Us1 - rhoL * uL);
    F[2] = FL[2] + SL * (Us2 - rhoL * vL);
    F[3] = FL[3] + SL * (Us3 - EL);
  } else {
    const double c = rhoR * (SR - uR) / (SR - Sstar);
    const double Us0 = c;
    const double Us1 = c * Sstar;
    const double Us2 = c * vR;
    const double Us3 =
        c * (ER / rhoR + (Sstar - uR) * (Sstar + pR / (rhoR * (SR - uR))));
    F[0] = FR[0] + SR * (Us0 - rhoR);
    F[1] = FR[1] + SR * (Us1 - rhoR * uR);
    F[2] = FR[2] + SR * (Us2 - rhoR * vR);
    F[3] = FR[3] + SR * (Us3 - ER);
  }
}

inline void flux_hllc(double rhoL, double uL, double pL, double rhoR, double uR,
                      double pR, double gamma, double F[3]) {
  double G[4];
  flux_hllc_t(rhoL, uL, 0.0, pL, rhoR, uR, 0.0, pR, gamma, G);
  F[0] = G[0];
  F[1] = G[1];
  F[2] = G[3];
}

// ---------------------------------------------------------------------------
// AUSM+-up (Liou 2006, JCP 214:137-170).
// Split the interface flux into convective mass-flux and pressure terms, each split by Mach number.
// Recommended constants: Kp=0.25, Ku=0.75, sigma=1, beta=1/8; alpha depends on fa.
// Mco is the cutoff Mach number, representing the reference/freestream Mach number rather than a tuning parameter.
// Use 1 for compressible transonic/supersonic cases; a small value amplifies pressure diffusion by 1/fa.
// ---------------------------------------------------------------------------
namespace ausm_detail {
inline double M1(double M, int s) { return 0.5 * (M + s * std::fabs(M)); }
inline double M2(double M, int s) { return s * 0.25 * (M + s) * (M + s); }
inline double M4(double M, int s, double beta) {
  if (std::fabs(M) >= 1.0) return M1(M, s);
  return M2(M, s) * (1.0 - s * 16.0 * beta * M2(M, -s));
}
inline double P5(double M, int s, double alpha) {
  if (std::fabs(M) >= 1.0)
    return 0.5 * (1.0 + s * (M > 0.0 ? 1.0 : (M < 0.0 ? -1.0 : 0.0)));
  return M2(M, s) * ((s * 2.0 - M) - s * 16.0 * alpha * M * M2(M, -s));
}
}  // namespace ausm_detail

inline void flux_ausmpup_t(double rhoL, double uL, double vL, double pL,
                           double rhoR, double uR, double vR, double pR,
                           double gamma, double Mco, double F[4]) {
  using namespace ausm_detail;
  const double Kp = 0.25, Ku = 0.75, sigma = 1.0, beta = 0.125;

  const double aL = std::sqrt(gamma * pL / rhoL);
  const double aR = std::sqrt(gamma * pR / rhoR);
  const double a12 = 0.5 * (aL + aR);
  const double rho12 = 0.5 * (rhoL + rhoR);

  const double ML = uL / a12, MR = uR / a12;
  const double Mbar2 = 0.5 * (uL * uL + uR * uR) / (a12 * a12);
  const double M02 = std::min(1.0, std::max(Mbar2, Mco * Mco));
  const double M0 = std::sqrt(M02);
  const double fa = M0 * (2.0 - M0);
  const double alpha = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa);

  // Interface Mach number, including pressure diffusion.
  const double M12 = M4(ML, +1, beta) + M4(MR, -1, beta) -
                     (Kp / fa) * std::max(1.0 - sigma * Mbar2, 0.0) * (pR - pL) /
                         (rho12 * a12 * a12);

  // Interface pressure, including velocity diffusion.
  const double Pp = P5(ML, +1, alpha), Pm = P5(MR, -1, alpha);
  const double p12 = Pp * pL + Pm * pR -
                     Ku * Pp * Pm * (rhoL + rhoR) * (fa * a12) * (uR - uL);

  // Mass flux, upwinded according to the sign of M12.
  const double mdot = a12 * M12 * (M12 > 0.0 ? rhoL : rhoR);
  const double rho = (M12 > 0.0) ? rhoL : rhoR;
  const double u = (M12 > 0.0) ? uL : uR;
  const double v = (M12 > 0.0) ? vL : vR;
  const double p = (M12 > 0.0) ? pL : pR;
  const double E = p / (gamma - 1.0) + 0.5 * rho * u * u + 0.5 * rho * v * v;
  const double H = (E + p) / rho;

  F[0] = mdot;
  F[1] = mdot * u + p12;
  F[2] = mdot * v;
  F[3] = mdot * H;
}

inline void flux_ausmpup(double rhoL, double uL, double pL, double rhoR,
                         double uR, double pR, double gamma, double Mco,
                         double F[3]) {
  double G[4];
  flux_ausmpup_t(rhoL, uL, 0.0, pL, rhoR, uR, 0.0, pR, gamma, Mco, G);
  F[0] = G[0];
  F[1] = G[1];
  F[2] = G[3];
}

}  // namespace cfd

#endif  // CFD_FLUX_H
