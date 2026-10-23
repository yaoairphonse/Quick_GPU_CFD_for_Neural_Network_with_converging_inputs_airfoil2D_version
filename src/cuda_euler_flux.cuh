// Device flux definitions copied from the existing validated CUDA solver.
#ifndef CFD_CUDA_EULER_FLUX_CUH
#define CFD_CUDA_EULER_FLUX_CUH
namespace cfd { namespace gpu_flux {
__device__ inline double d_m1(double m, int s) { return 0.5 * (m + s * fabs(m)); }
__device__ inline double d_m2(double m, int s) { return s * 0.25 * (m + s) * (m + s); }
__device__ inline double d_m4(double m, int s) {
  if (fabs(m) >= 1.0) return d_m1(m, s);
  return d_m2(m, s) * (1.0 - s * 2.0 * d_m2(m, -s));
}
__device__ inline double d_p5(double m, int s, double alpha) {
  if (fabs(m) >= 1.0) return 0.5 * (1.0 + s * (m > 0.0 ? 1.0 : (m < 0.0 ? -1.0 : 0.0)));
  return d_m2(m, s) * ((s * 2.0 - m) - s * 16.0 * alpha * m * d_m2(m, -s));
}

__device__ inline void flux_exact(double r, double u, double v, double p, double gm, double* f) {
  const double E = p / (gm - 1.0) + 0.5 * r * (u * u + v * v);
  f[0] = r * u; f[1] = r * u * u + p; f[2] = r * u * v; f[3] = u * (E + p);
}

__device__ inline void flux_hllc(double rL, double uL, double vL, double pL,
                          double rR, double uR, double vR, double pR,
                          double gm, double* f) {
  const double aL = sqrt(gm * pL / rL), aR = sqrt(gm * pR / rR);
  const double EL = pL / (gm - 1.0) + 0.5 * rL * (uL * uL + vL * vL);
  const double ER = pR / (gm - 1.0) + 0.5 * rR * (uR * uR + vR * vR);
  const double qL = sqrt(rL), qR = sqrt(rR);
  const double HL = (EL + pL) / rL, HR = (ER + pR) / rR;
  const double uT = (qL * uL + qR * uR) / (qL + qR);
  const double vT = (qL * vL + qR * vR) / (qL + qR);
  const double HT = (qL * HL + qR * HR) / (qL + qR);
  const double aT = sqrt(fmax((gm - 1.0) * (HT - 0.5 * (uT * uT + vT * vT)), 1e-300));
  const double SL = fmin(uL - aL, uT - aT), SR = fmax(uR + aR, uT + aT);
  double fL[4], fR[4];
  flux_exact(rL, uL, vL, pL, gm, fL); flux_exact(rR, uR, vR, pR, gm, fR);
  if (SL >= 0.0) { for (int v = 0; v < 4; ++v) f[v] = fL[v]; return; }
  if (SR <= 0.0) { for (int v = 0; v < 4; ++v) f[v] = fR[v]; return; }
  const double Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) /
                    (rL * (SL - uL) - rR * (SR - uR));
  if (Ss >= 0.0) {
    const double a = rL * (SL - uL) / (SL - Ss);
    const double us[4] = {a, a * Ss, a * vL,
      a * (EL / rL + (Ss - uL) * (Ss + pL / (rL * (SL - uL))))};
    const double ul[4] = {rL, rL * uL, rL * vL, EL};
    for (int v = 0; v < 4; ++v) f[v] = fL[v] + SL * (us[v] - ul[v]);
  } else {
    const double a = rR * (SR - uR) / (SR - Ss);
    const double us[4] = {a, a * Ss, a * vR,
      a * (ER / rR + (Ss - uR) * (Ss + pR / (rR * (SR - uR))))};
    const double ur[4] = {rR, rR * uR, rR * vR, ER};
    for (int v = 0; v < 4; ++v) f[v] = fR[v] + SR * (us[v] - ur[v]);
  }
}

__device__ inline void flux_ausm(double rL, double uL, double vL, double pL,
                          double rR, double uR, double vR, double pR,
                          double gm, double mco, double* f) {
  const double a12 = 0.5 * (sqrt(gm * pL / rL) + sqrt(gm * pR / rR));
  const double rho12 = 0.5 * (rL + rR), ML = uL / a12, MR = uR / a12;
  const double mb2 = 0.5 * (uL * uL + uR * uR) / (a12 * a12);
  const double m02 = fmin(1.0, fmax(mb2, mco * mco));
  const double fa = sqrt(m02) * (2.0 - sqrt(m02));
  const double alpha = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa);
  const double m12 = d_m4(ML, 1) + d_m4(MR, -1) -
    (0.25 / fa) * fmax(1.0 - mb2, 0.0) * (pR - pL) / (rho12 * a12 * a12);
  const double pp = d_p5(ML, 1, alpha), pm = d_p5(MR, -1, alpha);
  const double p12 = pp * pL + pm * pR - 0.75 * pp * pm * (rL + rR) * fa * a12 * (uR - uL);
  const bool left = m12 > 0.0;
  const double r = left ? rL : rR, u = left ? uL : uR;
  const double v = left ? vL : vR, p = left ? pL : pR;
  const double mdot = a12 * m12 * r;
  const double E = p / (gm - 1.0) + 0.5 * r * (u * u + v * v);
  f[0] = mdot; f[1] = mdot * u + p12; f[2] = mdot * v; f[3] = mdot * (E + p) / r;
}

} }
#endif
