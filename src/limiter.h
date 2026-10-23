// limiter.h -- Shared slope and limiter implementation for 1D and 2D.
//
// Return a limited slope for three consecutive cell-center values (ql, qc, qr) and their spacings.
// The 2D solver applies the limiter separately in the x and y directions.
#ifndef CFD_LIMITER_H
#define CFD_LIMITER_H

#include <algorithm>
#include <cmath>

#include "types.h"

namespace cfd {

inline double minmod2(double a, double b) {
  if (a * b <= 0.0) return 0.0;
  return (std::fabs(a) < std::fabs(b)) ? a : b;
}

// Smooth limiter function of Venkatakrishnan (1993).
//   d2 = unlimited face extrapolation minus the cell value; d1 = same-sign neighbor extremum minus the cell value.
//   Return phi (possibly slightly above 1; the caller takes the minimum); eps2 = (K*h)^3.
inline double venkat_phi(double d1, double d2, double eps2) {
  if (std::fabs(d2) < 1e-300) return 1.0;
  const double num = (d1 * d1 + eps2) * d2 + 2.0 * d2 * d2 * d1;
  const double den = d2 * (d1 * d1 + 2.0 * d2 * d2 + d1 * d2 + eps2);
  return num / den;
}

// Second-order centered slope on a nonuniform grid, with hl = xc - xl and hr = xr - xc:
//   s = ( hl^2 (qr - qc) + hr^2 (qc - ql) ) / ( hl*hr*(hl+hr) )
inline double central_slope(double ql, double qc, double qr, double hl,
                            double hr) {
  return (hl * hl * (qr - qc) + hr * hr * (qc - ql)) / (hl * hr * (hl + hr));
}

// Limited slope. h is the central cell width used in the Venkatakrishnan parameter eps^2 = (K h)^3.
inline double limited_slope(double ql, double qc, double qr, double hl,
                            double hr, double h, int limiter, double K) {
  if (limiter == LIM_FIRST_ORDER) return 0.0;
  if (limiter == LIM_NONE) return central_slope(ql, qc, qr, hl, hr);
  if (limiter == LIM_MINMOD) return minmod2((qc - ql) / hl, (qr - qc) / hr);

  // LIM_VENKAT
  const double s0 = central_slope(ql, qc, qr, hl, hr);
  const double qmax = std::max(qc, std::max(ql, qr));
  const double qmin = std::min(qc, std::min(ql, qr));
  const double eps2 = std::pow(K * h, 3.0);
  double phi = 1.0;
  const double d2R = +0.5 * h * s0;  // Extrapolation to the right face.
  const double d2L = -0.5 * h * s0;  // Extrapolation to the left face.
  if (d2R > 0.0)
    phi = std::min(phi, venkat_phi(qmax - qc, d2R, eps2));
  else if (d2R < 0.0)
    phi = std::min(phi, venkat_phi(qmin - qc, d2R, eps2));
  if (d2L > 0.0)
    phi = std::min(phi, venkat_phi(qmax - qc, d2L, eps2));
  else if (d2L < 0.0)
    phi = std::min(phi, venkat_phi(qmin - qc, d2L, eps2));
  return phi * s0;
}

}  // namespace cfd

#endif  // CFD_LIMITER_H
