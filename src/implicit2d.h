// Internal shared arithmetic for CPU/CUDA backward-Euler defect relaxation.
#ifndef CFD_IMPLICIT2D_H
#define CFD_IMPLICIT2D_H

#include <cmath>
#include "euler2d.h"

#ifdef __CUDACC__
#define CFD_IMPLICIT_HD __host__ __device__
#else
#define CFD_IMPLICIT_HD
#endif

namespace cfd {

CFD_IMPLICIT_HD inline bool implicit_finite(double x) {
#ifdef __CUDA_ARCH__
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

CFD_IMPLICIT_HD inline bool implicit_admissible(double r, double mu, double mv,
                                               double E, double gm) {
  if (!(r > 0.0) || !implicit_finite(r) || !implicit_finite(mu) ||
      !implicit_finite(mv) || !implicit_finite(E)) return false;
  const double p = (gm - 1.0) * (E - 0.5 * (mu * mu + mv * mv) / r);
  return p > 0.0 && implicit_finite(p);
}

// Spectral radius bound used ONLY as a pseudo-time preconditioner.  Actual
// volume/apertures matter at cut cells; full-cell CFL chooses physical dt.
CFD_IMPLICIT_HD inline double implicit_spectral_rate(
    double r, double mu, double mv, double E, double gm, double dx, double dy,
    bool cut, double vol, double ae, double aw, double an, double as,
    double wax, double way) {
  const double u = mu / r, v = mv / r;
  const double p = (gm - 1.0) * (E - 0.5 * (mu * mu + mv * mv) / r);
  const double a = sqrt(gm * p / r);
  if (!cut) return (fabs(u) + a) / dx + (fabs(v) + a) / dy;
  const double wall = sqrt(wax * wax + way * way);
  return ((fabs(u) + a) * (ae + aw) + (fabs(v) + a) * (an + as) +
          fabs(u * wax + v * way) + a * wall) / (2.0 * vol);
}

CFD_IMPLICIT_HD inline double implicit_defect(double old, double q, double R, double dt) {
  return dt * R - (q - old);
}

// Scale is frozen at the start of the physical step.  Check D itself, never
// omega*D: tiny cells can make omega small even for a large equation defect.
CFD_IMPLICIT_HD inline double implicit_scaled_defect(double old, double defect) {
  return fabs(defect) / fmax(1.0, fabs(old));
}

bool implicit2d_validate(const Solver2D* s, double tend, long maxsteps);
bool implicit2d_converged(Solver2D* s, const double* norms, double* initial, bool first);
bool implicit2d_physical_dt(Solver2D* s, int c, double rate, double tend);
void implicit2d_advance_time(Solver2D* s, double tend);

} // namespace cfd
#undef CFD_IMPLICIT_HD
#endif
