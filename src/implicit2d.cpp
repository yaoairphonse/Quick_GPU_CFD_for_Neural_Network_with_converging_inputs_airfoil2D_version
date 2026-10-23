#include "implicit2d.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

namespace cfd {

bool implicit2d_validate(const Solver2D* s, double tend, long maxsteps) {
  const Params2D& p = s->par;
  if (p.limiter != LIM_FIRST_ORDER || s->ib != nullptr) {
    std::fprintf(stderr, "backward Euler currently requires first-order regular/cut-cell grids (no legacy IB)\n");
    return false;
  }
  if (!std::isfinite(tend) || maxsteps < 0 || !(p.gamma > 1.0) ||
      !std::isfinite(p.gamma) || !(p.cfl > 0.0) || !std::isfinite(p.cfl) ||
      !(p.dt_fixed >= 0.0) || !std::isfinite(p.dt_fixed) ||
      !(p.pseudo_cfl > 0.0) || !std::isfinite(p.pseudo_cfl) ||
      p.implicit_maxiter < 1 || !(p.implicit_rtol >= 0.0) ||
      !std::isfinite(p.implicit_rtol) || !(p.implicit_atol > 0.0) ||
      !std::isfinite(p.implicit_atol) ||
      (s->cut != nullptr && s->cut->nbad_nbhd != 0)) {
    std::fprintf(stderr, "invalid backward-Euler controls or cut-cell geometry\n");
    return false;
  }
  for (int c = 0; c < s->ncase; ++c) {
    if (!std::isfinite(s->t[c]) || !(s->mco[c] > 0.0) || !std::isfinite(s->mco[c])) {
      std::fprintf(stderr, "invalid backward-Euler case time or reference Mach\n");
      return false;
    }
  }
  return true;
}

bool implicit2d_converged(Solver2D* s, const double* norms, double* initial, bool first) {
  bool done = true;
  s->implicit.initial_defect = s->implicit.final_defect = s->implicit.tolerance = 0.0;
  for (int c = 0; c < s->ncase; ++c) {
    if (first) initial[c] = norms[c];
    const double tol = std::max(s->par.implicit_atol, s->par.implicit_rtol * initial[c]);
    s->implicit.initial_defect = std::max(s->implicit.initial_defect, initial[c]);
    s->implicit.final_defect = std::max(s->implicit.final_defect, norms[c]);
    s->implicit.tolerance = std::max(s->implicit.tolerance, tol);
    if (!std::isfinite(norms[c]) || !std::isfinite(tol) || norms[c] > tol) done = false;
  }
  return done;
}

bool implicit2d_physical_dt(Solver2D* s, int c, double rate, double tend) {
  const double remain = tend - s->t[c];
  if (remain <= 0.0) { s->dt[c] = 0.0; return true; }
  double dt = s->par.dt_fixed > 0.0 ? s->par.dt_fixed : s->par.cfl / rate;
  if (!(dt > 0.0) || !std::isfinite(dt)) return false;
  const double roundoff = 8*std::numeric_limits<double>::epsilon()*
                          std::max(std::fabs(tend),std::fabs(s->t[c]));
  // Complete the last step rather than creating a roundoff-sized extra step.
  if (dt >= remain || remain-dt <= roundoff) dt = remain;
  s->dt[c] = dt;
  return s->t[c]+dt > s->t[c];
}

void implicit2d_advance_time(Solver2D* s, double tend) {
  for (int c = 0; c < s->ncase; ++c) {
    if (s->dt[c] <= 0.0) continue;
    s->t[c] = s->dt[c] == tend-s->t[c] ? tend : s->t[c]+s->dt[c];
  }
}

static bool fluid(const Solver2D* s, size_t k) {
  return s->cut == nullptr || s->cut->vol[k] > 0.0;
}

static bool admissible(const Solver2D* s, const double* U) {
  const Grid2D& g = s->grid;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  for (int c = 0; c < s->ncase; ++c)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        const size_t k = static_cast<size_t>(j + g.ng) * g.mx + i + g.ng;
        if (!fluid(s, k)) continue;
        const size_t b = static_cast<size_t>(c) * NVAR2D * cells + k;
        if (!implicit_admissible(U[b], U[b + cells], U[b + 2*cells], U[b + 3*cells], s->par.gamma))
          return false;
      }
  return true;
}

static double spectral_rate(const Solver2D* s, const double* Q, int c, int i, int j, bool local) {
  const Grid2D& g = s->grid;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  const size_t k = static_cast<size_t>(j + g.ng) * g.mx + i + g.ng;
  const size_t b = static_cast<size_t>(c) * NVAR2D * cells + k;
  const Cut2D* ct = local ? s->cut : nullptr;
  const size_t fe = static_cast<size_t>(j) * (g.nx + 1) + i;
  const size_t fn = static_cast<size_t>(j) * g.nx + i;
  return implicit_spectral_rate(Q[b], Q[b+cells], Q[b+2*cells], Q[b+3*cells],
    s->par.gamma, g.dx[i+g.ng], g.dy[j+g.ng], ct != nullptr,
    ct ? ct->vol[k] : 0.0, ct ? ct->axl[fe+1] : 0.0, ct ? ct->axl[fe] : 0.0,
    ct ? ct->ayl[fn+g.nx] : 0.0, ct ? ct->ayl[fn] : 0.0,
    ct ? ct->wax[k] : 0.0, ct ? ct->way[k] : 0.0);
}

int solver2d_run_implicit(Solver2D* s, double tend, long maxsteps) {
  if (!implicit2d_validate(s, tend, maxsteps)) return 3;
  if (!admissible(s, s->U.data())) {
    std::fprintf(stderr, "backward Euler: invalid initial fluid state\n");
    return 2;
  }
  const Grid2D& g = s->grid;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  const size_t states = s->U.size();
  std::vector<double> norms(s->ncase), initial(s->ncase);
  for (long step = 0; ; ++step) {
    bool done = true;
    for (int c = 0; c < s->ncase; ++c) if (s->t[c] < tend) done = false;
    if (done) return 0;
    if (step >= maxsteps) return 1;
    for (int c = 0; c < s->ncase; ++c) {
      double rate = 0.0;
      for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
          const size_t k = static_cast<size_t>(j + g.ng) * g.mx + i + g.ng;
          if (fluid(s, k)) rate = std::max(rate, spectral_rate(s, s->U.data(), c, i, j, false));
        }
      if (!implicit2d_physical_dt(s,c,rate,tend)) {
        std::fprintf(stderr, "backward Euler: physical dt is invalid or cannot advance time\n");
        return 3;
      }
    }
    double* Q = s->U1.data();
    double* trial = s->U2.data();
    std::copy(s->U.data(), s->U.data() + states, Q);
    std::copy(s->U.data(), s->U.data() + states, trial);
    double pseudo_cfl = s->par.pseudo_cfl;
    for (int iter = 0; ; ++iter) {
      solver2d_apply_bc(s, Q);
      solver2d_residual(s, Q, s->R.data(), s->W.data(), s->SX.data(), s->SY.data(),
                        s->FX.data(), s->FY.data());
      for (int c = 0; c < s->ncase; ++c) {
        double norm = 0.0;
        if (s->dt[c] > 0.0) {
          for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
              const size_t k = static_cast<size_t>(j + g.ng) * g.mx + i + g.ng;
              if (!fluid(s, k)) continue;
              for (int v = 0; v < NVAR2D; ++v) {
                const size_t b = (static_cast<size_t>(c) * NVAR2D + v) * cells + k;
                const double d = implicit_defect(s->U[b], Q[b], s->R[b], s->dt[c]);
                if (!std::isfinite(d)) return 2;
                norm = std::max(norm, implicit_scaled_defect(s->U[b], d));
              }
            }
        }
        norms[c] = norm;
      }
      if (implicit2d_converged(s, norms.data(), initial.data(), iter == 0)) break;
      if (iter >= s->par.implicit_maxiter) {
        std::fprintf(stderr, "backward Euler not converged: step=%ld updates=%d initial=%.6e defect=%.6e tolerance=%.6e; physical step not accepted\n",
                     s->nsteps, iter, s->implicit.initial_defect, s->implicit.final_defect,
                     s->implicit.tolerance);
        return 4;
      }
      int retries = 0;
      for (;;) {
        for (int c = 0; c < s->ncase; ++c) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
          for (int ij = 0; ij < g.nx * g.ny; ++ij) {
            const int i = ij % g.nx, j = ij / g.nx;
            const size_t k = static_cast<size_t>(j + g.ng) * g.mx + i + g.ng;
            if (!fluid(s, k) || s->dt[c] <= 0.0) continue;
            const double rate = spectral_rate(s, Q, c, i, j, true);
            const double omega = pseudo_cfl / (pseudo_cfl + s->dt[c] * rate);
            for (int v = 0; v < NVAR2D; ++v) {
              const size_t b = (static_cast<size_t>(c) * NVAR2D + v) * cells + k;
              trial[b] = Q[b] + omega * implicit_defect(s->U[b], Q[b], s->R[b], s->dt[c]);
            }
          }
        }
        if (admissible(s, trial)) break;
        ++s->implicit.retries;
        if (++retries > 12) {
          std::fprintf(stderr, "backward Euler: nonphysical pseudo candidate after 12 retries; physical step not accepted\n");
          return 2;
        }
        pseudo_cfl *= 0.5;
      }
      std::swap(Q, trial);
      ++s->implicit.updates;
    }
    // Commit only after every case meets its own raw BE defect tolerance.
    for (int c = 0; c < s->ncase; ++c) {
      if (s->dt[c] <= 0.0) continue;
      for (int v = 0; v < NVAR2D; ++v)
        for (int j = 0; j < g.ny; ++j)
          for (int i = 0; i < g.nx; ++i) {
            const size_t b = u2idx(g, c, v, i, j);
            s->U[b] = Q[b];
          }
    }
    implicit2d_advance_time(s,tend);
    ++s->nsteps;
  }
}

} // namespace cfd
