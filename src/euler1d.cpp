#include "euler1d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "flux.h"
#include "limiter.h"

namespace cfd {

// ---------------------------------------------------------------------------
// Indexing: U[(c*NVAR + v)*m + (i+ng)]; interior index i may be negative for ghosts.
// ---------------------------------------------------------------------------
static inline size_t uidx(const Grid1D& g, int c, int v, int i) {
  return (static_cast<size_t>(c) * NVAR1D + v) * g.m + (i + g.ng);
}

Params1D params1d_default() {
  Params1D p;
  p.gamma = 1.4;
  p.cfl = 0.5;
  p.flux = FLUX_HLLC;
  p.limiter = LIM_VENKAT;
  // K tradeoff measured on 2026-10-08: K=0.3 gives a smooth-solution L2 rate of about 1.75,
  // because eps^2=(Kh)^3 limits a region of width ~h^0.5 near extrema; K=10 gives ~1.95 but ~0.5% Sod overshoot.
  p.venkat_k = 5.0;
  // The AUSM+-up cutoff is the reference/freestream Mach number, not a tuning parameter.
  // Compressible transonic/supersonic cases use O(1), giving fa=1. Small values are for low speeds,
  // but they amplify pressure diffusion by 1/fa and can produce negative pressure in strong shock tubes.
  // Measurements on 2026-10-08: Mco=0.1 failed on the first step for Sod and Woodward-Colella.
  p.ausm_mco = 1.0;
  p.bc = BC_TRANSMISSIVE;
  p.dt_fixed = 0.0;
  return p;
}

void solver1d_init(Solver1D* s, int n, double xmin, double xmax, int ncase,
                   const Params1D& par) {
  s->par = par;
  s->ncase = ncase;
  s->nsteps = 0;

  Grid1D& g = s->grid;
  g.n = n;
  g.ng = 2;
  g.m = n + 2 * g.ng;
  g.xmin = xmin;
  g.xmax = xmax;
  g.xc.allocate(g.m);
  g.dx.allocate(g.m);
  const double h = (xmax - xmin) / n;
  for (int i = -g.ng; i < n + g.ng; ++i) {
    g.xc[i + g.ng] = xmin + (i + 0.5) * h;
    g.dx[i + g.ng] = h;
  }

  const size_t sz = static_cast<size_t>(ncase) * NVAR1D * g.m;
  s->U.allocate(sz);
  s->U1.allocate(sz);
  s->U2.allocate(sz);
  s->R.allocate(sz);
  s->W.allocate(sz);
  s->S.allocate(sz);
  s->t.allocate(ncase);
  s->dt.allocate(ncase);
}

void solver1d_set_primitive(Solver1D* s, int c, int i, double rho, double u,
                            double p) {
  const Grid1D& g = s->grid;
  const double gm = s->par.gamma;
  s->U[uidx(g, c, 0, i)] = rho;
  s->U[uidx(g, c, 1, i)] = rho * u;
  s->U[uidx(g, c, 2, i)] = p / (gm - 1.0) + 0.5 * rho * u * u;
}

void solver1d_set_riemann(Solver1D* s, int c, double x0, double rhoL,
                          double uL, double pL, double rhoR, double uR,
                          double pR) {
  const Grid1D& g = s->grid;
  for (int i = 0; i < g.n; ++i) {
    if (g.xc[i + g.ng] < x0)
      solver1d_set_primitive(s, c, i, rhoL, uL, pL);
    else
      solver1d_set_primitive(s, c, i, rhoR, uR, pR);
  }
}

void solver1d_get_primitive(const Solver1D* s, int c, int i, double* rho,
                            double* u, double* p) {
  const Grid1D& g = s->grid;
  const double gm = s->par.gamma;
  const double r = s->U[uidx(g, c, 0, i)];
  const double mom = s->U[uidx(g, c, 1, i)];
  const double E = s->U[uidx(g, c, 2, i)];
  *rho = r;
  *u = mom / r;
  *p = (gm - 1.0) * (E - 0.5 * mom * mom / r);
}

double solver1d_integral(const Solver1D* s, int c, int var) {
  const Grid1D& g = s->grid;
  double sum = 0.0;
  for (int i = 0; i < g.n; ++i)
    sum += s->U[uidx(g, c, var, i)] * g.dx[i + g.ng];
  return sum;
}

// ---------------------------------------------------------------------------
// Fill ghost cells according to the boundary conditions.
// ---------------------------------------------------------------------------
void solver1d_apply_bc(const Solver1D* s, double* U) {
  const Grid1D& g = s->grid;
  for (int c = 0; c < s->ncase; ++c) {
    for (int v = 0; v < NVAR1D; ++v) {
      double* q = U + (static_cast<size_t>(c) * NVAR1D + v) * g.m + g.ng;
      if (s->par.bc == BC_PERIODIC) {
        for (int k = 1; k <= g.ng; ++k) {
          q[-k] = q[g.n - k];
          q[g.n - 1 + k] = q[k - 1];
        }
      } else if (s->par.bc == BC_REFLECT) {
        // Mirror the state and reverse momentum (v == 1).
        const double sgn = (v == 1) ? -1.0 : 1.0;
        for (int k = 1; k <= g.ng; ++k) {
          q[-k] = sgn * q[k - 1];
          q[g.n - 1 + k] = sgn * q[g.n - k];
        }
      } else {
        for (int k = 1; k <= g.ng; ++k) {
          q[-k] = q[0];
          q[g.n - 1 + k] = q[g.n - 1];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Residual: R_i = -(F_{i+1/2} - F_{i-1/2}) / dx_i.
// W stores primitive variables (rho,u,p); S stores limited slopes. Both have the same shape as U.
// ---------------------------------------------------------------------------
void solver1d_residual(const Solver1D* s, const double* U, double* R,
                       double* W, double* S) {
  const Grid1D& g = s->grid;
  const Params1D& par = s->par;
  const double gm = par.gamma;
  const int n = g.n, ng = g.ng, m = g.m;

  for (int c = 0; c < s->ncase; ++c) {
    const size_t base = static_cast<size_t>(c) * NVAR1D * m;
    const double* Uc = U + base;
    double* Rc = R + base;
    double* Wr = W + base;       // rho
    double* Wu = W + base + m;   // u
    double* Wp = W + base + 2 * m;  // p
    double* Sr = S + base;
    double* Su = S + base + m;
    double* Sp = S + base + 2 * m;

    // 1) Conserved to primitive variables, including ghosts.
    for (int j = 0; j < m; ++j) {
      const double r = Uc[j];
      const double mom = Uc[m + j];
      const double E = Uc[2 * m + j];
      Wr[j] = r;
      Wu[j] = mom / r;
      Wp[j] = (gm - 1.0) * (E - 0.5 * mom * mom / r);
    }

    // 2) Slopes for cells -1..n, since each face requires both neighboring slopes.
    double* Wq[3] = {Wr, Wu, Wp};
    double* Sq[3] = {Sr, Su, Sp};
    for (int v = 0; v < 3; ++v) {
      const double* q = Wq[v];
      double* sl = Sq[v];
      for (int i = -1; i <= n; ++i) {
        const int j = i + ng;
        const double hl = g.xc[j] - g.xc[j - 1];
        const double hr = g.xc[j + 1] - g.xc[j];
        sl[j] = limited_slope(q[j - 1], q[j], q[j + 1], hl, hr, g.dx[j],
                              par.limiter, par.venkat_k);
      }
    }

    // 3) Fluxes at faces i+1/2 for i = -1..n-1, giving n+1 faces.
    // Temporarily store face fluxes in the R workspace instead of allocating another large array.
    // Store F_{i+1/2} at the matching Rc index j = i+ng, then take differences.
    // Three variables use Rc[v*m + j].
    for (int i = -1; i <= n - 1; ++i) {
      const int jl = i + ng, jr = i + 1 + ng;
      const double hl = 0.5 * g.dx[jl], hr = 0.5 * g.dx[jr];
      const double rL = Wr[jl] + hl * Sr[jl];
      const double uL = Wu[jl] + hl * Su[jl];
      const double pL = Wp[jl] + hl * Sp[jl];
      const double rR = Wr[jr] - hr * Sr[jr];
      const double uR = Wu[jr] - hr * Su[jr];
      const double pR = Wp[jr] - hr * Sp[jr];
      double F[3];
      if (par.flux == FLUX_AUSMPUP)
        flux_ausmpup(rL, uL, pL, rR, uR, pR, gm, par.ausm_mco, F);
      else
        flux_hllc(rL, uL, pL, rR, uR, pR, gm, F);
      for (int v = 0; v < 3; ++v) Rc[v * m + jl] = F[v];
    }

    // 4) Flux differences: F_{i+1/2} is at i+ng, F_{i-1/2} at i-1+ng.
    // Read the right-face flux before overwriting its location while marching from i=0 to n-1.
    // R_i and F_{i+1/2} share index i+ng, so temporary storage is required.
    // March left to right, retaining the previous face flux in Fprev.
    double Fprev[3] = {Rc[0 * m + (-1 + ng)], Rc[1 * m + (-1 + ng)],
                       Rc[2 * m + (-1 + ng)]};
    for (int i = 0; i < n; ++i) {
      const int j = i + ng;
      const double Fcur[3] = {Rc[0 * m + j], Rc[1 * m + j], Rc[2 * m + j]};
      const double inv = 1.0 / g.dx[j];
      for (int v = 0; v < 3; ++v) Rc[v * m + j] = -(Fcur[v] - Fprev[v]) * inv;
      for (int v = 0; v < 3; ++v) Fprev[v] = Fcur[v];
    }
  }
}

// ---------------------------------------------------------------------------
// Time-step selection and SSP-RK3.
// ---------------------------------------------------------------------------
static void compute_dt(Solver1D* s, double tend) {
  const Grid1D& g = s->grid;
  const double gm = s->par.gamma;
  for (int c = 0; c < s->ncase; ++c) {
    double smax = 0.0;
    for (int i = 0; i < g.n; ++i) {
      const int j = i + g.ng;
      const double r = s->U[uidx(g, c, 0, i)];
      const double mom = s->U[uidx(g, c, 1, i)];
      const double E = s->U[uidx(g, c, 2, i)];
      const double u = mom / r;
      const double p = (gm - 1.0) * (E - 0.5 * mom * mom / r);
      const double a = std::sqrt(gm * p / r);
      smax = std::max(smax, (std::fabs(u) + a) / g.dx[j]);
    }
    double dt = (s->par.dt_fixed > 0.0) ? s->par.dt_fixed : s->par.cfl / smax;
    const double remain = tend - s->t[c];
    if (remain <= 0.0)
      dt = 0.0;
    else if (dt > remain)
      dt = remain;
    s->dt[c] = dt;
  }
}

static bool has_bad_state(const Solver1D* s, const double* U) {
  const Grid1D& g = s->grid;
  const double gm = s->par.gamma;
  for (int c = 0; c < s->ncase; ++c) {
    for (int i = 0; i < g.n; ++i) {
      const size_t b = (static_cast<size_t>(c) * NVAR1D) * g.m + (i + g.ng);
      const double r = U[b];
      const double mom = U[b + g.m];
      const double E = U[b + 2 * g.m];
      const double p = (gm - 1.0) * (E - 0.5 * mom * mom / r);
      if (!(r > 0.0) || !(p > 0.0) || !std::isfinite(r) || !std::isfinite(p))
        return true;
    }
  }
  return false;
}

// out = a*A + b*(B + dt*R), using the time step of each case.
static void rk_combine(const Solver1D* s, double a, const double* A, double b,
                       const double* B, const double* R, double* out) {
  const Grid1D& g = s->grid;
  for (int c = 0; c < s->ncase; ++c) {
    const double dt = s->dt[c];
    for (int v = 0; v < NVAR1D; ++v) {
      const size_t base = (static_cast<size_t>(c) * NVAR1D + v) * g.m;
      for (int i = 0; i < g.n; ++i) {
        const size_t k = base + (i + g.ng);
        out[k] = a * A[k] + b * (B[k] + dt * R[k]);
      }
    }
  }
}

int solver1d_run(Solver1D* s, double tend, long maxsteps) {
  double* U = s->U.data();
  double* U1 = s->U1.data();
  double* U2 = s->U2.data();
  double* R = s->R.data();
  double* W = s->W.data();
  double* S = s->S.data();

  for (long step = 0; step < maxsteps; ++step) {
    bool done = true;
    for (int c = 0; c < s->ncase; ++c)
      if (s->t[c] < tend) done = false;
    if (done) return 0;

    solver1d_apply_bc(s, U);
    compute_dt(s, tend);

    // stage 1
    solver1d_residual(s, U, R, W, S);
    rk_combine(s, 0.0, U, 1.0, U, R, U1);
    solver1d_apply_bc(s, U1);
    // stage 2
    solver1d_residual(s, U1, R, W, S);
    rk_combine(s, 0.75, U, 0.25, U1, R, U2);
    solver1d_apply_bc(s, U2);
    // stage 3
    solver1d_residual(s, U2, R, W, S);
    rk_combine(s, 1.0 / 3.0, U, 2.0 / 3.0, U2, R, U1);

    // Copy the result back to U, updating only interior cells.
    const Grid1D& g = s->grid;
    for (int c = 0; c < s->ncase; ++c) {
      if (s->dt[c] <= 0.0) continue;  // This case has reached tend; skip its update.
      for (int v = 0; v < NVAR1D; ++v) {
        const size_t base = (static_cast<size_t>(c) * NVAR1D + v) * g.m + g.ng;
        for (int i = 0; i < g.n; ++i) U[base + i] = U1[base + i];
      }
      s->t[c] += s->dt[c];
    }
    s->nsteps++;

    if (has_bad_state(s, U)) {
      std::fprintf(stderr, "solver1d_run: non-physical state at step %ld\n",
                   s->nsteps);
      return 2;
    }
  }
  return 1;  // maxsteps exceeded.
}

}  // namespace cfd
