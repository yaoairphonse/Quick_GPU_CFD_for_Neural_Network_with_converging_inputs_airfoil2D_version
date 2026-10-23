#include "euler2d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "flux.h"
#include "limiter.h"

namespace cfd {

Params2D params2d_default() {
  Params2D p;
  p.gamma = 1.4;
  p.cfl = 0.4;
  p.flux = FLUX_HLLC;
  p.limiter = LIM_VENKAT;
  p.venkat_k = 5.0;
  p.ausm_mco = 1.0;  // Reference Mach number: use 1 for compressible cases; see flux.h.
  for (int k = 0; k < 4; ++k) p.bc[k] = BC_TRANSMISSIVE;
  p.dt_fixed = 0.0;
  p.time_integrator = TIME_SSP_RK3;
  p.pseudo_cfl = 0.4;
  p.implicit_maxiter = 2000;
  p.implicit_rtol = 1e-8;
  p.implicit_atol = 1e-11;
  p.sx.type = STRETCH_UNIFORM;
  p.sx.param = 0.0;
  p.sy.type = STRETCH_UNIFORM;
  p.sy.param = 0.0;
  return p;
}

// ---------------------------------------------------------------------------
// Grid construction.
// ---------------------------------------------------------------------------
void grid2d_stretch_faces(int n, double a, double b, Stretch s, double* xf) {
  const double L = b - a;
  for (int i = 0; i <= n; ++i) {
    const double xi = static_cast<double>(i) / n;
    double f = xi;
    switch (s.type) {
      case STRETCH_GEOMETRIC:
        if (std::fabs(s.param - 1.0) > 1e-12)
          f = (std::pow(s.param, i) - 1.0) / (std::pow(s.param, n) - 1.0);
        break;
      case STRETCH_TANH_ENDS:
        if (s.param > 1e-12)
          f = 0.5 * (1.0 + std::tanh(s.param * (2.0 * xi - 1.0)) / std::tanh(s.param));
        break;
      case STRETCH_SINH_CENTER:
        if (s.param > 1e-12)
          f = 0.5 * (1.0 + std::sinh(s.param * (2.0 * xi - 1.0)) / std::sinh(s.param));
        break;
      default:
        break;
    }
    xf[i] = a + L * f;
  }
  xf[0] = a;
  xf[n] = b;
}

// Build cell centers, widths, and ghost geometry from face coordinates; translate periodic ghosts and mirror the others.
static void build_axis(int n, int ng, double a, double b, int bc_lo, int bc_hi,
                       bool uniform, Array& xf, Array& xc, Array& dx) {
  const int m = n + 2 * ng;
  xc.allocate(m);
  dx.allocate(m);
  if (uniform) {
    // On a uniform axis, each width equals h exactly; subtracting xf values would introduce ~1e-16 roundoff.
    // This gives bitwise-identical geometry to the uniform 1D solver and exact directional symmetry.
    const double h = (b - a) / n;
    // Use the same linear formula for ghost centers as in 1D to avoid one-ulp differences in Venkatakrishnan hl/hr.
    for (int i = -ng; i < n + ng; ++i) {
      xc[i + ng] = a + (i + 0.5) * h;
      dx[i + ng] = h;
    }
  } else {
    for (int i = 0; i < n; ++i) {
      xc[i + ng] = 0.5 * (xf[i] + xf[i + 1]);
      dx[i + ng] = xf[i + 1] - xf[i];
    }
  }
  const bool per_lo = (bc_lo == BC_PERIODIC), per_hi = (bc_hi == BC_PERIODIC);
  if (per_lo != per_hi) {
    std::fprintf(stderr, "periodic boundary must be set on both sides of an axis\n");
    std::abort();
  }
  const double L = b - a;
  for (int k = 1; k <= ng && !uniform; ++k) {
    if (per_lo) {
      xc[ng - k] = xc[ng + n - k] - L;
      dx[ng - k] = dx[ng + n - k];
      xc[ng + n - 1 + k] = xc[ng + k - 1] + L;
      dx[ng + n - 1 + k] = dx[ng + k - 1];
    } else {
      xc[ng - k] = 2.0 * a - xc[ng + k - 1];
      dx[ng - k] = dx[ng + k - 1];
      xc[ng + n - 1 + k] = 2.0 * b - xc[ng + n - k];
      dx[ng + n - 1 + k] = dx[ng + n - k];
    }
  }
}

void solver2d_init(Solver2D* s, int nx, int ny, double xmin, double xmax,
                   double ymin, double ymax, int ncase, const Params2D& par) {
  s->par = par;
  s->ncase = ncase;
  s->nsteps = 0;
  s->nreject = 0;
  s->implicit = {};
  s->ib = nullptr;
  s->cut = nullptr;
  Grid2D& g = s->grid;
  g.nx = nx;
  g.ny = ny;
  g.ng = 2;
  g.mx = nx + 2 * g.ng;
  g.my = ny + 2 * g.ng;
  g.xmin = xmin;
  g.xmax = xmax;
  g.ymin = ymin;
  g.ymax = ymax;
  g.xf.allocate(nx + 1);
  g.yf.allocate(ny + 1);
  grid2d_stretch_faces(nx, xmin, xmax, par.sx, g.xf.data());
  grid2d_stretch_faces(ny, ymin, ymax, par.sy, g.yf.data());
  build_axis(nx, g.ng, xmin, xmax, par.bc[0], par.bc[1], par.sx.type == STRETCH_UNIFORM, g.xf, g.xc, g.dx);
  build_axis(ny, g.ng, ymin, ymax, par.bc[2], par.bc[3], par.sy.type == STRETCH_UNIFORM, g.yf, g.yc, g.dy);

  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  const size_t sz = static_cast<size_t>(ncase) * NVAR2D * cells;
  s->U.allocate(sz);
  s->U1.allocate(sz);
  s->U2.allocate(sz);
  s->R.allocate(sz);
  s->W.allocate(sz);
  s->SX.allocate(sz);
  s->SY.allocate(sz);
  s->FX.allocate(static_cast<size_t>(ncase) * NVAR2D * (nx + 1) * ny);
  s->FY.allocate(static_cast<size_t>(ncase) * NVAR2D * nx * (ny + 1));
  s->t.allocate(ncase);
  s->dt.allocate(ncase);
  s->mco.allocate(ncase);
  for (int c = 0; c < ncase; ++c) s->mco[c] = par.ausm_mco;
}

void solver2d_set_ib(Solver2D* s, const IB2D* ib) { s->ib = ib; }
void solver2d_set_cut(Solver2D* s, const Cut2D* cut) { s->cut = cut; }

void solver2d_set_mco(Solver2D* s, int c, double mco) { s->mco[c] = mco; }

void solver2d_set_primitive(Solver2D* s, int c, int i, int j, double rho,
                            double u, double v, double p) {
  const Grid2D& g = s->grid;
  s->U[u2idx(g, c, 0, i, j)] = rho;
  s->U[u2idx(g, c, 1, i, j)] = rho * u;
  s->U[u2idx(g, c, 2, i, j)] = rho * v;
  s->U[u2idx(g, c, 3, i, j)] =
      p / (s->par.gamma - 1.0) + 0.5 * rho * u * u + 0.5 * rho * v * v;
}

void solver2d_get_primitive(const Solver2D* s, int c, int i, int j, double* rho,
                            double* u, double* v, double* p) {
  const Grid2D& g = s->grid;
  const double r = s->U[u2idx(g, c, 0, i, j)];
  const double mu = s->U[u2idx(g, c, 1, i, j)];
  const double mv = s->U[u2idx(g, c, 2, i, j)];
  const double E = s->U[u2idx(g, c, 3, i, j)];
  *rho = r;
  *u = mu / r;
  *v = mv / r;
  *p = (s->par.gamma - 1.0) * (E - 0.5 * mu * mu / r - 0.5 * mv * mv / r);
}

double solver2d_integral(const Solver2D* s, int c, int var) {
  const Grid2D& g = s->grid;
  double sum = 0.0;
  for (int j = 0; j < g.ny; ++j)
    for (int i = 0; i < g.nx; ++i)
      sum += s->U[u2idx(g, c, var, i, j)] *
             (s->cut != nullptr ? s->cut->vol[static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng)]
                                : g.dx[i + g.ng] * g.dy[j + g.ng]);
  return sum;
}

// ---------------------------------------------------------------------------
// Immersed-boundary ghost values are reconstructed from image-point flow states.
// The image point mirrors the ghost across the wall; interpolate primitives using the fluid-cell stencil.
// Copy density/pressure (zero normal gradients) and reverse normal velocity for a slip wall.
// ---------------------------------------------------------------------------
static void apply_ib(const Solver2D* s, double* U) {
  const IB2D* ib = s->ib;
  if (ib == nullptr) return;
  const Grid2D& g = s->grid;
  const double gm = s->par.gamma;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  for (int c = 0; c < s->ncase; ++c) {
    double* Uc = U + static_cast<size_t>(c) * NVAR2D * cells;
    for (int k = 0; k < ib->nghost; ++k) {
      double r = 0.0, u = 0.0, v = 0.0, p = 0.0;
      for (int m = 0; m < IB_NS; ++m) {
        const double w = ib->sw[IB_NS * k + m];
        if (w == 0.0) continue;
        const size_t q = static_cast<size_t>(ib->sidx[IB_NS * k + m]);
        const double rq = Uc[q], mu = Uc[cells + q], mv = Uc[2 * cells + q], E = Uc[3 * cells + q];
        const double uq = mu / rq, vq = mv / rq;
        r += w * rq;
        u += w * uq;
        v += w * vq;
        p += w * ((gm - 1.0) * (E - 0.5 * rq * uq * uq - 0.5 * rq * vq * vq));
      }
      double pg = p, rg = r;
      if (ib->use_p_extrap) {
        double p2 = 0.0;
        for (int m = 0; m < IB_NS; ++m) {
          const double w = ib->sw2[IB_NS * k + m];
          if (w == 0.0) continue;
          const size_t q2 = static_cast<size_t>(ib->sidx2[IB_NS * k + m]);
          const double rq = Uc[q2], mu = Uc[cells + q2], mv = Uc[2 * cells + q2], E = Uc[3 * cells + q2];
          p2 += w * ((gm - 1.0) * (E - 0.5 * mu * mu / rq - 0.5 * mv * mv / rq));
        }
        const double pe = p - ib->d2[k] * (p2 - p) / ib->dprobe;
        if (pe > 0.0) {
          pg = pe;
          rg = r * std::pow(pe / p, 1.0 / gm);
        } else {
          ++ib->nguard;
        }
      }
      const double nx = ib->nrm[2 * k], ny = ib->nrm[2 * k + 1];
      const double un = u * nx + v * ny;
      const double ug = u - 2.0 * un * nx, vg = v - 2.0 * un * ny;
      const size_t q = static_cast<size_t>(ib->gidx[k]);
      Uc[q] = rg;
      Uc[cells + q] = rg * ug;
      Uc[2 * cells + q] = rg * vg;
      Uc[3 * cells + q] = pg / (gm - 1.0) + 0.5 * rg * ug * ug + 0.5 * rg * vg * vg;
    }
  }
}

// ---------------------------------------------------------------------------
// Apply boundary conditions in x on interior rows, then in y on rows including x ghosts,
// so that all four corner ghost regions are defined.
// ---------------------------------------------------------------------------
void solver2d_apply_bc(const Solver2D* s, double* U) {
  const Grid2D& g = s->grid;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  for (int c = 0; c < s->ncase; ++c) {
    for (int v = 0; v < NVAR2D; ++v) {
      double* base = U + (static_cast<size_t>(c) * NVAR2D + v) * cells;
      // Use xmin/xmax conditions independently; periodic boundaries must be paired.
      for (int j = 0; j < g.ny; ++j) {
        double* q = base + static_cast<size_t>(j + g.ng) * g.mx + g.ng;
        const double sgn = (v == 1) ? -1.0 : 1.0;
        // Handle xmin/xmax separately, allowing reflection on one side and transmission on the other.
        for (int k = 1; k <= g.ng; ++k) {
          // Left side.
          if (s->par.bc[0] == BC_PERIODIC)
            q[-k] = q[g.nx - k];
          else if (s->par.bc[0] == BC_REFLECT)
            q[-k] = sgn * q[k - 1];
          else
            q[-k] = q[0];
          // Right side.
          if (s->par.bc[1] == BC_PERIODIC)
            q[g.nx - 1 + k] = q[k - 1];
          else if (s->par.bc[1] == BC_REFLECT)
            q[g.nx - 1 + k] = sgn * q[g.nx - k];
          else
            q[g.nx - 1 + k] = q[g.nx - 1];
        }
      }
      // Apply y conditions over the full row, including x ghosts: i = -ng..nx+ng-1.
      const double sgny = (v == 2) ? -1.0 : 1.0;
      for (int i = -g.ng; i < g.nx + g.ng; ++i) {
        double* q = base + static_cast<size_t>(g.ng) * g.mx + (i + g.ng);
        for (int k = 1; k <= g.ng; ++k) {
          if (s->par.bc[2] == BC_PERIODIC)
            q[-static_cast<long>(k) * g.mx] = q[static_cast<long>(g.ny - k) * g.mx];
          else if (s->par.bc[2] == BC_REFLECT)
            q[-static_cast<long>(k) * g.mx] = sgny * q[static_cast<long>(k - 1) * g.mx];
          else
            q[-static_cast<long>(k) * g.mx] = q[0];
          if (s->par.bc[3] == BC_PERIODIC)
            q[static_cast<long>(g.ny - 1 + k) * g.mx] = q[static_cast<long>(k - 1) * g.mx];
          else if (s->par.bc[3] == BC_REFLECT)
            q[static_cast<long>(g.ny - 1 + k) * g.mx] = sgny * q[static_cast<long>(g.ny - k) * g.mx];
          else
            q[static_cast<long>(g.ny - 1 + k) * g.mx] = q[static_cast<long>(g.ny - 1) * g.mx];
        }
      }
    }
  }
  apply_ib(s, U);
}

// ---------------------------------------------------------------------------
// Residual assembly.
// 1) Conserved to primitive; 2) directional limited slopes; 3) x/y face fluxes; 4) gather four faces per cell.
// Compute each face flux once with L on its left/bottom and R on its right/top; neighboring cells
// use the same bits with opposite signs, ensuring algebraic conservation without atomic addition.
// ---------------------------------------------------------------------------
static inline void face_flux(const Params2D& par, double mco, double rL,
                             double unL, double utL, double pL, double rR,
                             double unR, double utR, double pR, double F[4]) {
  if (par.flux == FLUX_AUSMPUP)
    flux_ausmpup_t(rL, unL, utL, pL, rR, unR, utR, pR, par.gamma, mco, F);
  else
    flux_hllc_t(rL, unL, utL, pL, rR, unR, utR, pR, par.gamma, F);
}

void solver2d_residual(const Solver2D* s, const double* U, double* R, double* W,
                       double* SX, double* SY, double* FX, double* FY) {
  if (s->cut != nullptr) {
    cut2d_residual(s, U, R, W, SX, SY, FX, FY);
    return;
  }
  const Grid2D& g = s->grid;
  const Params2D& par = s->par;
  const double gm = par.gamma;
  const int nx = g.nx, ny = g.ny, ng = g.ng, mx = g.mx, my = g.my;
  const size_t cells = static_cast<size_t>(mx) * my;
  const size_t nfx = static_cast<size_t>(nx + 1) * ny;
  const size_t nfy = static_cast<size_t>(nx) * (ny + 1);

  for (int c = 0; c < s->ncase; ++c) {
    const size_t base = static_cast<size_t>(c) * NVAR2D * cells;
    const double* Uc = U + base;
    double* Rc = R + base;
    double* Wc = W + base;
    double* SXc = SX + base;
    double* SYc = SY + base;
    double* FXc = FX + static_cast<size_t>(c) * NVAR2D * nfx;
    double* FYc = FY + static_cast<size_t>(c) * NVAR2D * nfy;
    const double mco = s->mco[c];

    // 1) Primitive variables including ghosts: W = (rho,u,v,p).
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (size_t k = 0; k < cells; ++k) {
      const double r = Uc[k];
      const double mu = Uc[cells + k];
      const double mv = Uc[2 * cells + k];
      const double E = Uc[3 * cells + k];
      Wc[k] = r;
      Wc[cells + k] = mu / r;
      Wc[2 * cells + k] = mv / r;
      Wc[3 * cells + k] = (gm - 1.0) * (E - 0.5 * mu * mu / r - 0.5 * mv * mv / r);
    }

    // 2) Slopes: x for each interior row j with i=-1..nx; y for each interior column i with j=-1..ny.
    // Flatten (variable,row/column) into independent tasks so one case can use all CPU cores.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int task = 0; task < NVAR2D * ny; ++task) {
      const int v = task / ny, j = task % ny;
      const double* q = Wc + static_cast<size_t>(v) * cells;
      double* sx = SXc + static_cast<size_t>(v) * cells;
        const size_t row = static_cast<size_t>(j + ng) * mx;
        for (int i = -1; i <= nx; ++i) {
          const int a = i + ng;
          const double hl = g.xc[a] - g.xc[a - 1];
          const double hr = g.xc[a + 1] - g.xc[a];
          sx[row + a] = limited_slope(q[row + a - 1], q[row + a], q[row + a + 1], hl,
                                      hr, g.dx[a], par.limiter, par.venkat_k);
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int task = 0; task < NVAR2D * nx; ++task) {
      const int v = task / nx, i = task % nx;
      const double* q = Wc + static_cast<size_t>(v) * cells;
      double* sy = SYc + static_cast<size_t>(v) * cells;
        for (int j = -1; j <= ny; ++j) {
          const int b = j + ng;
          const size_t id = static_cast<size_t>(b) * mx + (i + ng);
          const double hl = g.yc[b] - g.yc[b - 1];
          const double hr = g.yc[b + 1] - g.yc[b];
          sy[id] = limited_slope(q[id - mx], q[id], q[id + mx], hl, hr, g.dy[b],
                                 par.limiter, par.venkat_k);
        }
    }

    // Set slopes to zero in IB ghosts and solid cells, giving first-order ghost states on wall-side faces.
    if (s->ib != nullptr) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
      for (size_t q = 0; q < NVAR2D * cells; ++q) {
        const size_t k = q % cells;
        if (s->ib->fluid[k] == 0.0) {
          SXc[q] = 0.0;
          SYc[q] = 0.0;
        }
      }
    }

    const double* Wr = Wc;
    const double* Wu = Wc + cells;
    const double* Wv = Wc + 2 * cells;
    const double* Wp = Wc + 3 * cells;
    const double* Sr = SXc;
    const double* Su = SXc + cells;
    const double* Sv = SXc + 2 * cells;
    const double* Sp = SXc + 3 * cells;

    // 3a) x face i=0..nx lies between cells i-1 and i; normal velocity u, tangential velocity v.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int j = 0; j < ny; ++j) {
      const size_t row = static_cast<size_t>(j + ng) * mx;
      for (int i = 0; i <= nx; ++i) {
        const int aL = i - 1 + ng, aR = i + ng;
        const double hL = 0.5 * g.dx[aL], hR = 0.5 * g.dx[aR];
        const size_t L = row + aL, Rr = row + aR;
        double F[4];
        face_flux(par, mco, Wr[L] + hL * Sr[L], Wu[L] + hL * Su[L],
                  Wv[L] + hL * Sv[L], Wp[L] + hL * Sp[L], Wr[Rr] - hR * Sr[Rr],
                  Wu[Rr] - hR * Su[Rr], Wv[Rr] - hR * Sv[Rr],
                  Wp[Rr] - hR * Sp[Rr], F);
        const size_t f = static_cast<size_t>(j) * (nx + 1) + i;
        FXc[0 * nfx + f] = F[0];  // Mass.
        FXc[1 * nfx + f] = F[1];  // x momentum = normal momentum.
        FXc[2 * nfx + f] = F[2];  // y momentum = tangential momentum.
        FXc[3 * nfx + f] = F[3];  // Energy.
      }
    }

    // 3b) y face j=0..ny lies between cells j-1 and j; normal velocity v, tangential velocity u.
    const double* Tr = SYc;
    const double* Tu = SYc + cells;
    const double* Tv = SYc + 2 * cells;
    const double* Tp = SYc + 3 * cells;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int j = 0; j <= ny; ++j) {
      const int bL = j - 1 + ng, bR = j + ng;
      const double hL = 0.5 * g.dy[bL], hR = 0.5 * g.dy[bR];
      for (int i = 0; i < nx; ++i) {
        const size_t L = static_cast<size_t>(bL) * mx + (i + ng);
        const size_t Rr = static_cast<size_t>(bR) * mx + (i + ng);
        double F[4];
        face_flux(par, mco, Wr[L] + hL * Tr[L], Wv[L] + hL * Tv[L],
                  Wu[L] + hL * Tu[L], Wp[L] + hL * Tp[L], Wr[Rr] - hR * Tr[Rr],
                  Wv[Rr] - hR * Tv[Rr], Wu[Rr] - hR * Tu[Rr],
                  Wp[Rr] - hR * Tp[Rr], F);
        const size_t f = static_cast<size_t>(j) * nx + i;
        FYc[0 * nfy + f] = F[0];  // Mass.
        FYc[1 * nfy + f] = F[2];  // x momentum = tangential momentum.
        FYc[2 * nfy + f] = F[1];  // y momentum = normal momentum.
        FYc[3 * nfy + f] = F[3];  // Energy.
      }
    }

    // 4) Each cell gathers its four faces.
    const int residual_tasks = NVAR2D * ny;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int task = 0; task < residual_tasks; ++task) {
      const int v = task / ny, j = task % ny;
      const double* fx = FXc + static_cast<size_t>(v) * nfx;
      const double* fy = FYc + static_cast<size_t>(v) * nfy;
      double* r = Rc + static_cast<size_t>(v) * cells;
      for (int i = 0; i < nx; ++i) {
          const double idx = 1.0 / g.dx[i + ng];
          const double idy = 1.0 / g.dy[j + ng];
          const size_t fe = static_cast<size_t>(j) * (nx + 1) + i;
          const size_t fn = static_cast<size_t>(j) * nx + i;
          const size_t kc = static_cast<size_t>(j + ng) * mx + (i + ng);
          r[kc] = -(fx[fe + 1] - fx[fe]) * idx - (fy[fn + nx] - fy[fn]) * idy;
          if (s->ib != nullptr && s->ib->fluid[kc] == 0.0) r[kc] = 0.0;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Time-step selection and SSP-RK3.
// ---------------------------------------------------------------------------
static void compute_dt(Solver2D* s, double tend) {
  const Grid2D& g = s->grid;
  const double gm = s->par.gamma;
  for (int c = 0; c < s->ncase; ++c) {
    double smax = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(max:smax) schedule(static) if(g.nx * g.ny >= 4096)
#endif
    for (int ij = 0; ij < g.nx * g.ny; ++ij) {
        const int i = ij % g.nx, j = ij / g.nx;
        if (s->ib != nullptr && s->ib->fluid[static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng)] == 0.0) continue;
        if (s->cut != nullptr && s->cut->vol[static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng)] <= 0.0) continue;
        const double r = s->U[u2idx(g, c, 0, i, j)];
        const double mu = s->U[u2idx(g, c, 1, i, j)];
        const double mv = s->U[u2idx(g, c, 2, i, j)];
        const double E = s->U[u2idx(g, c, 3, i, j)];
        const double u = mu / r, v = mv / r;
        const double p = (gm - 1.0) * (E - 0.5 * mu * mu / r - 0.5 * mv * mv / r);
        const double a = std::sqrt(gm * p / r);
        // 2D explicit stability rate: sum directional wave speeds divided by cell widths.
        smax = std::max(smax, (std::fabs(u) + a) / g.dx[i + g.ng] +
                                  (std::fabs(v) + a) / g.dy[j + g.ng]);
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

static bool has_bad_state(const Solver2D* s, const double* U) {
  const Grid2D& g = s->grid;
  const double gm = s->par.gamma;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  for (int c = 0; c < s->ncase; ++c)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        const size_t k = static_cast<size_t>(c) * NVAR2D * cells +
                         static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng);
        const size_t kk = static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng);
        if (s->ib != nullptr && s->ib->fluid[kk] == 0.0) continue;
        if (s->cut != nullptr && s->cut->vol[kk] <= 0.0) continue;
        const double r = U[k], mu = U[cells + k], mv = U[2 * cells + k],
                     E = U[3 * cells + k];
        const double p = (gm - 1.0) * (E - 0.5 * mu * mu / r - 0.5 * mv * mv / r);
        if (!(r > 0.0) || !(p > 0.0) || !std::isfinite(r) || !std::isfinite(p))
          return true;
      }
  return false;
}

static void report_bad_state(const Solver2D* s, const double* U, const char* stage) {
  const Grid2D& g = s->grid;
  const double gm = s->par.gamma;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  for (int c = 0; c < s->ncase; ++c)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        const size_t kk = static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng);
        if (s->ib != nullptr && s->ib->fluid[kk] == 0.0) continue;
        if (s->cut != nullptr && s->cut->vol[kk] <= 0.0) continue;
        const size_t k = static_cast<size_t>(c) * NVAR2D * cells + kk;
        const double r = U[k], mu = U[cells + k], mv = U[2 * cells + k], E = U[3 * cells + k];
        const double p = (gm - 1.0) * (E - 0.5 * mu * mu / r - 0.5 * mv * mv / r);
        if (!(r > 0.0) || !(p > 0.0) || !std::isfinite(r) || !std::isfinite(p)) {
          const double r0 = s->U[k], mu0 = s->U[cells + k], mv0 = s->U[2 * cells + k];
          const double E0 = s->U[3 * cells + k];
          const double p0 = (gm - 1.0) * (E0 - 0.5 * mu0 * mu0 / r0 - 0.5 * mv0 * mv0 / r0);
          const double vol = s->cut != nullptr ? s->cut->vol[kk] : g.dx[i + g.ng] * g.dy[j + g.ng];
          const double flag = s->cut != nullptr ? s->cut->flag[kk] : 0.0;
          std::fprintf(stderr,
                       "bad state: stage=%s case=%d i=%d j=%d x=%.17e y=%.17e vol=%.17e flag=%.0f rho=%.17e p=%.17e accepted_rho=%.17e accepted_p=%.17e dt=%.17e\n",
                       stage, c, i, j, g.xc[i + g.ng], g.yc[j + g.ng], vol, flag, r, p,
                       r0, p0, s->dt[c]);
          return;
        }
      }
}

// out = a*A + b*(B + dt*R), with a separate dt for each case; update interior cells only.
static void rk_combine(const Solver2D* s, double a, const double* A, double b,
                       const double* B, const double* R, double* out) {
  const Grid2D& g = s->grid;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  for (int c = 0; c < s->ncase; ++c) {
    const double dt = s->dt[c];
    const int tasks = NVAR2D * g.ny;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int task = 0; task < tasks; ++task) {
      const int v = task / g.ny, j = task % g.ny;
      const size_t base = (static_cast<size_t>(c) * NVAR2D + v) * cells;
      const size_t row = base + static_cast<size_t>(j + g.ng) * g.mx + g.ng;
      for (int i = 0; i < g.nx; ++i) {
          const size_t k = row + i;
          out[k] = a * A[k] + b * (B[k] + dt * R[k]);
      }
    }
  }
}

int solver2d_run(Solver2D* s, double tend, long maxsteps) {
  if (s->par.time_integrator == TIME_BACKWARD_EULER)
    return solver2d_run_implicit(s, tend, maxsteps);
  if (s->par.time_integrator != TIME_SSP_RK3) return 3;
  double* U = s->U.data();
  double* U1 = s->U1.data();
  double* U2 = s->U2.data();
  double* R = s->R.data();
  const Grid2D& g = s->grid;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;

  for (long step = 0; step < maxsteps; ++step) {
    bool done = true;
    for (int c = 0; c < s->ncase; ++c)
      if (s->t[c] < tend) done = false;
    if (done) return 0;

    solver2d_apply_bc(s, U);
    compute_dt(s, tend);

    int retries = 0;
    for (;;) {
      const char* bad_stage = "none";
      const double* bad_U = U1;
      solver2d_residual(s, U, R, s->W.data(), s->SX.data(), s->SY.data(),
                        s->FX.data(), s->FY.data());
      rk_combine(s, 0.0, U, 1.0, U, R, U1);
      bool bad = has_bad_state(s, U1);
      if (bad) bad_stage = "rk1-pre-srd";
      if (!bad && s->cut != nullptr) {
        cut2d_srd(s, U1);
        bad = has_bad_state(s, U1);
        if (bad) bad_stage = "rk1-post-srd";
      }

      if (!bad) {
        solver2d_apply_bc(s, U1);
        solver2d_residual(s, U1, R, s->W.data(), s->SX.data(), s->SY.data(),
                          s->FX.data(), s->FY.data());
        rk_combine(s, 0.75, U, 0.25, U1, R, U2);
        bad = has_bad_state(s, U2);
        bad_U = U2;
        if (bad) bad_stage = "rk2-pre-srd";
        if (!bad && s->cut != nullptr) {
          cut2d_srd(s, U2);
          bad = has_bad_state(s, U2);
          if (bad) bad_stage = "rk2-post-srd";
        }
      }

      if (!bad) {
        solver2d_apply_bc(s, U2);
        solver2d_residual(s, U2, R, s->W.data(), s->SX.data(), s->SY.data(),
                          s->FX.data(), s->FY.data());
        rk_combine(s, 1.0 / 3.0, U, 2.0 / 3.0, U2, R, U1);
        bad = has_bad_state(s, U1);
        bad_U = U1;
        if (bad) bad_stage = "rk3-pre-srd";
        if (!bad && s->cut != nullptr) {
          cut2d_srd(s, U1);
          bad = has_bad_state(s, U1);
          if (bad) bad_stage = "rk3-post-srd";
        }
      }

      if (!bad) break;

      ++s->nreject;
      ++retries;
      bool usable_dt = retries <= 12;
      for (int c = 0; c < s->ncase; ++c) {
        if (s->dt[c] <= 0.0) continue;
        s->dt[c] *= 0.5;
        if (!(s->dt[c] > 0.0) || !std::isfinite(s->dt[c])) usable_dt = false;
      }
      if (!usable_dt) {
        std::fprintf(stderr,
                     "solver2d_run: non-physical RK candidate after %d step retries at accepted step %ld (%s)\n",
                     retries, s->nsteps, bad_stage);
        report_bad_state(s, bad_U, bad_stage);
        return 2;
      }
    }

    for (int c = 0; c < s->ncase; ++c) {
      if (s->dt[c] <= 0.0) continue;  // This case has reached tend.
      const int tasks = NVAR2D * g.ny;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
      for (int task = 0; task < tasks; ++task) {
        const int v = task / g.ny, j = task % g.ny;
        const size_t base = (static_cast<size_t>(c) * NVAR2D + v) * cells;
        const size_t row = base + static_cast<size_t>(j + g.ng) * g.mx + g.ng;
        for (int i = 0; i < g.nx; ++i) U[row + i] = U1[row + i];
      }
      s->t[c] += s->dt[c];
    }
    s->nsteps++;

    if (has_bad_state(s, U)) return 2;  // defensive: accepted candidates were checked above
  }
  return 1;
}

int solver2d_run_backend(Solver2D* s, double tend, long maxsteps, int backend) {
  if (backend == BACKEND_CUDA) return solver2d_run_cuda(s, tend, maxsteps);
  return solver2d_run(s, tend, maxsteps);
}

#ifndef CFD_HAVE_CUDA
int solver2d_run_cuda(Solver2D*, double, long) {
  std::fprintf(stderr, "CUDA backend was not enabled in this build (configure with -DCFD_CUDA=ON)\n");
  return 3;
}

bool solver2d_cuda_available(char* reason, size_t reason_size) {
  if (reason != nullptr && reason_size > 0)
    std::snprintf(reason, reason_size, "CUDA backend was not enabled in this build");
  return false;
}
#endif

}  // namespace cfd
