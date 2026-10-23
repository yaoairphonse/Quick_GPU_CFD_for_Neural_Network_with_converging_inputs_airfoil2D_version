// ib2d.cpp -- Immersed-boundary geometry, image-point ghosts, and wall forces; see IB2D in euler2d.h.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "euler2d.h"
#include "sdf.h"

namespace cfd {

// Find the interior-center interval [xc[i0],xc[i0+1]] containing x; clamp i0 to 0..n-2.
static int find_cell(const Array& xc, int ng, int n, double x) {
  int lo = 0, hi = n - 1;
  if (x <= xc[ng]) return 0;
  if (x >= xc[ng + n - 1]) return n - 2;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (xc[ng + mid] <= x) lo = mid; else hi = mid;
  }
  return lo;
}

// Build a fluid interpolation stencil at (x,y), with at most IB_NS cells and weights summing to 1.
//   Mode 0: all four enclosing cells are fluid; use bilinear interpolation.
//   Mode 1: otherwise fit a weighted least-squares linear field using nearby 4x4 or 6x6 fluid cells.
//           Weights reproduce linear fields (second order); geometry alone determines the stencil,
//           so symmetric configurations receive symmetric weights regardless of index ordering.
//   Mode 2: too few or collinear cells; inverse-distance-squared weights reproduce constants only.
// Return the mode, or -1 if no fluid cell exists. Unused idx entries have zero weight.
static int stencil_at(const Grid2D& g, const double* fluid, double x, double y,
                      long idx[IB_NS], double w[IB_NS]) {
  for (int m = 0; m < IB_NS; ++m) { idx[m] = 0; w[m] = 0.0; }
  const int i0 = find_cell(g.xc, g.ng, g.nx, x), j0 = find_cell(g.yc, g.ng, g.ny, y);
  {
    const int ci[4] = {i0, i0 + 1, i0, i0 + 1}, cj[4] = {j0, j0, j0 + 1, j0 + 1};
    bool all_fluid = true;
    long id4[4];
    for (int m = 0; m < 4; ++m) {
      id4[m] = static_cast<long>(cj[m] + g.ng) * g.mx + (ci[m] + g.ng);
      if (fluid[id4[m]] == 0.0) all_fluid = false;
    }
    if (all_fluid) {
      double tx = (x - g.xc[g.ng + i0]) / (g.xc[g.ng + i0 + 1] - g.xc[g.ng + i0]);
      double ty = (y - g.yc[g.ng + j0]) / (g.yc[g.ng + j0 + 1] - g.yc[g.ng + j0]);
      tx = tx < 0.0 ? 0.0 : (tx > 1.0 ? 1.0 : tx);
      ty = ty < 0.0 ? 0.0 : (ty > 1.0 ? 1.0 : ty);
      const double bw[4] = {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};
      for (int m = 0; m < 4; ++m) { idx[m] = id4[m]; w[m] = bw[m]; }
      return 0;
    }
  }
  const double h2 = 0.25 * (g.dx[g.ng + i0] * g.dx[g.ng + i0] + g.dy[g.ng + j0] * g.dy[g.ng + j0]);
  for (int pass = 0; pass < 2; ++pass) {
    const int lo = pass == 0 ? -1 : -2, hi = pass == 0 ? 2 : 3;
    long ids[IB_NS];
    double X[IB_NS], Y[IB_NS];
    int cnt = 0;
    for (int dj = lo; dj <= hi; ++dj)
      for (int di = lo; di <= hi; ++di) {
        const int ii = i0 + di, jj = j0 + dj;
        if (ii < 0 || ii >= g.nx || jj < 0 || jj >= g.ny) continue;
        const long id = static_cast<long>(jj + g.ng) * g.mx + (ii + g.ng);
        if (fluid[id] == 0.0) continue;
        ids[cnt] = id;
        X[cnt] = g.xc[g.ng + ii] - x;
        Y[cnt] = g.yc[g.ng + jj] - y;
        ++cnt;
      }
    if (cnt < 3) continue;
    double M[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    double W[IB_NS];
    for (int q = 0; q < cnt; ++q) {
      W[q] = 1.0 / (X[q] * X[q] + Y[q] * Y[q] + h2);
      const double a[3] = {1.0, X[q], Y[q]};
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) M[r][c] += W[q] * a[r] * a[c];
    }
    const double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                       M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                       M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    const double scale = M[0][0] * M[1][1] * M[2][2];
    if (!(std::fabs(det) > 1e-9 * scale)) continue;  // Collinear stencil; expand the neighborhood.
    // Solve M z = e1 using the first-column cofactors in Cramer's rule.
    const double z0 = (M[1][1] * M[2][2] - M[1][2] * M[2][1]) / det;
    const double z1 = -(M[1][0] * M[2][2] - M[1][2] * M[2][0]) / det;
    const double z2 = (M[1][0] * M[2][1] - M[1][1] * M[2][0]) / det;
    for (int q = 0; q < cnt; ++q) {
      idx[q] = ids[q];
      w[q] = W[q] * (z0 + z1 * X[q] + z2 * Y[q]);
    }
    return 1;
  }
  int cnt = 0;
  double sum = 0.0;
  for (int dj = -2; dj <= 3; ++dj)
    for (int di = -2; di <= 3; ++di) {
      const int ii = i0 + di, jj = j0 + dj;
      if (ii < 0 || ii >= g.nx || jj < 0 || jj >= g.ny) continue;
      const long id = static_cast<long>(jj + g.ng) * g.mx + (ii + g.ng);
      if (fluid[id] == 0.0) continue;
      const double dx = g.xc[g.ng + ii] - x, dy = g.yc[g.ng + jj] - y;
      idx[cnt] = id;
      w[cnt] = 1.0 / (dx * dx + dy * dy + 1e-30);
      sum += w[cnt];
      ++cnt;
    }
  if (cnt == 0) return -1;
  for (int m = 0; m < cnt; ++m) w[m] /= sum;
  return 2;
}

// Fluid-directed wall normal: use the nearest segment and choose the direction that steps outside the polygon.
static void segment_normal_to_fluid(const double* px, const double* py, int n, int seg,
                                    double cx, double cy, double eps, double* nx, double* ny) {
  const int a = seg, b = (seg + 1) % n;
  const double tx = px[b] - px[a], ty = py[b] - py[a];
  const double L = std::sqrt(tx * tx + ty * ty);
  double mx = ty / L, my = -tx / L;
  if (point_in_polygon(px, py, n, cx + eps * mx, cy + eps * my)) {
    mx = -mx;
    my = -my;
  }
  *nx = mx;
  *ny = my;
}

int ib2d_build_polygon(const Solver2D* s, const double* px, const double* py, int n, IB2D* ib) {
  const Grid2D& g = s->grid;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  const int nx = g.nx, ny = g.ny;

  // 1) SDF and classification in interior cells.
  std::vector<double> phi(static_cast<size_t>(nx) * ny);
  std::vector<unsigned char> type(static_cast<size_t>(nx) * ny);
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
      phi[static_cast<size_t>(j) * nx + i] =
          sdf_polygon(px, py, n, g.xc[i + g.ng], g.yc[j + g.ng]);
  const int nghost = ib_classify(phi.data(), nx, ny, type.data());

  ib->nghost = nghost;
  ib->nfallback = 0;
  ib->fluid.allocate(cells);
  for (size_t k = 0; k < cells; ++k) ib->fluid[k] = 1.0;  // Domain-boundary ghosts count as fluid.
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
      if (type[static_cast<size_t>(j) * nx + i] != CELL_FLUID)
        ib->fluid[static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng)] = 0.0;

  ib->gidx.allocate(nghost);
  ib->sidx.allocate(IB_NS * static_cast<size_t>(nghost));
  ib->sw.allocate(IB_NS * static_cast<size_t>(nghost));
  ib->nrm.allocate(2 * static_cast<size_t>(nghost));
  ib->sidx2.allocate(IB_NS * static_cast<size_t>(nghost));
  ib->sw2.allocate(IB_NS * static_cast<size_t>(nghost));
  ib->d2.allocate(nghost);
  ib->use_p_extrap = 1;
  ib->nguard = 0;

  double hmin = 1e300;
  for (int i = 0; i < nx; ++i) hmin = std::fmin(hmin, g.dx[i + g.ng]);
  for (int j = 0; j < ny; ++j) hmin = std::fmin(hmin, g.dy[j + g.ng]);

  ib->dprobe = hmin;

  // 2) Ghosts: nearest wall point, normal, image point, and interpolation stencil.
  int k = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      if (type[static_cast<size_t>(j) * nx + i] != CELL_GHOST) continue;
      const double x = g.xc[i + g.ng], y = g.yc[j + g.ng];
      double cx = 0.0, cy = 0.0;
      int seg = 0;
      const double ph = sdf_polygon_closest(px, py, n, x, y, &cx, &cy, &seg);
      double d = std::fabs(ph), nxn, nyn;
      const int sb = (seg + 1) % n;
      const double stx = px[sb] - px[seg], sty = py[sb] - py[seg];
      const double tpar = ((cx - px[seg]) * stx + (cy - py[seg]) * sty) / (stx * stx + sty * sty);
      if (tpar > 1e-9 && tpar < 1.0 - 1e-9) {
        // For an interior segment projection, use its face normal; this is more accurate than (c-x)/d for small d.
        // A point exactly on a vertex-angle bisector ties the two nearest faces; their normals differ by pi/M.
        segment_normal_to_fluid(px, py, n, seg, cx, cy, 1e-7 * hmin, &nxn, &nyn);
      } else if (d > 1e-10 * hmin) {
        nxn = (cx - x) / d;  // The ghost is inside solid: the wall direction points into fluid.
        nyn = (cy - y) / d;
        if (ph > 0.0) { nxn = -nxn; nyn = -nyn; }  // Guard against an unexpected positive ghost phi.
      } else {
        segment_normal_to_fluid(px, py, n, seg, cx, cy, 1e-7 * hmin, &nxn, &nyn);
        d = 0.0;
      }
      const double ix = x + 2.0 * d * nxn, iy = y + 2.0 * d * nyn;  // Image point.
      long idx[IB_NS];
      double w[IB_NS];
      const int mode = stencil_at(g, ib->fluid.data(), ix, iy, idx, w);
      if (mode < 0) {
        std::fprintf(stderr, "ib2d_build_polygon: no fluid cell near image point of ghost (%d,%d)\n", i, j);
        return 1;
      }
      if (mode >= 1) ++ib->nfallback;
      ib->gidx[k] = static_cast<double>(static_cast<long>(j + g.ng) * g.mx + (i + g.ng));
      for (int m = 0; m < IB_NS; ++m) {
        ib->sidx[IB_NS * k + m] = static_cast<double>(idx[m]);
        ib->sw[IB_NS * k + m] = w[m];
      }
      {
        long idx2[IB_NS];
        double w2[IB_NS];
        const int mode2 = stencil_at(g, ib->fluid.data(), ix + ib->dprobe * nxn, iy + ib->dprobe * nyn, idx2, w2);
        if (mode2 < 0) {
          std::fprintf(stderr, "ib2d_build_polygon: no fluid cell near second probe of ghost (%d,%d)\n", i, j);
          return 1;
        }
        for (int m = 0; m < IB_NS; ++m) {
          ib->sidx2[IB_NS * k + m] = static_cast<double>(idx2[m]);
          ib->sw2[IB_NS * k + m] = w2[m];
        }
        ib->d2[k] = 2.0 * d;
      }
      ib->nrm[2 * k] = nxn;
      ib->nrm[2 * k + 1] = nyn;
      ++k;
    }
  return 0;
}

static double hmin_of(const Grid2D& g) {
  double h = 1e300;
  for (int i = 0; i < g.nx; ++i) h = std::fmin(h, g.dx[i + g.ng]);
  for (int j = 0; j < g.ny; ++j) h = std::fmin(h, g.dy[j + g.ng]);
  return h;
}

// Interpolate density and pressure at (x,y).
static int rho_p_at(const Solver2D* s, int c, double x, double y, double* rho, double* p) {
  const Grid2D& g = s->grid;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  const double gm = s->par.gamma;
  const double* Uc = s->U.data() + static_cast<size_t>(c) * NVAR2D * cells;
  long idx[IB_NS];
  double w[IB_NS];
  if (stencil_at(g, s->ib->fluid.data(), x, y, idx, w) < 0) return 1;
  double r = 0.0, pp = 0.0;
  for (int m = 0; m < IB_NS; ++m) {
    if (w[m] == 0.0) continue;
    const size_t q = static_cast<size_t>(idx[m]);
    const double rq = Uc[q], mu = Uc[cells + q], mv = Uc[2 * cells + q], E = Uc[3 * cells + q];
    r += w[m] * rq;
    pp += w[m] * ((gm - 1.0) * (E - 0.5 * mu * mu / rq - 0.5 * mv * mv / rq));
  }
  *rho = r;
  *p = pp;
  return 0;
}

int ib2d_wall_state(const Solver2D* s, int c, double wx, double wy, double nx, double ny,
                    double* rho, double* p) {
  const double h = hmin_of(s->grid);
  double r1, p1, r2, p2;
  if (rho_p_at(s, c, wx + 1.5 * h * nx, wy + 1.5 * h * ny, &r1, &p1)) return 1;
  if (rho_p_at(s, c, wx + 2.5 * h * nx, wy + 2.5 * h * ny, &r2, &p2)) return 1;
  *rho = r1 - 1.5 * (r2 - r1);  // Slope (2-1)/h, extrapolated back by 1.5h.
  *p = p1 - 1.5 * (p2 - p1);
  return 0;
}

void ib2d_wall_force(const Solver2D* s, int c, const double* px, const double* py, int n,
                     double* fx, double* fy) {
  const double hmin = hmin_of(s->grid);
  double Fx = 0.0, Fy = 0.0;
  for (int a = 0; a < n; ++a) {
    const int b = (a + 1) % n;
    const double tx = px[b] - px[a], ty = py[b] - py[a];
    const double L = std::sqrt(tx * tx + ty * ty);
    const int ns = static_cast<int>(std::ceil(L / (0.5 * hmin))) + 1;
    double nxn, nyn;
    segment_normal_to_fluid(px, py, n, a, 0.5 * (px[a] + px[b]), 0.5 * (py[a] + py[b]), 1e-7 * hmin, &nxn, &nyn);
    for (int q = 0; q < ns; ++q) {
      const double f = (q + 0.5) / ns;
      double rw, pw;
      if (ib2d_wall_state(s, c, px[a] + f * tx, py[a] + f * ty, nxn, nyn, &rw, &pw)) continue;
      Fx += -pw * nxn * (L / ns);
      Fy += -pw * nyn * (L / ns);
    }
  }
  *fx = Fx;
  *fy = Fy;
}

}  // namespace cfd
