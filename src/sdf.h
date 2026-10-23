// sdf.h -- 2D signed-distance functions and immersed-boundary cell classification.
//
// Convention: SDF<0 in solid, >0 in fluid; |SDF| is the Euclidean distance to the polygon wall.
// The polygon is a closed polyline, connecting the last vertex to the first; either orientation is accepted.
// Small header-only functions; arrays are caller-owned and no new/delete is used.
#ifndef CFD_SDF_H
#define CFD_SDF_H

#include <cmath>
#include <cstddef>

namespace cfd {

// Squared minimum distance from (x,y) to segment (ax,ay)-(bx,by).
inline double seg_dist2(double x, double y, double ax, double ay, double bx, double by) {
  const double dx = bx - ax, dy = by - ay;
  const double l2 = dx * dx + dy * dy;
  double t = 0.0;
  if (l2 > 0.0) {
    t = ((x - ax) * dx + (y - ay) * dy) / l2;
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
  }
  const double px = ax + t * dx - x, py = ay + t * dy - y;
  return px * px + py * py;
}

// As sdf_polygon, also returning the nearest wall point (*cx,*cy) and segment index *seg.
inline double sdf_polygon_closest(const double* px, const double* py, int n, double x,
                                  double y, double* cx, double* cy, int* seg);

// Ray casting with odd/even intersections: test whether a point lies inside the closed polygon.
inline bool point_in_polygon(const double* px, const double* py, int n, double x, double y) {
  bool in = false;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const bool straddle = (py[i] > y) != (py[j] > y);
    if (straddle) {
      const double xc = px[j] + (y - py[j]) * (px[i] - px[j]) / (py[i] - py[j]);
      if (x < xc) in = !in;
    }
  }
  return in;
}

// Polygon signed distance, negative inside. An O(n) search is sufficient for these low-fidelity grids.
inline double sdf_polygon(const double* px, const double* py, int n, double x, double y) {
  double d2 = 1e300;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const double s = seg_dist2(x, y, px[j], py[j], px[i], py[i]);
    if (s < d2) d2 = s;
  }
  const double d = std::sqrt(d2);
  return point_in_polygon(px, py, n, x, y) ? -d : d;
}

inline double sdf_polygon_closest(const double* px, const double* py, int n, double x,
                                  double y, double* cx, double* cy, int* seg) {
  double best = 1e300;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const double dx = px[i] - px[j], dy = py[i] - py[j];
    const double l2 = dx * dx + dy * dy;
    double t = 0.0;
    if (l2 > 0.0) {
      t = ((x - px[j]) * dx + (y - py[j]) * dy) / l2;
      t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    }
    const double qx = px[j] + t * dx, qy = py[j] + t * dy;
    const double d2 = (qx - x) * (qx - x) + (qy - y) * (qy - y);
    if (d2 < best) {
      best = d2;
      *cx = qx;
      *cy = qy;
      *seg = j;  // Segment j -> i.
    }
  }
  const double d = std::sqrt(best);
  return point_in_polygon(px, py, n, x, y) ? -d : d;
}

// Absolute polygon area from the shoelace formula.
inline double polygon_area(const double* px, const double* py, int n) {
  double a = 0.0;
  for (int i = 0, j = n - 1; i < n; j = i++) a += px[j] * py[i] - px[i] * py[j];
  return 0.5 * std::fabs(a);
}

// NACA symmetric 00tt half-thickness for unit chord and closed trailing-edge coefficient -0.1036; x in [0,1].
inline double naca_halfthickness(double x, double t) {
  const double s = std::sqrt(x > 0.0 ? x : 0.0);
  return 5.0 * t * (0.2969 * s - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x -
                    0.1036 * x * x * x * x);
}

// Generate a closed symmetric NACA polygon from trailing edge along upper surface to leading edge and back along lower surface.
// Cosine spacing clusters near the leading edge. npan panels per surface give 2*npan+1 vertices:
// npan upper vertices and npan+1 lower vertices including both edges. Caller supplies px,py of length 2*npan+1; return vertex count.
// Rotate geometry by aoa_deg about (0.25,0) only for wall-geometry tests; the pipeline rotates freestream instead.
inline int naca_polygon(double t, int npan, double aoa_deg, double* px, double* py) {
  const double pi = 3.14159265358979323846;
  int k = 0;
  for (int i = 0; i < npan; ++i) {  // Upper surface: x decreases from 1 toward 0, excluding the leading edge.
    const double th = pi * (1.0 - static_cast<double>(i) / npan);
    const double x = 0.5 * (1.0 - std::cos(th));
    px[k] = x; py[k] = naca_halfthickness(x, t); ++k;
  }
  for (int i = 0; i <= npan; ++i) {  // Lower surface: x increases from 0 at the leading edge to 1 at the trailing edge.
    const double th = pi * static_cast<double>(i) / npan;
    const double x = 0.5 * (1.0 - std::cos(th));
    px[k] = x; py[k] = -naca_halfthickness(x, t); ++k;
  }
  if (aoa_deg != 0.0) {
    const double a = -aoa_deg * pi / 180.0, c = std::cos(a), s = std::sin(a);
    for (int i = 0; i < k; ++i) {
      const double x = px[i] - 0.25, y = py[i];
      px[i] = 0.25 + c * x - s * y;
      py[i] = s * x + c * y;
    }
  }
  return k;
}

// Inscribed m-vertex circle polygon, for comparisons with the analytic SDF r-R.
inline void circle_polygon(double cx, double cy, double R, int m, double* px, double* py,
                           double phase = 0.0) {
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < m; ++i) {
    px[i] = cx + R * std::cos(2.0 * pi * i / m + phase);
    py[i] = cy + R * std::sin(2.0 * pi * i / m + phase);
  }
}

// Cell types.
enum CellType { CELL_FLUID = 0, CELL_GHOST = 1, CELL_SOLID = 2 };

// Classify from the cell-center SDF; phi/type have length nx*ny with i varying fastest:
//   phi > 0                     -> fluid.
//   phi <= 0 with a fluid face neighbor -> ghost, one layer inside the wall.
//   Other phi <= 0              -> solid interior, excluded from the calculation.
// Return the ghost count. Neighbors outside the domain are absent.
inline int ib_classify(const double* phi, int nx, int ny, unsigned char* type) {
  int nghost = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const int id = j * nx + i;
      if (phi[id] > 0.0) { type[id] = CELL_FLUID; continue; }
      bool touch = false;
      if (i > 0 && phi[id - 1] > 0.0) touch = true;
      if (i < nx - 1 && phi[id + 1] > 0.0) touch = true;
      if (j > 0 && phi[id - nx] > 0.0) touch = true;
      if (j < ny - 1 && phi[id + nx] > 0.0) touch = true;
      type[id] = touch ? CELL_GHOST : CELL_SOLID;
      nghost += touch ? 1 : 0;
    }
  return nghost;
}

}  // namespace cfd

#endif  // CFD_SDF_H
