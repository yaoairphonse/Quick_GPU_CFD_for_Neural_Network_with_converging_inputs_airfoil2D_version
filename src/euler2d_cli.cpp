// euler2d_cli -- Command-line interface for Python tests and plots of the 2D Euler solver.
//
// Problems:
//   vortex     Periodic Shu isentropic vortex, translated by the freestream, for convergence studies.
//   riemann_x  Riemann discontinuity along x, uniform in y, for comparison with the 1D solver.
//   riemann_y  Riemann discontinuity along y, uniform in x.
//   blast      Circular blast: left state inside radius r0, right state outside.
//   body       Uniform --freestream past --body with an immersed-boundary slip wall.
//              --wall-out saves x,y,nx,ny,rho,p and force at each wall-face midpoint.
//
// Example:
//   euler2d_cli --problem vortex --nx 64 --ny 64 --xmin -8 --xmax 8 --ymin -8 --ymax 8
//       --bcx periodic --bcy periodic --tend 4 --out vortex.csv
//
// CSV columns: case,x,y,p,rho,u,v; x,y are cell centers, following the data contract in Section 4 of the plan.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <sstream>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "euler2d.h"
#include "sdf.h"

using namespace cfd;

static void usage() {
  std::fprintf(
      stderr,
      "usage: euler2d_cli --problem vortex|riemann_x|riemann_y|blast --out file.csv\n"
      "  --nx N --ny N [--xmin a --xmax b --ymin a --ymax b] [--tend T]\n"
      "  [--riemann rhoL,uL,vL,pL,rhoR,uR,vR,pR]  (repeatable; one case per occurrence)\n"
      "  [--x0 0.5]       discontinuity location: x for riemann_x, y for riemann_y\n"
      "  [--xc0 0 --yc0 0 --r0 0.4]               blast/vortex center and radius\n"
      "  [--beta 5 --u0 1 --v0 1]                  vortex strength and freestream velocity\n"
      "  [--stretch-x uniform|geom:r|tanh:d|sinh:d] [--stretch-y ...]\n"
      "  [--bcx periodic|transmissive|reflect] [--bcy ...]\n"
      "  [--flux hllc|ausm] [--limiter first|none|minmod|venkat] [--k K]\n"
      "  [--cfl C] [--gamma G] [--mco M] [--maxsteps M]\n"
      "  [--time-integrator rk3|be] [--dt physical_dt]  (0/default: physical CFL)\n"
      "  [--pseudo-cfl 0.4] [--implicit-maxiter 2000]\n"
      "  [--implicit-rtol 1e-8] [--implicit-atol 1e-11]  (BE: first-order, no legacy IB)\n"
      "  [--backend cpu|cuda]  (CUDA: fp64 first-order regular or cut-cell+SRD)\n"
      "  [--no-output]         run without writing the field CSV (scaling gate)\n"
      "  [--restart field.csv --t0 T]  bilinear nested-grid initialization\n"
      "  body: --problem body --freestream rho,u,v,p --body circle:cx,cy,R | wedge:x0,y0,len,halfangle_deg\n"
      "        | naca:t,aoa_deg,x0,y0,chord | polygon:path.csv\n"
      "        [--wall ib|cut] [--npoly M] [--wall-out wall.csv]\n"
      "        cut: [--surface-out surface.vtp] [--surface-integrals loads.csv]\n"
      "             [--surface-markers edges.csv] [--surface-ref x,y]\n"
      "        surface exports case 0; markers CSV: edge_id,patch_id (all edges, zero-based)\n");
}

static bool load_surface_markers(const char* path, size_t edges, std::vector<int>* patches) {
  FILE* file = std::fopen(path, "r");
  if (!file) { std::fprintf(stderr, "cannot open surface markers %s\n", path); return false; }
  patches->assign(edges, -1);
  char line[4096];
  bool valid = true;
  while (std::fgets(line, sizeof(line), file)) {
    std::string row(line);
    const size_t start = row.find_first_not_of(" \t\r\n");
    if (start == std::string::npos || row[start] == '#') continue;
    for (char& ch : row) if (ch == ',') ch = ' ';
    std::istringstream in(row);
    std::string first, second, extra;
    in >> first >> second;
    if (first == "edge_id" && second == "patch_id" && !(in >> extra)) continue;
    std::istringstream numeric(row);
    long long edge, patch;
    if (!(numeric >> edge >> patch) || (numeric >> extra) || edge < 0 ||
        static_cast<unsigned long long>(edge) >= edges || patch < 0 || patch > INT_MAX ||
        (*patches)[static_cast<size_t>(edge)] != -1) { valid = false; break; }
    (*patches)[static_cast<size_t>(edge)] = static_cast<int>(patch);
  }
  valid = valid && !std::ferror(file);
  std::fclose(file);
  for (int patch : *patches) if (patch < 0) valid = false;
  if (!valid) std::fprintf(stderr, "invalid surface markers: need exactly one nonnegative patch per edge\n");
  return valid;
}

static bool load_polygon_file(const char* path, std::vector<double>* px,
                              std::vector<double>* py) {
  FILE* f = std::fopen(path, "r");
  if (!f) {
    std::fprintf(stderr, "cannot open polygon file %s\n", path);
    return false;
  }
  char line[4096];
  while (std::fgets(line, sizeof(line), f)) {
    if (line[0] == '#') continue;
    for (char* q = line; *q; ++q)
      if (*q == ',') *q = ' ';
    double x, y;
    if (std::sscanf(line, " %lf %lf", &x, &y) != 2) continue;
    if (!std::isfinite(x) || !std::isfinite(y)) {
      std::fclose(f);
      std::fprintf(stderr, "non-finite polygon vertex in %s\n", path);
      return false;
    }
    px->push_back(x);
    py->push_back(y);
  }
  std::fclose(f);
  if (px->size() >= 2 && px->front() == px->back() && py->front() == py->back()) {
    px->pop_back();
    py->pop_back();
  }
  if (px->size() < 3) {
    std::fprintf(stderr, "polygon file %s has fewer than three numeric vertices\n", path);
    return false;
  }
  return true;
}

struct RestartField {
  int nx = 0, ny = 0;
  std::vector<double> x, y, p, rho, u, v;
};

struct RestartBracket {
  int lo, hi;
  double w;
};

static RestartBracket restart_bracket(const std::vector<double>& a, double z) {
  if (z <= a.front()) return {0, 0, 0.0};
  if (z >= a.back()) {
    const int last = static_cast<int>(a.size()) - 1;
    return {last, last, 0.0};
  }
  const int hi = static_cast<int>(std::upper_bound(a.begin(), a.end(), z) - a.begin());
  const int lo = hi - 1;
  return {lo, hi, (z - a[lo]) / (a[hi] - a[lo])};
}

static bool load_restart_field(const char* path, RestartField* r) {
  FILE* f = std::fopen(path, "r");
  if (!f) {
    std::fprintf(stderr, "cannot open restart field %s\n", path);
    return false;
  }
  char line[4096];
  if (!std::fgets(line, sizeof(line), f) ||
      std::strstr(line, "case,x,y,p,rho,u,v") == nullptr) {
    std::fclose(f);
    std::fprintf(stderr, "restart field has an unexpected header: %s\n", path);
    return false;
  }
  std::vector<double> xx, yy;
  while (std::fgets(line, sizeof(line), f)) {
    int c;
    double x, y, p, rho, u, v;
    if (std::sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%lf", &c, &x, &y, &p,
                    &rho, &u, &v) != 7)
      continue;
    if (c != 0 || !(rho > 0.0) || !(p > 0.0) || !std::isfinite(x) ||
        !std::isfinite(y) || !std::isfinite(rho) || !std::isfinite(p) ||
        !std::isfinite(u) || !std::isfinite(v)) {
      std::fclose(f);
      std::fprintf(stderr, "restart field contains an invalid state: %s\n", path);
      return false;
    }
    xx.push_back(x); yy.push_back(y); r->p.push_back(p); r->rho.push_back(rho);
    r->u.push_back(u); r->v.push_back(v);
  }
  std::fclose(f);
  if (xx.empty()) {
    std::fprintf(stderr, "restart field is empty: %s\n", path);
    return false;
  }
  const double y0 = yy.front();
  while (r->nx < static_cast<int>(yy.size()) && yy[r->nx] == y0) ++r->nx;
  if (r->nx < 2 || static_cast<int>(yy.size()) % r->nx != 0) {
    std::fprintf(stderr, "restart field is not a rectangular Cartesian grid: %s\n", path);
    return false;
  }
  r->ny = static_cast<int>(yy.size()) / r->nx;
  if (r->ny < 2) {
    std::fprintf(stderr, "restart field needs at least 2x2 cells: %s\n", path);
    return false;
  }
  r->x.assign(xx.begin(), xx.begin() + r->nx);
  r->y.resize(r->ny);
  for (int j = 0; j < r->ny; ++j) {
    r->y[j] = yy[static_cast<size_t>(j) * r->nx];
    if (j > 0 && !(r->y[j] > r->y[j - 1])) {
      std::fprintf(stderr, "restart y coordinates are not strictly increasing: %s\n", path);
      return false;
    }
    for (int i = 0; i < r->nx; ++i) {
      const size_t k = static_cast<size_t>(j) * r->nx + i;
      if (xx[k] != r->x[i] || yy[k] != r->y[j] ||
          (i > 0 && !(r->x[i] > r->x[i - 1]))) {
        std::fprintf(stderr, "restart rows do not form a sorted Cartesian grid: %s\n", path);
        return false;
      }
    }
  }
  return true;
}

static double bilinear_sample(const RestartField& r, const std::vector<double>& q,
                              const RestartBracket& bx, const RestartBracket& by) {
  const int i0 = bx.lo, i1 = bx.hi, j0 = by.lo, j1 = by.hi;
  const double wx = bx.w, wy = by.w;
  const auto at = [&](int i, int j) { return q[static_cast<size_t>(j) * r.nx + i]; };
  const double a = (1.0 - wx) * at(i0, j0) + wx * at(i1, j0);
  const double b = (1.0 - wx) * at(i0, j1) + wx * at(i1, j1);
  return (1.0 - wy) * a + wy * b;
}

static Stretch parse_stretch(const std::string& s) {
  Stretch r;
  r.type = STRETCH_UNIFORM;
  r.param = 0.0;
  if (s == "uniform") return r;
  const size_t c = s.find(':');
  if (c == std::string::npos) {
    usage();
    std::exit(2);
  }
  const std::string t = s.substr(0, c);
  r.param = std::atof(s.c_str() + c + 1);
  if (t == "geom") r.type = STRETCH_GEOMETRIC;
  else if (t == "tanh") r.type = STRETCH_TANH_ENDS;
  else if (t == "sinh") r.type = STRETCH_SINH_CENTER;
  else {
    usage();
    std::exit(2);
  }
  return r;
}

static int parse_bc(const std::string& s) {
  if (s == "periodic") return BC_PERIODIC;
  if (s == "reflect") return BC_REFLECT;
  return BC_TRANSMISSIVE;
}

int main(int argc, char** argv) {
  std::string problem = "riemann_x", outpath;
  int nx = 64, ny = 64;
  double xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0, tend = 0.2, x0 = 0.5;
  double xc0 = 0.0, yc0 = 0.0, r0 = 0.4, beta = 5.0, u0 = 1.0, v0 = 1.0;
  double mco = -1.0, t0 = 0.0;
  long maxsteps = 10000000;
  Params2D par = params2d_default();
  std::vector<std::vector<double> > riem;
  std::string body_spec, wall_out, wall_method = "ib", restart_path;
  std::string surface_out, surface_integrals, surface_markers;
  double surface_xref = 0.0, surface_yref = 0.0;
  std::string backend_name = "cpu";
  double fs[4] = {1.0, 0.0, 0.0, 1.0};
  int npoly = 360;
  bool no_output = false;

  for (int a = 1; a < argc; ++a) {
    const std::string k = argv[a];
    auto need = [&](void) -> const char* {
      if (a + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++a];
    };
    if (k == "--problem") problem = need();
    else if (k == "--nx") nx = std::atoi(need());
    else if (k == "--ny") ny = std::atoi(need());
    else if (k == "--xmin") xmin = std::atof(need());
    else if (k == "--xmax") xmax = std::atof(need());
    else if (k == "--ymin") ymin = std::atof(need());
    else if (k == "--ymax") ymax = std::atof(need());
    else if (k == "--tend") tend = std::atof(need());
    else if (k == "--t0") t0 = std::atof(need());
    else if (k == "--x0") x0 = std::atof(need());
    else if (k == "--xc0") xc0 = std::atof(need());
    else if (k == "--yc0") yc0 = std::atof(need());
    else if (k == "--r0") r0 = std::atof(need());
    else if (k == "--beta") beta = std::atof(need());
    else if (k == "--u0") u0 = std::atof(need());
    else if (k == "--v0") v0 = std::atof(need());
    else if (k == "--cfl") par.cfl = std::atof(need());
    else if (k == "--dt") par.dt_fixed = std::atof(need());
    else if (k == "--pseudo-cfl") par.pseudo_cfl = std::atof(need());
    else if (k == "--implicit-maxiter") par.implicit_maxiter = std::atoi(need());
    else if (k == "--implicit-rtol") par.implicit_rtol = std::atof(need());
    else if (k == "--implicit-atol") par.implicit_atol = std::atof(need());
    else if (k == "--time-integrator") {
      const std::string value = need();
      if (value == "rk3") par.time_integrator = TIME_SSP_RK3;
      else if (value == "be") par.time_integrator = TIME_BACKWARD_EULER;
      else { usage(); return 2; }
    }
    else if (k == "--gamma") par.gamma = std::atof(need());
    else if (k == "--k") par.venkat_k = std::atof(need());
    else if (k == "--mco") mco = std::atof(need());
    else if (k == "--maxsteps") maxsteps = std::atol(need());
    else if (k == "--out") outpath = need();
    else if (k == "--no-output") no_output = true;
    else if (k == "--restart") restart_path = need();
    else if (k == "--backend") backend_name = need();
    else if (k == "--body") body_spec = need();
    else if (k == "--wall") wall_method = need();
    else if (k == "--wall-out") wall_out = need();
    else if (k == "--surface-out") surface_out = need();
    else if (k == "--surface-integrals") surface_integrals = need();
    else if (k == "--surface-markers") surface_markers = need();
    else if (k == "--surface-ref") {
      char extra;
      if (std::sscanf(need(), "%lf,%lf %c", &surface_xref, &surface_yref, &extra) != 2 ||
          !std::isfinite(surface_xref) || !std::isfinite(surface_yref)) return 2;
    }
    else if (k == "--npoly") npoly = std::atoi(need());
    else if (k == "--freestream") {
      if (std::sscanf(need(), "%lf,%lf,%lf,%lf", &fs[0], &fs[1], &fs[2], &fs[3]) != 4) {
        usage();
        return 2;
      }
    }
    else if (k == "--stretch-x") par.sx = parse_stretch(need());
    else if (k == "--stretch-y") par.sy = parse_stretch(need());
    else if (k == "--bcx") par.bc[0] = par.bc[1] = parse_bc(need());
    else if (k == "--bcy") par.bc[2] = par.bc[3] = parse_bc(need());
    else if (k == "--flux") {
      const std::string v = need();
      par.flux = (v == "ausm") ? FLUX_AUSMPUP : FLUX_HLLC;
    } else if (k == "--limiter") {
      const std::string v = need();
      par.limiter = (v == "first") ? LIM_FIRST_ORDER
                                    : (v == "none" ? LIM_NONE
                                                   : (v == "minmod" ? LIM_MINMOD : LIM_VENKAT));
    } else if (k == "--riemann") {
      std::vector<double> v(8);
      if (std::sscanf(need(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &v[0], &v[1],
                      &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) != 8) {
        usage();
        return 2;
      }
      riem.push_back(v);
    } else {
      usage();
      return 2;
    }
  }
  if (outpath.empty() && !no_output) {
    usage();
    return 2;
  }
  if ((!surface_out.empty() || !surface_integrals.empty() || !surface_markers.empty()) &&
      (problem != "body" || wall_method != "cut")) {
    std::fprintf(stderr, "surface options require --problem body --wall cut\n");
    return 2;
  }
  const std::string outputs[] = {no_output ? "" : outpath, wall_out, surface_out, surface_integrals};
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j)
      if (!outputs[i].empty() && outputs[i] == outputs[j]) {
        std::fprintf(stderr, "output paths must be distinct\n"); return 2;
      }
  if (mco > 0.0) par.ausm_mco = mco;
  if (!(t0 >= 0.0) || !(tend > t0)) {
    std::fprintf(stderr, "require 0 <= t0 < tend\n");
    return 2;
  }
  if (wall_method != "ib" && wall_method != "cut") {
    usage();
    return 2;
  }
  if (backend_name != "cpu" && backend_name != "cuda") {
    usage();
    return 2;
  }

  int ncase = 1;
  if (problem == "vortex") {
    par.bc[0] = par.bc[1] = par.bc[2] = par.bc[3] = BC_PERIODIC;
  } else if (problem == "body") {
    if (body_spec.empty()) {
      usage();
      return 2;
    }
  } else if (problem == "riemann_x" || problem == "riemann_y" || problem == "blast") {
    if (riem.empty()) {
      const double sod[8] = {1.0, 0.0, 0.0, 1.0, 0.125, 0.0, 0.0, 0.1};
      riem.push_back(std::vector<double>(sod, sod + 8));
    }
    ncase = static_cast<int>(riem.size());
  } else {
    usage();
    return 2;
  }

  Solver2D s;
  solver2d_init(&s, nx, ny, xmin, xmax, ymin, ymax, ncase, par);
  const Grid2D& g = s.grid;
  const double gm = par.gamma;

  for (int c = 0; c < ncase; ++c)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const double x = g.xc[i + g.ng], y = g.yc[j + g.ng];
        double rho, u, v, p;
        if (problem == "body") {
          continue;  // Initialize all of these cells with the freestream below.
        } else if (problem == "vortex") {
          const double dx = x - xc0, dy = y - yc0, r2 = dx * dx + dy * dy;
          const double e = std::exp(0.5 * (1.0 - r2));
          u = u0 - beta / (2.0 * M_PI) * dy * e;
          v = v0 + beta / (2.0 * M_PI) * dx * e;
          const double T = 1.0 - (gm - 1.0) * beta * beta / (8.0 * gm * M_PI * M_PI) * e * e;
          rho = std::pow(T, 1.0 / (gm - 1.0));
          p = std::pow(rho, gm);
        } else {
          const std::vector<double>& r = riem[c];
          bool left;
          if (problem == "riemann_x") left = (x < x0);
          else if (problem == "riemann_y") left = (y < x0);
          else left = ((x - xc0) * (x - xc0) + (y - yc0) * (y - yc0) < r0 * r0);
          const int o = left ? 0 : 4;
          rho = r[o]; u = r[o + 1]; v = r[o + 2]; p = r[o + 3];
        }
        solver2d_set_primitive(&s, c, i, j, rho, u, v, p);
      }

  std::vector<double> bpx, bpy;
  IB2D ib;
  Cut2D cut;
  const bool use_cut = wall_method == "cut";
  if (problem == "body") {
    for (int c = 0; c < ncase; ++c)
      for (int j = -g.ng; j < ny + g.ng; ++j)
        for (int i = -g.ng; i < nx + g.ng; ++i)
          solver2d_set_primitive(&s, c, i, j, fs[0], fs[1], fs[2], fs[3]);
    const size_t col = body_spec.find(':');
    if (col == std::string::npos) { usage(); return 2; }
    const std::string kind = body_spec.substr(0, col);
    const char* a = body_spec.c_str() + col + 1;
    if (kind == "circle") {
      double cx, cy, R;
      if (std::sscanf(a, "%lf,%lf,%lf", &cx, &cy, &R) != 3) { usage(); return 2; }
      bpx.resize(npoly); bpy.resize(npoly);
      circle_polygon(cx, cy, R, npoly, bpx.data(), bpy.data(), M_PI / npoly);
    } else if (kind == "wedge") {
      double x0, y0, len, ha;
      if (std::sscanf(a, "%lf,%lf,%lf,%lf", &x0, &y0, &len, &ha) != 4) { usage(); return 2; }
      const double h = len * std::tan(ha * M_PI / 180.0);
      bpx = {x0, x0 + len, x0 + len};
      bpy = {y0, y0 + h, y0 - h};
    } else if (kind == "naca") {
      double t, aoa, x0, y0, ch;
      if (std::sscanf(a, "%lf,%lf,%lf,%lf,%lf", &t, &aoa, &x0, &y0, &ch) != 5) { usage(); return 2; }
      const int npan = npoly / 2;
      bpx.resize(2 * npan + 1); bpy.resize(2 * npan + 1);
      const int np = naca_polygon(t, npan, aoa, bpx.data(), bpy.data());
      for (int q = 0; q < np; ++q) { bpx[q] = x0 + ch * bpx[q]; bpy[q] = y0 + ch * bpy[q]; }
      bpx.resize(np); bpy.resize(np);
    } else if (kind == "polygon") {
      if (!load_polygon_file(a, &bpx, &bpy)) return 2;
    } else {
      usage();
      return 2;
    }
    if (use_cut) {
      std::vector<int> patches;
      if (!surface_markers.empty() &&
          !load_surface_markers(surface_markers.c_str(), bpx.size(), &patches)) return 2;
      if (cut2d_build_polygon(&s, bpx.data(), bpy.data(), static_cast<int>(bpx.size()), &cut,
                              patches.empty() ? nullptr : patches.data()) != 0) return 5;
      if (cut.nbad_nbhd != 0) {
        std::fprintf(stderr, "cut-cell geometry has %d invalid SRD neighborhoods\n", cut.nbad_nbhd);
        return 5;
      }
      solver2d_set_cut(&s, &cut);
      std::fprintf(stderr, "body: %zu vertices, %d cut cells, %d solid cells, %d small cells, %d bad neighborhoods\n",
                   bpx.size(), cut.ncut, cut.nsolid, cut.nsmall, cut.nbad_nbhd);
    } else {
      if (ib2d_build_polygon(&s, bpx.data(), bpy.data(), static_cast<int>(bpx.size()), &ib) != 0) return 5;
      solver2d_set_ib(&s, &ib);
      std::fprintf(stderr, "body: %zu vertices, %d ghost cells, %d with fallback stencil\n",
                   bpx.size(), ib.nghost, ib.nfallback);
    }
  }

  if (!restart_path.empty()) {
    if (ncase != 1) {
      std::fprintf(stderr, "restart currently supports exactly one case\n");
      return 2;
    }
    RestartField restart;
    if (!load_restart_field(restart_path.c_str(), &restart)) return 2;
    // Brackets depend only on the destination axis, not on the field variable.
    // Precompute them once; the old path repeated two upper_bound searches for
    // each of four primitive variables in every destination cell.
    std::vector<RestartBracket> bx(nx), by(ny);
    for (int i = 0; i < nx; ++i) bx[i] = restart_bracket(restart.x, g.xc[i + g.ng]);
    for (int j = 0; j < ny; ++j) by[j] = restart_bracket(restart.y, g.yc[j + g.ng]);
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const double rho = bilinear_sample(restart, restart.rho, bx[i], by[j]);
        const double u = bilinear_sample(restart, restart.u, bx[i], by[j]);
        const double v = bilinear_sample(restart, restart.v, bx[i], by[j]);
        const double p = bilinear_sample(restart, restart.p, bx[i], by[j]);
        solver2d_set_primitive(&s, 0, i, j, rho, u, v, p);
      }
    s.t[0] = t0;
    std::fprintf(stderr,
                 "restart: %dx%d -> %dx%d bilinear primitive initialization at t=%.17g\n",
                 restart.nx, restart.ny, nx, ny, t0);
  }

  const int backend = backend_name == "cuda" ? BACKEND_CUDA : BACKEND_CPU;
  if (backend == BACKEND_CUDA) {
    char reason[256];
    if (!solver2d_cuda_available(reason, sizeof(reason))) {
      std::fprintf(stderr, "CUDA backend unavailable: %s\n", reason);
      return 6;
    }
  }
  const auto solve_start = std::chrono::steady_clock::now();
  const int status = solver2d_run_backend(&s, tend, maxsteps, backend);
  if (par.time_integrator == TIME_BACKWARD_EULER) {
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - solve_start).count();
    std::fprintf(stderr,
      "implicit: method=backward-euler spatial=raw-fv physical_steps=%ld pseudo_updates=%ld pseudo_retries=%ld initial_defect=%.9e final_defect=%.9e tolerance=%.9e solve_wall=%.6f s\n",
      s.nsteps, s.implicit.updates, s.implicit.retries, s.implicit.initial_defect,
      s.implicit.final_defect, s.implicit.tolerance, seconds);
  }
  if (status != 0) {
    std::fprintf(stderr, "euler2d_cli: solver returned status %d\n", status);
    return 3;
  }

  if (!surface_out.empty() && cut2d_write_surface_vtp(&s, 0, surface_out.c_str()) != 0) {
    std::fprintf(stderr, "surface export failed: %s\n", surface_out.c_str()); return 4;
  }
  if (!surface_integrals.empty() && cut2d_write_surface_integrals(&s, 0,
      surface_integrals.c_str(), surface_xref, surface_yref) != 0) {
    std::fprintf(stderr, "surface integral export failed: %s\n", surface_integrals.c_str()); return 4;
  }
  if (!no_output) {
    FILE* f = std::fopen(outpath.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "cannot open %s\n", outpath.c_str());
      return 4;
    }
    std::fprintf(f, "case,x,y,p,rho,u,v\n");
    double rho_min = std::numeric_limits<double>::infinity();
    double rho_max = -std::numeric_limits<double>::infinity();
    double p_min = std::numeric_limits<double>::infinity();
    double p_max = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < ncase; ++c)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
          double rho, u, v, p;
          solver2d_get_primitive(&s, c, i, j, &rho, &u, &v, &p);
          const size_t q = static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng);
          if (!use_cut || cut.vol[q] > 0.0) {
            rho_min = std::min(rho_min, rho); rho_max = std::max(rho_max, rho);
            p_min = std::min(p_min, p); p_max = std::max(p_max, p);
          }
          std::fprintf(f, "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n", c,
                       g.xc[i + g.ng], g.yc[j + g.ng], p, rho, u, v);
        }
    std::fclose(f);
    double symmetry = 0.0;
    if (ncase == 1) {
      for (int j = 0; j < ny / 2; ++j)
        for (int i = 0; i < nx; ++i) {
          const int jj = ny - 1 - j;
          const size_t a = static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng);
          const size_t b = static_cast<size_t>(jj + g.ng) * g.mx + (i + g.ng);
          if (use_cut && (cut.vol[a] <= 0.0 || cut.vol[b] <= 0.0)) continue;
          double r1, u1, v1, p1, r2, u2, v2, p2;
          solver2d_get_primitive(&s, 0, i, j, &r1, &u1, &v1, &p1);
          solver2d_get_primitive(&s, 0, i, jj, &r2, &u2, &v2, &p2);
          symmetry = std::max(symmetry, std::max(std::fabs(r1 - r2),
            std::max(std::fabs(u1 - u2), std::max(std::fabs(v1 + v2), std::fabs(p1 - p2)))));
        }
    }
    std::fprintf(stderr,
      "field: rho_min=%.17e rho_max=%.17e p_min=%.17e p_max=%.17e y_symmetry=%.3e\n",
      rho_min, rho_max, p_min, p_max, symmetry);
  }
  if (problem == "body" && !use_cut)
    std::fprintf(stderr, "ib: pressure-extrapolation guard hits = %ld\n", ib.nguard);
  if (problem == "body" && !wall_out.empty()) {
    FILE* w = std::fopen(wall_out.c_str(), "w");
    if (!w) return 4;
    double fx = 0.0, fy = 0.0;
    if (use_cut) cut2d_wall_force(&s, 0, &fx, &fy);
    else ib2d_wall_force(&s, 0, bpx.data(), bpy.data(), static_cast<int>(bpx.size()), &fx, &fy);
    std::fprintf(w, "# fx=%.17e fy=%.17e\n", fx, fy);
    std::fprintf(w, "x,y,nx,ny,rho,p,len\n");
    if (use_cut) {
      const size_t cells = static_cast<size_t>(g.mx) * g.my;
      const double* Wr = s.W.data();
      const double* Wp = s.W.data() + 3 * cells;
      const double* Sr = s.SX.data();
      const double* Sp = s.SX.data() + 3 * cells;
      const double* Tr = s.SY.data();
      const double* Tp = s.SY.data() + 3 * cells;
      // Keep legacy CSV normal pointing into the fluid; VTP explicitly uses the opposite convention.
      for (const WallFragment2D& fragment : cut.wall_fragments) {
        const size_t q = fragment.owner_cell;
        const double mx = 0.5*(fragment.x0+fragment.x1), my = 0.5*(fragment.y0+fragment.y1);
        double rw = Wr[q] + Sr[q]*(mx-cut.cx[q]) + Tr[q]*(my-cut.cy[q]);
        double pw = Wp[q] + Sp[q]*(cut.wcx[q]-cut.cx[q]) + Tp[q]*(cut.wcy[q]-cut.cy[q]);
        if (!(rw > 0.0)) rw = Wr[q];
        if (!(pw > 0.0)) pw = Wp[q];
        std::fprintf(w, "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                     mx, my, -fragment.nx, -fragment.ny, rw, pw, fragment.length);
      }
    } else {
      const int nv = static_cast<int>(bpx.size());
      for (int a2 = 0; a2 < nv; ++a2) {
        const int b2 = (a2 + 1) % nv;
        const double mx = 0.5 * (bpx[a2] + bpx[b2]), my = 0.5 * (bpy[a2] + bpy[b2]);
        const double tx = bpx[b2] - bpx[a2], ty = bpy[b2] - bpy[a2];
        const double L = std::sqrt(tx * tx + ty * ty);
        double nx2 = ty / L, ny2 = -tx / L;
        if (point_in_polygon(bpx.data(), bpy.data(), nv, mx + 1e-7 * nx2, my + 1e-7 * ny2)) { nx2 = -nx2; ny2 = -ny2; }
        double rw, pw;
        if (ib2d_wall_state(&s, 0, mx, my, nx2, ny2, &rw, &pw)) continue;
        std::fprintf(w, "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n", mx, my, nx2, ny2, rw, pw, L);
      }
    }
    std::fclose(w);
  }
  std::fprintf(stderr, "ok: backend=%s ncase=%d nx=%d ny=%d steps=%ld rejected=%ld\n",
               backend_name.c_str(), ncase, nx, ny, s.nsteps, s.nreject);
  return 0;
}
