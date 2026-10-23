// euler1d_cli -- Command-line interface for Python tests and plots of the 1D Euler solver.
//
// Example:
//   euler1d_cli --problem riemann --n 400 --tend 0.2
//       --riemann 1,0,1,0.125,0,0.1 --flux hllc --limiter venkat --out sod.csv
//   euler1d_cli --problem simple_wave --n 200 --tend 0.2 --amp 0.1 --out w.csv
//
// CSV columns: case,x,rho,u,p; all quantities are nondimensional.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "euler1d.h"

using namespace cfd;

static void usage() {
  std::fprintf(stderr,
               "usage: euler1d_cli --problem riemann|simple_wave --n N --tend T\n"
               "  [--riemann rhoL,uL,pL,rhoR,uR,pR]  (repeatable; one case per occurrence)\n"
               "  [--x0 0.5] [--xmin 0] [--xmax 1] [--amp A]\n"
               "  [--flux hllc|ausm] [--limiter first|none|minmod|venkat] [--k K]\n"
               "  [--cfl C] [--gamma G] [--bc periodic|transmissive]\n"
               "  [--maxsteps M] --out file.csv\n");
}

int main(int argc, char** argv) {
  std::string problem = "riemann", outpath;
  int n = 200;
  double tend = 0.2, x0 = 0.5, xmin = 0.0, xmax = 1.0, amp = 0.1;
  long maxsteps = 10000000;
  Params1D par = params1d_default();
  std::vector<std::vector<double> > riem;

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
    else if (k == "--n") n = std::atoi(need());
    else if (k == "--tend") tend = std::atof(need());
    else if (k == "--x0") x0 = std::atof(need());
    else if (k == "--xmin") xmin = std::atof(need());
    else if (k == "--xmax") xmax = std::atof(need());
    else if (k == "--amp") amp = std::atof(need());
    else if (k == "--cfl") par.cfl = std::atof(need());
    else if (k == "--gamma") par.gamma = std::atof(need());
    else if (k == "--k") par.venkat_k = std::atof(need());
    else if (k == "--maxsteps") maxsteps = std::atol(need());
    else if (k == "--out") outpath = need();
    else if (k == "--flux") {
      const std::string v = need();
      par.flux = (v == "ausm") ? FLUX_AUSMPUP : FLUX_HLLC;
    } else if (k == "--limiter") {
      const std::string v = need();
      par.limiter = (v == "first") ? LIM_FIRST_ORDER
                                    : (v == "none" ? LIM_NONE
                                                   : (v == "minmod" ? LIM_MINMOD : LIM_VENKAT));
    } else if (k == "--bc") {
      const std::string v = need();
      par.bc = (v == "periodic") ? BC_PERIODIC : BC_TRANSMISSIVE;
    } else if (k == "--riemann") {
      std::vector<double> v(6);
      if (std::sscanf(need(), "%lf,%lf,%lf,%lf,%lf,%lf", &v[0], &v[1], &v[2],
                      &v[3], &v[4], &v[5]) != 6) {
        usage();
        return 2;
      }
      riem.push_back(v);
    } else {
      usage();
      return 2;
    }
  }
  if (outpath.empty()) {
    usage();
    return 2;
  }

  int ncase = 1;
  if (problem == "riemann") {
    if (riem.empty()) {
      // Default Sod problem.
      const double sod[6] = {1.0, 0.0, 1.0, 0.125, 0.0, 0.1};
      riem.push_back(std::vector<double>(sod, sod + 6));
    }
    ncase = static_cast<int>(riem.size());
  } else if (problem == "simple_wave") {
    par.bc = BC_PERIODIC;
  } else {
    usage();
    return 2;
  }

  Solver1D s;
  solver1d_init(&s, n, xmin, xmax, ncase, par);

  if (problem == "riemann") {
    for (int c = 0; c < ncase; ++c)
      solver1d_set_riemann(&s, c, x0, riem[c][0], riem[c][1], riem[c][2],
                           riem[c][3], riem[c][4], riem[c][5]);
  } else {
    // Initial right-running simple wave: u0 = A sin(2 pi x), a0 = 1, rho0 = 1, p0 = rho0 a0^2 / gamma.
    // Isentropic with a constant left-running Riemann invariant: a = a0 + (gamma-1)/2 * u.
    const double gm = par.gamma;
    const double a0 = 1.0, rho0 = 1.0, p0 = rho0 * a0 * a0 / gm;
    for (int i = 0; i < n; ++i) {
      const double x = s.grid.xc[i + s.grid.ng];
      const double u = amp * std::sin(2.0 * M_PI * x);
      const double a = a0 + 0.5 * (gm - 1.0) * u;
      const double rho = rho0 * std::pow(a / a0, 2.0 / (gm - 1.0));
      const double p = p0 * std::pow(rho / rho0, gm);
      solver1d_set_primitive(&s, 0, i, rho, u, p);
    }
  }

  const int status = solver1d_run(&s, tend, maxsteps);
  if (status != 0) {
    std::fprintf(stderr, "euler1d_cli: solver returned status %d\n", status);
    return 3;
  }

  FILE* f = std::fopen(outpath.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", outpath.c_str());
    return 4;
  }
  std::fprintf(f, "case,x,rho,u,p\n");
  for (int c = 0; c < ncase; ++c)
    for (int i = 0; i < n; ++i) {
      double r, u, p;
      solver1d_get_primitive(&s, c, i, &r, &u, &p);
      std::fprintf(f, "%d,%.17e,%.17e,%.17e,%.17e\n", c,
                   s.grid.xc[i + s.grid.ng], r, u, p);
    }
  std::fclose(f);
  std::fprintf(stderr, "ok: ncase=%d n=%d steps=%ld\n", ncase, n, s.nsteps);
  return 0;
}
