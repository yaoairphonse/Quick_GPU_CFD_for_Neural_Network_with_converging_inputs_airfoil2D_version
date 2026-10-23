// euler1d.h -- 1D compressible Euler solver: conservative, cell-centered finite volumes.
//
// The initial 1D test bed validates fluxes, reconstruction, limiters, time integration, and batching
// before extending them to 2D. The data layout already follows the final design:
//   U[(case * NVAR + var) * m + (i + ng)], m = n + 2*ng; ng is the ghost-cell count.
// Each case has its own time and time step; batched and individual results must match bit for bit.
#ifndef CFD_EULER1D_H
#define CFD_EULER1D_H

#include "array.h"
#include "types.h"

namespace cfd {

enum { NVAR1D = 3 };  // rho, rho*u, rho*E

struct Params1D {
  double gamma;
  double cfl;
  int flux;       // FluxType
  int limiter;    // LimiterType
  double venkat_k;  // Venkatakrishnan K; eps^2 = (K*dx)^3. Larger K means weaker limiting.
  double ausm_mco;  // AUSM+-up cutoff Mach number.
  int bc;         // BCType
  double dt_fixed;  // Positive: fixed time step for tests, such as bitwise 1D/2D comparisons; zero: CFL-based stepping.
};

struct Grid1D {
  int n;    // Number of interior cells.
  int ng;   // Ghost layers, currently fixed at 2.
  int m;    // n + 2*ng
  double xmin, xmax;
  Array xc;  // Cell centers including ghosts, length m.
  Array dx;  // Cell widths including ghosts, length m.
};

struct Solver1D {
  Grid1D grid;
  Params1D par;
  int ncase;
  Array U, U1, U2;  // Conserved states; three arrays for SSP-RK3.
  Array R;          // Residual dU/dt.
  Array W;          // Primitive variables (rho,u,p).
  Array S;          // Limited slopes of (rho,u,p).
  Array t;          // Current time for each case.
  Array dt;         // Current time step for each case.
  long nsteps;      // Number of steps advanced, taking the maximum over cases.
};

Params1D params1d_default();

// Construct a uniform-grid solver; ncase >= 1.
void solver1d_init(Solver1D* s, int n, double xmin, double xmax, int ncase,
                   const Params1D& par);

// Set primitive variables (rho,u,p) in interior cell i = 0..n-1 of a case.
void solver1d_set_primitive(Solver1D* s, int c, int i, double rho, double u,
                            double p);

// Set Riemann initial data at x0: left state for x < x0, right state otherwise.
void solver1d_set_riemann(Solver1D* s, int c, double x0, double rhoL,
                          double uL, double pL, double rhoR, double uR,
                          double pR);

// Advance each case to tend; shorten the final step to reach tend exactly.
// Return 0 on success; nonzero if maxsteps is exceeded or a nonphysical state occurs.
int solver1d_run(Solver1D* s, double tend, long maxsteps);

// Read primitive variables in interior cells.
void solver1d_get_primitive(const Solver1D* s, int c, int i, double* rho,
                            double* u, double* p);

// Integrate a conserved variable over interior cells (sum U * dx) to check conservation.
double solver1d_integral(const Solver1D* s, int c, int var);

// Public residual calculation for unit tests. Boundary conditions must already be applied to U.
void solver1d_apply_bc(const Solver1D* s, double* U);
void solver1d_residual(const Solver1D* s, const double* U, double* R,
                       double* W, double* S);

}  // namespace cfd

#endif  // CFD_EULER1D_H
