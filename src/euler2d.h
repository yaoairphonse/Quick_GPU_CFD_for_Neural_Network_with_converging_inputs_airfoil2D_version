// euler2d.h -- 2D compressible Euler solver with cell-centered finite volumes on stretched rectangular grids.
//
// Design:
//   * Independent 1D stretching in x/y forms a tensor-product grid with four neighbors per cell.
//     Face loops need no atomic addition. All cases in a batch share the same grid.
//   * Storage: U[((case*NVAR + var) * my + (j+ng)) * mx + (i+ng)], mx=nx+2ng, my=ny+2ng.
//   * Reconstruction uses directional slopes and limiter.h independently along each axis;
//     flux.h provides HLLC/AUSM+-up, giving bitwise 1D agreement for axis-aligned problems.
//   * Every case has its own time, time step, and AUSM+-up reference Mach number mco.
#ifndef CFD_EULER2D_H
#define CFD_EULER2D_H

#include <cstddef>
#include <vector>

#include "array.h"
#include "types.h"

namespace cfd {

enum { NVAR2D = 4 };  // rho, rho*u, rho*v, rho*E

enum SolverBackend { BACKEND_CPU = 0, BACKEND_CUDA = 1 };
enum TimeIntegrator2D { TIME_SSP_RK3 = 0, TIME_BACKWARD_EULER = 1 };

// 1D grid stretching, applied independently in x or y.
enum StretchType {
  STRETCH_UNIFORM = 0,
  STRETCH_GEOMETRIC = 1,     // Fixed adjacent-width ratio param; >1 widens toward max and clusters near min.
  STRETCH_TANH_ENDS = 2,     // Refine both ends; param=delta>0, with larger values increasing clustering.
  STRETCH_SINH_CENTER = 3    // Refine the center; param=delta>0, with larger values increasing clustering.
};
struct Stretch {
  int type;
  double param;
};

struct Params2D {
  bool low_mach_preconditioning = false; // steady pseudo-time only
  bool local_pseudo_time = false;
  double low_mach_reference = .2, low_mach_floor = .02;
  double gamma;
  double cfl;
  int flux;         // FluxType
  int limiter;      // LimiterType
  int amr_spatial_order = 1; // static cut-cell solver only: 1 constant, 2 limited least squares
  double venkat_k;  // Venkatakrishnan K.
  double ausm_mco;  // AUSM+-up reference Mach number, overridable per case.
  int bc[4];        // BCType for xmin,xmax,ymin,ymax; periodic boundaries must be paired.
  double dt_fixed;  // > 0 specifies physical dt; 0 = use full-cell CFL
  int time_integrator; // TimeIntegrator2D; explicit SSP-RK3 remains the default
  double pseudo_cfl;   // local pseudo-time relaxation, independent of physical CFL
  int implicit_maxiter;
  double implicit_rtol, implicit_atol; // tolerances on the actual BE defect
  Stretch sx, sy;
};

struct Grid2D {
  int nx, ny, ng, mx, my;
  double xmin, xmax, ymin, ymax;
  Array xf, yf;  // Face coordinates, lengths nx+1 and ny+1.
  Array xc, dx;  // Cell centers and widths including ghosts, length mx.
  Array yc, dy;  // Length my.
};

// Immersed-boundary slip-wall data: ghosts and image points, built by ib2d_build_polygon.
// Integer indices are stored exactly as doubles below 2^53 to keep a single Array type.
enum { IB_NS = 36 };  // Maximum number of image-point stencil cells (6x6).

struct IB2D {
  int nghost;
  int nfallback;  // Ghosts using a linearly reproducing least-squares stencil when the surrounding four cells are not all fluid.
  Array fluid;    // Length mx*my: 1 for fluid and domain ghosts, 0 for IB ghosts or solid.
  Array gidx;     // nghost ghost-cell linear indices (j+ng)*mx+(i+ng).
  Array sidx;     // nghost*IB_NS interpolation-cell indices.
  Array sw;       // nghost*IB_NS weights summing to 1; unused entries have zero weight.
  Array nrm;      // nghost*2 wall normals pointing into the fluid.
  // Linear normal pressure extrapolation uses a second probe dprobe beyond the image point.
  // p_ghost = p_I - 2d*(p_2-p_I)/dprobe; isentropic density: rho_g = rho_I*(p_g/p_I)^(1/gamma).
  // This corrects the curved-wall O(h) error of copying image pressure; dp/dn = rho*u_t^2*curvature.
  Array sidx2;    // nghost*IB_NS second-probe stencil indices.
  Array sw2;      // nghost*IB_NS second-probe weights.
  Array d2;       // nghost entries of 2d, the ghost-to-image distance.
  double dprobe;  // Distance from image point to the second probe.
  int use_p_extrap;           // 1 enables pressure extrapolation; 0 copies image pressure, the original first-order behavior.
  mutable long nguard;        // Cumulative count of extrapolated nonpositive pressures replaced by image pressure.
};

// Embedded cut-cell data from cut2d_build_polygon; see cutcell2d.cpp.
// Face normals point outward; an eastern x face points +x. Wall vector Aw points into solid and includes wall length.
struct WallFragment2D {
  size_t owner_cell;
  int source_edge_id, body_id, patch_id;
  double x0, y0, x1, y1, nx, ny, length; // normal: fluid -> solid
};
struct WallCell2D {
  size_t owner_cell, begin, count;
};
struct SurfaceIntegral2D {
  double length, fx, fy, mz; // pressure only, per unit span
};
struct Cut2D {
  std::vector<WallFragment2D> wall_fragments;
  std::vector<WallCell2D> wall_cells; // sparse cell ranges, not a full-grid allocation
  int nsmall;       // Small cells requiring SRD neighborhoods.
  int nbad_nbhd;    // Small cells without a sufficiently large neighborhood; should be zero.
  int nsolid, ncut; // Cell statistics.
  Array vol;        // mx*my fluid areas; 0 denotes solid.
  Array cx, cy;     // mx*my fluid centroids.
  Array flag;       // mx*my classification: 0 full fluid, 1 cut, 2 solid.
  Array axl, axy;   // (nx+1)*ny x-face aperture lengths and y centroids, indexed j*(nx+1)+i.
  Array ayl, ayx;   // nx*(ny+1) y-face aperture lengths and x centroids, indexed j*nx+i.
  Array wax, way;   // mx*my wall vectors pointing into solid.
  Array wcx, wcy;   // mx*my wall centroids.
  Array nbrcount;   // mx*my N_rs, the number of neighborhoods containing the cell.
  Array smallpos;   // mx*my positions in the small-cell list; -1 denotes a regular cell.
  Array sidx;       // nsmall small-cell linear indices.
  Array scnt;       // nsmall neighborhood sizes.
  Array smem;       // nsmall*25 neighborhood member indices.
  Array svhat, sxhat, syhat;  // nsmall weighted volumes and centroids.
};

struct ImplicitStats2D {
  long updates, retries; // pseudo updates/trial retries, separate from physical steps
  double initial_defect, final_defect, tolerance; // latest physical step, scaled Linf
};

struct Solver2D {
  Grid2D grid;
  Params2D par;
  int ncase;
  Array U, U1, U2;  // Conserved states; three arrays for SSP-RK3.
  Array R;          // Residual dU/dt.
  Array W;          // Primitive variables (rho,u,v,p).
  Array SX, SY;     // Limited directional slopes.
  Array FX, FY;     // x/y face fluxes.
  Array t, dt, mco; // Per-case time, time step, and reference Mach number.
  long nsteps;
  long nreject;       // rejected RK candidates; state is never clipped or committed
  ImplicitStats2D implicit;
  const IB2D* ib;  // nullptr means no immersed body.
  const Cut2D* cut;  // nullptr means no cut-cell body.
};

Params2D params2d_default();

// Generate n+1 face coordinates for n cells on [a,b], preserving both endpoints exactly.
void grid2d_stretch_faces(int n, double a, double b, Stretch s, double* xf);

// Field indexing: i,j are interior indices and may be negative for ghosts.
inline size_t u2idx(const Grid2D& g, int c, int v, int i, int j) {
  return ((static_cast<size_t>(c) * NVAR2D + v) * g.my + (j + g.ng)) * g.mx +
         (i + g.ng);
}

void solver2d_init(Solver2D* s, int nx, int ny, double xmin, double xmax,
                   double ymin, double ymax, int ncase, const Params2D& par);

void solver2d_set_primitive(Solver2D* s, int c, int i, int j, double rho,
                            double u, double v, double p);
void solver2d_get_primitive(const Solver2D* s, int c, int i, int j, double* rho,
                            double* u, double* v, double* p);
void solver2d_set_mco(Solver2D* s, int c, double mco);

// Attach IB data without copying; the caller must keep it alive longer than the solver.
void solver2d_set_ib(Solver2D* s, const IB2D* ib);

// Attach cut-cell data without copying; it cannot be combined with IB data.
void solver2d_set_cut(Solver2D* s, const Cut2D* cut);

// Build cut-cell geometry and SRD neighborhoods from a closed solid polygon; return 0 on success.
// Legacy-grid restriction: offset edges coincident with grid lines by 1e-9 and keep three cells of boundary clearance.
int cut2d_build_polygon(const Solver2D* s, const double* px, const double* py, int n, Cut2D* cut,
                        const int* edge_patch_ids = nullptr);
// Surface pressure follows the existing residual's one-pressure-per-cut-cell rule.
// patch_id=-1 selects all patches; xref/yref are the moment reference point.
int cut2d_surface_integral(Solver2D* s, int c, int patch_id, double xref, double yref,
                           SurfaceIntegral2D* result);
int cut2d_write_surface_vtp(Solver2D* s, int c, const char* path);
int cut2d_write_surface_integrals(Solver2D* s, int c, const char* path,
                                 double xref, double yref);

// Cut-cell residual, called by solver2d_residual when s->cut is non-null.
void cut2d_residual(const Solver2D* s, const double* U, double* R, double* W, double* SX,
                    double* SY, double* FX, double* FY);

// Apply Berger & Giuliani (2020) state redistribution in place after an RK stage, for all cases.
void cut2d_srd(const Solver2D* s, double* U);

// Pressure force on the body: sum p_w*Aw, where Aw points into solid.
void cut2d_wall_force(const Solver2D* s, int c, double* fx, double* fy);

// Build IB data from a closed solid polygon: SDF classification, ghosts, and image-point stencils.
// Return 0 on success. Geometry is built once on CPU; RK stages only interpolate stencils.
int ib2d_build_polygon(const Solver2D* s, const double* px, const double* py, int n, IB2D* ib);

// Sample density/pressure at wall point (wx,wy), with fluid-directed normal (nx,ny), using probes at 1.5h and 2.5h,
// then linearly extrapolate the interpolated fluid values to the wall; return 0 on success.
int ib2d_wall_state(const Solver2D* s, int c, double wx, double wy, double nx, double ny,
                    double* rho, double* p);

// Integrate wall pressure by sampling each polygon segment and interpolating surrounding fluid cells.
// Return body force (fx,fy), with units pressure*length, for case c.
void ib2d_wall_force(const Solver2D* s, int c, const double* px, const double* py, int n, double* fx, double* fy);

// Advance to tend. Return 0 success, 1 maxsteps, 2 nonphysical, 3 unsupported/invalid.
// 4 implicit inner iteration did not converge (unaccepted physical step is rolled back).
int solver2d_run(Solver2D* s, double tend, long maxsteps);
int solver2d_run_implicit(Solver2D* s, double tend, long maxsteps);

// CUDA backend is optional at build time.  It currently implements the fp64,
// first-order regular-grid and cut-cell+SRD paths used by the aircraft run.
// Return codes match solver2d_run; 3 means the CUDA backend/runtime is unavailable
// or the requested configuration is not implemented.
int solver2d_run_cuda(Solver2D* s, double tend, long maxsteps);
int solver2d_run_backend(Solver2D* s, double tend, long maxsteps, int backend);
bool solver2d_cuda_available(char* reason, size_t reason_size);

// Integrate a conserved variable (sum U*dx*dy) to check conservation.
double solver2d_integral(const Solver2D* s, int c, int var);

// Public for unit tests; boundary conditions must already be applied to U.
void solver2d_apply_bc(const Solver2D* s, double* U);
void solver2d_residual(const Solver2D* s, const double* U, double* R, double* W,
                       double* SX, double* SY, double* FX, double* FY);

}  // namespace cfd

#endif  // CFD_EULER2D_H
