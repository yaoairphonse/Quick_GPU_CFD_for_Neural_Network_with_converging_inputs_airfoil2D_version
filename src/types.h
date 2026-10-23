// types.h -- Shared option enumerations for the 1D and 2D solvers.
#ifndef CFD_TYPES_H
#define CFD_TYPES_H

namespace cfd {

enum FluxType { FLUX_HLLC = 0, FLUX_AUSMPUP = 1 };

enum LimiterType { LIM_NONE = 0, LIM_MINMOD = 1, LIM_VENKAT = 2, LIM_FIRST_ORDER = 3 };

// Boundary conditions:
//   PERIODIC     Periodic.
//   TRANSMISSIVE Zero-gradient extrapolation; appropriate for outflow or before waves reach the boundary.
//   REFLECT      Slip wall/symmetry: mirror the interior state and reverse normal momentum.
enum BCType { BC_PERIODIC = 0, BC_TRANSMISSIVE = 1, BC_REFLECT = 2 };

}  // namespace cfd

#endif  // CFD_TYPES_H
