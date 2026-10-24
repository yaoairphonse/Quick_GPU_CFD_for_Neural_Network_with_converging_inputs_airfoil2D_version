#ifndef CFD_STATIC_AMR2D_H
#define CFD_STATIC_AMR2D_H

#include "euler2d.h"
#include "static_amr2d_boundary.h"
#include "static_amr2d_viscous.h"
#include <array>
#include <string>
#include <vector>

namespace cfd {
struct RectMetrics2D {
  double volume, cx, cy;
  std::vector<WallFragment2D> walls;
  std::vector<std::array<double,2>> boundary; // only needed for multi-component background leaves
};
// Geometry primitives shared with the existing polygon cut-cell implementation.
int cut2d_rect_metrics(const double* px, const double* py, int n,
                       double x0, double x1, double y0, double y1,
                       const int* patches, RectMetrics2D* result);
int cut2d_rect_components(const double*,const double*,int,double,double,double,double,
                          const int*,std::vector<RectMetrics2D>*);
std::vector<std::array<double,2>> cut2d_open_intervals(const double*,const double*,int,int,double,double,double);
void cut2d_open_segment(const double* px, const double* py, int n, int axis,
                         double coordinate, double lo, double hi, double* length, double* centroid);
double cut2d_rect_wall_distance(const double* px, const double* py, int n,
                                double x0, double x1, double y0, double y1);

struct AmrBox2D { double x0, x1, y0, y1; int level; };
struct StaticAmrOptions2D {
  int nx = 16, ny = 12, levels = 3, buffer = 4;
  double xmin = -4, xmax = 4, ymin = -3, ymax = 3;
  size_t max_cells = 2000000;
  bool agglomerate_small=false; // opt-in: merge connected small finite-volume cells
  std::vector<AmrBox2D> boxes;
};
struct AmrCell2D {
  int ix, iy, span, level;
  double x0, x1, y0, y1, volume, cx, cy;
  size_t wall_begin, wall_count;
};
struct AmrFace2D {
  int left, right, axis, boundary; // -1 outside; normal points +axis
  double coordinate, lo, hi, length, centroid;
};
struct AmrReconstructionTerm2D { int neighbor; double x, y; };
struct StaticAmrMesh2D {
  StaticAmrOptions2D options;
  AmrBoundary2D boundaries[4]; // xmin/xmax/ymin/ymax; immutable during a GPU session
  AmrTransport2D transport;
  AmrViscousTable2D viscous;
  std::vector<double> body_x,body_y; // original polygon for visible fluid-side wall samples
  std::vector<AmrCell2D> cells; // active leaves only, no covered parent states
  // Compact background partition, including solid leaves, for coverage/provenance audits.
  // ix, iy, span, level, first active_cell_id (-1 for solid); leaf_components
  // holds every control volume when a background leaf has disconnected fluid.
  std::vector<std::array<int,5>> partition;
  std::vector<std::array<size_t,2>> leaf_components; // background leaf id, fluid component id
  size_t split_leaves=0;
  std::vector<std::array<double,9>> agglomeration; // old_id,new_id,x0,x1,y0,y1,V,cx,cy
  std::vector<AmrFace2D> faces; // one entry per shared subface
  std::vector<WallFragment2D> walls;
  std::vector<size_t> face_offsets, cell_faces;
  std::vector<size_t> reconstruction_offsets;
  std::vector<AmrReconstructionTerm2D> reconstruction_terms;
  std::vector<size_t> group_offsets, group_members, member_offsets, member_groups;
  std::vector<double> group_volume;
  size_t background_leaves = 0, coarse_fine_faces = 0, small_cells = 0;
};
struct StaticAmrState2D {
  std::vector<double> u; // SoA: u[var * cells + cell]
  double time = 0;
  long steps = 0, rejected = 0;
};
int static_amr2d_build(const StaticAmrOptions2D& options,
                       const std::vector<double>& px, const std::vector<double>& py,
                       const std::vector<int>& patches, StaticAmrMesh2D* mesh, std::string* error);
int static_amr2d_initialize(const StaticAmrMesh2D& mesh, const Params2D& params,
                            const std::array<double,4>& primitive, StaticAmrState2D* state);
int static_amr2d_run(const StaticAmrMesh2D& mesh, const Params2D& params,
                     double tend, long maxsteps, StaticAmrState2D* state, std::string* error);
// Diagnostics call the same operators used by the time integrator.
int static_amr2d_rhs(const StaticAmrMesh2D& mesh, const Params2D& params,
                     const std::vector<double>& u, std::vector<double>* residual,
                     std::vector<double>* integrated_face_flux = nullptr);
int static_amr2d_redistribute(const StaticAmrMesh2D& mesh, std::vector<double>* u);
void static_amr2d_reconstruct(const StaticAmrMesh2D& mesh, const std::vector<double>& primitive,
                              std::vector<double>* slopes);
int static_amr2d_wall_pressure(const StaticAmrMesh2D& mesh, const Params2D& params,
                               const std::vector<double>& conserved, std::vector<double>* pressure);
int static_amr2d_write_boundaries(const StaticAmrMesh2D& mesh, const Params2D& params,
                                  const std::string& prefix, std::string* error);
// Output is geometry-aware: VTU cell outlines are background rectangles; alpha identifies cut volume.
int static_amr2d_write(const StaticAmrMesh2D& mesh, const Params2D& params,
                       const StaticAmrState2D& state, const std::string& prefix, std::string* error,
                       const char* backend="cpu");
} // namespace cfd
#endif
