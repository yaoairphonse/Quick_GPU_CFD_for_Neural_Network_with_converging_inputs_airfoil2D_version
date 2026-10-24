#ifndef CFD_STATIC_AMR2D_RECONSTRUCTION_H
#define CFD_STATIC_AMR2D_RECONSTRUCTION_H

#include "static_amr2d.h"

namespace cfd {
#ifdef __CUDACC__
#define AMR_RECON_HD __host__ __device__
#else
#define AMR_RECON_HD
#endif
AMR_RECON_HD inline double amr_face_x(const AmrFace2D& f) { return f.axis==0?f.coordinate:f.centroid; }
AMR_RECON_HD inline double amr_face_y(const AmrFace2D& f) { return f.axis==1?f.coordinate:f.centroid; }
AMR_RECON_HD inline double amr_wall_x(const WallFragment2D& w) { return .5*(w.x0+w.x1); }
AMR_RECON_HD inline double amr_wall_y(const WallFragment2D& w) { return .5*(w.y0+w.y1); }
AMR_RECON_HD inline double amr_point_value(const double* primitive,const double* slopes,size_t n,
                                            int variable,size_t cell,const AmrCell2D& c,double x,double y) {
  return primitive[variable*n+cell]+slopes[(2*variable)*n+cell]*(x-c.cx)+slopes[(2*variable+1)*n+cell]*(y-c.cy);
}
AMR_RECON_HD inline double amr_limiter(double center,double lower,double upper,double delta) {
  if(delta>0&&center+delta>upper)return (upper-center)/delta;
  if(delta<0&&center+delta<lower)return (lower-center)/delta;
  return 1;
}
#undef AMR_RECON_HD
}
#endif
