#ifndef CFD_STATIC_AMR2D_VISCOUS_H
#define CFD_STATIC_AMR2D_VISCOUS_H
#include <cmath>
#include <string>
#include <vector>
#ifdef __CUDACC__
#define AMR_VIS_HD __host__ __device__
#else
#define AMR_VIS_HD
#endif
namespace cfd {
enum AmrFlowModel { AMR_MODEL_EULER=0, AMR_MODEL_LAMINAR=1 };
enum AmrViscosityLaw { AMR_MU_CONSTANT=0, AMR_MU_SUTHERLAND=1 };
enum AmrViscousStabilization { AMR_VISCOUS_VOLUME=0, AMR_VISCOUS_SRD=1 };
struct AmrTransport2D {
  int model=AMR_MODEL_EULER,law=AMR_MU_CONSTANT;
  int stabilization=AMR_VISCOUS_VOLUME;
  double mu_ref=0,gas_constant=1,temperature_ref=1,sutherland=0,viscous_cfl=.1;
  int sa_model=0; // 0 off, 1 NASA SA-neg (ft2, Note1c), mechanical coupling, kappa=0
  double sa_nu_inf=0,sa_cfl=.1;
  double wall_probe_min_h=0; // optional geometric donor selection, never distance clipping
};
struct AmrGradientTerm2D {int source,side;double x,y;}; // source >=0 neighbor; -1 body; -2 exterior wall
struct AmrWallSample2D {int donor;double distance;};
struct AmrViscousFaceGeometry2D {double dx,dy,ex,ey,distance;};
struct AmrViscousTable2D {
  std::vector<size_t> offsets;
  std::vector<AmrGradientTerm2D> terms;
  std::vector<AmrWallSample2D> wall_samples,face_samples;
  std::vector<AmrViscousFaceGeometry2D> face_geometry;
  std::vector<double> wall_distance;
};
struct AmrViscousFields2D {
  std::vector<double> gradient,mu,face_flux,wall_flux,residual,cell_dt; // gradient SoA ux,uy,vx,vy
};
AMR_VIS_HD inline bool amr_vis_finite(double x){
#ifdef __CUDA_ARCH__
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}
AMR_VIS_HD inline bool amr_transport_valid(const AmrTransport2D& t){
  if(t.model==AMR_MODEL_EULER)return t.sa_model==0;
  return (t.sa_model==0||(t.sa_model==1&&amr_vis_finite(t.sa_nu_inf)&&t.sa_nu_inf>=0&&amr_vis_finite(t.sa_cfl)&&t.sa_cfl>0&&t.sa_cfl<=.1))&&t.model==AMR_MODEL_LAMINAR&&(t.law==AMR_MU_CONSTANT||t.law==AMR_MU_SUTHERLAND)&&
    amr_vis_finite(t.mu_ref)&&t.mu_ref>0&&amr_vis_finite(t.gas_constant)&&t.gas_constant>0&&
    amr_vis_finite(t.temperature_ref)&&t.temperature_ref>0&&amr_vis_finite(t.sutherland)&&t.sutherland>=0&&
    amr_vis_finite(t.viscous_cfl)&&t.viscous_cfl>0&&t.viscous_cfl<=.2&&
    (t.stabilization==AMR_VISCOUS_VOLUME||t.stabilization==AMR_VISCOUS_SRD)&&
    amr_vis_finite(t.wall_probe_min_h)&&t.wall_probe_min_h>=0&&t.wall_probe_min_h<=2;
}
AMR_VIS_HD inline double amr_dynamic_viscosity(const AmrTransport2D& t,double temperature){
  if(!(temperature>0)||!amr_vis_finite(temperature))return -1;
  if(t.law==AMR_MU_CONSTANT)return t.mu_ref;
  const double ratio=temperature/t.temperature_ref;
  return t.mu_ref*ratio*sqrt(ratio)*(t.temperature_ref+t.sutherland)/(temperature+t.sutherland);
}
AMR_VIS_HD inline void amr_stress(double mu,const double* grad,double* stress){
  const double dil=(2.0/3)*(grad[0]+grad[3]);
  stress[0]=mu*(2*grad[0]-dil);stress[1]=mu*(grad[1]+grad[2]);stress[2]=mu*(2*grad[3]-dil);
}
AMR_VIS_HD inline void amr_viscous_flux(double mu,const double* grad,double nx,double ny,double u,double v,double* f){
  double tau[3];amr_stress(mu,grad,tau);f[0]=0;f[1]=tau[0]*nx+tau[1]*ny;f[2]=tau[1]*nx+tau[2]*ny;f[3]=u*f[1]+v*f[2];
}
AMR_VIS_HD inline void amr_correct_gradient(const double* left,const double* right,double dx,double dy,
                                            double du,double dv,double* grad){
  const double d2=dx*dx+dy*dy;
  for(int q=0;q<4;++q)grad[q]=.5*(left[q]+right[q]);
  const double cu=(du-grad[0]*dx-grad[1]*dy)/d2,cv=(dv-grad[2]*dx-grad[3]*dy)/d2;
  grad[0]+=cu*dx;grad[1]+=cu*dy;grad[2]+=cv*dx;grad[3]+=cv*dy;
}
AMR_VIS_HD inline void amr_wall_gradient(double nx,double ny,double distance,double du,double dv,double* grad){
  grad[0]=nx*du/distance;grad[1]=ny*du/distance;grad[2]=nx*dv/distance;grad[3]=ny*dv/distance;
}
struct StaticAmrMesh2D;struct Params2D;struct StaticAmrState2D;
int static_amr2d_prepare_viscous(StaticAmrMesh2D*,const Params2D&,std::string*);
bool static_amr2d_viscous_ready(const StaticAmrMesh2D&);
int static_amr2d_viscous_evaluate(const StaticAmrMesh2D&,const Params2D&,const std::vector<double>& primitive,AmrViscousFields2D*,const std::vector<double>* conserved=nullptr);
int static_amr2d_viscous_diagnostics(const StaticAmrMesh2D&,const Params2D&,const std::vector<double>& conserved,AmrViscousFields2D*);
int static_amr2d_write_viscous_snapshot(const StaticAmrMesh2D&,const Params2D&,const StaticAmrState2D&,const std::string&,std::string*);
int static_amr2d_write_viscous(const StaticAmrMesh2D&,const Params2D&,const std::string&,std::string*);
}
#undef AMR_VIS_HD
#endif
