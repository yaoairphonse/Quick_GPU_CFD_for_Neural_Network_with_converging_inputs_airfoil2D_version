#ifndef CFD_SA_MODEL_H
#define CFD_SA_MODEL_H
#include <cmath>
#ifdef __CUDACC__
#define SA_HD __host__ __device__
#else
#define SA_HD
#endif
namespace cfd {
// NASA TMR SA-neg, including ft2 and Allmaras-2012 modified S (Note 1c).
// No compressibility/rotation/roughness corrections; no clipping of nutilde.
constexpr double sa_cb1=.1355,sa_cb2=.622,sa_sigma=2.0/3,sa_kappa=.41;
constexpr double sa_cw1=sa_cb1/(sa_kappa*sa_kappa)+(1+sa_cb2)/sa_sigma;
struct SaModelPoint {double fv1,modified_vorticity,fw,source,source_rate,diffusion;};
SA_HD inline double sa_eddy_kinematic(double working,double molecular){
 if(working<=0)return 0;
 const double chi=working/molecular,cube=chi*chi*chi;
 return working*cube/(cube+7.1*7.1*7.1);
}
SA_HD inline SaModelPoint sa_neg_point(double working,double molecular,double distance,double omega){
 SaModelPoint a{};const double chi=working/molecular,cube=chi*chi*chi,d2=distance*distance;
 if(working<0){const double fn=(16+cube)/(16-cube);a.diffusion=(molecular+working*fn)/sa_sigma;
  a.source=sa_cb1*(1-1.2)*omega*working+sa_cw1*working*working/d2;
  a.source_rate=sa_cb1*.2*omega+2*sa_cw1*fabs(working)/d2;return a;}
 a.fv1=cube/(cube+7.1*7.1*7.1);const double fv2=1-chi/(1+chi*a.fv1);
 const double sbar=working*fv2/(sa_kappa*sa_kappa*d2);
 // Equivalent Note1c form avoids underflow in omega*N for subnormal vorticity.
 a.modified_vorticity=sbar>=-.7*omega?omega+sbar:omega*((-.01*omega-.1*sbar)/(-.5*omega-sbar));
 const double r=a.modified_vorticity==0?10:fmin(working/(a.modified_vorticity*sa_kappa*sa_kappa*d2),10.);
 const double r6=r*r*r*r*r*r,g=r+.3*(r6-r),g6=g*g*g*g*g*g;
 a.fw=g*pow(65/(g6+64),1.0/6);const double ft2=1.2*exp(-.5*chi*chi);
 const double production=sa_cb1*(1-ft2)*a.modified_vorticity,destruction=sa_cw1*a.fw-sa_cb1*ft2/(sa_kappa*sa_kappa);
 a.source=production*working-destruction*working*working/d2;
 a.source_rate=fabs(production)+2*fabs(destruction)*working/d2;
 a.diffusion=(molecular+working)/sa_sigma;return a;
}
SA_HD inline double sa_density_corrected_source(const SaModelPoint& a,double rho,double gx,double gy,double rx,double ry){
 return rho*(a.source+sa_cb2/sa_sigma*(gx*gx+gy*gy))-a.diffusion*(rx*gx+ry*gy);
}
} // namespace cfd
#undef SA_HD
#endif
