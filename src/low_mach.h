// Turkel pressure preconditioning in entropy variables, shared by CPU and CUDA.
// Steady pseudo-time only: the physical conservative residual is kept for diagnostics.
// Reference: Turkel & Vatsa (2005), https://doi.org/10.1051/m2an:2005021.
#ifndef CFD_LOW_MACH_H
#define CFD_LOW_MACH_H
#include <cmath>
#ifdef __CUDACC__
#define LM_HD __host__ __device__
#else
#define LM_HD
#endif
namespace cfd {
LM_HD inline double lm_beta2(double speed2,double a2,double reference,double floor) {
  return fmin(1.,fmax(speed2/a2,fmax(reference*reference,floor*floor)));
}
LM_HD inline double lm_spectral(double un,double a,double beta2) {
  return .5*((1+beta2)*fabs(un)+sqrt((1-beta2)*(1-beta2)*un*un+4*beta2*a*a));
}
LM_HD inline double lm_face_speed(const double* l,const double* r,int normal,int energy,
                                  double gm,double reference,double floor) {
  double b=0;
  for(int side=0;side<2;++side){const double* w=side?r:l;double q2=0;
    for(int j=1;j<energy;++j)q2+=w[j]*w[j];
    b=fmax(b,lm_beta2(q2,gm*w[energy]/w[0],reference,floor));}
  b=fmax(b,fmin(1.,fabs(r[energy]-l[energy])/fmin(l[energy],r[energy])));
  return fmax(lm_spectral(l[normal],sqrt(gm*l[energy]/l[0]),b),lm_spectral(r[normal],sqrt(gm*r[energy]/r[0]),b));
}
// Face-normal spectral bound using the same Roe state and pressure sensor as
// lm_flux.  Including both endpoint states prevents an asymmetric face rate
// from underestimating the update of either adjacent control volume.
LM_HD inline double lm_face_speed_normal(const double* l,const double* r,const double* n,
                                         int dim,int energy,double gm,double reference,double floor) {
  double ql=0,qr=0,unl=0,unr=0;
  for(int j=0;j<dim;++j){ql+=l[j+1]*l[j+1];qr+=r[j+1]*r[j+1];unl+=l[j+1]*n[j];unr+=r[j+1]*n[j];}
  const double sl=sqrt(l[0]),sr=sqrt(r[0]),denominator=sl+sr;
  double q2=0,un=0;
  for(int j=0;j<dim;++j){const double velocity=(sl*l[j+1]+sr*r[j+1])/denominator;q2+=velocity*velocity;un+=velocity*n[j];}
  const double hl=gm*l[energy]/((gm-1)*l[0])+.5*ql;
  const double hr=gm*r[energy]/((gm-1)*r[0])+.5*qr;
  const double h=(sl*hl+sr*hr)/denominator,a2=(gm-1)*(h-.5*q2);
  double b=lm_beta2(q2,a2,reference,floor);
  b=fmax(b,lm_beta2(ql,gm*l[energy]/l[0],reference,floor));
  b=fmax(b,lm_beta2(qr,gm*r[energy]/r[0],reference,floor));
  b=fmax(b,fmin(1.,fabs(r[energy]-l[energy])/fmin(l[energy],r[energy])));
  const double center=lm_spectral(un,sqrt(a2),b);
  const double left=lm_spectral(unl,sqrt(gm*l[energy]/l[0]),b);
  const double right=lm_spectral(unr,sqrt(gm*r[energy]/r[0]),b);
  return fmax(center,fmax(left,right));
}
// The dissipation is J P^-1 |P A| J^-1 dU, P=diag(beta^2,1,1,1,1)
// in (p,u,v,w,entropy). Physical central fluxes remain conservative.
LM_HD inline void lm_flux(const double* l,const double* r,const double* n,
                         double gm,double reference,double floor,double* f) {
  double ul[5],ur[5],fl[5],fr[5],unl=0,unr=0,ql=0,qr=0;
  for(int j=0;j<3;++j){unl+=l[j+1]*n[j];unr+=r[j+1]*n[j];ql+=l[j+1]*l[j+1];qr+=r[j+1]*r[j+1];}
  ul[0]=l[0];ur[0]=r[0];ul[4]=l[4]/(gm-1)+.5*l[0]*ql;ur[4]=r[4]/(gm-1)+.5*r[0]*qr;
  fl[0]=l[0]*unl;fr[0]=r[0]*unr;
  for(int j=0;j<3;++j){ul[j+1]=l[0]*l[j+1];ur[j+1]=r[0]*r[j+1];fl[j+1]=ul[j+1]*unl+l[4]*n[j];fr[j+1]=ur[j+1]*unr+r[4]*n[j];}
  fl[4]=(ul[4]+l[4])*unl;fr[4]=(ur[4]+r[4])*unr;
  const double sl=sqrt(l[0]),sr=sqrt(r[0]),rho=sl*sr;
  double u[3],un=0,q2=0,du[3],dm=0;
  for(int j=0;j<3;++j){u[j]=(sl*l[j+1]+sr*r[j+1])/(sl+sr);un+=u[j]*n[j];q2+=u[j]*u[j];dm+=u[j]*(ur[j+1]-ul[j+1]);}
  const double H=(sl*(ul[4]+l[4])/l[0]+sr*(ur[4]+r[4])/r[0])/(sl+sr),a2=(gm-1)*(H-.5*q2),a=sqrt(a2);
  const double dr=ur[0]-ul[0],dp=(gm-1)*(ur[4]-ul[4]-dm+.5*q2*dr);
  double dun=0;for(int j=0;j<3;++j){du[j]=(ur[j+1]-ul[j+1]-u[j]*dr)/rho;dun+=du[j]*n[j];}
  // Pressure jumps restore acoustic dissipation near shocks.
  const double b=fmax(lm_beta2(q2,a2,reference,floor),fmin(1.,fabs(r[4]-l[4])/fmin(l[4],r[4])));
  const double disc=sqrt((1-b)*(1-b)*un*un+4*b*a2),lp=.5*((1+b)*un+disc),lm=.5*((1+b)*un-disc);
  const double c1=(fabs(lp)-fabs(lm))/disc,c0=(lp*fabs(lm)-lm*fabs(lp))/disc;
  const double zp=((c0+c1*b*un)*dp+c1*b*rho*a2*dun)/b;
  const double zu=c1*dp/rho+(c0+c1*un)*dun;
  double d[5],work=0;d[0]=zp/a2+fabs(un)*(dr-dp/a2);
  for(int j=0;j<3;++j){const double dv=zu*n[j]+fabs(un)*(du[j]-dun*n[j]);d[j+1]=u[j]*d[0]+rho*dv;work+=rho*u[j]*dv;}
  d[4]=zp/(gm-1)+.5*q2*d[0]+work;
  for(int j=0;j<5;++j)f[j]=.5*(fl[j]+fr[j]-d[j]);
}
// Apply J P J^-1 to a conservative residual, including passive rho*nutilde.
LM_HD inline void lm_apply(double* r,const double* w,double gm,double reference,double floor,int dim,bool sa) {
  double q2=0,work=0;for(int j=0;j<dim;++j){q2+=w[j+1]*w[j+1];work+=w[j+1]*r[j+1];}
  const int energy=dim+1;const double a2=gm*w[energy]/w[0];
  const double dp=(gm-1)*(r[energy]-work+.5*q2*r[0]);
  const double change=(lm_beta2(q2,a2,reference,floor)-1)*dp/a2;
  r[0]+=change;for(int j=0;j<dim;++j)r[j+1]+=w[j+1]*change;
  r[energy]+=change*(a2/(gm-1)+.5*q2);
  if(sa)r[energy+1]+=w[energy+1]*change;
}
} // namespace cfd
#undef LM_HD
#endif
