// Persistent AMR CUDA state. Control, CFL and RK rejection stay on-device.
#include "static_amr2d_gpu.h"
#include "static_amr2d_reconstruction.h"
#include "sa_model.h"
#include "low_mach.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include "cuda_euler_flux.cuh"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>

namespace cfd {
namespace {
constexpr int threads=256;
struct Control {
  double time,target,dt;
  long steps,rejected,maxsteps;
  int bad,fatal,retrying,attempt,finished,active;
  double cfl=.15,local_scale=1;
};
struct Config {
  bool precondition=false,local_time=false;double mach_reference=.2,mach_floor=.02;
  size_t n,nf,nw,nv;
  double gamma,cfl,fixed_dt,mco;
  int flux,bc[4],small,agglomerated,spatial_order;
  AmrBoundary2D boundary[4];
  AmrTransport2D transport;
};
struct Data {
  AmrCell2D* cells=nullptr;
  AmrFace2D* faces=nullptr;
  WallFragment2D* walls=nullptr;
  size_t *offset=nullptr,*face_ids=nullptr,*goff=nullptr,*members=nullptr,*moff=nullptr,*groups=nullptr;
  double *gvol=nullptr,*u=nullptr,*u1=nullptr,*u2=nullptr,*w=nullptr,*r0=nullptr,*r=nullptr;
  size_t* reconstruction_offsets=nullptr;AmrReconstructionTerm2D* reconstruction_terms=nullptr;double* slopes=nullptr;
  double *flux=nullptr,*speed=nullptr,*cell_dt=nullptr,*qhat=nullptr;
  size_t* visc_offsets=nullptr;
  AmrGradientTerm2D* visc_terms=nullptr;
  AmrWallSample2D *wall_samples=nullptr,*face_samples=nullptr;
  AmrViscousFaceGeometry2D* face_geometry=nullptr;
  double *gradient=nullptr,*mu=nullptr,*wall_flux=nullptr,*face_coefficient=nullptr,*wall_coefficient=nullptr;
  double *diffusion_rate=nullptr,*diffusion_group=nullptr;
  double *sa_phi=nullptr,*eddy=nullptr,*wall_distance=nullptr,*sa_grad=nullptr,*sa_diff=nullptr,*sa_source=nullptr,*sa_source_rate=nullptr;
  double *sa_face_flux=nullptr,*sa_face_coefficient=nullptr,*sa_wall_flux=nullptr,*sa_wall_coefficient=nullptr;
  Control* control=nullptr;
};
int blocks(size_t n){return static_cast<int>((n+threads-1)/threads);}
__device__ bool live(const Control* c){return c->active&&!c->fatal;}
__device__ bool healthy(const Control* c){return live(c)&&!c->bad;}
#include "static_amr2d_sa_cuda.cuh"

__global__ void target_kernel(Control* c,double target,long maxsteps) {
  c->target=target;c->maxsteps=maxsteps;c->finished=c->time>=target;
}
__global__ void cfl_kernel(Control* c,double value){c->cfl=value;}
__global__ void begin_kernel(Control* c) {
  c->active=0;if(c->fatal||c->finished)return;
  if(c->time>=c->target){c->finished=1;return;}
  if(c->steps>=c->maxsteps){c->fatal=3;return;}
  c->bad=0;c->active=1;
}
__global__ void primitive_kernel(Config cfg,Data d,const double* u,bool base) {
  const size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=cfg.n||!healthy(d.control))return;
  const size_t n=cfg.n;const double rho=u[k],mx=u[n+k],my=u[2*n+k],energy=u[3*n+k];
  const double vx=mx/rho,vy=my/rho,p=(cfg.gamma-1)*(energy-.5*(mx*vx+my*vy));
  if(!isfinite(rho)||!(rho>0)||!isfinite(mx)||!isfinite(my)||!isfinite(energy)||!isfinite(p)||!(p>0)) {
    atomicOr(&d.control->bad,1);if(base)atomicExch(&d.control->fatal,2);return;
  }
  d.w[k]=rho;d.w[n+k]=vx;d.w[2*n+k]=vy;d.w[3*n+k]=p;
  if(cfg.transport.sa_model){d.sa_phi[k]=u[4*n+k]/rho;if(!isfinite(d.sa_phi[k])){atomicOr(&d.control->bad,1);if(base)atomicExch(&d.control->fatal,2);}}
}
__global__ void reconstruction_kernel(Config cfg,Data d) {
  const size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=cfg.n||!healthy(d.control))return;
  const auto c=d.cells[k];
  for(int v=0;v<4;++v){const double center=d.w[v*cfg.n+k];double gx=0,gy=0,lower=center,upper=center;
    for(size_t pos=d.reconstruction_offsets[k];pos<d.reconstruction_offsets[k+1];++pos){const auto t=d.reconstruction_terms[pos];const double other=d.w[v*cfg.n+t.neighbor],diff=other-center;
      gx+=t.x*diff;gy+=t.y*diff;lower=fmin(lower,other);upper=fmax(upper,other);}
    if(v==0||v==3)lower=fmax(lower,.1*center);
    double theta=1;
    for(size_t pos=d.offset[k];pos<d.offset[k+1];++pos){const auto f=d.faces[d.face_ids[pos]];
      theta=fmin(theta,amr_limiter(center,lower,upper,gx*(amr_face_x(f)-c.cx)+gy*(amr_face_y(f)-c.cy)));}
    for(size_t pos=c.wall_begin;pos<c.wall_begin+c.wall_count;++pos){const auto wall=d.walls[pos];
      theta=fmin(theta,amr_limiter(center,lower,upper,gx*(amr_wall_x(wall)-c.cx)+gy*(amr_wall_y(wall)-c.cy)));}
    d.slopes[(2*v)*cfg.n+k]=theta*gx;d.slopes[(2*v+1)*cfg.n+k]=theta*gy;
  }
}
template<bool JoinSA>
__global__ void velocity_gradient_kernel(Config cfg,Data d) {
  const size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=cfg.n||!healthy(d.control))return;
  d.mu[k]=amr_dynamic_viscosity(cfg.transport,d.w[3*cfg.n+k]/(d.w[k]*cfg.transport.gas_constant));
  if(!(d.mu[k]>0)||!isfinite(d.mu[k])){atomicOr(&d.control->bad,1);return;}
  if(cfg.transport.sa_model)d.eddy[k]=d.w[k]*sa_eddy_kinematic(d.sa_phi[k],d.mu[k]/d.w[k]);
  double grad[4]={},gx=0,gy=0,rx=0,ry=0;
  const double density=d.w[k],working=JoinSA?d.sa_phi[k]:0,molecular=d.mu[k];
  for(size_t pos=d.visc_offsets[k];pos<d.visc_offsets[k+1];++pos){const auto term=d.visc_terms[pos];double u=0,v=0;
    if(term.source>=0){u=d.w[cfg.n+term.source];v=d.w[2*cfg.n+term.source];}
    else if(term.source==-2){u=cfg.boundary[term.side].wall_velocity[0];v=cfg.boundary[term.side].wall_velocity[1];}
    const double du=u-d.w[cfg.n+k],dv=v-d.w[2*cfg.n+k];grad[0]+=term.x*du;grad[1]+=term.y*du;grad[2]+=term.x*dv;grad[3]+=term.y*dv;
    if constexpr(JoinSA){const double phi=term.source>=0?d.sa_phi[term.source]:0,rho=term.source>=0?d.w[term.source]:density;
      gx+=term.x*(phi-working);gy+=term.y*(phi-working);rx+=term.x*(rho-density);ry+=term.y*(rho-density);}}
  for(int v=0;v<4;++v)d.gradient[v*cfg.n+k]=grad[v];
  if constexpr(JoinSA){d.sa_grad[k]=gx;d.sa_grad[cfg.n+k]=gy;
    const auto model=sa_neg_point(working,molecular/density,d.wall_distance[k],fabs(grad[2]-grad[1]));
    d.sa_diff[k]=model.diffusion;d.sa_source[k]=sa_density_corrected_source(model,density,gx,gy,rx,ry);d.sa_source_rate[k]=model.source_rate;
    if(!(model.diffusion>0)||!isfinite(d.sa_source[k])||!isfinite(model.source_rate))atomicOr(&d.control->bad,1);}
}
__device__ void viscous_face(Config cfg,Data d,size_t q,double* flux) {
  const auto f=d.faces[q];const auto geo=d.face_geometry[q];const int k=f.left>=0?f.left:f.right;double grad[4],mu=d.mu[k]+(cfg.transport.sa_model?d.eddy[k]:0),uf=d.w[cfg.n+k],vf=d.w[2*cfg.n+k],nu=mu/d.w[k],distance=geo.distance;
  for(int v=0;v<4;++v)grad[v]=d.gradient[v*cfg.n+k];
  if(f.boundary<0){const int l=f.left,r=f.right;double gl[4],gr[4];for(int v=0;v<4;++v){gl[v]=d.gradient[v*cfg.n+l];gr[v]=d.gradient[v*cfg.n+r];}
    const double dx=geo.dx,dy=geo.dy;
    amr_correct_gradient(gl,gr,dx,dy,d.w[cfg.n+r]-d.w[cfg.n+l],d.w[2*cfg.n+r]-d.w[2*cfg.n+l],grad);
    const double ex=geo.ex,ey=geo.ey;
    uf=.5*(d.w[cfg.n+l]+d.w[cfg.n+r])+grad[0]*ex+grad[1]*ey;vf=.5*(d.w[2*cfg.n+l]+d.w[2*cfg.n+r])+grad[2]*ex+grad[3]*ey;
    const double temp=.5*(d.w[3*cfg.n+l]/d.w[l]+d.w[3*cfg.n+r]/d.w[r])/cfg.transport.gas_constant;mu=amr_dynamic_viscosity(cfg.transport,temp)+(cfg.transport.sa_model?.5*(d.eddy[l]+d.eddy[r]):0);nu=fmax((d.mu[l]+(cfg.transport.sa_model?d.eddy[l]:0))/d.w[l],(d.mu[r]+(cfg.transport.sa_model?d.eddy[r]:0))/d.w[r]);
  }else {distance=d.face_samples[q].distance;const int bc=cfg.boundary[f.boundary].kind<0?cfg.bc[f.boundary]:cfg.boundary[f.boundary].kind;const double sign=f.boundary%2?1:-1;
    if(bc==AMR_BC_NOSLIP){mu=d.mu[k];uf=cfg.boundary[f.boundary].wall_velocity[0];vf=cfg.boundary[f.boundary].wall_velocity[1];amr_wall_gradient(f.axis==0?sign:0,f.axis==1?sign:0,distance,uf-d.w[cfg.n+k],vf-d.w[2*cfg.n+k],grad);}
    else if(bc==AMR_BC_REFLECT){grad[1]=0;grad[2]=0;if(f.axis==0)uf=0;else vf=0;}
    else {grad[f.axis]=0;grad[2+f.axis]=0;uf+=grad[0]*geo.ex+grad[1]*geo.ey;vf+=grad[2]*geo.ex+grad[3]*geo.ey;}
  }
  amr_viscous_flux(mu,grad,f.axis==0?1:0,f.axis==1?1:0,uf,vf,flux);d.face_coefficient[q]=(4.0/3)*nu/distance;
}
__device__ void viscous_wall_point(Config cfg,Data d,size_t q) {
  const auto wall=d.walls[q];const auto sample=d.wall_samples[q];const int k=sample.donor,owner=wall.owner_cell;double grad[4],flux[4];
  amr_wall_gradient(wall.nx,wall.ny,sample.distance,-d.w[cfg.n+k],-d.w[2*cfg.n+k],grad);amr_viscous_flux(d.mu[k],grad,wall.nx,wall.ny,0,0,flux);
  for(int v=0;v<4;++v){d.wall_flux[v*cfg.nw+q]=flux[v]*wall.length;if(!isfinite(flux[v]))atomicOr(&d.control->bad,1);}
  d.wall_coefficient[q]=(4.0/3)*fmax(d.mu[k]/d.w[k],d.mu[owner]/d.w[owner])/sample.distance;
}
__global__ void viscous_wall_kernel(Config cfg,Data d) {
  const size_t q=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(q<cfg.nw&&healthy(d.control))viscous_wall_point(cfg,d,q);
}
__device__ void flux_point(Config cfg,Data d,size_t q) {
  const auto face=d.faces[q];const int inside=face.left>=0?face.left:face.right;
  double a[4],b[4];for(int v=0;v<4;++v){
    a[v]=cfg.spatial_order==2&&face.left>=0?amr_point_value(d.w,d.slopes,cfg.n,v,face.left,d.cells[face.left],amr_face_x(face),amr_face_y(face)):d.w[v*cfg.n+(face.left>=0?face.left:inside)];
    b[v]=cfg.spatial_order==2&&face.right>=0?amr_point_value(d.w,d.slopes,cfg.n,v,face.right,d.cells[face.right],amr_face_x(face),amr_face_y(face)):d.w[v*cfg.n+(face.right>=0?face.right:inside)];}
  if(face.boundary>=0) {
    if(!amr_bc_state(cfg.boundary[face.boundary],cfg.bc[face.boundary],cfg.gamma,face.boundary,
                     face.left<0?b:a,face.left<0?a:b)){atomicOr(&d.control->bad,1);return;}
  }
  const int normal=1+face.axis,tangent=2-face.axis;double f[4];
  const int bc=face.boundary<0?-1:(cfg.boundary[face.boundary].kind<0?cfg.bc[face.boundary]:cfg.boundary[face.boundary].kind);
  if(bc==AMR_BC_NOSLIP){f[0]=0;f[1]=d.w[3*cfg.n+inside];f[2]=0;f[3]=0;}
  else if(cfg.precondition){const double al[5]={a[0],a[normal],a[tangent],0,a[3]},br[5]={b[0],b[normal],b[tangent],0,b[3]},nn[3]={1,0,0};double f5[5];
    lm_flux(al,br,nn,cfg.gamma,cfg.mach_reference,cfg.mach_floor,f5);f[0]=f5[0];f[1]=f5[1];f[2]=f5[2];f[3]=f5[4];}
  else if(cfg.flux==FLUX_HLLC)gpu_flux::flux_hllc(a[0],a[normal],a[tangent],a[3],b[0],b[normal],b[tangent],b[3],cfg.gamma,f);
  else gpu_flux::flux_ausm(a[0],a[normal],a[tangent],a[3],b[0],b[normal],b[tangent],b[3],cfg.gamma,cfg.mco,f);
  for(int v=0;v<4;++v)if(!isfinite(f[v])){atomicOr(&d.control->bad,1);return;}
  d.flux[q]=f[0]*face.length;d.flux[normal*cfg.nf+q]=f[1]*face.length;
  d.flux[tangent*cfg.nf+q]=f[2]*face.length;d.flux[3*cfg.nf+q]=f[3]*face.length;
  if(cfg.transport.model==AMR_MODEL_LAMINAR){double fv[4];viscous_face(cfg,d,q,fv);for(int v=0;v<4;++v){if(!isfinite(fv[v]))atomicOr(&d.control->bad,1);d.flux[v*cfg.nf+q]-=fv[v]*face.length;}}
  d.speed[q]=fmax(fabs(a[normal])+sqrt(cfg.gamma*a[3]/a[0]),fabs(b[normal])+sqrt(cfg.gamma*b[3]/b[0]));
  if(cfg.precondition)d.speed[q]=lm_face_speed(a,b,normal,3,cfg.gamma,cfg.mach_reference,cfg.mach_floor);
}
__global__ void flux_kernel(Config cfg,Data d) {
  const size_t q=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(q<cfg.nf&&healthy(d.control))flux_point(cfg,d,q);
}
template<bool JoinSA>
__global__ void flux_wall_kernel(Config cfg,Data d) {
  const size_t q=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(!healthy(d.control))return;
  if(q<cfg.nf)flux_point(cfg,d,q);else if(q<cfg.nf+cfg.nw)viscous_wall_point(cfg,d,q-cfg.nf);
  if constexpr(JoinSA)sa_face_wall_point(cfg,d,q);
}
template<bool JoinSA>
__global__ void collect_kernel(Config cfg,Data d,double* r,bool base) {
  const size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=cfg.n||!healthy(d.control))return;
  const auto cell=d.cells[k];double sum[4]={},spectral=0,diffusion=0,sa_net=0,sa_spectral=0;
  for(size_t pos=d.offset[k];pos<d.offset[k+1];++pos){const size_t q=d.face_ids[pos];const double sign=d.faces[q].left==static_cast<int>(k)?1.0:-1.0;
    for(int v=0;v<4;++v)sum[v]+=sign*d.flux[v*cfg.nf+q];
    if constexpr(JoinSA){sa_net+=sign*d.sa_face_flux[q];if(base)sa_spectral+=d.sa_face_coefficient[q];}
    if(base){spectral+=d.speed[q]*d.faces[q].length;if(cfg.transport.model==AMR_MODEL_LAMINAR)diffusion+=d.face_coefficient[q]*d.faces[q].length;}
  }
  const double pressure=d.w[3*cfg.n+k],sound=sqrt(cfg.gamma*pressure/d.w[k]);
  const double a=cfg.precondition?sound*sqrt(lm_beta2(d.w[cfg.n+k]*d.w[cfg.n+k]+d.w[2*cfg.n+k]*d.w[2*cfg.n+k],sound*sound,cfg.mach_reference,cfg.mach_floor)):sound;
  for(size_t q=cell.wall_begin;q<cell.wall_begin+cell.wall_count;++q){const auto wall=d.walls[q];const double wall_pressure=cfg.spatial_order==2?amr_point_value(d.w,d.slopes,cfg.n,3,k,cell,amr_wall_x(wall),amr_wall_y(wall)):pressure;
    sum[1]+=wall_pressure*wall.nx*wall.length;sum[2]+=wall_pressure*wall.ny*wall.length;if(base)spectral+=a*wall.length;if(cfg.transport.model==AMR_MODEL_LAMINAR){for(int v=0;v<4;++v)sum[v]-=d.wall_flux[v*cfg.nw+q];if(base)diffusion+=d.wall_coefficient[q]*wall.length;}
    if constexpr(JoinSA){sa_net-=d.sa_wall_flux[q];if(base)sa_spectral+=d.sa_wall_coefficient[q];}}
  for(int v=0;v<4;++v)r[v*cfg.n+k]=-sum[v]/cell.volume;
  if constexpr(JoinSA){r[4*cfg.n+k]=-sa_net/cell.volume+d.sa_source[k];if(!isfinite(r[4*cfg.n+k]))atomicOr(&d.control->bad,1);}
  if(base){d.cell_dt[k]=spectral>0?d.control->cfl*(cfg.agglomerated?cell.volume:(cell.x1-cell.x0)*(cell.y1-cell.y0))/spectral:CUDART_INF;
    if(cfg.transport.model==AMR_MODEL_LAMINAR){d.diffusion_rate[k]=diffusion/cell.volume;
      if(cfg.transport.stabilization==AMR_VISCOUS_VOLUME&&diffusion>0)d.cell_dt[k]=fmin(d.cell_dt[k],cfg.transport.viscous_cfl*cell.volume/diffusion);}
    if constexpr(JoinSA){const double rate=2*sa_spectral/(d.w[k]*cell.volume)+d.sa_source_rate[k];if(rate>0)d.cell_dt[k]=fmin(d.cell_dt[k],cfg.transport.sa_cfl/rate);}}
}
__global__ void diffusion_group_kernel(Config cfg,Data d) {
  const size_t g=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(g>=cfg.n||!healthy(d.control))return;
  double sum=0;for(size_t pos=d.goff[g];pos<d.goff[g+1];++pos){const size_t k=d.members[pos];sum+=d.cells[k].volume*d.diffusion_rate[k]/(d.moff[k+1]-d.moff[k]);}
  d.diffusion_group[g]=sum/d.gvol[g];
}
__global__ void precondition_kernel(Config cfg,Data d,double* rhs) {
  const size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=cfg.n||!healthy(d.control))return;
  double rr[5],ww[5]={d.w[k],d.w[cfg.n+k],d.w[2*cfg.n+k],d.w[3*cfg.n+k],cfg.transport.sa_model?d.sa_phi[k]:0};
  for(int v=0;v<cfg.nv;++v)rr[v]=rhs[v*cfg.n+k];
  lm_apply(rr,ww,cfg.gamma,cfg.mach_reference,cfg.mach_floor,2,cfg.transport.sa_model);
  for(int v=0;v<cfg.nv;++v)rhs[v*cfg.n+k]=rr[v];
}
__global__ void choose_dt_kernel(Config cfg,Data d) {
  __shared__ double values[threads];double dt=CUDART_INF;
  if(healthy(d.control))for(size_t k=threadIdx.x;k<cfg.n;k+=blockDim.x){double local=d.cell_dt[k];
    if(cfg.transport.model==AMR_MODEL_LAMINAR&&cfg.transport.stabilization==AMR_VISCOUS_SRD){double rate=d.diffusion_rate[k];
      if(cfg.small){double sum=0;for(size_t pos=d.moff[k];pos<d.moff[k+1];++pos)sum+=d.diffusion_group[d.groups[pos]];rate=sum/(d.moff[k+1]-d.moff[k]);}
      if(rate>0)local=fmin(local,cfg.transport.viscous_cfl/rate);}
    dt=fmin(dt,local);}
  values[threadIdx.x]=dt;__syncthreads();
  for(int stride=threads/2;stride>0;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]=fmin(values[threadIdx.x],values[threadIdx.x+stride]);__syncthreads();}
  if(threadIdx.x||!live(d.control))return;
  auto* c=d.control;
  if(c->bad){c->fatal=2;return;}
  if(!c->retrying){c->dt=cfg.fixed_dt>0?cfg.fixed_dt:values[0];c->attempt=0;}
  const double remaining=c->target-c->time;c->dt=fmin(c->dt,remaining);
  const double tol=32*DBL_EPSILON*fmax(fmax(fabs(c->target),fabs(c->time)),c->dt);
  if(remaining-c->dt<=tol)c->dt=remaining;
  c->local_scale=c->dt/values[0];
  if(!isfinite(c->dt)||!(c->dt>0)||c->time+c->dt==c->time)c->fatal=4;
}
__global__ void rk_kernel(Config cfg,Data d,int stage) {
  const size_t q=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(q>=cfg.nv*cfg.n||!healthy(d.control))return;const double dt=cfg.local_time?d.cell_dt[q%cfg.n]*d.control->local_scale:d.control->dt;
  if(stage==0)d.u1[q]=d.u[q]+dt*d.r0[q];
  else if(stage==1)d.u2[q]=.75*d.u[q]+.25*(d.u1[q]+dt*d.r[q]);
  else d.u1[q]=d.u[q]/3+(2.0/3)*(d.u2[q]+dt*d.r[q]);
}
__global__ void group_kernel(Config cfg,Data d,const double* u) {
  const size_t g=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(g>=cfg.n||!healthy(d.control))return;
  for(int v=0;v<4;++v){double sum=0;for(size_t pos=d.goff[g];pos<d.goff[g+1];++pos){const size_t k=d.members[pos];sum+=d.cells[k].volume*u[v*cfg.n+k]/(d.moff[k+1]-d.moff[k]);}d.qhat[v*cfg.n+g]=sum/d.gvol[g];}
}
__global__ void redistribute_kernel(Config cfg,Data d,double* u,bool convert) {
  const size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=cfg.n||!healthy(d.control))return;
  for(int v=0;v<4;++v){double sum=0;for(size_t pos=d.moff[k];pos<d.moff[k+1];++pos)sum+=d.qhat[v*cfg.n+d.groups[pos]];u[v*cfg.n+k]=sum/(d.moff[k+1]-d.moff[k]);}
  if(convert){const double rho=u[k],vx=u[cfg.n+k]/rho,vy=u[2*cfg.n+k]/rho,p=(cfg.gamma-1)*(u[3*cfg.n+k]-.5*(u[cfg.n+k]*vx+u[2*cfg.n+k]*vy));
    if(!isfinite(rho)||!(rho>0)||!isfinite(vx)||!isfinite(vy)||!isfinite(p)||!(p>0)){atomicOr(&d.control->bad,1);return;}
    d.w[k]=rho;d.w[cfg.n+k]=vx;d.w[2*cfg.n+k]=vy;d.w[3*cfg.n+k]=p;}
}
__global__ void commit_state_kernel(Config cfg,Data d) {
  const size_t q=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(q<cfg.nv*cfg.n&&healthy(d.control))d.u[q]=d.u1[q];
}
__global__ void commit_control_kernel(Control* c) {
  if(!live(c))return;
  if(c->bad){++c->rejected;++c->attempt;c->retrying=1;c->dt*=.5;if(c->attempt>=21)c->fatal=5;}
  else {const double remaining=c->target-c->time;c->time=c->dt==remaining?c->target:c->time+c->dt;++c->steps;c->retrying=0;c->attempt=0;c->finished=c->time>=c->target;}
}
} // namespace

struct StaticAmrGpu {
  Config cfg{};Data d{};AmrGpuOptions options;AmrGpuStats stats;
  cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
  cudaEvent_t batch_start=nullptr,batch_end=nullptr;
  std::vector<void*> allocations;Control* host_control=nullptr;double* host_state=nullptr;
  ~StaticAmrGpu(){if(stream)cudaStreamSynchronize(stream);if(batch_start)cudaEventDestroy(batch_start);if(batch_end)cudaEventDestroy(batch_end);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);
    for(void* p:allocations)cudaFree(p);if(host_control)cudaFreeHost(host_control);if(host_state)cudaFreeHost(host_state);if(stream)cudaStreamDestroy(stream);}
};
namespace {
bool checked(cudaError_t status,std::string* error,const char* operation) {
  if(status==cudaSuccess)return true;if(error)*error=std::string(operation)+": "+cudaGetErrorString(status);return false;
}
template<class T> bool alloc(StaticAmrGpu* s,T** p,size_t count,std::string* error) {
  if(count==0)return true;
  if(!checked(cudaMalloc(reinterpret_cast<void**>(p),count*sizeof(T)),error,"AMR cudaMalloc"))return false;
  s->allocations.push_back(*p);s->stats.allocated_bytes+=count*sizeof(T);return true;
}
template<class T> bool upload(StaticAmrGpu* s,T** p,const std::vector<T>& values,std::string* error) {
  if(!alloc(s,p,values.size(),error))return false;if(values.empty())return true;
  if(!checked(cudaMemcpyAsync(*p,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,s->stream),error,"AMR geometry H2D"))return false;
  s->stats.geometry_h2d_bytes+=values.size()*sizeof(T);return true;
}
void launch_trial(StaticAmrGpu* s) {
  const auto c=s->cfg;const auto d=s->d;const auto stream=s->stream;const int nc=blocks(c.n),nf=blocks(c.nf),nu=blocks(c.nv*c.n);
  const bool fused=s->options.fused&&c.transport.model==AMR_MODEL_LAMINAR;
  const bool joined=fused&&s->options.sa_fused&&c.transport.sa_model;
  const auto gradients=[&](){if(c.transport.model==AMR_MODEL_LAMINAR){
    if(joined)velocity_gradient_kernel<true><<<nc,threads,0,stream>>>(c,d);
    else velocity_gradient_kernel<false><<<nc,threads,0,stream>>>(c,d);}
    if(c.transport.sa_model&&!joined)sa_gradient_kernel<<<nc,threads,0,stream>>>(c,d);
    if(c.spatial_order==2)reconstruction_kernel<<<nc,threads,0,stream>>>(c,d);};
  const auto face_work=[&](){if(fused){
    if(joined)flux_wall_kernel<true><<<blocks(c.nf+c.nw),threads,0,stream>>>(c,d);
    else flux_wall_kernel<false><<<blocks(c.nf+c.nw),threads,0,stream>>>(c,d);
  }else{if(c.transport.model==AMR_MODEL_LAMINAR&&c.nw)viscous_wall_kernel<<<blocks(c.nw),threads,0,stream>>>(c,d);flux_kernel<<<nf,threads,0,stream>>>(c,d);}};
  const auto collect=[&](double* rhs,bool base){
    if(joined)collect_kernel<true><<<nc,threads,0,stream>>>(c,d,rhs,base);
    else {collect_kernel<false><<<nc,threads,0,stream>>>(c,d,rhs,base);
      if(c.transport.sa_model){sa_face_wall_kernel<<<blocks(c.nf+c.nw),threads,0,stream>>>(c,d);sa_collect_kernel<<<nc,threads,0,stream>>>(c,d,rhs,base);}}
    if(c.precondition)precondition_kernel<<<nc,threads,0,stream>>>(c,d,rhs);};
  begin_kernel<<<1,1,0,stream>>>(d.control);
  primitive_kernel<<<nc,threads,0,stream>>>(c,d,d.u,true);
  gradients();face_work();collect(d.r0,true);
  if(c.transport.model==AMR_MODEL_LAMINAR&&c.transport.stabilization==AMR_VISCOUS_SRD&&c.small)diffusion_group_kernel<<<nc,threads,0,stream>>>(c,d);
  choose_dt_kernel<<<1,threads,0,stream>>>(c,d);
  for(int stage=0;stage<3;++stage){rk_kernel<<<nu,threads,0,stream>>>(c,d,stage);double* candidate=stage==1?d.u2:d.u1;
    if(c.small){group_kernel<<<nc,threads,0,stream>>>(c,d,candidate);redistribute_kernel<<<nc,threads,0,stream>>>(c,d,candidate,fused);}
    if(!(c.small&&fused))primitive_kernel<<<nc,threads,0,stream>>>(c,d,candidate,false);
    if(stage<2){gradients();face_work();collect(d.r,false);}
  }
  commit_state_kernel<<<nu,threads,0,stream>>>(c,d);commit_control_kernel<<<1,1,0,stream>>>(d.control);
}
void host_metadata(const StaticAmrGpu* s,StaticAmrState2D* state) {
  state->time=s->host_control->time;state->steps=s->host_control->steps;state->rejected=s->host_control->rejected;
}
}

int static_amr_gpu_create(const StaticAmrMesh2D& m,const Params2D& p,const StaticAmrState2D& initial,
                          const AmrGpuOptions& options,StaticAmrGpu** result,std::string* error) {
  if(!result)return 1;*result=nullptr;
  if(!static_amr2d_viscous_ready(m)||m.cells.empty()||m.faces.empty()||initial.u.size()!=(m.transport.sa_model?5:4)*m.cells.size()||!std::isfinite(initial.time)||initial.time<0||
     options.batch_trials<1||options.batch_trials>1024||(p.amr_spatial_order!=1&&p.amr_spatial_order!=2)||p.time_integrator!=TIME_SSP_RK3||
     !std::isfinite(p.gamma)||!(p.gamma>1)||!(p.cfl>0)||p.cfl>.5||!std::isfinite(p.dt_fixed)||p.dt_fixed<0||
     !std::isfinite(p.ausm_mco)||p.ausm_mco<=0||(p.flux!=FLUX_HLLC&&p.flux!=FLUX_AUSMPUP)) {if(error)*error="invalid AMR GPU configuration";return 1;}
  for(int side=0;side<4;++side)if(!amr_bc_valid(m.boundaries[side],p.bc[side],side)){if(error)*error="invalid AMR GPU boundary configuration";return 1;}
  const auto start=std::chrono::steady_clock::now();std::unique_ptr<StaticAmrGpu> owner(new StaticAmrGpu);auto* s=owner.get();s->options=options;
  s->cfg.n=m.cells.size();s->cfg.nv=m.transport.sa_model?5:4;s->cfg.nf=m.faces.size();s->cfg.nw=m.walls.size();s->cfg.transport=m.transport;s->cfg.gamma=p.gamma;s->cfg.cfl=p.cfl;s->cfg.fixed_dt=p.dt_fixed;s->cfg.mco=p.ausm_mco;s->cfg.flux=p.flux;s->cfg.small=m.small_cells>0;
  s->cfg.agglomerated=m.options.agglomerate_small;
  s->cfg.precondition=p.low_mach_preconditioning;s->cfg.mach_reference=p.low_mach_reference;s->cfg.mach_floor=p.low_mach_floor;
  s->cfg.local_time=p.local_pseudo_time;
  s->cfg.spatial_order=p.amr_spatial_order;
  for(int i=0;i<4;++i){s->cfg.bc[i]=p.bc[i];s->cfg.boundary[i]=m.boundaries[i];}
  if(!checked(cudaStreamCreateWithFlags(&s->stream,cudaStreamNonBlocking),error,"AMR stream"))return 1;
  if(options.event_timing&&(!checked(cudaEventCreate(&s->batch_start),error,"AMR timing start")||
     !checked(cudaEventCreate(&s->batch_end),error,"AMR timing end")))return 1;
  auto& d=s->d;
  if(!upload(s,&d.cells,m.cells,error)||!upload(s,&d.faces,m.faces,error)||!upload(s,&d.walls,m.walls,error)||
     !upload(s,&d.offset,m.face_offsets,error)||!upload(s,&d.face_ids,m.cell_faces,error)||!upload(s,&d.goff,m.group_offsets,error)||
     !upload(s,&d.members,m.group_members,error)||!upload(s,&d.moff,m.member_offsets,error)||!upload(s,&d.groups,m.member_groups,error)||!upload(s,&d.gvol,m.group_volume,error))return 1;
  if(s->cfg.spatial_order==2){if(!upload(s,&d.reconstruction_offsets,m.reconstruction_offsets,error)||
      !upload(s,&d.reconstruction_terms,m.reconstruction_terms,error)||!alloc(s,&d.slopes,8*s->cfg.n,error))return 1;}
  if(m.transport.model==AMR_MODEL_LAMINAR){
    if(!upload(s,&d.visc_offsets,m.viscous.offsets,error)||!upload(s,&d.visc_terms,m.viscous.terms,error)||!upload(s,&d.wall_samples,m.viscous.wall_samples,error)||!upload(s,&d.face_samples,m.viscous.face_samples,error)||!upload(s,&d.face_geometry,m.viscous.face_geometry,error))return 1;
    if(!alloc(s,&d.gradient,4*s->cfg.n,error)||!alloc(s,&d.mu,s->cfg.n,error)||!alloc(s,&d.wall_flux,4*s->cfg.nw,error)||!alloc(s,&d.face_coefficient,s->cfg.nf,error)||!alloc(s,&d.wall_coefficient,s->cfg.nw,error)||!alloc(s,&d.diffusion_rate,s->cfg.n,error)||!alloc(s,&d.diffusion_group,s->cfg.n,error))return 1;
  }
  const size_t n=s->cfg.n,nf=s->cfg.nf;
  if(m.transport.sa_model){
    if(!upload(s,&d.wall_distance,m.viscous.wall_distance,error))return 1;
    for(double** buffer:{&d.sa_phi,&d.eddy,&d.sa_diff,&d.sa_source,&d.sa_source_rate})if(!alloc(s,buffer,n,error))return 1;
    if(!alloc(s,&d.sa_grad,2*n,error)||!alloc(s,&d.sa_face_flux,nf,error)||!alloc(s,&d.sa_face_coefficient,nf,error)||!alloc(s,&d.sa_wall_flux,s->cfg.nw,error)||!alloc(s,&d.sa_wall_coefficient,s->cfg.nw,error))return 1;
  }
  for(double** buffer:{&d.u,&d.u1,&d.u2,&d.w,&d.r0,&d.r,&d.qhat})if(!alloc(s,buffer,s->cfg.nv*n,error))return 1;
  if(!alloc(s,&d.flux,4*nf,error)||!alloc(s,&d.speed,nf,error)||!alloc(s,&d.cell_dt,n,error)||!alloc(s,&d.control,1,error))return 1;
  if(!checked(cudaHostAlloc(reinterpret_cast<void**>(&s->host_control),sizeof(Control),cudaHostAllocDefault),error,"AMR pinned control")||
     !checked(cudaHostAlloc(reinterpret_cast<void**>(&s->host_state),s->cfg.nv*n*sizeof(double),cudaHostAllocDefault),error,"AMR pinned state"))return 1;
  s->stats.pinned_bytes=sizeof(Control)+s->cfg.nv*n*sizeof(double);
  *s->host_control={initial.time,initial.time,0,initial.steps,initial.rejected,0,0,0,0,0,1,0};
  s->host_control->cfl=p.cfl;
  std::copy(initial.u.begin(),initial.u.end(),s->host_state);
  if(!checked(cudaMemcpyAsync(d.control,s->host_control,sizeof(Control),cudaMemcpyHostToDevice,s->stream),error,"AMR initial control")||
     !checked(cudaMemcpyAsync(d.u,s->host_state,s->cfg.nv*n*sizeof(double),cudaMemcpyHostToDevice,s->stream),error,"AMR initial state")||
     !checked(cudaStreamSynchronize(s->stream),error,"AMR setup sync"))return 1;
  s->stats.state_h2d_bytes=s->cfg.nv*n*sizeof(double);
  s->stats.control_h2d_bytes=sizeof(Control);
  if(options.graph) {
    if(!checked(cudaStreamBeginCapture(s->stream,cudaStreamCaptureModeThreadLocal),error,"AMR graph begin"))return 1;
    launch_trial(s);
    if(!checked(cudaStreamEndCapture(s->stream,&s->graph),error,"AMR graph end")||
       !checked(cudaGraphInstantiate(&s->executable,s->graph,nullptr,nullptr,0),error,"AMR graph instantiate"))return 1;
    size_t nodes=0;if(!checked(cudaGraphGetNodes(s->graph,nullptr,&nodes),error,"AMR graph node count"))return 1;
    s->stats.graph_nodes_per_trial=nodes;
  }
  s->stats.setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  *result=owner.release();return 0;
}

int static_amr_gpu_set_cfl(StaticAmrGpu* s,double value,std::string* error) {
  if(!s||!std::isfinite(value)||!(value>0)||value>.5){if(error)*error="CFL must be in (0,.5] for explicit RK3";return 1;}
  cfl_kernel<<<1,1,0,s->stream>>>(s->d.control,value);return checked(cudaGetLastError(),error,"CFL update")?0:1;
}
int static_amr_gpu_advance(StaticAmrGpu* s,double tend,long maxsteps,StaticAmrState2D* state,std::string* error,double wall_budget_seconds) {
  if(!s||!state||!std::isfinite(tend)||tend<s->host_control->time||maxsteps<1||!std::isfinite(wall_budget_seconds)||wall_budget_seconds<0){if(error)*error="invalid AMR GPU advance";return 1;}
  const auto start=std::chrono::steady_clock::now();auto last=start;bool expired=false;
  target_kernel<<<1,1,0,s->stream>>>(s->d.control,tend,maxsteps);
  for(;;) {
    if(s->options.event_timing&&!checked(cudaEventRecord(s->batch_start,s->stream),error,"AMR timing record start"))return 1;
    for(int trial=0;trial<s->options.batch_trials;++trial) {
      if(s->options.graph){if(!checked(cudaGraphLaunch(s->executable,s->stream),error,"AMR graph launch"))return 1;}
      else launch_trial(s);
    }
    if(s->options.event_timing&&!checked(cudaEventRecord(s->batch_end,s->stream),error,"AMR timing record end"))return 1;
    if(!checked(cudaGetLastError(),error,"AMR trial kernels")||
       !checked(cudaMemcpyAsync(s->host_control,s->d.control,sizeof(Control),cudaMemcpyDeviceToHost,s->stream),error,"AMR control D2H")||
       !checked(cudaStreamSynchronize(s->stream),error,"AMR batch sync"))return 1;
    if(s->options.event_timing){float ms=0;if(!checked(cudaEventElapsedTime(&ms,s->batch_start,s->batch_end),error,"AMR timing elapsed"))return 1;
      s->stats.advance_event_seconds+=ms*.001;++s->stats.event_batches;}
    s->stats.control_d2h_bytes+=sizeof(Control);++s->stats.status_polls;host_metadata(s,state);
    const auto now=std::chrono::steady_clock::now();
    if(std::chrono::duration<double>(now-last).count()>=2){std::fprintf(stderr,"amr-cuda: t=%.9g steps=%ld rejected=%ld polls=%llu\n",state->time,state->steps,state->rejected,static_cast<unsigned long long>(s->stats.status_polls));last=now;}
    if(s->host_control->fatal||s->host_control->finished)break;
    if(wall_budget_seconds>0&&std::chrono::duration<double>(now-start).count()>=wall_budget_seconds){expired=true;break;}
  }
  s->stats.advance_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  if(s->host_control->fatal){if(error)*error="AMR GPU stopped with status "+std::to_string(s->host_control->fatal)+"; last accepted state retained";return 1;}
  if(expired)return 2;
  return 0;
}

int static_amr_gpu_download(StaticAmrGpu* s,StaticAmrState2D* state,std::string* error) {
  if(!s||!state){if(error)*error="invalid AMR GPU download";return 1;}
  const auto start=std::chrono::steady_clock::now();const size_t bytes=s->cfg.nv*s->cfg.n*sizeof(double);
  if(!checked(cudaMemcpyAsync(s->host_state,s->d.u,bytes,cudaMemcpyDeviceToHost,s->stream),error,"AMR state D2H")||
     !checked(cudaStreamSynchronize(s->stream),error,"AMR download sync"))return 1;
  state->u.assign(s->host_state,s->host_state+s->cfg.nv*s->cfg.n);host_metadata(s,state);
  s->stats.state_d2h_bytes+=bytes;++s->stats.full_downloads;
  s->stats.download_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return 0;
}
AmrGpuStats static_amr_gpu_stats(const StaticAmrGpu* s){return s?s->stats:AmrGpuStats{};}
void static_amr_gpu_destroy(StaticAmrGpu* s){delete s;}
} // namespace cfd
