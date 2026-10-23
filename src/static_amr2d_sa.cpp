#include "static_amr2d_sa.h"
#include <algorithm>
#include <limits>
namespace cfd {
int static_amr2d_prepare_sa(StaticAmrMesh2D* mesh,const Params2D& p,std::string* error){
 if(!mesh)return 1;
 auto& m=*mesh;m.viscous.wall_distance.clear();if(!m.transport.sa_model)return 0;
 if(!m.options.agglomerate_small||m.small_cells||m.transport.stabilization!=AMR_VISCOUS_VOLUME){if(error)*error="SA-neg requires agglomerated true-volume mesh without SRD";return 1;}
 for(const auto& c:m.cells){double d=INFINITY;
  for(size_t j=0;j<m.body_x.size();++j){const size_t k=(j+1)%m.body_x.size();const double ax=m.body_x[j],ay=m.body_y[j],dx=m.body_x[k]-ax,dy=m.body_y[k]-ay;
   const double t=std::max(0.,std::min(1.,((c.cx-ax)*dx+(c.cy-ay)*dy)/(dx*dx+dy*dy)));d=std::min(d,std::hypot(c.cx-ax-t*dx,c.cy-ay-t*dy));}
  for(int side=0;side<4;++side)if((m.boundaries[side].kind<0?p.bc[side]:m.boundaries[side].kind)==AMR_BC_NOSLIP){const double v=side<2?c.cx:c.cy;const double wall=side==0?m.options.xmin:side==1?m.options.xmax:side==2?m.options.ymin:m.options.ymax;d=std::min(d,std::fabs(v-wall));}
  if(!(d>0)){if(error)*error="SA geometric wall distance is nonpositive";return 1;}m.viscous.wall_distance.push_back(d);
 }
 return 0;
}
int static_amr2d_sa_evaluate(const StaticAmrMesh2D& m,const Params2D& p,const std::vector<double>& u,
 const std::vector<double>& w,const AmrViscousFields2D& mean,const std::vector<double>& meanflux,SaFields2D* out){
 const size_t n=m.cells.size(),nf=m.faces.size(),nw=m.walls.size();if(!out||u.size()!=5*n||m.viscous.wall_distance.size()!=n)return 1;auto& a=*out;
 a.gradient.assign(2*n,0);a.density_gradient.assign(2*n,0);a.diffusion.resize(n);a.source.resize(n);a.source_rate.resize(n);a.face_flux.resize(nf);a.wall_flux.resize(nw);a.residual.resize(n);a.cell_dt.resize(n);
 std::vector<double> phi(n),fc(nf),wc(nw);for(size_t k=0;k<n;++k){phi[k]=u[4*n+k]/w[k];if(!std::isfinite(phi[k]))return 1;}
 for(size_t k=0;k<n;++k){for(size_t q=m.viscous.offsets[k];q<m.viscous.offsets[k+1];++q){const auto t=m.viscous.terms[q];const double ph=t.source>=0?phi[t.source]:0,rho=t.source>=0?w[t.source]:w[k];
   a.gradient[k]+=t.x*(ph-phi[k]);a.gradient[n+k]+=t.y*(ph-phi[k]);a.density_gradient[k]+=t.x*(rho-w[k]);a.density_gradient[n+k]+=t.y*(rho-w[k]);}
  const auto model=sa_neg_point(phi[k],mean.mu[k]/w[k],m.viscous.wall_distance[k],std::fabs(mean.gradient[2*n+k]-mean.gradient[n+k]));a.diffusion[k]=model.diffusion;a.source_rate[k]=model.source_rate;
  a.source[k]=sa_density_corrected_source(model,w[k],a.gradient[k],a.gradient[n+k],a.density_gradient[k],a.density_gradient[n+k]);
  if(!(a.diffusion[k]>0)||!std::isfinite(a.source[k])||!std::isfinite(a.source_rate[k]))return 1;
 }
 for(size_t q=0;q<nf;++q){const auto f=m.faces[q];const auto geo=m.viscous.face_geometry[q];const int inside=f.left>=0?f.left:f.right;const double mass=meanflux[q];
  const double left=f.left>=0?phi[f.left]:m.transport.sa_nu_inf,right=f.right>=0?phi[f.right]:m.transport.sa_nu_inf;
  double diff=0,coefficient=0;
  if(f.boundary<0){const int l=f.left,r=f.right;const double gl[4]={a.gradient[l],a.gradient[n+l],0,0},gr[4]={a.gradient[r],a.gradient[n+r],0,0};double grad[4];amr_correct_gradient(gl,gr,geo.dx,geo.dy,phi[r]-phi[l],0,grad);
   coefficient=.5*(w[l]*a.diffusion[l]+w[r]*a.diffusion[r]);diff=coefficient*grad[f.axis]*f.length;fc[q]=coefficient/geo.distance*f.length;
  }else{const int bc=m.boundaries[f.boundary].kind<0?p.bc[f.boundary]:m.boundaries[f.boundary].kind;const bool inflow=f.left<0?mass>0:mass<0;const double sign=f.boundary%2?1:-1;
   if(bc==AMR_BC_NOSLIP||inflow){const double wall=bc==AMR_BC_NOSLIP?0:m.transport.sa_nu_inf;coefficient=bc==AMR_BC_NOSLIP?mean.mu[inside]/sa_sigma:w[inside]*a.diffusion[inside];diff=sign*coefficient*(wall-phi[inside])/m.viscous.face_samples[q].distance*f.length;fc[q]=coefficient/m.viscous.face_samples[q].distance*f.length;}else fc[q]=0;
  }
  a.face_flux[q]=mass*(mass>=0?left:right)-diff;
 }
 for(size_t q=0;q<nw;++q){const auto sample=m.viscous.wall_samples[q];const auto wall=m.walls[q];const int k=sample.donor;wc[q]=mean.mu[k]/sa_sigma/sample.distance*wall.length;a.wall_flux[q]=-wc[q]*phi[k];}
 for(size_t k=0;k<n;++k){const auto c=m.cells[k];double net=0,spectral=0;for(size_t pos=m.face_offsets[k];pos<m.face_offsets[k+1];++pos){size_t q=m.cell_faces[pos];net+=(m.faces[q].left==static_cast<int>(k)?1:-1)*a.face_flux[q];spectral+=fc[q];}
  for(size_t q=c.wall_begin;q<c.wall_begin+c.wall_count;++q){net-=a.wall_flux[q];spectral+=wc[q];}
  a.residual[k]=-net/c.volume+a.source[k];const double rate=2*spectral/(w[k]*c.volume)+a.source_rate[k];a.cell_dt[k]=rate>0?m.transport.sa_cfl/rate:INFINITY;
  if(!std::isfinite(a.residual[k]))return 1;
 }
 return 0;
}
} // namespace cfd
