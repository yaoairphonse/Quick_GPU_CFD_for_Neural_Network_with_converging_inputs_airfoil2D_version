// Included inside the static AMR CUDA implementation's anonymous namespace.
__global__ void sa_gradient_kernel(Config cfg,Data d){
 const size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=cfg.n||!healthy(d.control))return;
 double gx=0,gy=0,rx=0,ry=0;
 for(size_t pos=d.visc_offsets[k];pos<d.visc_offsets[k+1];++pos){const auto t=d.visc_terms[pos];const double ph=t.source>=0?d.sa_phi[t.source]:0,rho=t.source>=0?d.w[t.source]:d.w[k];
  gx+=t.x*(ph-d.sa_phi[k]);gy+=t.y*(ph-d.sa_phi[k]);rx+=t.x*(rho-d.w[k]);ry+=t.y*(rho-d.w[k]);}
 d.sa_grad[k]=gx;d.sa_grad[cfg.n+k]=gy;
 const auto model=sa_neg_point(d.sa_phi[k],d.mu[k]/d.w[k],d.wall_distance[k],fabs(d.gradient[2*cfg.n+k]-d.gradient[cfg.n+k]));
 d.sa_diff[k]=model.diffusion;d.sa_source[k]=sa_density_corrected_source(model,d.w[k],gx,gy,rx,ry);d.sa_source_rate[k]=model.source_rate;
 if(!(model.diffusion>0)||!isfinite(d.sa_source[k])||!isfinite(model.source_rate))atomicOr(&d.control->bad,1);
}
__device__ __forceinline__ void sa_face_wall_point(Config cfg,Data d,size_t q){
 if(!healthy(d.control))return;
 if(q>=cfg.nf){const size_t j=q-cfg.nf;if(j>=cfg.nw)return;const auto sample=d.wall_samples[j];const int k=sample.donor;
  const double coefficient=d.mu[k]/sa_sigma/sample.distance*d.walls[j].length;d.sa_wall_coefficient[j]=coefficient;d.sa_wall_flux[j]=-coefficient*d.sa_phi[k];return;}
 const auto f=d.faces[q];const auto geo=d.face_geometry[q];const int inside=f.left>=0?f.left:f.right;const double mass=d.flux[q];
 const double left=f.left>=0?d.sa_phi[f.left]:cfg.transport.sa_nu_inf,right=f.right>=0?d.sa_phi[f.right]:cfg.transport.sa_nu_inf;
 double diff=0,fc=0;
 if(f.boundary<0){const int l=f.left,r=f.right;double gl[4]={d.sa_grad[l],d.sa_grad[cfg.n+l],0,0},gr[4]={d.sa_grad[r],d.sa_grad[cfg.n+r],0,0},grad[4];
  amr_correct_gradient(gl,gr,geo.dx,geo.dy,d.sa_phi[r]-d.sa_phi[l],0,grad);const double coeff=.5*(d.w[l]*d.sa_diff[l]+d.w[r]*d.sa_diff[r]);diff=coeff*grad[f.axis]*f.length;fc=coeff/geo.distance*f.length;
 }else{const int bc=cfg.boundary[f.boundary].kind<0?cfg.bc[f.boundary]:cfg.boundary[f.boundary].kind;const bool inflow=f.left<0?mass>0:mass<0;const double sign=f.boundary%2?1:-1;
  if(bc==AMR_BC_NOSLIP||inflow){const double wall=bc==AMR_BC_NOSLIP?0:cfg.transport.sa_nu_inf,coeff=bc==AMR_BC_NOSLIP?d.mu[inside]/sa_sigma:d.w[inside]*d.sa_diff[inside];
   diff=sign*coeff*(wall-d.sa_phi[inside])/d.face_samples[q].distance*f.length;fc=coeff/d.face_samples[q].distance*f.length;}}
 d.sa_face_flux[q]=mass*(mass>=0?left:right)-diff;d.sa_face_coefficient[q]=fc;
 if(!isfinite(d.sa_face_flux[q]))atomicOr(&d.control->bad,1);
}
__global__ void sa_face_wall_kernel(Config cfg,Data d){
 const size_t q=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;sa_face_wall_point(cfg,d,q);
}
__global__ void sa_collect_kernel(Config cfg,Data d,double* rhs,bool base){
 const size_t k=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(k>=cfg.n||!healthy(d.control))return;
 const auto c=d.cells[k];double net=0,spectral=0;
 for(size_t pos=d.offset[k];pos<d.offset[k+1];++pos){size_t q=d.face_ids[pos];net+=(d.faces[q].left==static_cast<int>(k)?1:-1)*d.sa_face_flux[q];spectral+=d.sa_face_coefficient[q];}
 for(size_t q=c.wall_begin;q<c.wall_begin+c.wall_count;++q){net-=d.sa_wall_flux[q];spectral+=d.sa_wall_coefficient[q];}
 rhs[4*cfg.n+k]=-net/c.volume+d.sa_source[k];if(!isfinite(rhs[4*cfg.n+k]))atomicOr(&d.control->bad,1);
 if(base){const double rate=2*spectral/(d.w[k]*c.volume)+d.sa_source_rate[k];if(rate>0)d.cell_dt[k]=fmin(d.cell_dt[k],cfg.transport.sa_cfl/rate);}
}
