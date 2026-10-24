#include "static_amr2d.h"
#include "static_amr2d_sa.h"
#include "static_amr2d_binary.h"
#include <map>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>

namespace cfd {
namespace {
int failed(std::string* error,const std::string& message){if(error)*error=message;return 1;}
int kind(const StaticAmrMesh2D& m,const Params2D& p,int side){return m.boundaries[side].kind<0?p.bc[side]:m.boundaries[side].kind;}
double cross(double x,double y,double a,double b){return x*b-y*a;}
bool inside_body(const StaticAmrMesh2D& m,double x,double y){
  bool inside=false;const size_t n=m.body_x.size();
  for(size_t i=0,j=n?n-1:0;i<n;j=i++)if((m.body_y[i]>y)!=(m.body_y[j]>y)){
    const double hit=(m.body_x[j]-m.body_x[i])*(y-m.body_y[i])/(m.body_y[j]-m.body_y[i])+m.body_x[i];if(x<hit)inside=!inside;
  }
  return inside;
}
bool visible(const StaticAmrMesh2D& m,double x,double y,const AmrCell2D& cell){
  if(inside_body(m,cell.cx,cell.cy))return false;
  const double dx=cell.cx-x,dy=cell.cy-y;
  for(size_t i=0;i<m.body_x.size();++i){const size_t j=(i+1)%m.body_x.size();const double ex=m.body_x[j]-m.body_x[i],ey=m.body_y[j]-m.body_y[i];
    const double det=cross(dx,dy,ex,ey);if(std::fabs(det)<=1e-14*std::hypot(dx,dy)*std::hypot(ex,ey))continue;
    const double ax=m.body_x[i]-x,ay=m.body_y[i]-y,t=cross(ax,ay,ex,ey)/det,u=cross(ax,ay,dx,dy)/det;
    if(t>1e-10&&t<=1+1e-12&&u>=-1e-12&&u<=1+1e-12)return false;
  }
  return true;
}
}
bool static_amr2d_viscous_ready(const StaticAmrMesh2D& m){
  if(m.transport.model==AMR_MODEL_EULER)for(const auto& b:m.boundaries)if(b.kind==AMR_BC_NOSLIP)return false;
  return (!m.transport.sa_model||m.viscous.wall_distance.size()==m.cells.size())&&amr_transport_valid(m.transport)&&(m.transport.model==AMR_MODEL_EULER||
    (m.viscous.offsets.size()==m.cells.size()+1&&m.viscous.wall_samples.size()==m.walls.size()&&m.viscous.face_samples.size()==m.faces.size()&&m.viscous.face_geometry.size()==m.faces.size()));
}
int static_amr2d_prepare_viscous(StaticAmrMesh2D* mesh,const Params2D& p,std::string* error){
  if(!mesh)return failed(error,"null viscous mesh");
  auto& m=*mesh;m.viscous=AmrViscousTable2D{};
  if(!amr_transport_valid(m.transport))return failed(error,"invalid laminar transport parameters");
  if(m.transport.model==AMR_MODEL_EULER)return static_amr2d_viscous_ready(m)?0:failed(error,"no-slip wall requires laminar model");
  const size_t n=m.cells.size();auto& table=m.viscous;table.offsets.push_back(0);
  for(size_t k=0;k<n;++k){const auto& c=m.cells[k];struct Sample{int source,side;double dx,dy;};std::vector<Sample> samples;
    for(size_t pos=m.face_offsets[k];pos<m.face_offsets[k+1];++pos){const auto& f=m.faces[m.cell_faces[pos]];const int other=f.left==static_cast<int>(k)?f.right:f.left;
      if(other>=0)samples.push_back({other,-1,m.cells[other].cx-c.cx,m.cells[other].cy-c.cy});
      else if(kind(m,p,f.boundary)==AMR_BC_NOSLIP){const double x=f.axis==0?f.coordinate:f.centroid,y=f.axis==1?f.coordinate:f.centroid;samples.push_back({-2,f.boundary,x-c.cx,y-c.cy});}
    }
    for(size_t pos=c.wall_begin;pos<c.wall_begin+c.wall_count;++pos){const auto& w=m.walls[pos];samples.push_back({-1,-1,.5*(w.x0+w.x1)-c.cx,.5*(w.y0+w.y1)-c.cy});}
    double xx=0,xy=0,yy=0;for(const auto& s:samples){const double d2=s.dx*s.dx+s.dy*s.dy;if(!(d2>0))return failed(error,"zero gradient sample distance");xx+=s.dx*s.dx/d2;xy+=s.dx*s.dy/d2;yy+=s.dy*s.dy/d2;}
    const double det=xx*yy-xy*xy;if(!(det>1e-12*(xx+yy)*(xx+yy)))return failed(error,"singular/ill-conditioned velocity gradient stencil at cell "+std::to_string(k));
    for(const auto& s:samples){const double d2=s.dx*s.dx+s.dy*s.dy;table.terms.push_back({s.source,s.side,(yy*s.dx-xy*s.dy)/(det*d2),(xx*s.dy-xy*s.dx)/(det*d2)});}
    table.offsets.push_back(table.terms.size());
  }
  for(const auto& w:m.walls){const double x=.5*(w.x0+w.x1),y=.5*(w.y0+w.y1);const auto& own=m.cells[w.owner_cell];const double h=std::min(own.x1-own.x0,own.y1-own.y0);
    std::queue<std::pair<int,int>> todo;std::vector<bool> visited(n,false);todo.push({w.owner_cell,0});visited[w.owner_cell]=true;int donor=-1;double best=INFINITY,distance=0;
    while(!todo.empty()){const auto [k,depth]=todo.front();todo.pop();const auto& c=m.cells[k];const double dx=x-c.cx,dy=y-c.cy,dn=dx*w.nx+dy*w.ny,d2=dx*dx+dy*dy;
      if(dn>1e-12*h&&dn>=m.transport.wall_probe_min_h*h&&d2<best&&visible(m,x,y,c)){donor=k;distance=dn;best=d2;}
      if(depth==4)continue;
      for(size_t pos=m.face_offsets[k];pos<m.face_offsets[k+1];++pos){const auto& f=m.faces[m.cell_faces[pos]];const int other=f.left==k?f.right:f.left;
        if(other>=0&&!visited[other]){visited[other]=true;todo.push({other,depth+1});}}
    }
    if(donor<0)return failed(error,"no visible fluid-side wall donor within four face rings");
    table.wall_samples.push_back({donor,distance});
  }
  for(const auto& f:m.faces){AmrWallSample2D sample{-1,0};if(f.boundary>=0){const int owner=f.left>=0?f.left:f.right;const auto& c=m.cells[owner];const double sign=f.boundary%2?1:-1;
      sample={owner,sign*(f.coordinate-(f.axis==0?c.cx:c.cy))};if(!(sample.distance>0))return failed(error,"invalid exterior face distance");}table.face_samples.push_back(sample);
    AmrViscousFaceGeometry2D geo{};const double x=f.axis==0?f.coordinate:f.centroid,y=f.axis==1?f.coordinate:f.centroid;
    if(f.boundary<0){const auto& l=m.cells[f.left];const auto& r=m.cells[f.right];geo={r.cx-l.cx,r.cy-l.cy,x-.5*(l.cx+r.cx),y-.5*(l.cy+r.cy),std::hypot(r.cx-l.cx,r.cy-l.cy)};}
    else{const auto& c=m.cells[sample.donor];geo={0,0,x-c.cx,y-c.cy,sample.distance};}
    if(!(geo.distance>0)||!std::isfinite(geo.distance))return failed(error,"invalid fixed viscous face geometry");
    table.face_geometry.push_back(geo);
  }
  return static_amr2d_prepare_sa(mesh,p,error);
}
int static_amr2d_viscous_evaluate(const StaticAmrMesh2D& m,const Params2D& p,const std::vector<double>& w,AmrViscousFields2D* out,const std::vector<double>* conserved){
  if(!out||!static_amr2d_viscous_ready(m)||m.transport.model!=AMR_MODEL_LAMINAR)return 1;
  const size_t n=m.cells.size(),nf=m.faces.size(),nw=m.walls.size();if(w.size()!=4*n)return 1;auto& a=*out;
  a.gradient.assign(4*n,0);a.mu.resize(n);a.face_flux.assign(4*nf,0);a.wall_flux.assign(4*nw,0);a.residual.assign(4*n,0);a.cell_dt.resize(n);
  std::vector<double> fc(nf),wc(nw),eddy(n,0);int bad=0;
  if(m.transport.sa_model&&(!conserved||conserved->size()!=5*n))return 1;
  #pragma omp parallel for reduction(|:bad)
  for(size_t k=0;k<n;++k){a.mu[k]=amr_dynamic_viscosity(m.transport,w[3*n+k]/(w[k]*m.transport.gas_constant));if(!(a.mu[k]>0)||!std::isfinite(a.mu[k]))bad=1;
    if(m.transport.sa_model){const double ph=(*conserved)[4*n+k]/w[k];if(!std::isfinite(ph)){bad=1;continue;}eddy[k]=w[k]*sa_eddy_kinematic(ph,a.mu[k]/w[k]);}
    for(size_t q=m.viscous.offsets[k];q<m.viscous.offsets[k+1];++q){const auto& term=m.viscous.terms[q];double u=0,v=0;
      if(term.source>=0){u=w[n+term.source];v=w[2*n+term.source];}
      else if(term.source==-2){u=m.boundaries[term.side].wall_velocity[0];v=m.boundaries[term.side].wall_velocity[1];}
      const double du=u-w[n+k],dv=v-w[2*n+k];a.gradient[k]+=term.x*du;a.gradient[n+k]+=term.y*du;a.gradient[2*n+k]+=term.x*dv;a.gradient[3*n+k]+=term.y*dv;}
  }
  if(bad)return 1;
  #pragma omp parallel for reduction(|:bad)
  for(size_t q=0;q<nf;++q){const auto& f=m.faces[q];const auto& geo=m.viscous.face_geometry[q];const int k=f.left>=0?f.left:f.right;double grad[4],mu=a.mu[k]+eddy[k],uf=w[n+k],vf=w[2*n+k],nu=mu/w[k],distance=geo.distance;
    for(int v=0;v<4;++v)grad[v]=a.gradient[v*n+k];
    if(f.boundary<0){const int l=f.left,r=f.right;double gl[4],gr[4];for(int v=0;v<4;++v){gl[v]=a.gradient[v*n+l];gr[v]=a.gradient[v*n+r];}
      const double dx=geo.dx,dy=geo.dy;amr_correct_gradient(gl,gr,dx,dy,w[n+r]-w[n+l],w[2*n+r]-w[2*n+l],grad);
      const double ex=geo.ex,ey=geo.ey;
      uf=.5*(w[n+l]+w[n+r])+grad[0]*ex+grad[1]*ey;vf=.5*(w[2*n+l]+w[2*n+r])+grad[2]*ex+grad[3]*ey;
      const double temp=.5*(w[3*n+l]/w[l]+w[3*n+r]/w[r])/m.transport.gas_constant;mu=amr_dynamic_viscosity(m.transport,temp)+.5*(eddy[l]+eddy[r]);nu=std::max((a.mu[l]+eddy[l])/w[l],(a.mu[r]+eddy[r])/w[r]);
    }else {distance=m.viscous.face_samples[q].distance;const int bc=kind(m,p,f.boundary);const double sign=f.boundary%2?1:-1;
      if(bc==AMR_BC_NOSLIP){mu=a.mu[k];uf=m.boundaries[f.boundary].wall_velocity[0];vf=m.boundaries[f.boundary].wall_velocity[1];amr_wall_gradient(f.axis==0?sign:0,f.axis==1?sign:0,distance,uf-w[n+k],vf-w[2*n+k],grad);}
      else if(bc==AMR_BC_REFLECT){if(f.axis==0){grad[1]=0;grad[2]=0;uf=0;}else{grad[1]=0;grad[2]=0;vf=0;}}
      else {grad[f.axis]=0;grad[2+f.axis]=0;uf+=grad[0]*geo.ex+grad[1]*geo.ey;vf+=grad[2]*geo.ex+grad[3]*geo.ey;}
    }
    double flux[4];amr_viscous_flux(mu,grad,f.axis==0?1:0,f.axis==1?1:0,uf,vf,flux);for(int v=0;v<4;++v){a.face_flux[v*nf+q]=flux[v]*f.length;if(!std::isfinite(flux[v]))bad=1;}fc[q]=(4.0/3)*nu/distance;
  }
  #pragma omp parallel for reduction(|:bad)
  for(size_t q=0;q<nw;++q){const auto& wall=m.walls[q];const auto& sample=m.viscous.wall_samples[q];const int k=sample.donor,owner=wall.owner_cell;double grad[4],flux[4];
    amr_wall_gradient(wall.nx,wall.ny,sample.distance,-w[n+k],-w[2*n+k],grad);amr_viscous_flux(a.mu[k],grad,wall.nx,wall.ny,0,0,flux);
    for(int v=0;v<4;++v){a.wall_flux[v*nw+q]=flux[v]*wall.length;if(!std::isfinite(flux[v]))bad=1;}
    wc[q]=(4.0/3)*std::max(a.mu[k]/w[k],a.mu[owner]/w[owner])/sample.distance;
  }
  if(bad)return 1;
  #pragma omp parallel for
  for(size_t k=0;k<n;++k){const auto& c=m.cells[k];double sum[4]={},spectral=0;
    for(size_t pos=m.face_offsets[k];pos<m.face_offsets[k+1];++pos){const size_t q=m.cell_faces[pos];const double sign=m.faces[q].left==static_cast<int>(k)?1:-1;
      for(int v=0;v<4;++v)sum[v]+=sign*a.face_flux[v*nf+q];
      spectral+=fc[q]*m.faces[q].length;}
    for(size_t q=c.wall_begin;q<c.wall_begin+c.wall_count;++q){for(int v=0;v<4;++v)sum[v]+=a.wall_flux[v*nw+q];spectral+=wc[q]*m.walls[q].length;}
    for(int v=0;v<4;++v)a.residual[v*n+k]=sum[v]/c.volume;
    a.cell_dt[k]=spectral>0?m.transport.viscous_cfl*c.volume/spectral:INFINITY;
  }
  if(m.transport.stabilization==AMR_VISCOUS_SRD&&m.small_cells){
    // Apply exactly the existing nonnegative SRD map to the diffusion rate.
    // Unlike substituting background volume, this includes overlap counts.
    std::vector<double> rates(n),groups(n);
    for(size_t k=0;k<n;++k)rates[k]=m.transport.viscous_cfl/a.cell_dt[k];
    for(size_t g=0;g<n;++g){double sum=0;
      for(size_t pos=m.group_offsets[g];pos<m.group_offsets[g+1];++pos){const size_t k=m.group_members[pos];sum+=m.cells[k].volume*rates[k]/(m.member_offsets[k+1]-m.member_offsets[k]);}
      groups[g]=sum/m.group_volume[g];}
    for(size_t k=0;k<n;++k){double sum=0;
      for(size_t pos=m.member_offsets[k];pos<m.member_offsets[k+1];++pos)sum+=groups[m.member_groups[pos]];
      const double rate=sum/(m.member_offsets[k+1]-m.member_offsets[k]);a.cell_dt[k]=rate>0?m.transport.viscous_cfl/rate:INFINITY;}
  }
  return 0;
}
int static_amr2d_viscous_diagnostics(const StaticAmrMesh2D& m,const Params2D& p,const std::vector<double>& u,AmrViscousFields2D* out){
  const size_t n=m.cells.size();if(u.size()!=(m.transport.sa_model?5:4)*n)return 1;std::vector<double>w(4*n);
  for(size_t k=0;k<n;++k){w[k]=u[k];w[n+k]=u[n+k]/u[k];w[2*n+k]=u[2*n+k]/u[k];w[3*n+k]=(p.gamma-1)*(u[3*n+k]-.5*(u[n+k]*w[n+k]+u[2*n+k]*w[2*n+k]));
    if(!(w[k]>0)||!(w[3*n+k]>0)||!std::isfinite(w[k])||!std::isfinite(w[3*n+k]))return 1;}
  return static_amr2d_viscous_evaluate(m,p,w,out,&u);
}
int static_amr2d_write_viscous_snapshot(const StaticAmrMesh2D& m,const Params2D& p,const StaticAmrState2D& state,const std::string& prefix,std::string* error){
  if(m.transport.model==AMR_MODEL_EULER)return 0;
  AmrViscousFields2D a;if(static_amr2d_viscous_diagnostics(m,p,state.u,&a))return failed(error,"viscous snapshot diagnostic failed");
  // Independent diagnostics of both limits; does not change time integration.
  std::vector<double> speed(m.cells.size(),0);
  auto primitive=[&](int k,double* w){w[0]=state.u[k];w[1]=state.u[m.cells.size()+k]/w[0];w[2]=state.u[2*m.cells.size()+k]/w[0];w[3]=(p.gamma-1)*(state.u[3*m.cells.size()+k]-.5*w[0]*(w[1]*w[1]+w[2]*w[2]));};
  for(const auto& face:m.faces){const int owner=face.left>=0?face.left:face.right;double l[4],r[4];primitive(face.left>=0?face.left:owner,l);primitive(face.right>=0?face.right:owner,r);
    if(face.boundary>=0&&!amr_bc_state(m.boundaries[face.boundary],p.bc[face.boundary],p.gamma,face.boundary,face.left<0?r:l,face.left<0?l:r))return failed(error,"timestep boundary diagnostic failed");
    const int normal=1+face.axis;const double rate=std::max(std::fabs(l[normal])+std::sqrt(p.gamma*l[3]/l[0]),std::fabs(r[normal])+std::sqrt(p.gamma*r[3]/r[0]))*face.length;
    if(face.left>=0)speed[face.left]+=rate;
    if(face.right>=0)speed[face.right]+=rate;
  }
  for(const auto& wall:m.walls){double w[4];primitive(wall.owner_cell,w);speed[wall.owner_cell]+=std::sqrt(p.gamma*w[3]/w[0])*wall.length;}
  double convective_dt=INFINITY;for(size_t k=0;k<m.cells.size();++k){const auto& c=m.cells[k];if(speed[k]>0)convective_dt=std::min(convective_dt,p.cfl*(m.options.agglomerate_small?c.volume:(c.x1-c.x0)*(c.y1-c.y0))/speed[k]);}
  std::ofstream limits(prefix+".timestep_limits.json");limits<<std::setprecision(17)<<"{\"convective_dt\":"<<convective_dt<<",\"viscous_dt\":"<<*std::min_element(a.cell_dt.begin(),a.cell_dt.end())<<",\"mode\":\""<<(m.transport.stabilization==AMR_VISCOUS_SRD?"srd-rate":"true-volume")<<"\",\"fixed_dt\":"<<p.dt_fixed<<"}\n";limits.close();if(!limits)return failed(error,"timestep limit output failed");
  const size_t n=m.cells.size(),nw=m.walls.size();std::vector<double> rows;rows.reserve(14*nw);std::map<int,std::array<double,6>> loads;std::array<double,6> total={};
  std::vector<double> wall_pressure;if(static_amr2d_wall_pressure(m,p,state.u,&wall_pressure))return failed(error,"viscous snapshot wall pressure failed");
  for(size_t q=0;q<nw;++q){const auto& w=m.walls[q];const auto& sample=m.viscous.wall_samples[q];const size_t k=w.owner_cell;const double pressure=wall_pressure[q];
    const double x=.5*(w.x0+w.x1),y=.5*(w.y0+w.y1),px=pressure*w.nx*w.length,py=pressure*w.ny*w.length,vx=-a.wall_flux[nw+q],vy=-a.wall_flux[2*nw+q];
    const std::array<double,6> values={px,py,x*py-y*px,vx,vy,x*vy-y*vx};auto& patch=loads[w.patch_id];for(int v=0;v<6;++v){patch[v]+=values[v];total[v]+=values[v];}
    for(double value:{static_cast<double>(q),static_cast<double>(k),static_cast<double>(sample.donor),sample.distance,a.mu[sample.donor],a.wall_flux[nw+q]/w.length,a.wall_flux[2*nw+q]/w.length,(-w.ny*a.wall_flux[nw+q]+w.nx*a.wall_flux[2*nw+q])/w.length,pressure,px,py,vx,vy,values[5]})rows.push_back(value);
  }
  if(static_amr_write_npy(prefix+".wall_stress.npy",rows.data(),nw,14,error))return 1;
  std::ofstream f(prefix+".viscous_loads.json");f<<std::setprecision(17)<<"{\"time\":"<<state.time<<",\"model\":\""<<(m.transport.sa_model?"sa-neg-mechanical":"laminar")<<"\",\"conductivity\":0,\"load_columns\":[\"fx_pressure\",\"fy_pressure\",\"mz_pressure\",\"fx_viscous\",\"fy_viscous\",\"mz_viscous\"],\"total\":[";
  for(int v=0;v<6;++v){if(v)f<<',';f<<total[v];}f<<"],\"patches\":[";bool first=true;for(const auto& [id,value]:loads){if(!first)f<<',';first=false;f<<"{\"patch\":"<<id<<",\"loads\":[";for(int v=0;v<6;++v){if(v)f<<',';f<<value[v];}f<<"]}";}
  f<<"],\"viscous_dt_min\":"<<*std::min_element(a.cell_dt.begin(),a.cell_dt.end())<<",\"wall_stress_columns\":[\"wall_id\",\"owner\",\"donor\",\"normal_distance\",\"mu\",\"tau_n_x\",\"tau_n_y\",\"tau_t_signed\",\"p\",\"fx_pressure\",\"fy_pressure\",\"fx_viscous\",\"fy_viscous\",\"mz_viscous\"],\"moment_reference\":[0,0],\"wall_normal\":\"fluid to solid\",\"wall_energy_flux\":0}\n";f.close();
  if(m.transport.sa_model){std::vector<double> rows;rows.reserve(3*n);for(size_t k=0;k<n;++k){const double phi=state.u[4*n+k]/state.u[k];rows.push_back(phi);rows.push_back(state.u[k]*sa_eddy_kinematic(phi,a.mu[k]/state.u[k]));rows.push_back(m.viscous.wall_distance[k]);}
    if(static_amr_write_npy(prefix+".sa.npy",rows.data(),n,3,error))return 1;
    std::ofstream desc(prefix+".sa.json");desc<<"{\"model\":\"SA-neg-ft2-Note1c-mechanical\",\"columns\":[\"nutilde\",\"mu_t\",\"wall_distance\"],\"nu_inf\":"<<m.transport.sa_nu_inf<<",\"wall_nutilde\":0,\"thermal_conductivity\":0}\n";
    if(!desc)return failed(error,"SA metadata failed");}
  return f?0:failed(error,"viscous load output failed");
}
int static_amr2d_write_viscous(const StaticAmrMesh2D& m,const Params2D&,const std::string& prefix,std::string* error){
  std::ofstream f(prefix+".transport.json");f<<std::setprecision(17);const auto& t=m.transport;
  f<<"{\"model\":\""<<(t.model==AMR_MODEL_EULER?"euler":t.sa_model?"sa-neg-mechanical":"laminar")<<"\",\"viscosity_law\":\""<<(t.law==AMR_MU_CONSTANT?"constant":"sutherland")<<"\",\"mu_ref\":"<<t.mu_ref<<",\"sa_model\":"<<t.sa_model<<",\"sa_nu_inf\":"<<t.sa_nu_inf<<",\"Rgas\":"<<t.gas_constant<<",\"T_ref\":"<<t.temperature_ref<<",\"S\":"<<t.sutherland<<",\"viscous_cfl\":"<<t.viscous_cfl<<",\"conductivity\":0,\"viscous_work\":true,\"body_wall\":\""<<(t.model==AMR_MODEL_EULER?"slip":"stationary no-slip")<<"\",\"open_viscous_bc\":\"zero normal velocity derivative, tangential derivative retained\",\"gradient_terms\":"<<m.viscous.terms.size()<<",\"wall_samples\":"<<m.viscous.wall_samples.size()<<"}\n";
  f.close();if(!f)return failed(error,"cannot write transport metadata");
  // Separate explicit metadata so downstream readers cannot confuse timestep modes.
  std::ofstream mode(prefix+".viscous_stabilization.json");
  mode<<"{\"mode\":\""<<(t.stabilization==AMR_VISCOUS_SRD?"srd-rate":"true-volume")<<"\",\"redistribution\":\"original zero-degree SRD\",\"wall_probe_min_h\":"<<t.wall_probe_min_h<<",\"general_stability_proof\":false}\n";
  mode.close();if(!mode)return failed(error,"cannot write viscous stabilization metadata");
  if(t.model==AMR_MODEL_EULER)return 0;
  std::ofstream g(prefix+".gradient_table.csv");g<<std::setprecision(17)<<"owner,source,side,cx,cy\n";for(size_t k=0;k<m.cells.size();++k)for(size_t q=m.viscous.offsets[k];q<m.viscous.offsets[k+1];++q){const auto& s=m.viscous.terms[q];g<<k<<','<<s.source<<','<<s.side<<','<<s.x<<','<<s.y<<'\n';}g.close();
  std::ofstream w(prefix+".wall_samples.csv");w<<std::setprecision(17)<<"wall_id,owner,donor,normal_distance\n";for(size_t q=0;q<m.walls.size();++q)w<<q<<','<<m.walls[q].owner_cell<<','<<m.viscous.wall_samples[q].donor<<','<<m.viscous.wall_samples[q].distance<<'\n';w.close();
  return g&&w?0:failed(error,"viscous table output failed");
}
}
