// Full 3D compressible finite volumes with Newtonian stress and NASA SA-neg.
// Shared CPU/CUDA operators; RK3 and matrix-free DP-LUR steady integration.
// The faceted no-slip sphere has zero Fourier heat flux (kappa=0).
#include "sa_model.h"
#include "low_mach.h"
#include "mesh_format.h"
#include "spatial_operators.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif
using cfd::sa_neg_point;
struct Config {
 double gamma=1.4,mach=1.65,mu=0,S=.36,cfl=.15,step_scale=1,dplur_omega=.8,mean_step_ratio=1,sa_step_ratio=1,reconstruction_blend=1,sa_diffusion_blend=1;
 bool precondition=false,dplur=false,dplur_block_coupling=false,node_gradient=false;
 int spatial_order=1,sa_spatial_order=1;
 double reference[6],rlo[3],rhi[3],moment_ref[3];
};
struct Data {
 int n,nf;Cell* cells;Face* faces;Term* terms;int* ids;
 double *u,*u1,*u2,*w,*gradient,*flux,*rates,*r,*raw,*dt,*mu,*eddy,*diff,*source,*source_rate;
 int* bad;
 int np=0;int *cell_nodes=nullptr,*node_cells=nullptr,*face_nodes=nullptr;uint64_t* node_offsets=nullptr;
 double *points=nullptr,*node_weights=nullptr,*cell_gradients=nullptr,*node_w=nullptr;
 double *partial_cells=nullptr,*partial_faces=nullptr,*report=nullptr;
 int mg_n1=0,mg_n2=0;
 int *mg_map01=nullptr,*mg_map12=nullptr,*mg_offsets1=nullptr,*mg_children1=nullptr,*mg_offsets2=nullptr,*mg_children2=nullptr;
 double *mg_volume1=nullptr,*mg_volume2=nullptr,*mg_residual1=nullptr,*mg_residual2=nullptr;
};
struct AgglomerationHost {
 int n1=0,n2=0;
 std::vector<int> map01,map12,offsets1,children1,offsets2,children2;
 std::vector<double> volume1,volume2,residual1,residual2;
};
struct NodeGradientHost {
 int np=0;
 std::vector<int> cell_nodes,node_cells,face_nodes;
 std::vector<uint64_t> node_offsets;
 std::vector<double> points,node_weights,cell_gradients,node_w;
};
template<class T>void read_table(std::ifstream& input,uint64_t offset,std::vector<T>& values){
 input.seekg(static_cast<std::streamoff>(offset));input.read(reinterpret_cast<char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(T)));
 if(!input)throw std::runtime_error("truncated companion table");
}
NodeGradientHost load_node_gradient(const std::filesystem::path& connectivity_path,const std::filesystem::path& gradient_path,size_t cell_count,size_t face_count){
 static_assert(sizeof(int)==sizeof(int32_t),"solver requires 32-bit int indices");std::ifstream connectivity(connectivity_path,std::ios::binary);ConnectivityHeader ch;connectivity.read(reinterpret_cast<char*>(&ch),sizeof(ch));
 if(!connectivity||std::string(ch.magic,7)!="QCFDNC1"||ch.version!=1||ch.endian!=0x01020304||ch.cell_count!=cell_count||ch.incidence_count!=4*ch.cell_count||ch.point_count>static_cast<uint64_t>(INT_MAX)||std::filesystem::file_size(connectivity_path)!=ch.file_bytes)throw std::runtime_error("incompatible node-gradient connectivity");
 NodeGradientHost data;data.np=static_cast<int>(ch.point_count);data.points.resize(3*ch.point_count);data.cell_nodes.resize(4*cell_count);data.node_offsets.resize(ch.point_count+1);data.node_cells.resize(ch.incidence_count);
 read_table(connectivity,ch.points_offset,data.points);read_table(connectivity,ch.cell_nodes_offset,data.cell_nodes);read_table(connectivity,ch.node_offsets_offset,data.node_offsets);read_table(connectivity,ch.node_cells_offset,data.node_cells);connectivity.close();
 std::ifstream gradient(gradient_path,std::ios::binary);NodeGradientHeader gh;gradient.read(reinterpret_cast<char*>(&gh),sizeof(gh));
 if(!gradient||std::string(gh.magic,7)!="QCFDNG1"||gh.version!=2||gh.endian!=0x01020304||gh.point_count!=ch.point_count||gh.cell_count!=ch.cell_count||gh.face_count!=face_count||gh.incidence_count!=ch.incidence_count||std::filesystem::file_size(gradient_path)!=gh.file_bytes)throw std::runtime_error("incompatible node-gradient coefficient file");
 data.node_weights.resize(ch.incidence_count);data.cell_gradients.resize(12*cell_count);data.face_nodes.resize(3*face_count);read_table(gradient,gh.node_weights_offset,data.node_weights);read_table(gradient,gh.cell_gradients_offset,data.cell_gradients);read_table(gradient,gh.face_nodes_offset,data.face_nodes);gradient.close();data.node_w.assign(6*ch.point_count,0);return data;
}
void build_child_csr(const std::vector<int>& parent,int parent_count,std::vector<int>& offsets,std::vector<int>& children){
 offsets.assign(static_cast<size_t>(parent_count)+1,0);for(int value:parent){if(value<0||value>=parent_count)throw std::runtime_error("agglomeration parent outside range");++offsets[static_cast<size_t>(value)+1];}
 for(int i=0;i<parent_count;++i)offsets[i+1]+=offsets[i];children.resize(parent.size());std::vector<int> cursor=offsets;
 for(size_t child=0;child<parent.size();++child)children[cursor[parent[child]]++]=static_cast<int>(child);
 for(int i=0;i<parent_count;++i)if(offsets[i]==offsets[i+1])throw std::runtime_error("empty agglomeration parent");
}
AgglomerationHost load_agglomeration(const std::filesystem::path& path,const std::vector<Cell>& cells){
 static_assert(sizeof(int)==sizeof(int32_t),"solver requires 32-bit int indices");std::ifstream input(path,std::ios::binary);AgglomerationHeader header;input.read(reinterpret_cast<char*>(&header),sizeof(header));
 if(!input||std::string(header.magic,7)!="QCFDMG1"||header.version!=1||header.endian!=0x01020304||header.level_count!=3||header.fine_cell_count!=cells.size()||std::filesystem::file_size(path)!=header.file_bytes)throw std::runtime_error("incompatible three-level agglomeration hierarchy");
 AgglomerationHost hierarchy;AgglomerationLevelHeader first;input.read(reinterpret_cast<char*>(&first),sizeof(first));if(first.child_count!=cells.size()||first.parent_count>static_cast<uint64_t>(INT32_MAX))throw std::runtime_error("invalid first agglomeration map");
 hierarchy.n1=static_cast<int>(first.parent_count);hierarchy.map01.resize(cells.size());input.read(reinterpret_cast<char*>(hierarchy.map01.data()),static_cast<std::streamsize>(hierarchy.map01.size()*sizeof(int)));
 AgglomerationLevelHeader second;input.read(reinterpret_cast<char*>(&second),sizeof(second));if(second.child_count!=first.parent_count||second.parent_count>static_cast<uint64_t>(INT32_MAX))throw std::runtime_error("invalid second agglomeration map");
 hierarchy.n2=static_cast<int>(second.parent_count);hierarchy.map12.resize(hierarchy.n1);input.read(reinterpret_cast<char*>(hierarchy.map12.data()),static_cast<std::streamsize>(hierarchy.map12.size()*sizeof(int)));if(!input||input.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("truncated or trailing agglomeration data");
 build_child_csr(hierarchy.map01,hierarchy.n1,hierarchy.offsets1,hierarchy.children1);build_child_csr(hierarchy.map12,hierarchy.n2,hierarchy.offsets2,hierarchy.children2);
 hierarchy.volume1.assign(hierarchy.n1,0);for(size_t k=0;k<cells.size();++k)hierarchy.volume1[hierarchy.map01[k]]+=cells[k].volume;
 hierarchy.volume2.assign(hierarchy.n2,0);for(int k=0;k<hierarchy.n1;++k)hierarchy.volume2[hierarchy.map12[k]]+=hierarchy.volume1[k];
 if(*std::min_element(hierarchy.volume1.begin(),hierarchy.volume1.end())<=0||*std::min_element(hierarchy.volume2.begin(),hierarchy.volume2.end())<=0)throw std::runtime_error("non-positive aggregate volume");
 hierarchy.residual1.assign(static_cast<size_t>(6)*hierarchy.n1,0);hierarchy.residual2.assign(static_cast<size_t>(6)*hierarchy.n2,0);return hierarchy;
}
HD bool finite_value(double x){
#ifdef __CUDA_ARCH__
 return isfinite(x);
#else
 return std::isfinite(x);
#endif
}
HD void fail_state(Data d,int code=1){
#ifdef __CUDA_ARCH__
 atomicCAS(d.bad,0,code);
#else
 if(!*d.bad)*d.bad=code;
#endif
}
HD double dot(const double* a,const double* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
HD void conservative(const double* w,double gm,double* u){
 u[0]=w[0];for(int j=0;j<3;++j)u[j+1]=w[0]*w[j+1];u[4]=w[4]/(gm-1)+.5*w[0]*dot(w+1,w+1);u[5]=w[0]*w[5];
}
HD bool farfield(const double* interior,const double* n,Config c,double* outside){
 for(int v=0;v<6;++v)outside[v]=interior[v];
 double un=dot(interior+1,n),ai=sqrt(c.gamma*interior[4]/interior[0]);
 if(un>=ai)return true;
 if(un<=-ai){for(int v=0;v<6;++v)outside[v]=c.reference[v];return true;}
 const double ur=dot(c.reference+1,n),ar=sqrt(c.gamma*c.reference[4]/c.reference[0]),gm=c.gamma-1;
 const double plus=un+2*ai/gm,minus=ur-2*ar/gm,ab=.25*gm*(plus-minus),ub=.5*(plus+minus);
 if(!(ab>0))return false;
 const double* adv=un>=0?interior:c.reference;const double entropy=adv[4]/pow(adv[0],c.gamma),ua=dot(adv+1,n);
 outside[0]=pow(ab*ab/(c.gamma*entropy),1/gm);outside[4]=entropy*pow(outside[0],c.gamma);outside[5]=adv[5];
 for(int j=0;j<3;++j)outside[j+1]=adv[j+1]+(ub-ua)*n[j];
 return outside[0]>0&&outside[4]>0&&finite_value(outside[4]);
}
HD void hllc(const double* l,const double* r,const double* n,double gm,double* f){
 double ul[6],ur[6],fl[5],fr[5];conservative(l,gm,ul);conservative(r,gm,ur);
 const double vl=dot(l+1,n),vr=dot(r+1,n),al=sqrt(gm*l[4]/l[0]),ar=sqrt(gm*r[4]/r[0]);
 fl[0]=l[0]*vl;fr[0]=r[0]*vr;
 for(int j=0;j<3;++j){fl[j+1]=ul[j+1]*vl+l[4]*n[j];fr[j+1]=ur[j+1]*vr+r[4]*n[j];}
 fl[4]=(ul[4]+l[4])*vl;fr[4]=(ur[4]+r[4])*vr;
 const double sl=fmin(vl-al,vr-ar),sr=fmax(vl+al,vr+ar);
 if(sl>=0){for(int j=0;j<5;++j)f[j]=fl[j];return;}
 if(sr<=0){for(int j=0;j<5;++j)f[j]=fr[j];return;}
 const double sm=(r[4]-l[4]+l[0]*vl*(sl-vl)-r[0]*vr*(sr-vr))/(l[0]*(sl-vl)-r[0]*(sr-vr));
 const bool left=sm>=0;const double* w=left?l:r;const double* u=left?ul:ur;const double* ff=left?fl:fr;
 const double s=left?sl:sr,v=left?vl:vr,rhostar=w[0]*(s-v)/(s-sm);double star[5];star[0]=rhostar;
 for(int j=0;j<3;++j)star[j+1]=rhostar*(w[j+1]+(sm-v)*n[j]);
 star[4]=rhostar*(u[4]/w[0]+(sm-v)*(sm+w[4]/(w[0]*(s-v))));
 for(int j=0;j<5;++j)f[j]=ff[j]+s*(star[j]-u[j]);
}
HD void primitive_point(Config c,Data d,const double* u,int k){
 if(*d.bad)return;const double* q=u+6*k;double* w=d.w+6*k;
 if(!(q[0]>0)||!finite_value(q[0])){fail_state(d,k+1);return;}
 w[0]=q[0];for(int j=0;j<3;++j)w[j+1]=q[j+1]/q[0];w[4]=(c.gamma-1)*(q[4]-.5*q[0]*dot(w+1,w+1));w[5]=q[5]/q[0];
 for(int j=0;j<6;++j)if(!finite_value(w[j])){fail_state(d,k+1);return;}
 if(!(w[4]>0))fail_state(d,k+1);
}
HD void node_primitive_point(Data d,int node){
 if(*d.bad)return;double* value=d.node_w+6*node;for(int v=0;v<6;++v)value[v]=0;
 for(uint64_t index=d.node_offsets[node];index<d.node_offsets[node+1];++index){const double weight=d.node_weights[index];const double* cell=d.w+6*d.node_cells[index];for(int v=0;v<6;++v)value[v]+=weight*cell[v];}
 for(int v=0;v<6;++v)if(!finite_value(value[v]))fail_state(d);
}
HD void gradient_point(Config c,Data d,int k){
 if(*d.bad)return;const Cell cell=d.cells[k];const double* w=d.w+6*k;double* g=d.gradient+18*k;double lower[6],upper[6];for(int i=0;i<18;++i)g[i]=0;for(int v=0;v<6;++v)lower[v]=upper[v]=w[v];
 if(c.node_gradient)for(int vertex=0;vertex<4;++vertex){const double* nodal=d.node_w+6*d.cell_nodes[4*k+vertex];const double* shape=d.cell_gradients+12*k+3*vertex;for(int v=0;v<6;++v)for(int j=0;j<3;++j)g[3*v+j]+=nodal[v]*shape[j];}
 for(int ti=cell.term_first;ti<cell.term_first+cell.term_count;++ti){const Term t=d.terms[ti];double wb[6];const double* neighbor;
  if(t.source>=0)neighbor=d.w+6*t.source;
  else{for(int v=0;v<6;++v)wb[v]=w[v];
   if(t.source==-1){wb[1]=wb[2]=wb[3]=wb[5]=0;}
   else if(t.source==-3){const auto f=d.faces[t.face];const double un=dot(w+1,f.n);for(int j=0;j<3;++j)wb[j+1]-=un*f.n[j];}
   else if(!farfield(w,d.faces[t.face].n,c,wb)){fail_state(d);return;}
   neighbor=wb;}
  if(!c.node_gradient)for(int v=0;v<6;++v)for(int j=0;j<3;++j)g[3*v+j]+=t.coefficient[j]*(neighbor[v]-w[v]);
  if(c.spatial_order==2)for(int v=0;v<6;++v){lower[v]=fmin(lower[v],neighbor[v]);upper[v]=fmax(upper[v],neighbor[v]);}
 }
 if(c.spatial_order==2){double limiter[6]={1,1,1,1,1,1};
  for(int ii=cell.first;ii<cell.first+cell.count;++ii){const Face f=d.faces[d.ids[ii]];double offset[3];for(int j=0;j<3;++j)offset[j]=f.x[j]-cell.x[j];
   for(int v=0;v<6;++v){const double change=dot(g+3*v,offset);
    if(change>0)limiter[v]=fmin(limiter[v],(upper[v]-w[v])/change);
    else if(change<0)limiter[v]=fmin(limiter[v],(lower[v]-w[v])/change);
   }
  }
  for(int v=0;v<6;++v)d.raw[6*k+v]=fmax(0.,fmin(1.,limiter[v]));
 }
 const double temperature=w[4]/w[0],tr=1/c.gamma;d.mu[k]=c.mu*pow(temperature/tr,1.5)*(tr+c.S)/(temperature+c.S);
 const double omega=sqrt((g[10]-g[8])*(g[10]-g[8])+(g[5]-g[9])*(g[5]-g[9])+(g[6]-g[4])*(g[6]-g[4]));
 const auto model=sa_neg_point(w[5],d.mu[k]/w[0],cell.dwall,omega);
 d.eddy[k]=w[0]*cfd::sa_eddy_kinematic(w[5],d.mu[k]/w[0]);d.diff[k]=model.diffusion;d.source_rate[k]=model.source_rate;
 d.source[k]=w[0]*(model.source+cfd::sa_cb2/cfd::sa_sigma*dot(g+15,g+15))-model.diffusion*dot(g,g+15);
 if(!(d.mu[k]>0)||!(d.diff[k]>0)||!finite_value(d.source[k])||!finite_value(d.eddy[k]))fail_state(d,-k-1);
}
HD void reconstruct_face_state(Config c,Data d,int cell,int neighbor,const Face& f,double* state){
 const double* center=d.w+6*cell;for(int v=0;v<6;++v)state[v]=center[v];
 if(c.spatial_order==1)return;
 double offset[3];for(int j=0;j<3;++j)offset[j]=f.x[j]-d.cells[cell].x[j];
 const int reconstructed_variables=c.sa_spatial_order==2?6:5;for(int v=0;v<reconstructed_variables;++v){
  const double increment=dot(d.gradient+18*cell+3*v,offset);
  state[v]+=c.reconstruction_blend*d.raw[6*cell+v]*increment;
 }
 if(!(state[0]>0&&state[4]>0&&finite_value(state[0])&&finite_value(state[4]))){state[0]=center[0];state[4]=center[4];}
}
HD bool mitchell_gradient_point(Data d,int face_index,int variable,double boundary_value,double* gradient){
 const Face face=d.faces[face_index];const int* nodes=d.face_nodes+3*face_index;const double* p0=d.points+3*nodes[0];double edge1[3],edge2[3],direction[3];
 for(int j=0;j<3;++j){edge1[j]=d.points[3*nodes[1]+j]-p0[j];edge2[j]=d.points[3*nodes[2]+j]-p0[j];direction[j]=(face.right>=0?d.cells[face.right].x[j]:face.x[j])-d.cells[face.left].x[j];}
 const double center_difference=face.right>=0?d.w[6*face.right+variable]-d.w[6*face.left+variable]:boundary_value-d.w[6*face.left+variable];
 const double edge1_difference=face.right>=0?d.node_w[6*nodes[1]+variable]-d.node_w[6*nodes[0]+variable]:0;
 const double edge2_difference=face.right>=0?d.node_w[6*nodes[2]+variable]-d.node_w[6*nodes[0]+variable]:0;
 return qcfd::mitchell_face_gradient(direction,edge1,edge2,center_difference,edge1_difference,edge2_difference,gradient);
}
HD void face_point(Config c,Data d,int fi){
 if(*d.bad)return;const Face f=d.faces[fi];const int l=f.left,r=f.right;const double* wl_cell=d.w+6*l;const double* wr_cell=r>=0?d.w+6*r:wl_cell;double wl[6],wr[6];
 reconstruct_face_state(c,d,l,r,f,wl);if(*d.bad)return;if(r>=0){reconstruct_face_state(c,d,r,l,f,wr);if(*d.bad)return;}else for(int v=0;v<6;++v)wr[v]=wl[v];
 if(r<0&&f.bc==2&&!farfield(wl,f.n,c,wr)){fail_state(d);return;}
 if(r<0&&f.bc==3){const double un=dot(wl+1,f.n);for(int j=0;j<3;++j)wr[j+1]-=2*un*f.n[j];}
 double flux[6]={},g[18],vel[3],mu=d.mu[l]+d.eddy[l],rho=wl_cell[0],diff=d.diff[l];
 for(int v=0;v<18;++v)g[v]=d.gradient[18*l+v];for(int j=0;j<3;++j)vel[j]=.5*(wl_cell[j+1]+wr_cell[j+1]);
 if(r<0&&(f.bc==1||f.bc==3)){
  for(int j=0;j<3;++j){flux[j+1]=wl[4]*f.n[j];vel[j]=0;}
  if(f.bc==1){
  if(c.node_gradient){for(int v=1;v<=3;++v)if(!mitchell_gradient_point(d,fi,v,0,g+3*v)){fail_state(d);return;}if(!mitchell_gradient_point(d,fi,5,0,g+15)){fail_state(d);return;}}
  else{for(int v=0;v<3;++v)for(int j=0;j<3;++j)g[3*(v+1)+j]=-wl_cell[v+1]*f.n[j]/f.distance;for(int j=0;j<3;++j)g[15+j]=-wl_cell[5]*f.n[j]/f.distance;}
  mu=d.mu[l];diff=mu/(rho*cfd::sa_sigma);flux[5]=-rho*diff*dot(g+15,f.n);
  }
 }else{
  if(c.precondition)cfd::lm_flux(wl,wr,f.n,c.gamma,c.mach,.02,flux);else hllc(wl,wr,f.n,c.gamma,flux);
  flux[5]=flux[0]*(flux[0]>=0?wl[5]:wr[5]);
  if(r>=0){
   if(c.node_gradient){for(int v=0;v<6;++v)if(!mitchell_gradient_point(d,fi,v,0,g+3*v)){fail_state(d);return;}}
   else{double delta[3];for(int j=0;j<3;++j)delta[j]=d.cells[r].x[j]-d.cells[l].x[j];for(int v=0;v<6;++v){double avg[3];for(int j=0;j<3;++j)avg[j]=.5*(d.gradient[18*l+3*v+j]+d.gradient[18*r+3*v+j]);qcfd::corrected_face_gradient(avg,delta,wr_cell[v]-wl_cell[v]);for(int j=0;j<3;++j)g[3*v+j]=avg[j];}}
   mu=.5*(d.mu[l]+d.eddy[l]+d.mu[r]+d.eddy[r]);rho=.5*(wl_cell[0]+wr_cell[0]);diff=.5*(d.diff[l]+d.diff[r]);
   // Diagnostic homotopy: only blend=1 recovers the selected high-order SA operator.
   double sa_normal=dot(g+15,f.n);
   if(c.sa_diffusion_blend<1){const double two_point=(wr_cell[5]-wl_cell[5])/f.distance;
    sa_normal=c.sa_diffusion_blend*sa_normal+(1-c.sa_diffusion_blend)*two_point;}
   flux[5]-=rho*diff*sa_normal;
  }
 }
 if(r>=0||f.bc==1){
  const double div=g[3]+g[7]+g[11];double stress[3]={};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)stress[i]+=mu*(g[3*(i+1)+j]+g[3*(j+1)+i]-(i==j?2*div/3:0))*f.n[j];
  for(int j=0;j<3;++j)flux[j+1]-=stress[j];flux[4]-=dot(stress,vel);
 }
 double speed=fmax(fabs(dot(wl+1,f.n))+sqrt(c.gamma*wl[4]/wl[0]),fabs(dot(wr+1,f.n))+sqrt(c.gamma*wr[4]/wr[0]));
 if(c.precondition)speed=cfd::lm_face_speed_normal(wl,wr,f.n,3,4,c.gamma,c.mach,.02);
 d.rates[3*fi]=speed*f.area;d.rates[3*fi+1]=(r>=0||f.bc==1)?(4./3)*mu/rho*f.area/f.distance:0;
 d.rates[3*fi+2]=(r>=0||f.bc==1)?rho*diff*f.area/f.distance:0;
 for(int v=0;v<6;++v){d.flux[6*fi+v]=flux[v]*f.area;if(!finite_value(flux[v]))fail_state(d);}
}
HD void residual_point(Config c,Data d,int k,bool base){
 if(*d.bad)return;const Cell cell=d.cells[k];double rr[6]={},spectral=0,viscous=0,sa=0;
 for(int i=cell.first;i<cell.first+cell.count;++i){int fi=d.ids[i];const Face f=d.faces[fi];const double sign=f.left==k?1:-1;
  for(int v=0;v<6;++v)rr[v]-=sign*d.flux[6*fi+v]/cell.volume;
  spectral+=d.rates[3*fi];viscous+=d.rates[3*fi+1];sa+=d.rates[3*fi+2];}
 rr[5]+=d.source[k];for(int v=0;v<6;++v)d.raw[6*k+v]=rr[v];
 if(c.precondition)cfd::lm_apply(rr,d.w+6*k,c.gamma,c.mach,.02,3,true);
 for(int v=0;v<6;++v){d.r[6*k+v]=rr[v];if(!finite_value(rr[v]))fail_state(d);}
 if(base){
  if(c.dplur){
   const double conv=spectral/cell.volume;
   const double mean_rate=conv+2*viscous/cell.volume;
   const double sa_rate=conv+2*sa/(d.w[6*k]*cell.volume)+fmax(d.source_rate[k],0.);
   d.dt[k]=c.step_scale*c.cfl/(fmax(mean_rate,sa_rate)+1e-100);
  }else d.dt[k]=c.step_scale*fmin(c.cfl*cell.volume/spectral,fmin(viscous>0?.1*cell.volume/viscous:1e100,
                         .1/(2*sa/(d.w[6*k]*cell.volume)+d.source_rate[k]+1e-100)));
 }
}
HD void update_point(Data d,int k,int stage){
 if(*d.bad)return;const double dt=d.dt[k];for(int v=0;v<6;++v){const int q=6*k+v;
  if(stage==0)d.u1[q]=d.u[q]+dt*d.r[q];
  else if(stage==1)d.u2[q]=.75*d.u[q]+.25*(d.u1[q]+dt*d.r[q]);
  else d.u1[q]=d.u[q]/3+(2./3)*(d.u2[q]+dt*d.r[q]);}
}
// Apply the 3-D Euler normal-flux Jacobian to a conservative perturbation.
// This supplies pressure/velocity/energy coupling without storing 25 values
// per face.  The implicit operator remains a compact first-order Rusanov
// approximation; the nonlinear residual keeps its selected physical flux.
HD void euler_flux_jacobian_action(const double* w,const double* dq,const double* n,double gm,double* action){
 const double rho=w[0],dr=dq[0],un=dot(w+1,n),q2=dot(w+1,w+1),energy=w[4]/(gm-1)+.5*rho*q2;
 double du[3],dun=0,velocity_work=0;
 for(int j=0;j<3;++j){du[j]=(dq[j+1]-w[j+1]*dr)/rho;dun+=du[j]*n[j];velocity_work+=w[j+1]*dq[j+1];}
 const double dp=(gm-1)*(dq[4]-velocity_work+.5*q2*dr);
 action[0]=dq[1]*n[0]+dq[2]*n[1]+dq[3]*n[2];
 for(int j=0;j<3;++j)action[j+1]=dq[j+1]*un+rho*w[j+1]*dun+dp*n[j];
 action[4]=(dq[4]+dp)*un+(energy+w[4])*dun;
}
// Matrix-free data-parallel implicit relaxation.  The diagonal contains a
// conservative spectral bound.  The optional block neighbor action retains
// the Euler cross-variable coupling; viscous and SA couplings stay scalar.
// Fixed Jacobi sweeps avoid a global reduction per sweep.
HD void dplur_diagonal(Config c,Data d,int k,double* diagonal){
 const Cell cell=d.cells[k];double conv=0,viscous=0,sa=0;
 for(int i=cell.first;i<cell.first+cell.count;++i){const int fi=d.ids[i];conv+=d.rates[3*fi];viscous+=d.rates[3*fi+1];sa+=d.rates[3*fi+2];}
 conv/=cell.volume;viscous/=cell.volume;sa/=d.w[6*k]*cell.volume;
 diagonal[0]=conv;for(int v=1;v<5;++v)diagonal[v]=conv+2*viscous;
 diagonal[5]=conv+2*sa+fmax(d.source_rate[k],0.);
 for(int v=0;v<6;++v)diagonal[v]=1+d.dt[k]*(v==5?c.sa_step_ratio:c.mean_step_ratio)*diagonal[v];
}
HD void dplur_initialize_point(Config c,Data d,int k){
 if(*d.bad)return;double diagonal[6];dplur_diagonal(c,d,k,diagonal);
 for(int v=0;v<6;++v)d.u1[6*k+v]=d.dt[k]*(v==5?c.sa_step_ratio:c.mean_step_ratio)*d.r[6*k+v]/diagonal[v];
}
HD void dplur_relax_point(Config c,Data d,int k){
 if(*d.bad)return;const Cell cell=d.cells[k];double diagonal[6],sum[6]={};dplur_diagonal(c,d,k,diagonal);
 for(int i=cell.first;i<cell.first+cell.count;++i){const int fi=d.ids[i];const Face face=d.faces[fi];const int neighbor=face.left==k?face.right:face.left;if(neighbor<0)continue;
  const double conv=.5*d.rates[3*fi]/cell.volume,viscous=d.rates[3*fi+1]/cell.volume,sa=d.rates[3*fi+2]/(d.w[6*k]*cell.volume);
  const double* neighbor_update=d.u1+6*neighbor;
  if(c.dplur_block_coupling){
   double outward_normal[3],jacobian_action[5];const double orientation=face.left==k?1.:-1.;
   for(int j=0;j<3;++j)outward_normal[j]=orientation*face.n[j];
   euler_flux_jacobian_action(d.w+6*neighbor,neighbor_update,outward_normal,c.gamma,jacobian_action);
   const double face_scale=.5*face.area/cell.volume;
   for(int v=0;v<5;++v)sum[v]+=conv*neighbor_update[v]-face_scale*jacobian_action[v];
  }else for(int v=0;v<5;++v)sum[v]+=conv*neighbor_update[v];
  for(int v=1;v<5;++v)sum[v]+=viscous*neighbor_update[v];sum[5]+=(conv+sa)*neighbor_update[5];
 }
 for(int v=0;v<6;++v){const int q=6*k+v;const double pseudo_dt=d.dt[k]*(v==5?c.sa_step_ratio:c.mean_step_ratio),next=pseudo_dt*(d.r[q]+sum[v])/diagonal[v];d.u2[q]=(1-c.dplur_omega)*d.u1[q]+c.dplur_omega*next;if(!finite_value(d.u2[q]))fail_state(d);}
}
HD void dplur_apply_point(Data d,int k){
 if(*d.bad)return;for(int v=0;v<6;++v){const int q=6*k+v;d.u2[q]=d.u[q]+d.u1[q];}
}
HD void restrict_level0_point(Data d,int parent){
 double integrated[6]={};for(int index=d.mg_offsets1[parent];index<d.mg_offsets1[parent+1];++index){const int child=d.mg_children1[index],q=6*child;const double volume=d.cells[child].volume;for(int v=0;v<6;++v)integrated[v]+=volume*d.r[q+v];}
 for(int v=0;v<6;++v)d.mg_residual1[6*parent+v]=integrated[v]/d.mg_volume1[parent];
}
HD void restrict_level1_point(Data d,int parent){
 double integrated[6]={};for(int index=d.mg_offsets2[parent];index<d.mg_offsets2[parent+1];++index){const int child=d.mg_children2[index],q=6*child;const double volume=d.mg_volume1[child];for(int v=0;v<6;++v)integrated[v]+=volume*d.mg_residual1[q+v];}
 for(int v=0;v<6;++v)d.mg_residual2[6*parent+v]=integrated[v]/d.mg_volume2[parent];
}
HD void smooth_residual_point(Data d,int k,double mean_weight,double sa_weight){
 const int parent1=d.mg_map01[k],parent2=d.mg_map12[parent1];for(int v=0;v<6;++v){const int q=6*k+v;const double weight=v==5?sa_weight:mean_weight,coarse=.5*(d.mg_residual1[6*parent1+v]+d.mg_residual2[6*parent2+v]);d.r[q]=(1-weight)*d.r[q]+weight*coarse;}
}
#ifdef __CUDACC__
__global__ void primitives(Config c,Data d,const double* u){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)primitive_point(c,d,u,k);}
__global__ void node_primitives(Data d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.np)node_primitive_point(d,k);}
__global__ void gradients(Config c,Data d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)gradient_point(c,d,k);}
__global__ void faces(Config c,Data d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.nf)face_point(c,d,k);}
__global__ void residuals(Config c,Data d,bool base){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)residual_point(c,d,k,base);}
__global__ void updates(Data d,int stage){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)update_point(d,k,stage);}
__global__ void dplur_initialize(Config c,Data d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)dplur_initialize_point(c,d,k);}
__global__ void dplur_relax(Config c,Data d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)dplur_relax_point(c,d,k);}
__global__ void dplur_apply(Data d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)dplur_apply_point(d,k);}
__global__ void restrict_level0(Data d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.mg_n1)restrict_level0_point(d,k);}
__global__ void restrict_level1(Data d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.mg_n2)restrict_level1_point(d,k);}
__global__ void smooth_residual(Data d,double mean_weight,double sa_weight){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)smooth_residual_point(d,k,mean_weight,sa_weight);}
// Diagnostics stay on-device: only thirteen reduced scalars cross PCIe.
__global__ void reduce_cells(Config c,Data d){
 __shared__ double values[7][128];const int lane=threadIdx.x,k=blockIdx.x*128+lane;double local[7]={};
 if(k<d.n){const Cell cell=d.cells[k];bool near=true;for(int j=0;j<3;++j)if(cell.x[j]<c.rlo[j]||cell.x[j]>c.rhi[j])near=false;
  if(near){local[0]=cell.volume;for(int v=0;v<6;++v)local[v+1]=cell.volume*fabs(d.raw[6*k+v]);}}
 for(int v=0;v<7;++v)values[v][lane]=local[v];__syncthreads();
 for(int stride=64;stride;stride/=2){if(lane<stride)for(int v=0;v<7;++v)values[v][lane]+=values[v][lane+stride];__syncthreads();}
 if(lane==0)for(int v=0;v<7;++v)d.partial_cells[7*blockIdx.x+v]=values[v][0];
}
__global__ void reduce_faces(Config c,Data d){
 __shared__ double values[6][128];const int lane=threadIdx.x,k=blockIdx.x*128+lane;double local[6]={};
 if(k<d.nf){const Face face=d.faces[k];if(face.right<0&&face.bc==1){double arm[3];for(int j=0;j<3;++j){local[j]=d.flux[6*k+j+1];arm[j]=face.x[j]-c.moment_ref[j];}
  for(int j=0;j<3;++j)local[j+3]=arm[(j+1)%3]*local[(j+2)%3]-arm[(j+2)%3]*local[(j+1)%3];}}
 for(int v=0;v<6;++v)values[v][lane]=local[v];__syncthreads();
 for(int stride=64;stride;stride/=2){if(lane<stride)for(int v=0;v<6;++v)values[v][lane]+=values[v][lane+stride];__syncthreads();}
 if(lane==0)for(int v=0;v<6;++v)d.partial_faces[6*blockIdx.x+v]=values[v][0];
}
__global__ void reduce_report(Data d){
 __shared__ double values[13][128];const int lane=threadIdx.x,nc=(d.n+127)/128,nf=(d.nf+127)/128;
 for(int v=0;v<13;++v){double sum=0;if(v<7){for(int k=lane;k<nc;k+=128)sum+=d.partial_cells[7*k+v];}
  else for(int k=lane;k<nf;k+=128)sum+=d.partial_faces[6*k+v-7];values[v][lane]=sum;}
 __syncthreads();for(int stride=64;stride;stride/=2){if(lane<stride)for(int v=0;v<13;++v)values[v][lane]+=values[v][lane+stride];__syncthreads();}
 if(lane==0)for(int v=0;v<13;++v)d.report[v]=values[v][0];
}
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#endif
int main(int argc,char** argv)try{
 const auto process_start=std::chrono::steady_clock::now();
 std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("all options need values");args[argv[i]]=argv[i+1];}
 const auto num=[&](std::string key,double value){return args.count(key)?std::stod(args[key]):value;};
 if(!args.count("--mesh")||!args.count("--out"))throw std::runtime_error("usage: sphere3d --mesh mesh.bin --out DIRECTORY [--gradient legacy|node --connectivity connectivity.bin --node-gradient node_gradient.bin --sa-diffusion-blend 0..1 --backend cuda|cpu --integrator rk3|dplur --mach 1.65 --re 10000 --budget 120 --iterations 50000 --residual-audit 1]");
 const bool gpu=args.count("--backend")&&args["--backend"]=="cuda";const std::string integrator=args.count("--integrator")?args["--integrator"]:"rk3";if(integrator!="rk3"&&integrator!="dplur")throw std::runtime_error("integrator must be rk3 or dplur");
 const std::string gradient_scheme=args.count("--gradient")?args["--gradient"]:"legacy";if(gradient_scheme!="legacy"&&gradient_scheme!="node")throw std::runtime_error("gradient must be legacy or node");
 Config c;c.dplur=integrator=="dplur";c.node_gradient=gradient_scheme=="node";c.mach=num("--mach",1.65);const double re=num("--re",10000),budget=num("--budget",120),tol=num("--tolerance",1e-4);
 c.mu=num("--mu-star",2*c.mach/re);c.cfl=num("--cfl",c.dplur?1.:.15);c.step_scale=num("--step-scale",1);c.dplur_omega=num("--dplur-omega",.8);c.mean_step_ratio=num("--mean-step-ratio",1);c.sa_step_ratio=num("--sa-step-ratio",1);c.sa_diffusion_blend=num("--sa-diffusion-blend",1);if(c.sa_diffusion_blend<0||c.sa_diffusion_blend>1)throw std::runtime_error("SA diffusion blend must be between 0 and 1");c.precondition=num("--precondition",0)!=0;c.spatial_order=static_cast<int>(num("--spatial-order",1));c.sa_spatial_order=static_cast<int>(num("--sa-spatial-order",1));c.dplur_block_coupling=num("--dplur-block-coupling",0)!=0;c.reconstruction_blend=num("--reconstruction-blend",c.spatial_order==2?1.:0.);c.S=110.4/(c.gamma*216.65);
 const double cfl_max=num("--cfl-max",c.dplur?20.:.4),cfl_growth=num("--cfl-growth",c.dplur?1.1:1.05),cfl_cut_ratio=num("--cfl-cut-ratio",1.05),cfl_reversal_ratio=num("--cfl-reversal-ratio",1.0005),rollback_ratio=num("--rollback-ratio",1.15),reconstruction_target=num("--reconstruction-target",1),reconstruction_step=num("--reconstruction-step",.1);
 const double agglomeration_smoothing=num("--agglomeration-smoothing",0),agglomeration_sa_smoothing=num("--agglomeration-sa-smoothing",0);const bool use_agglomeration=agglomeration_smoothing>0||agglomeration_sa_smoothing>0;
 const int dplur_sweeps=static_cast<int>(num("--dplur-sweeps",5)),cfl_growth_wait=static_cast<int>(num("--cfl-growth-wait",3)),cfl_reversal_wait=static_cast<int>(num("--cfl-reversal-wait",3)),cfl_cut_cooldown=static_cast<int>(num("--cfl-cut-cooldown",6)),reconstruction_wait=static_cast<int>(num("--reconstruction-wait",3)),controller_start_iteration=static_cast<int>(num("--controller-start-iteration",500));
 const bool reconstruction_ramp=num("--reconstruction-ramp",0)!=0,rollback_enabled=num("--rollback",1)!=0;
 const double area_ref=num("--reference-area",std::acos(-1.));
 const double moment_ref[3]={num("--moment-x",0),num("--moment-y",0),num("--moment-z",0)};
 const double angle=num("--aoa",0)*std::acos(-1.)/180;c.reference[0]=1;c.reference[1]=c.mach*cos(angle);c.reference[2]=0;c.reference[3]=c.mach*sin(angle);c.reference[4]=1/c.gamma;c.reference[5]=3*c.mu;
 if(!(c.mach>0&&re>0&&budget>0&&c.cfl>0&&c.cfl<=cfl_max&&cfl_max>0&&cfl_growth>1&&cfl_growth<=1.5&&cfl_cut_ratio>1&&cfl_reversal_ratio>1&&cfl_reversal_ratio<cfl_cut_ratio&&rollback_ratio>=cfl_cut_ratio&&cfl_growth_wait>=1&&cfl_reversal_wait>=1&&cfl_cut_cooldown>=0&&controller_start_iteration>=0&&c.step_scale>0&&c.step_scale<=1&&tol>0&&c.dplur_omega>0&&c.dplur_omega<=1&&c.mean_step_ratio>=0&&c.mean_step_ratio<=10&&c.sa_step_ratio>=0&&c.sa_step_ratio<=10&&(c.mean_step_ratio>0||c.sa_step_ratio>0)&&dplur_sweeps>=1&&dplur_sweeps<=50&&(c.spatial_order==1||c.spatial_order==2)&&(c.sa_spatial_order==1||c.sa_spatial_order==2)&&c.reconstruction_blend>=0&&c.reconstruction_blend<=reconstruction_target&&reconstruction_target<=1&&reconstruction_step>0&&reconstruction_step<=1&&reconstruction_wait>=1&&agglomeration_smoothing>=0&&agglomeration_smoothing<=1&&agglomeration_sa_smoothing>=0&&agglomeration_sa_smoothing<=1&&(!use_agglomeration||(c.dplur&&args.count("--agglomeration")))&&(!reconstruction_ramp||c.spatial_order==2)))throw std::runtime_error("invalid physical or iteration options");
 if(!c.dplur&&cfl_max>.45)throw std::runtime_error("RK3 cfl-max must not exceed 0.45");
 const bool restarted=args.count("--restart");
 if(restarted&&std::filesystem::weakly_canonical(args["--out"])==std::filesystem::weakly_canonical(std::filesystem::path(args["--restart"]).parent_path()))throw std::runtime_error("restart output must use a new directory to preserve the input checkpoint");
 int n,nf,ni,nt;std::ifstream mesh(args["--mesh"],std::ios::binary);char magic[8]={};mesh.read(magic,8);mesh.seekg(0);const bool binary_mesh=std::string(magic,7)=="QCFD3D1";
 if(binary_mesh){MeshHeader h;mesh.read(reinterpret_cast<char*>(&h),sizeof(h));if(h.n>100000000||h.nf>1000000000||h.ni>2000000000||h.nt>2000000000)throw std::runtime_error("mesh exceeds index limits");n=h.n;nf=h.nf;ni=h.ni;nt=h.nt;}
 else mesh>>n>>nf>>ni>>nt;
 if(!mesh||n<1||nf<1)throw std::runtime_error("invalid mesh header");
 std::vector<Cell> hc(n);std::vector<Face> hf(nf);std::vector<Term> ht(nt);std::vector<int> hi(ni);
 if(binary_mesh){mesh.read(reinterpret_cast<char*>(hc.data()),hc.size()*sizeof(Cell));mesh.read(reinterpret_cast<char*>(hf.data()),hf.size()*sizeof(Face));mesh.read(reinterpret_cast<char*>(hi.data()),hi.size()*sizeof(int));mesh.read(reinterpret_cast<char*>(ht.data()),ht.size()*sizeof(Term));}
 else{
 for(auto& a:hc)mesh>>a.x[0]>>a.x[1]>>a.x[2]>>a.volume>>a.dwall>>a.first>>a.count>>a.term_first>>a.term_count;
 for(auto& f:hf)mesh>>f.left>>f.right>>f.bc>>f.x[0]>>f.x[1]>>f.x[2]>>f.n[0]>>f.n[1]>>f.n[2]>>f.area>>f.distance;
 for(int& i:hi)mesh>>i;for(auto& t:ht)mesh>>t.source>>t.face>>t.coefficient[0]>>t.coefficient[1]>>t.coefficient[2];}
 if(!mesh)throw std::runtime_error("truncated mesh");
 const double mesh_read_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-process_start).count();mesh.close();
 NodeGradientHost node_gradient;if(c.node_gradient){if(!binary_mesh||!args.count("--connectivity")||!args.count("--node-gradient"))throw std::runtime_error("node gradient requires binary mesh, --connectivity, and --node-gradient");node_gradient=load_node_gradient(args["--connectivity"],args["--node-gradient"],hc.size(),hf.size());std::cout<<"Loaded node gradient for "<<node_gradient.np<<" points, "<<n<<" tetrahedra, and "<<nf<<" faces\n";}
 AgglomerationHost hierarchy;if(use_agglomeration)hierarchy=load_agglomeration(args["--agglomeration"],hc);
 double residual_lo[3]={1e100,1e100,1e100},residual_hi[3]={-1e100,-1e100,-1e100};
 for(const auto& f:hf)if(f.right<0&&f.bc==1)for(int j=0;j<3;++j){residual_lo[j]=std::min(residual_lo[j],f.x[j]);residual_hi[j]=std::max(residual_hi[j],f.x[j]);}
 for(int j=0;j<3;++j){residual_lo[j]-=num("--residual-padding",2);residual_hi[j]+=num("--residual-padding",2);}
 for(int j=0;j<3;++j){c.rlo[j]=residual_lo[j];c.rhi[j]=residual_hi[j];c.moment_ref[j]=moment_ref[j];}
 std::vector<double> hu(6*n),hu1(6*n),hu2(6*n),hw(6*n),hg(18*n),hflux(6*nf),hrates(3*nf),hr(6*n),hraw(6*n),hdt(n),hmu(n),heddy(n),hdiff(n),hsource(n),hsrate(n),hrollback(rollback_enabled?6*n:0);
 double initial[6];conservative(c.reference,c.gamma,initial);for(int k=0;k<n;++k)std::copy(initial,initial+6,hu.data()+6*k);int bad=0;
 // Legacy checkpoints have no mesh identity header. Reuse the exact mesh and cell ordering.
 // This is a solution restart; the CFL controller and convergence hold start afresh.
 if(restarted){
  const auto bytes=hu.size()*sizeof(double);
  if(std::filesystem::file_size(args["--restart"])!=bytes)throw std::runtime_error("restart size does not match six float64 conserved values per cell");
  std::ifstream state(args["--restart"],std::ios::binary);state.read(reinterpret_cast<char*>(hu.data()),bytes);
  if(!state)throw std::runtime_error("cannot read complete restart state");
  for(int k=0;k<n;++k){const double* q=hu.data()+6*k;for(int v=0;v<6;++v)if(!std::isfinite(q[v]))throw std::runtime_error("restart contains non-finite conserved values");
   if(!(q[0]>0&&q[4]-.5*dot(q+1,q+1)/q[0]>0))throw std::runtime_error("restart contains non-positive density or internal energy");}
  std::cout<<"Loaded restart state for "<<n<<" cells; controller history resets\n";
 }
 Data d{n,nf,hc.data(),hf.data(),ht.data(),hi.data(),hu.data(),hu1.data(),hu2.data(),hw.data(),hg.data(),hflux.data(),hrates.data(),hr.data(),hraw.data(),hdt.data(),hmu.data(),heddy.data(),hdiff.data(),hsource.data(),hsrate.data(),&bad};
 if(c.node_gradient){d.np=node_gradient.np;d.points=node_gradient.points.data();d.cell_nodes=node_gradient.cell_nodes.data();d.node_offsets=node_gradient.node_offsets.data();d.node_cells=node_gradient.node_cells.data();d.face_nodes=node_gradient.face_nodes.data();d.node_weights=node_gradient.node_weights.data();d.cell_gradients=node_gradient.cell_gradients.data();d.node_w=node_gradient.node_w.data();}
 if(use_agglomeration){d.mg_n1=hierarchy.n1;d.mg_n2=hierarchy.n2;d.mg_map01=hierarchy.map01.data();d.mg_map12=hierarchy.map12.data();d.mg_offsets1=hierarchy.offsets1.data();d.mg_children1=hierarchy.children1.data();d.mg_offsets2=hierarchy.offsets2.data();d.mg_children2=hierarchy.children2.data();d.mg_volume1=hierarchy.volume1.data();d.mg_volume2=hierarchy.volume2.data();d.mg_residual1=hierarchy.residual1.data();d.mg_residual2=hierarchy.residual2.data();}
#ifdef __CUDACC__
 std::vector<void*> allocations;size_t allocated_bytes=0;
 const auto upload=[&](auto*& ptr,size_t count){using T=std::remove_pointer_t<std::decay_t<decltype(ptr)>>;T* device=nullptr;checked(cudaMalloc(&device,count*sizeof(T)));if(ptr)checked(cudaMemcpy(device,ptr,count*sizeof(T),cudaMemcpyHostToDevice));ptr=device;allocations.push_back(device);allocated_bytes+=count*sizeof(T);};
 if(gpu){upload(d.cells,n);upload(d.faces,nf);upload(d.terms,nt);upload(d.ids,ni);for(double** ptr:{&d.u,&d.u1,&d.u2,&d.w,&d.r,&d.raw})upload(*ptr,6*n);
  upload(d.gradient,18*n);upload(d.flux,6*nf);upload(d.rates,3*nf);for(double** ptr:{&d.dt,&d.mu,&d.eddy,&d.diff,&d.source,&d.source_rate})upload(*ptr,n);upload(d.bad,1);
  if(c.node_gradient){upload(d.points,3*d.np);upload(d.cell_nodes,4*n);upload(d.node_offsets,d.np+1);upload(d.node_cells,4*n);upload(d.face_nodes,3*nf);upload(d.node_weights,4*n);upload(d.cell_gradients,12*n);upload(d.node_w,6*d.np);}
  upload(d.partial_cells,7*((n+127)/128));upload(d.partial_faces,6*((nf+127)/128));upload(d.report,13);
  if(use_agglomeration){upload(d.mg_map01,n);upload(d.mg_map12,d.mg_n1);upload(d.mg_offsets1,d.mg_n1+1);upload(d.mg_children1,n);upload(d.mg_offsets2,d.mg_n2+1);upload(d.mg_children2,d.mg_n1);upload(d.mg_volume1,d.mg_n1);upload(d.mg_volume2,d.mg_n2);upload(d.mg_residual1,6*d.mg_n1);upload(d.mg_residual2,6*d.mg_n2);}}
#else
 if(gpu)throw std::runtime_error("this binary has no CUDA backend");
#endif
 const auto set_bad=[&](int value){bad=value;
#ifdef __CUDACC__
  if(gpu)checked(cudaMemcpy(d.bad,&bad,sizeof(int),cudaMemcpyHostToDevice));
#endif
 };
 const auto check_bad=[&](){
#ifdef __CUDACC__
  if(gpu){checked(cudaGetLastError());checked(cudaMemcpy(&bad,d.bad,sizeof(int),cudaMemcpyDeviceToHost));}
#endif
  return bad!=0;
 };
 const auto prim=[&](const double* state){
#ifdef __CUDACC__
  if(gpu){primitives<<<(n+127)/128,128>>>(c,d,state);return;}
#endif
  for(int k=0;k<n;++k)primitive_point(c,d,state,k);
 };
 const auto rhs=[&](const double* state,bool base){prim(state);
#ifdef __CUDACC__
  if(gpu){if(c.node_gradient)node_primitives<<<(d.np+127)/128,128>>>(d);gradients<<<(n+127)/128,128>>>(c,d);faces<<<(nf+127)/128,128>>>(c,d);residuals<<<(n+127)/128,128>>>(c,d,base);return;}
#endif
  if(c.node_gradient)for(int k=0;k<d.np;++k)node_primitive_point(d,k);for(int k=0;k<n;++k)gradient_point(c,d,k);for(int k=0;k<nf;++k)face_point(c,d,k);for(int k=0;k<n;++k)residual_point(c,d,k,base);
 };
 const auto update=[&](int stage){
#ifdef __CUDACC__
  if(gpu){updates<<<(n+127)/128,128>>>(d,stage);return;}
#endif
  for(int k=0;k<n;++k)update_point(d,k,stage);
 };
 const auto multilevel_smooth=[&](){if(!use_agglomeration)return;
#ifdef __CUDACC__
  if(gpu){restrict_level0<<<(d.mg_n1+127)/128,128>>>(d);restrict_level1<<<(d.mg_n2+127)/128,128>>>(d);smooth_residual<<<(n+127)/128,128>>>(d,agglomeration_smoothing,agglomeration_sa_smoothing);return;}
#endif
  for(int k=0;k<d.mg_n1;++k)restrict_level0_point(d,k);for(int k=0;k<d.mg_n2;++k)restrict_level1_point(d,k);for(int k=0;k<n;++k)smooth_residual_point(d,k,agglomeration_smoothing,agglomeration_sa_smoothing);
 };
 const auto implicit_update=[&](){
  multilevel_smooth();
#ifdef __CUDACC__
  if(gpu){dplur_initialize<<<(n+127)/128,128>>>(c,d);for(int sweep=1;sweep<dplur_sweeps;++sweep){dplur_relax<<<(n+127)/128,128>>>(c,d);std::swap(d.u1,d.u2);}dplur_apply<<<(n+127)/128,128>>>(d);return;}
#endif
  for(int k=0;k<n;++k)dplur_initialize_point(c,d,k);for(int sweep=1;sweep<dplur_sweeps;++sweep){for(int k=0;k<n;++k)dplur_relax_point(c,d,k);std::swap(d.u1,d.u2);}for(int k=0;k<n;++k)dplur_apply_point(d,k);
 };
 const auto download=[&](){
#ifdef __CUDACC__
  if(gpu){checked(cudaMemcpy(hu.data(),d.u,6*n*sizeof(double),cudaMemcpyDeviceToHost));checked(cudaMemcpy(hw.data(),d.w,6*n*sizeof(double),cudaMemcpyDeviceToHost));checked(cudaMemcpy(hraw.data(),d.raw,6*n*sizeof(double),cudaMemcpyDeviceToHost));checked(cudaMemcpy(hflux.data(),d.flux,6*nf*sizeof(double),cudaMemcpyDeviceToHost));checked(cudaMemcpy(heddy.data(),d.eddy,n*sizeof(double),cudaMemcpyDeviceToHost));return;}
#endif
  if(d.u!=hu.data())std::copy(d.u,d.u+6*n,hu.data());
 };
 const auto save_rollback_state=[&](){if(!rollback_enabled)return;
#ifdef __CUDACC__
  if(gpu){checked(cudaMemcpy(hrollback.data(),d.u,6*n*sizeof(double),cudaMemcpyDeviceToHost));return;}
#endif
  std::copy(d.u,d.u+6*n,hrollback.data());
 };
 const auto restore_rollback_state=[&](){if(!rollback_enabled)throw std::runtime_error("rollback requested without storage");
#ifdef __CUDACC__
  if(gpu){checked(cudaMemcpy(d.u,hrollback.data(),6*n*sizeof(double),cudaMemcpyHostToDevice));return;}
#endif
  std::copy(hrollback.data(),hrollback.data()+6*n,d.u);
 };
 const std::filesystem::path out=args["--out"];std::filesystem::create_directories(out);std::ofstream history(out/"history.csv");history<<"iteration,seconds,cfl,reconstruction_blend,res_rho,res_mx,res_my,res_mz,res_E,res_SA,res_mean_max,res_total_max,CD,CL,CM,CY,CZ,rejected,step_scale,cfl_reductions,rollbacks,reconstruction_ramps,reversal_checks,growth_cooldown,controller_action\n";
 const auto start=std::chrono::steady_clock::now();const auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
 double norms[6]={},force[3]={},moment[3]={},imbalance[6]={},checkpoint_norms[6]={},initial_norm=0,previous_mean=INFINITY,previous_sa=INFINITY,previous_cd=0,previous_cl=0,previous_cm=0,controller_scale_mean=1,controller_scale_sa=1,best_controller_metric=INFINITY,checkpoint_blend=c.reconstruction_blend,checkpoint_cfl=c.cfl,checkpoint_cd=0,checkpoint_cl=0,checkpoint_cm=0;
 int hold=0,rejected=0,rejected_at_last_check=0,cfl_reductions=0,positivity_cfl_reductions=0,rollbacks=0,reconstruction_ramps=0,stable_checks=0,reversal_checks=0,growth_cooldown=0,checkpoint_iteration=0,iteration=0;bool converged=false,failed=false,controller_baseline=false,checkpoint_valid=false;
 const int maxiter=static_cast<int>(num("--iterations",50000)),check_period=static_cast<int>(num("--check-period",100));
 if(maxiter<0||check_period<1)throw std::runtime_error("iterations must be nonnegative and check-period positive");
 for(;;){
  if(iteration%check_period==0||iteration==maxiter||elapsed()>=budget){
   set_bad(0);rhs(d.u,false);if(check_bad())throw std::runtime_error("invalid accepted state");
   double volume=0;for(double& x:norms)x=0;for(double& x:force)x=0;for(double& x:moment)x=0;for(double& x:imbalance)x=0;
   bool device_report=false;
#ifdef __CUDACC__
   if(gpu){reduce_cells<<<(n+127)/128,128>>>(c,d);reduce_faces<<<(nf+127)/128,128>>>(c,d);reduce_report<<<1,128>>>(d);double report[13];checked(cudaGetLastError());checked(cudaMemcpy(report,d.report,sizeof(report),cudaMemcpyDeviceToHost));
    volume=report[0];for(int v=0;v<6;++v)norms[v]=report[v+1];for(int j=0;j<3;++j){force[j]=report[j+7];moment[j]=report[j+10];}device_report=true;}
#endif
   if(!device_report){download();
   for(int k=0;k<n;++k){bool near=true;for(int j=0;j<3;++j)if(hc[k].x[j]<residual_lo[j]||hc[k].x[j]>residual_hi[j])near=false;
    if(near){volume+=hc[k].volume;for(int v=0;v<6;++v)norms[v]+=hc[k].volume*fabs(hraw[6*k+v]);}
    for(int v=0;v<6;++v)imbalance[v]+=hc[k].volume*hraw[6*k+v];}
   for(int fi=0;fi<nf;++fi)if(hf[fi].right<0&&hf[fi].bc==1){double arm[3],load[3];for(int j=0;j<3;++j){load[j]=hflux[6*fi+j+1];force[j]+=load[j];arm[j]=hf[fi].x[j]-moment_ref[j];}
    for(int j=0;j<3;++j)moment[j]+=arm[(j+1)%3]*load[(j+2)%3]-arm[(j+2)%3]*load[(j+1)%3];}
   }
   if(!(volume>0))throw std::runtime_error("empty near-field residual region");
   for(int v=0;v<6;++v)norms[v]/=volume;norms[5]/=c.reference[5];
   const double qarea=.5*c.mach*c.mach*area_ref,cd=dot(force,c.reference+1)/(c.mach*qarea),cl=(-force[0]*sin(angle)+force[2]*cos(angle))/qarea,cm=moment[1]/qarea,meanmax=*std::max_element(norms,norms+5),rawmax=fmax(meanmax,norms[5]);
   if(iteration==0)initial_norm=rawmax;
   const bool stable_load=iteration>0&&fabs(cd-previous_cd)/fmax(.01,fabs(cd))<.002&&fabs(cl-previous_cl)/fmax(.1,fabs(cl))<.002&&fabs(cm-previous_cm)/fmax(.01,fabs(cm))<.002;
   if(iteration>=500&&rawmax<=tol&&stable_load)++hold;else hold=0;converged=hold>=5;
   const bool terminal=converged||iteration==maxiter||elapsed()>=budget;const double reported_cfl=c.cfl,reported_blend=c.reconstruction_blend;std::string controller_action=terminal?"terminal":"baseline";
   if(terminal&&rollback_enabled){const double terminal_metric=controller_baseline?fmax(meanmax/controller_scale_mean,norms[5]/controller_scale_sa):1;if(!checkpoint_valid||terminal_metric<best_controller_metric){save_rollback_state();best_controller_metric=terminal_metric;checkpoint_blend=c.reconstruction_blend;checkpoint_cfl=reported_cfl;checkpoint_iteration=iteration;checkpoint_cd=cd;checkpoint_cl=cl;checkpoint_cm=cm;for(int v=0;v<6;++v)checkpoint_norms[v]=norms[v];checkpoint_valid=true;}}
   if(!terminal&&iteration<controller_start_iteration)controller_action="warmup";
   else if(!terminal){
    const bool new_rejections=rejected>rejected_at_last_check;
    if(!controller_baseline){
     controller_baseline=true;stable_checks=0;previous_mean=meanmax;previous_sa=norms[5];controller_scale_mean=fmax(meanmax,1e-300);controller_scale_sa=fmax(norms[5],1e-300);best_controller_metric=1;controller_action="baseline";
     if(rollback_enabled){save_rollback_state();checkpoint_blend=c.reconstruction_blend;checkpoint_cfl=c.cfl;checkpoint_iteration=iteration;checkpoint_cd=cd;checkpoint_cl=cl;checkpoint_cm=cm;for(int v=0;v<6;++v)checkpoint_norms[v]=norms[v];checkpoint_valid=true;}
    }else{
     const double controller_metric=fmax(meanmax/controller_scale_mean,norms[5]/controller_scale_sa);
     const bool severe_growth=checkpoint_valid&&controller_metric>best_controller_metric*rollback_ratio;
     const bool meaningful_growth=meanmax>previous_mean*cfl_cut_ratio;
     const bool sa_growth=norms[5]>previous_sa*1.1;
     const bool joint_reversal=meanmax>previous_mean*cfl_reversal_ratio&&norms[5]>previous_sa*cfl_reversal_ratio;
     reversal_checks=joint_reversal?reversal_checks+1:0;if(growth_cooldown>0)--growth_cooldown;
     if(severe_growth&&rollback_enabled){
      restore_rollback_state();c.cfl=fmax(.01,.5*c.cfl);c.reconstruction_blend=checkpoint_blend;++cfl_reductions;++rollbacks;stable_checks=0;reversal_checks=0;growth_cooldown=cfl_cut_cooldown;controller_baseline=false;checkpoint_valid=false;controller_action="rollback_and_cut";
     }else{
      if(meaningful_growth){c.cfl=fmax(.01,.5*c.cfl);++cfl_reductions;stable_checks=0;reversal_checks=0;growth_cooldown=cfl_cut_cooldown;controller_action="cut";}
      else if(new_rejections){stable_checks=0;controller_action="rejection_hold";}
      else if(sa_growth){stable_checks=0;controller_action="SA_hold";}
      else if(reversal_checks>=cfl_reversal_wait){c.cfl=fmax(.01,c.cfl/cfl_growth);++cfl_reductions;stable_checks=0;reversal_checks=0;growth_cooldown=cfl_cut_cooldown;controller_action="reversal_cut";}
      else if(growth_cooldown>0){stable_checks=0;controller_action="cooldown";}
      else if(meanmax<=previous_mean*.999&&norms[5]<=previous_sa*.999){++stable_checks;controller_action="stable";}
      else{stable_checks=0;controller_action="hold";}
      if(!meaningful_growth&&!new_rejections&&reconstruction_ramp&&c.reconstruction_blend<reconstruction_target-1e-12&&stable_checks>=reconstruction_wait){
       c.reconstruction_blend=fmin(reconstruction_target,c.reconstruction_blend+reconstruction_step);c.cfl=fmax(.01,.75*c.cfl);++reconstruction_ramps;stable_checks=0;controller_baseline=false;checkpoint_valid=false;controller_action="ramp_reconstruction";
      }else if(!meaningful_growth&&!new_rejections&&stable_checks>=cfl_growth_wait){
       c.cfl=fmin(cfl_max,c.cfl*cfl_growth);c.step_scale=fmin(1.,c.step_scale*1.05);stable_checks=0;controller_action="grow_cfl";
      }
      previous_mean=meanmax;previous_sa=norms[5];
      if(controller_baseline&&rollback_enabled&&controller_metric<best_controller_metric){save_rollback_state();best_controller_metric=controller_metric;checkpoint_blend=c.reconstruction_blend;checkpoint_cfl=reported_cfl;checkpoint_iteration=iteration;checkpoint_cd=cd;checkpoint_cl=cl;checkpoint_cm=cm;for(int v=0;v<6;++v)checkpoint_norms[v]=norms[v];checkpoint_valid=true;}
     }
    }
   }
   history<<iteration<<','<<elapsed()<<','<<reported_cfl<<','<<reported_blend;for(double x:norms)history<<','<<x;history<<','<<meanmax<<','<<rawmax<<','<<cd<<','<<cl<<','<<cm<<','<<force[1]/qarea<<','<<force[2]/qarea<<','<<rejected<<','<<c.step_scale<<','<<cfl_reductions<<','<<rollbacks<<','<<reconstruction_ramps<<','<<reversal_checks<<','<<growth_cooldown<<','<<controller_action<<'\n';history.flush();
   std::cout<<"iteration="<<iteration<<" seconds="<<elapsed()<<" mean="<<meanmax<<" SA="<<norms[5]<<" CD="<<cd<<" CFL="<<reported_cfl<<" blend="<<reported_blend<<" action="<<controller_action<<" hold="<<hold<<'\n'<<std::flush;
   previous_cd=cd;previous_cl=cl;previous_cm=cm;rejected_at_last_check=rejected;if(terminal)break;
  }
  bool accepted=false;
  for(int trial=0;trial<12;++trial){set_bad(0);rhs(d.u,true);
   if(c.dplur){implicit_update();prim(d.u2);}else{update(0);rhs(d.u1,false);update(1);rhs(d.u2,false);update(2);prim(d.u1);}
   if(!check_bad()){if(c.dplur)std::swap(d.u,d.u2);else std::swap(d.u,d.u1);accepted=true;break;}
   const double rejected_cfl=c.cfl;c.cfl=fmax(1e-8,.5*c.cfl);++rejected;++cfl_reductions;++positivity_cfl_reductions;
   std::cerr<<"rejected iteration="<<iteration<<" trial="<<trial<<" CFL="<<rejected_cfl<<" next_CFL="<<c.cfl<<" failure_code="<<bad<<'\n';}
  if(!accepted){failed=true;std::cerr<<integrator<<" trials failed positivity; saving the last accepted state\n";break;}++iteration;
 }
 // Recompute the physical conservative residual and boundary fluxes on the final state.
 const double compute_seconds=elapsed();const auto output_start=std::chrono::steady_clock::now();
 set_bad(0);rhs(d.u,false);if(check_bad())throw std::runtime_error("invalid final state");download();
 if(num("--residual-audit",0)!=0){
#ifdef __CUDACC__
  if(gpu)checked(cudaMemcpy(hsource.data(),d.source,n*sizeof(double),cudaMemcpyDeviceToHost));
#endif
  const double limits[4]={1e-4,1e-3,1e-2,1e-1};
  double volume[5]={},sa[5]={},source[5]={},transport[5]={},total_volume=0,total_sa=0;
  uint64_t count[5]={},negative_sa=0;std::vector<std::pair<double,int>> contributors;contributors.reserve(n);
  for(int k=0;k<n;++k){const auto& cell=hc[k];bool near=true;
   for(int j=0;j<3;++j)if(cell.x[j]<residual_lo[j]||cell.x[j]>residual_hi[j])near=false;
   if(!near)continue;
   int band=0;while(band<4&&cell.dwall>=limits[band])++band;
   const double value=cell.volume*std::fabs(hraw[6*k+5]);
   volume[band]+=cell.volume;sa[band]+=value;
   source[band]+=cell.volume*std::fabs(hsource[k]);
   transport[band]+=cell.volume*std::fabs(hraw[6*k+5]-hsource[k]);
   total_volume+=cell.volume;total_sa+=value;++count[band];
   negative_sa+=hw[6*k+5]<0;contributors.emplace_back(value,k);
  }
  double diffusion_positive=0,diffusion_negative=0;uint64_t antidiffusive_internal=0,antidiffusive_wall=0;
  if(c.sa_spatial_order==1)for(int fi=0;fi<nf;++fi){const Face face=hf[fi];const int l=face.left,r=face.right;
   if(r<0&&face.bc!=1)continue;
   const double mass=hflux[6*fi],upwind=r>=0?(mass>=0?hw[6*l+5]:hw[6*r+5]):0;
   const double diffusive_flux=hflux[6*fi+5]-mass*upwind;
   const double jump=r>=0?hw[6*r+5]-hw[6*l+5]:-hw[6*l+5];
   const double energy=jump*diffusive_flux;
   if(energy>0){diffusion_positive+=energy;if(r>=0)++antidiffusive_internal;else ++antidiffusive_wall;}
   else diffusion_negative-=energy;
  }
  std::sort(contributors.begin(),contributors.end(),[](const auto& a,const auto& b){return a.first>b.first;});
  std::ofstream audit(out/"sa_residual_audit.json");if(!audit)throw std::runtime_error("cannot create SA residual audit");
  audit<<std::setprecision(17)<<"{\"cells\":"<<contributors.size()<<",\"negative_nutilde_cells\":"<<negative_sa
       <<",\"normalized_SA_residual\":"<<total_sa/(total_volume*c.reference[5])
       <<",\"diffusion_energy\":{\"available\":"<<(c.sa_spatial_order==1?"true":"false")
       <<",\"positive\":"<<diffusion_positive<<",\"negative_magnitude\":"<<diffusion_negative
       <<",\"net\":"<<diffusion_positive-diffusion_negative
       <<",\"antidiffusive_internal_faces\":"<<antidiffusive_internal
       <<",\"antidiffusive_wall_faces\":"<<antidiffusive_wall
       <<"},\"wall_distance_R_limits\":[0,0.0001,0.001,0.01,0.1],\"bands\":[";
  for(int b=0;b<5;++b){if(b)audit<<',';audit<<"{\"cells\":"<<count[b]<<",\"volume_fraction\":"<<volume[b]/total_volume
   <<",\"SA_L1_fraction\":"<<sa[b]/total_sa<<",\"source_L1_normalized\":"<<source[b]/(total_volume*c.reference[5])
   <<",\"transport_L1_normalized\":"<<transport[b]/(total_volume*c.reference[5])<<'}';}
  audit<<"],\"top_cell_fractions\":{";
  for(int top: {100,1000,10000}){double sum=0;for(int i=0;i<std::min(top,static_cast<int>(contributors.size()));++i)sum+=contributors[i].first;
   if(top!=100)audit<<',';audit<<'"'<<top<<"\":"<<sum/total_sa;}
  audit<<"},\"top_cells\":[";
  for(int i=0;i<std::min(20,static_cast<int>(contributors.size()));++i){const int k=contributors[i].second;const auto& cell=hc[k];
   if(i)audit<<',';audit<<"{\"cell\":"<<k<<",\"x\":["<<cell.x[0]<<','<<cell.x[1]<<','<<cell.x[2]
    <<"],\"wall_distance_R\":"<<cell.dwall<<",\"volume_R3\":"<<cell.volume
    <<",\"SA_residual\":"<<hraw[6*k+5]<<",\"SA_source\":"<<hsource[k]<<'}';}
  audit<<"]}\n";audit.close();if(!audit)throw std::runtime_error("incomplete SA residual audit");
 }
 const bool binary_output=num("--binary-output",0)!=0;const std::string field_output=args.count("--field-output")?args["--field-output"]:(binary_output?"binary":"csv");
 if(field_output!="none"&&field_output!="binary"&&field_output!="csv")throw std::runtime_error("field-output must be none, binary, or csv");
 bool best_checkpoint_written=false;
 const auto write_binary_state=[&](const std::filesystem::path& path,const std::vector<double>& values){const uintmax_t bytes=values.size()*sizeof(double);
  std::ofstream stream(path,std::ios::binary);if(!stream)throw std::runtime_error("cannot create binary state: "+path.string());
  stream.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(bytes));stream.close();
  if(!stream||!std::filesystem::is_regular_file(path)||std::filesystem::file_size(path)!=bytes)throw std::runtime_error("incomplete binary state: "+path.string());};
 if(field_output=="binary"){write_binary_state(out/"state.bin",hu);if(rollback_enabled&&checkpoint_valid){write_binary_state(out/"best_state.bin",hrollback);best_checkpoint_written=true;}}
 else if(field_output=="csv"){
 std::ofstream field(out/"field.csv");field<<std::setprecision(17)<<"x_R,y_R,z_R,volume_R3,rho,u_a,v_a,w_a,p_rhoa2,Mach,T_Tinf,nutilde,mu_t\n";
 for(int k=0;k<n;++k){const double* w=hw.data()+6*k;field<<hc[k].x[0]<<','<<hc[k].x[1]<<','<<hc[k].x[2]<<','<<hc[k].volume;for(int v=0;v<5;++v)field<<','<<w[v];field<<','<<sqrt(dot(w+1,w+1)/(c.gamma*w[4]/w[0]))<<','<<c.gamma*w[4]/w[0]<<','<<w[5]<<','<<heddy[k]<<'\n';}
 }
 std::ofstream surface(out/"surface.csv");surface<<std::setprecision(17)<<"x_R,y_R,z_R,nx,ny,nz,area_R2,p,Cp,force_x,force_y,force_z\n";
 for(int fi=0;fi<nf;++fi){const auto f=hf[fi];if(f.right>=0||f.bc!=1)continue;surface<<f.x[0]<<','<<f.x[1]<<','<<f.x[2]<<','<<f.n[0]<<','<<f.n[1]<<','<<f.n[2]<<','<<f.area<<','<<hw[6*f.left+4]<<','<<(hw[6*f.left+4]-1/c.gamma)/(.5*c.mach*c.mach);for(int j=0;j<3;++j)surface<<','<<hflux[6*fi+j+1];surface<<'\n';}
 surface.close();
 double balance_error[5]={};for(int v=0;v<5;++v){double sum=0,boundary=0;for(int k=0;k<n;++k)sum+=hc[k].volume*hraw[6*k+v];for(int fi=0;fi<nf;++fi)if(hf[fi].right<0)boundary+=hflux[6*fi+v];balance_error[v]=fabs(sum+boundary);}
 const double output_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-output_start).count();
 std::ofstream result(out/"result.json");result<<std::setprecision(17)<<"{\"restarted\":"<<(restarted?"true":"false")<<",\"integrator\":\""<<integrator<<"\",\"spatial_order\":"<<c.spatial_order<<",\"gradient\":\""<<(c.node_gradient?"node-adaptive-pseudo-laplacian-green-gauss":"legacy-cell-green-gauss")<<"\",\"viscous_face_gradient\":\"Mitchell-node-face-stencil-for-node-gradient; legacy-edge-normal-otherwise\",\"SA_convection_order\":"<<c.sa_spatial_order<<",\"SA_diffusion_blend\":"<<c.sa_diffusion_blend<<",\"final_reconstruction_blend\":"<<c.reconstruction_blend<<",\"reconstruction_ramp\":"<<(reconstruction_ramp?"true":"false")<<",\"reconstruction_ramps\":"<<reconstruction_ramps<<",\"dplur_sweeps\":"<<(c.dplur?dplur_sweeps:0)<<",\"dplur_omega\":"<<(c.dplur?c.dplur_omega:0)<<",\"dplur_block_coupling\":"<<(c.dplur_block_coupling?"true":"false")<<",\"mean_step_ratio\":"<<(c.dplur?c.mean_step_ratio:1)<<",\"SA_step_ratio\":"<<(c.dplur?c.sa_step_ratio:1)<<",\"final_CFL\":"<<c.cfl<<",\"CFL_max\":"<<cfl_max<<",\"CFL_growth_factor\":"<<cfl_growth<<",\"CFL_growth_wait\":"<<cfl_growth_wait<<",\"CFL_reversal_ratio\":"<<cfl_reversal_ratio<<",\"CFL_reversal_wait\":"<<cfl_reversal_wait<<",\"CFL_cut_cooldown\":"<<cfl_cut_cooldown<<",\"controller_start_iteration\":"<<controller_start_iteration<<",\"CFL_cut_ratio\":"<<cfl_cut_ratio<<",\"rollback_ratio\":"<<rollback_ratio<<",\"CFL_reductions\":"<<cfl_reductions<<",\"positivity_CFL_reductions\":"<<positivity_cfl_reductions<<",\"rollbacks\":"<<rollbacks<<",\"best_checkpoint_available\":"<<(best_checkpoint_written?"true":"false")<<",\"best_checkpoint_iteration\":"<<checkpoint_iteration<<",\"best_checkpoint_CFL\":"<<checkpoint_cfl<<",\"best_checkpoint_CD\":"<<checkpoint_cd<<",\"best_checkpoint_CL\":"<<checkpoint_cl<<",\"best_checkpoint_CM_y\":"<<checkpoint_cm<<",\"best_checkpoint_residual\":[";for(int v=0;v<6;++v)result<<(v?",":"")<<checkpoint_norms[v];result<<"],\"final_step_scale\":"<<c.step_scale<<",\"failed\":"<<(failed?"true":"false")<<",\"CM_y\":"<<moment[1]/(.5*c.mach*c.mach*area_ref)<<",\"dimensions\":3,\"model\":\"SA-neg mechanical viscous\",\"wall\":\"stationary no-slip\",\"conductivity\":0,\"time_kind\":\"local steady pseudo-time\",\"backend\":\""<<(gpu?"cuda":"cpu")<<"\",\"Mach\":"<<c.mach<<",\"Re_input\":"<<re<<",\"mu_inf_star\":"<<c.mu<<",\"cells\":"<<n<<",\"iterations\":"<<iteration<<",\"rejected\":"<<rejected<<",\"converged\":"<<(converged?"true":"false")<<",\"process_seconds\":"<<elapsed()<<",\"raw_tolerance\":"<<tol<<",\"initial_raw_max\":"<<initial_norm<<",\"raw_residual\":[";
 for(int v=0;v<6;++v)result<<(v?",":"")<<norms[v];result<<"],\"agglomeration_smoothing\":"<<agglomeration_smoothing<<",\"agglomeration_SA_smoothing\":"<<agglomeration_sa_smoothing<<",\"agglomeration_level_cells\":["<<n<<','<<(use_agglomeration?d.mg_n1:0)<<','<<(use_agglomeration?d.mg_n2:0)<<"],\"conservation_balance_error\":[";for(int v=0;v<5;++v)result<<(v?",":"")<<balance_error[v];result<<"],\"force_rhoa2R2\":["<<force[0]<<','<<force[1]<<','<<force[2]<<"],\"mesh_read_seconds\":"<<mesh_read_seconds<<",\"compute_seconds\":"<<compute_seconds<<",\"output_seconds\":"<<output_seconds<<",\"total_wall_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-process_start).count()<<",\"reference_area\":"<<area_ref<<",\"Re_reference\":"<<c.mach/c.mu<<",\"CL\":"<<(-force[0]*sin(angle)+force[2]*cos(angle))/(.5*c.mach*c.mach*area_ref)<<",\"CD\":"<<dot(force,c.reference+1)/(c.mach*.5*c.mach*c.mach*area_ref)<<",\"AOA_degrees\":"<<num("--aoa",0)<<",\"residual_region_min\":["<<residual_lo[0]<<','<<residual_lo[1]<<','<<residual_lo[2]<<"],\"residual_region_max\":["<<residual_hi[0]<<','<<residual_hi[1]<<','<<residual_hi[2]<<"],\"state_format\":\""<<(field_output=="binary"?"little-endian float64 AoS: rho,rhou,rhov,rhow,rhoE,rho_nutilde":field_output=="csv"?"CSV":"none")<<"\",\"scope\":\"local 3D development solver; boundary-layer and experimental drag accuracy not certified; molecular viscosity specified explicitly or through reference Re\"}\n";
#ifdef __CUDACC__
 if(gpu)for(void* p:allocations)cudaFree(p);
#endif
 return failed?2:converged?0:3;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
