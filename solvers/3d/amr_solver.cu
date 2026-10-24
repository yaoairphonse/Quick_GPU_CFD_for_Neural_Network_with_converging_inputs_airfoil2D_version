// Procedural Cartesian-AMR cut-cell backend.  The explicit sphere3d reader is
// intentionally unchanged so tetrahedral and general hybrid meshes coexist.
#include "amr_mesh_format.h"
#include "mesh_format.h"
#include "spatial_operators.h"
#include "sa_model.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define AMR_HD __host__ __device__
#else
#define AMR_HD
#endif

struct AmrConfig {
 double gamma=1.4,mach=.85,mu=1.7e-7,S=.36,cfl=.05;
 double reference[6]={};
 double reconstruction_blend=0;
 int spatial_order=1;
 bool uniform_transmissive=false;
};

struct AmrData {
 int blocks_count=0,procedural_count=0,n=0,interface_count=0,boundary_count=0,cut_count=0,wall_count=0,block_cells=8;
 AmrBlock* blocks=nullptr;AmrBlockInterface* interfaces=nullptr;AmrBoundarySide* boundaries=nullptr;AmrControl* controls=nullptr;
 int* procedural_control=nullptr;uint8_t* procedural_status=nullptr;Face* cut_faces=nullptr;Face* wall_faces=nullptr;
 double *u=nullptr,*u1=nullptr,*u2=nullptr,*w=nullptr,*gradient=nullptr,*balance=nullptr,*raw=nullptr,*dt=nullptr;
 double *mu=nullptr,*eddy=nullptr,*diffusion=nullptr,*source=nullptr,*source_rate=nullptr,*spectral=nullptr,*viscous_rate=nullptr,*sa_rate=nullptr;
 int* bad=nullptr;
};

template<class T>std::vector<T> read_exact(const std::filesystem::path& path,size_t count){
 std::vector<T> result(count);std::ifstream stream(path,std::ios::binary);stream.read(reinterpret_cast<char*>(result.data()),static_cast<std::streamsize>(count*sizeof(T)));
 if(!stream||stream.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("invalid table size: "+path.string());return result;
}

AMR_HD bool finite_value(double value){
#ifdef __CUDA_ARCH__
 return isfinite(value);
#else
 return std::isfinite(value);
#endif
}
AMR_HD double dot3(const double* a,const double* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
AMR_HD void add_value(double* target,double value){
#ifdef __CUDA_ARCH__
 atomicAdd(target,value);
#else
 *target+=value;
#endif
}
AMR_HD void fail(AmrData data,int code=1){
#ifdef __CUDA_ARCH__
 atomicCAS(data.bad,0,code);
#else
 if(!*data.bad)*data.bad=code;
#endif
}
AMR_HD void conservative(const double* w,double gamma,double* u){
 u[0]=w[0];for(int j=0;j<3;++j)u[j+1]=w[0]*w[j+1];u[4]=w[4]/(gamma-1)+.5*w[0]*dot3(w+1,w+1);u[5]=w[0]*w[5];
}
AMR_HD bool farfield(const double* interior,const double* normal,AmrConfig config,double* outside){
 for(int v=0;v<6;++v)outside[v]=interior[v];const double un=dot3(interior+1,normal),ai=sqrt(config.gamma*interior[4]/interior[0]);
 if(un>=ai)return true;if(un<=-ai){for(int v=0;v<6;++v)outside[v]=config.reference[v];return true;}
 const double ur=dot3(config.reference+1,normal),ar=sqrt(config.gamma*config.reference[4]/config.reference[0]),gm=config.gamma-1;
 const double plus=un+2*ai/gm,minus=ur-2*ar/gm,ab=.25*gm*(plus-minus),ub=.5*(plus+minus);if(!(ab>0))return false;
 const double* adv=un>=0?interior:config.reference;const double entropy=adv[4]/pow(adv[0],config.gamma),ua=dot3(adv+1,normal);
 outside[0]=pow(ab*ab/(config.gamma*entropy),1/gm);outside[4]=entropy*pow(outside[0],config.gamma);outside[5]=adv[5];
 for(int j=0;j<3;++j)outside[j+1]=adv[j+1]+(ub-ua)*normal[j];return outside[0]>0&&outside[4]>0&&finite_value(outside[4]);
}
AMR_HD void hllc(const double* left,const double* right,const double* normal,double gamma,double* flux){
 double ul[6],ur[6],fl[5],fr[5];conservative(left,gamma,ul);conservative(right,gamma,ur);
 const double vl=dot3(left+1,normal),vr=dot3(right+1,normal),al=sqrt(gamma*left[4]/left[0]),ar=sqrt(gamma*right[4]/right[0]);
 fl[0]=left[0]*vl;fr[0]=right[0]*vr;for(int j=0;j<3;++j){fl[j+1]=ul[j+1]*vl+left[4]*normal[j];fr[j+1]=ur[j+1]*vr+right[4]*normal[j];}
 fl[4]=(ul[4]+left[4])*vl;fr[4]=(ur[4]+right[4])*vr;const double sl=fmin(vl-al,vr-ar),sr=fmax(vl+al,vr+ar);
 if(sl>=0){for(int j=0;j<5;++j)flux[j]=fl[j];return;}if(sr<=0){for(int j=0;j<5;++j)flux[j]=fr[j];return;}
 const double sm=(right[4]-left[4]+left[0]*vl*(sl-vl)-right[0]*vr*(sr-vr))/(left[0]*(sl-vl)-right[0]*(sr-vr));
 const bool use_left=sm>=0;const double* w=use_left?left:right;const double* u=use_left?ul:ur;const double* base=use_left?fl:fr;
 const double s=use_left?sl:sr,v=use_left?vl:vr,rhostar=w[0]*(s-v)/(s-sm);double star[5];star[0]=rhostar;
 for(int j=0;j<3;++j)star[j+1]=rhostar*(w[j+1]+(sm-v)*normal[j]);star[4]=rhostar*(u[4]/w[0]+(sm-v)*(sm+w[4]/(w[0]*(s-v))));
 for(int j=0;j<5;++j)flux[j]=base[j]+s*(star[j]-u[j]);
}

AMR_HD int local_id(const int* coordinate,int block_cells){return (coordinate[2]*block_cells+coordinate[1])*block_cells+coordinate[0];}
AMR_HD void side_coordinate(int q,int side,int block_cells,int* coordinate){
 const int axis=side/2,first=axis==0?1:0,second=axis==2?1:2;coordinate[axis]=(side&1)?block_cells-1:0;coordinate[first]=q%block_cells;coordinate[second]=q/block_cells;
}
AMR_HD bool finish_generated_face(AmrData data,int first_procedural,int second_procedural,Face& face){
 if(first_procedural<0||second_procedural<0)return false;if(data.procedural_status[first_procedural]==2||data.procedural_status[second_procedural]==2)return false;
 face.left=data.procedural_control[first_procedural];face.right=data.procedural_control[second_procedural];face.bc=0;if(face.left<0&&face.right<0)return false;
 if(face.left<0||face.right<0){fail(data,11);return false;}if(face.left==face.right)return false;
 double delta[3];for(int j=0;j<3;++j)delta[j]=data.controls[face.right].x[j]-data.controls[face.left].x[j];face.distance=dot3(delta,face.n);
 if(!(face.distance>0)){face.distance=sqrt(dot3(delta,delta));if(!(face.distance>0)){fail(data,12);return false;}}return true;
}
AMR_HD bool internal_face(AmrData data,long long index,Face& face){
 const int b=data.block_cells,per_axis=(b-1)*b*b,per_block=3*per_axis,block=static_cast<int>(index/per_block),slot=static_cast<int>(index%per_block),axis=slot/per_axis;
 const int item=slot%per_axis,along=item/(b*b),remaining=item%(b*b);int c0[3],c1[3];const int first=axis==0?1:0,second=axis==2?1:2;
 c0[axis]=along;c0[first]=remaining%b;c0[second]=remaining/b;for(int j=0;j<3;++j)c1[j]=c0[j];++c1[axis];const AmrBlock& value=data.blocks[block];
 face={};face.x[0]=value.low[0]+(c0[0]+.5)*value.spacing;face.x[1]=value.low[1]+(c0[1]+.5)*value.spacing;face.x[2]=value.low[2]+(c0[2]+.5)*value.spacing;face.x[axis]=value.low[axis]+(along+1)*value.spacing;face.n[axis]=1;face.area=value.spacing*value.spacing;
 return finish_generated_face(data,value.first_procedural+local_id(c0,b),value.first_procedural+local_id(c1,b),face);
}
AMR_HD bool interface_face(AmrData data,long long index,Face& face){
 const int b=data.block_cells,record=static_cast<int>(index/(b*b)),q=static_cast<int>(index%(b*b));const AmrBlockInterface item=data.interfaces[record];
 const AmrBlock& first_block=data.blocks[item.first_block];const AmrBlock& second_block=data.blocks[item.second_block];const int side=item.first_side,axis=side/2;int first_coordinate[3],second_coordinate[3],fine_coordinate[3];
 if(item.level_difference==0){side_coordinate(q,side,b,first_coordinate);side_coordinate(q,side^1,b,second_coordinate);for(int j=0;j<3;++j)fine_coordinate[j]=first_coordinate[j];}
 else{
  const bool first_is_fine=item.level_difference<0;const AmrBlock& fine=first_is_fine?first_block:second_block;const AmrBlock& coarse=first_is_fine?second_block:first_block;const int fine_side=first_is_fine?side:side^1;side_coordinate(q,fine_side,b,fine_coordinate);int coarse_coordinate[3];coarse_coordinate[axis]=(fine_side&1)?0:b-1;
  for(int transverse=0;transverse<3;++transverse)if(transverse!=axis){const int half=(fine.fine_coordinate[transverse]-coarse.fine_coordinate[transverse])/fine.fine_span;coarse_coordinate[transverse]=(half*b+fine_coordinate[transverse])/2;}
  for(int j=0;j<3;++j){first_coordinate[j]=first_is_fine?fine_coordinate[j]:coarse_coordinate[j];second_coordinate[j]=first_is_fine?coarse_coordinate[j]:fine_coordinate[j];}
 }
 const AmrBlock& fine=item.level_difference==0?first_block:(item.level_difference<0?first_block:second_block);const int fine_side=item.level_difference==0?side:(item.level_difference<0?side:side^1);
 face={};for(int j=0;j<3;++j)face.x[j]=fine.low[j]+(fine_coordinate[j]+.5)*fine.spacing;face.x[axis]=fine.low[axis]+((fine_side&1)?b:0)*fine.spacing;face.n[axis]=(side&1)?1:-1;face.area=fine.spacing*fine.spacing;
 return finish_generated_face(data,first_block.first_procedural+local_id(first_coordinate,b),second_block.first_procedural+local_id(second_coordinate,b),face);
}
AMR_HD bool boundary_face(AmrData data,long long index,Face& face){
 const int b=data.block_cells,record=static_cast<int>(index/(b*b)),q=static_cast<int>(index%(b*b));const AmrBoundarySide item=data.boundaries[record];const AmrBlock& block=data.blocks[item.block];int coordinate[3];side_coordinate(q,item.side,b,coordinate);const int procedural=block.first_procedural+local_id(coordinate,b);if(data.procedural_status[procedural]==2)return false;
 face={};face.left=data.procedural_control[procedural];face.right=-1;face.bc=2;if(face.left<0)return false;const int axis=item.side/2;for(int j=0;j<3;++j)face.x[j]=block.low[j]+(coordinate[j]+.5)*block.spacing;face.x[axis]=block.low[axis]+((item.side&1)?b:0)*block.spacing;face.n[axis]=(item.side&1)?1:-1;face.area=block.spacing*block.spacing;
 double delta[3];for(int j=0;j<3;++j)delta[j]=face.x[j]-data.controls[face.left].x[j];face.distance=dot3(delta,face.n);if(!(face.distance>0)){fail(data,13);return false;}return true;
}

AMR_HD void geometry_face(AmrData data,const Face& face){
 const double phi=face.x[0]+2*face.x[1]-.5*face.x[2];for(int j=0;j<3;++j){const double value=face.area*face.n[j];add_value(data.gradient+18*face.left+j,value);add_value(data.gradient+18*face.left+3+j,phi*value);if(face.right>=0){add_value(data.gradient+18*face.right+j,-value);add_value(data.gradient+18*face.right+3+j,-phi*value);}}
 add_value(data.spectral+face.left,face.area);if(face.right>=0)add_value(data.spectral+face.right,face.area);
}
AMR_HD void primitive_point(AmrConfig config,AmrData data,const double* state,int cell){
 if(*data.bad)return;const double* q=state+6*cell;double* w=data.w+6*cell;if(!(q[0]>0)||!finite_value(q[0])){fail(data,cell+1);return;}w[0]=q[0];for(int j=0;j<3;++j)w[j+1]=q[j+1]/q[0];w[4]=(config.gamma-1)*(q[4]-.5*q[0]*dot3(w+1,w+1));w[5]=q[5]/q[0];if(!(w[4]>0)||!finite_value(w[4])||!finite_value(w[5]))fail(data,cell+1);
}
AMR_HD void gradient_face(AmrConfig config,AmrData data,const Face& face){
 const double* left=data.w+6*face.left;double boundary[6],value[6];const double* right;
 if(face.right>=0)right=data.w+6*face.right;
 else if(config.uniform_transmissive){right=left;}
 else if(face.bc==1){for(int v=0;v<6;++v)boundary[v]=left[v];boundary[1]=boundary[2]=boundary[3]=boundary[5]=0;right=boundary;}
 else{if(!farfield(left,face.n,config,boundary)){fail(data,21);return;}right=boundary;}
 if(face.right>=0||face.bc==2||config.uniform_transmissive){
  const AmrControl& lc=data.controls[face.left];double dl=0,dr=0;for(int j=0;j<3;++j){dl+=fabs((face.x[j]-lc.x[j])*face.n[j]);if(face.right>=0)dr+=fabs((data.controls[face.right].x[j]-face.x[j])*face.n[j]);}if(face.right<0)dr=dl;const double denominator=dl+dr;
  for(int v=0;v<6;++v)value[v]=denominator>0?(dr*left[v]+dl*right[v])/denominator:.5*(left[v]+right[v]);
 }else for(int v=0;v<6;++v)value[v]=right[v];
 for(int v=0;v<6;++v)for(int j=0;j<3;++j){const double contribution=value[v]*face.area*face.n[j];add_value(data.gradient+18*face.left+3*v+j,contribution);if(face.right>=0)add_value(data.gradient+18*face.right+3*v+j,-contribution);}
}
AMR_HD void finalize_gradient(AmrConfig config,AmrData data,int cell){
 if(*data.bad)return;double* gradient=data.gradient+18*cell;const double inverse=1/data.controls[cell].volume;for(int v=0;v<18;++v)gradient[v]*=inverse;const double* w=data.w+6*cell,temperature=w[4]/w[0],reference=1/config.gamma;
 data.mu[cell]=config.mu*pow(temperature/reference,1.5)*(reference+config.S)/(temperature+config.S);const double omega=sqrt((gradient[10]-gradient[8])*(gradient[10]-gradient[8])+(gradient[5]-gradient[9])*(gradient[5]-gradient[9])+(gradient[6]-gradient[4])*(gradient[6]-gradient[4]));
 const auto model=cfd::sa_neg_point(w[5],data.mu[cell]/w[0],data.controls[cell].dwall,omega);data.eddy[cell]=w[0]*cfd::sa_eddy_kinematic(w[5],data.mu[cell]/w[0]);data.diffusion[cell]=model.diffusion;data.source_rate[cell]=model.source_rate;data.source[cell]=w[0]*(model.source+cfd::sa_cb2/cfd::sa_sigma*dot3(gradient+15,gradient+15))-model.diffusion*dot3(gradient,gradient+15);
 if(!(data.mu[cell]>0)||!(data.diffusion[cell]>0)||!finite_value(data.source[cell]))fail(data,-cell-1);
}

AMR_HD void reconstruct_face_state(AmrConfig config,const AmrControl* controls,const double* primitive,const double* gradients,const Face& face,int cell,int neighbor,double* result){
 const double* center=primitive+6*cell;for(int v=0;v<6;++v)result[v]=center[v];if(config.spatial_order!=2||config.reconstruction_blend<=0)return;
 const double* other=neighbor>=0?primitive+6*neighbor:(face.bc==2?config.reference:nullptr);double offset[3];for(int j=0;j<3;++j)offset[j]=face.x[j]-controls[cell].x[j];
 for(int v=0;v<5;++v){double increment=dot3(gradients+18*cell+3*v,offset);if(other){const double difference=other[v]-center[v];if(increment*difference<=0)increment=0;else if(fabs(increment)>fabs(difference))increment=difference;}double candidate=center[v]+config.reconstruction_blend*increment;if((v==0||v==4)&&(!(candidate>.1*center[v])||candidate>10*center[v]))candidate=center[v];result[v]=candidate;}
}

AMR_HD void flux_face(AmrConfig config,AmrData data,const Face& face){
 const double* left_cell=data.w+6*face.left;const double* right_cell=face.right>=0?data.w+6*face.right:left_cell;double reconstructed_left[6],reconstructed_right[6],boundary[6],flux[6]={},gradient[18],velocity[3];reconstruct_face_state(config,data.controls,data.w,data.gradient,face,face.left,face.right,reconstructed_left);const double* left=reconstructed_left;const double* right=left;if(face.right>=0){reconstruct_face_state(config,data.controls,data.w,data.gradient,face,face.right,face.left,reconstructed_right);right=reconstructed_right;}else if(!config.uniform_transmissive&&face.bc==2){if(!farfield(left,face.n,config,boundary)){fail(data,31);return;}right=boundary;}double mu=data.mu[face.left]+data.eddy[face.left],rho=left_cell[0],diffusion=data.diffusion[face.left];
 for(int v=0;v<18;++v)gradient[v]=data.gradient[18*face.left+v];for(int j=0;j<3;++j)velocity[j]=.5*(left_cell[j+1]+right_cell[j+1]);
 if(!config.uniform_transmissive&&face.right<0&&face.bc==1){
  for(int j=0;j<3;++j){flux[j+1]=left[4]*face.n[j];velocity[j]=0;}const double distance=data.controls[face.left].dwall;
  for(int v=1;v<=3;++v)for(int j=0;j<3;++j)gradient[3*v+j]=-left_cell[v]*face.n[j]/distance;for(int j=0;j<3;++j)gradient[15+j]=-left_cell[5]*face.n[j]/distance;
  mu=data.mu[face.left];diffusion=mu/(rho*cfd::sa_sigma);flux[5]=-rho*diffusion*dot3(gradient+15,face.n);
 }else{
  hllc(left,right,face.n,config.gamma,flux);flux[5]=flux[0]*(flux[0]>=0?left[5]:right[5]);
  if(!config.uniform_transmissive&&face.right>=0){double delta[3];for(int j=0;j<3;++j)delta[j]=data.controls[face.right].x[j]-data.controls[face.left].x[j];for(int v=0;v<6;++v){double average[3];for(int j=0;j<3;++j)average[j]=.5*(data.gradient[18*face.left+3*v+j]+data.gradient[18*face.right+3*v+j]);qcfd::corrected_face_gradient(average,delta,right_cell[v]-left_cell[v]);for(int j=0;j<3;++j)gradient[3*v+j]=average[j];}
   mu=.5*(data.mu[face.left]+data.eddy[face.left]+data.mu[face.right]+data.eddy[face.right]);rho=.5*(left_cell[0]+right_cell[0]);diffusion=.5*(data.diffusion[face.left]+data.diffusion[face.right]);flux[5]-=rho*diffusion*dot3(gradient+15,face.n);}
 }
 if(!config.uniform_transmissive&&(face.right>=0||(face.right<0&&face.bc==1))){const double divergence=gradient[3]+gradient[7]+gradient[11];double stress[3]={};for(int i=0;i<3;++i)for(int j=0;j<3;++j)stress[i]+=mu*(gradient[3*(i+1)+j]+gradient[3*(j+1)+i]-(i==j?2*divergence/3:0))*face.n[j];for(int j=0;j<3;++j)flux[j+1]-=stress[j];flux[4]-=dot3(stress,velocity);}
 const double speed=fmax(fabs(dot3(left+1,face.n))+sqrt(config.gamma*left[4]/left[0]),fabs(dot3(right+1,face.n))+sqrt(config.gamma*right[4]/right[0]));const double integrated_speed=speed*face.area;
 double viscous=0,sa=0;if(!config.uniform_transmissive&&(face.right>=0||(face.right<0&&face.bc==1))){viscous=(4./3)*mu/rho*face.area/face.distance;sa=rho*diffusion*face.area/face.distance;}
 for(int v=0;v<6;++v){const double integrated=flux[v]*face.area;add_value(data.balance+6*face.left+v,integrated);if(face.right>=0)add_value(data.balance+6*face.right+v,-integrated);if(!finite_value(integrated))fail(data,32);}
 add_value(data.spectral+face.left,integrated_speed);add_value(data.viscous_rate+face.left,viscous);add_value(data.sa_rate+face.left,sa);if(face.right>=0){add_value(data.spectral+face.right,integrated_speed);add_value(data.viscous_rate+face.right,viscous);add_value(data.sa_rate+face.right,sa);}
}
AMR_HD void finalize_residual(AmrConfig config,AmrData data,int cell,bool base){
 const double volume=data.controls[cell].volume;for(int v=0;v<6;++v){double value=-data.balance[6*cell+v]/volume;if(v==5&&!config.uniform_transmissive)value+=data.source[cell];data.raw[6*cell+v]=data.balance[6*cell+v]=value;if(!finite_value(value))fail(data,41);}
 if(base){const double conv=data.spectral[cell]/volume,mean_rate=conv+2*data.viscous_rate[cell]/volume,sa_rate=conv+2*data.sa_rate[cell]/(data.w[6*cell]*volume)+fmax(data.source_rate[cell],0.);data.dt[cell]=config.cfl/(fmax(mean_rate,sa_rate)+1e-100);}
}
AMR_HD void update_point(AmrData data,int cell,int stage){const double dt=data.dt[cell];for(int v=0;v<6;++v){const int q=6*cell+v;if(stage==0)data.u1[q]=data.u[q]+dt*data.raw[q];else if(stage==1)data.u2[q]=.75*data.u[q]+.25*(data.u1[q]+dt*data.raw[q]);else data.u1[q]=data.u[q]/3+(2./3)*(data.u2[q]+dt*data.raw[q]);}}

AMR_HD void dplur_diagonal(AmrData data,int cell,double* diagonal){
 const double volume=data.controls[cell].volume,convective=data.spectral[cell]/volume,viscous=data.viscous_rate[cell]/volume,sa=data.sa_rate[cell]/(data.w[6*cell]*volume);
 diagonal[0]=1+data.dt[cell]*convective;for(int v=1;v<5;++v)diagonal[v]=1+data.dt[cell]*(convective+2*viscous);diagonal[5]=1+data.dt[cell]*(convective+2*sa+fmax(data.source_rate[cell],0.));
}
AMR_HD void dplur_initialize_point(AmrData data,int cell){double diagonal[6];dplur_diagonal(data,cell,diagonal);for(int v=0;v<6;++v)data.u1[6*cell+v]=data.dt[cell]*data.raw[6*cell+v]/diagonal[v];}
AMR_HD void dplur_face(AmrConfig config,AmrData data,const Face& face){
 if(face.right<0)return;const double* left=data.w+6*face.left;const double* right=data.w+6*face.right;const double speed=fmax(fabs(dot3(left+1,face.n))+sqrt(config.gamma*left[4]/left[0]),fabs(dot3(right+1,face.n))+sqrt(config.gamma*right[4]/right[0]));
 const double mu=.5*(data.mu[face.left]+data.eddy[face.left]+data.mu[face.right]+data.eddy[face.right]),rho=.5*(left[0]+right[0]),diffusion=.5*(data.diffusion[face.left]+data.diffusion[face.right]);const double integrated_convective=.5*speed*face.area,integrated_viscous=(4./3)*mu/rho*face.area/face.distance,integrated_sa=rho*diffusion*face.area/face.distance;
 const int cells[2]={face.left,face.right};for(int side=0;side<2;++side){const int cell=cells[side],neighbor=cells[side^1];const double inverse_volume=1/data.controls[cell].volume,convective=integrated_convective*inverse_volume,viscous=integrated_viscous*inverse_volume,sa=integrated_sa/(data.w[6*cell]*data.controls[cell].volume);const double* update=data.u1+6*neighbor;add_value(data.balance+6*cell,convective*update[0]);for(int v=1;v<5;++v)add_value(data.balance+6*cell+v,(convective+viscous)*update[v]);add_value(data.balance+6*cell+5,(convective+sa)*update[5]);}
}
AMR_HD void dplur_relax_point(AmrData data,int cell,double omega){double diagonal[6];dplur_diagonal(data,cell,diagonal);for(int v=0;v<6;++v){const int q=6*cell+v,next_q=6*cell+v;const double next=data.dt[cell]*(data.raw[q]+data.balance[q])/diagonal[v];data.u2[next_q]=(1-omega)*data.u1[q]+omega*next;}}
AMR_HD void dplur_apply_point(AmrData data,int cell){for(int v=0;v<6;++v)data.u2[6*cell+v]=data.u[6*cell+v]+data.u1[6*cell+v];}

#ifdef __CUDACC__
__global__ void primitive_kernel(AmrConfig c,AmrData d,const double* u){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)primitive_point(c,d,u,k);}
__global__ void finalize_gradient_kernel(AmrConfig c,AmrData d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)finalize_gradient(c,d,k);}
__global__ void finalize_residual_kernel(AmrConfig c,AmrData d,bool base){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)finalize_residual(c,d,k,base);}
__global__ void update_kernel(AmrData d,int stage){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)update_point(d,k,stage);}
__global__ void dplur_initialize_kernel(AmrData d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)dplur_initialize_point(d,k);}
__global__ void dplur_relax_kernel(AmrData d,double omega){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)dplur_relax_point(d,k,omega);}
__global__ void dplur_apply_kernel(AmrData d){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<d.n)dplur_apply_point(d,k);}
#define DEFINE_FACE_KERNEL(NAME,COUNT,GENERATOR,OP) __global__ void NAME(AmrConfig c,AmrData d){long long q=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;if(q<(COUNT)){Face f;if(GENERATOR){OP;}}}
DEFINE_FACE_KERNEL(internal_gradient_kernel,static_cast<long long>(d.blocks_count)*3*(d.block_cells-1)*d.block_cells*d.block_cells,internal_face(d,q,f),gradient_face(c,d,f))
DEFINE_FACE_KERNEL(interface_gradient_kernel,static_cast<long long>(d.interface_count)*d.block_cells*d.block_cells,interface_face(d,q,f),gradient_face(c,d,f))
DEFINE_FACE_KERNEL(boundary_gradient_kernel,static_cast<long long>(d.boundary_count)*d.block_cells*d.block_cells,boundary_face(d,q,f),gradient_face(c,d,f))
DEFINE_FACE_KERNEL(internal_flux_kernel,static_cast<long long>(d.blocks_count)*3*(d.block_cells-1)*d.block_cells*d.block_cells,internal_face(d,q,f),flux_face(c,d,f))
DEFINE_FACE_KERNEL(interface_flux_kernel,static_cast<long long>(d.interface_count)*d.block_cells*d.block_cells,interface_face(d,q,f),flux_face(c,d,f))
DEFINE_FACE_KERNEL(boundary_flux_kernel,static_cast<long long>(d.boundary_count)*d.block_cells*d.block_cells,boundary_face(d,q,f),flux_face(c,d,f))
DEFINE_FACE_KERNEL(internal_dplur_kernel,static_cast<long long>(d.blocks_count)*3*(d.block_cells-1)*d.block_cells*d.block_cells,internal_face(d,q,f),dplur_face(c,d,f))
DEFINE_FACE_KERNEL(interface_dplur_kernel,static_cast<long long>(d.interface_count)*d.block_cells*d.block_cells,interface_face(d,q,f),dplur_face(c,d,f))
__global__ void stored_gradient_kernel(AmrConfig c,AmrData d,const Face* faces,int count){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<count)gradient_face(c,d,faces[q]);}
__global__ void stored_flux_kernel(AmrConfig c,AmrData d,const Face* faces,int count){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<count)flux_face(c,d,faces[q]);}
__global__ void stored_dplur_kernel(AmrConfig c,AmrData d,const Face* faces,int count){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<count)dplur_face(c,d,faces[q]);}
__global__ void internal_geometry_kernel(AmrData d){long long q=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x,count=static_cast<long long>(d.blocks_count)*3*(d.block_cells-1)*d.block_cells*d.block_cells;if(q<count){Face f;if(internal_face(d,q,f))geometry_face(d,f);}}
__global__ void interface_geometry_kernel(AmrData d){long long q=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x,count=static_cast<long long>(d.interface_count)*d.block_cells*d.block_cells;if(q<count){Face f;if(interface_face(d,q,f))geometry_face(d,f);}}
__global__ void boundary_geometry_kernel(AmrData d){long long q=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x,count=static_cast<long long>(d.boundary_count)*d.block_cells*d.block_cells;if(q<count){Face f;if(boundary_face(d,q,f))geometry_face(d,f);}}
__global__ void stored_geometry_kernel(AmrData d,const Face* faces,int count){int q=blockIdx.x*blockDim.x+threadIdx.x;if(q<count)geometry_face(d,faces[q]);}
#undef DEFINE_FACE_KERNEL
#endif

template<class Function>void host_faces(AmrData data,Function function){
 const long long internal_count=static_cast<long long>(data.blocks_count)*3*(data.block_cells-1)*data.block_cells*data.block_cells;for(long long q=0;q<internal_count;++q){Face face;if(internal_face(data,q,face))function(face);}const long long interface_count=static_cast<long long>(data.interface_count)*data.block_cells*data.block_cells;for(long long q=0;q<interface_count;++q){Face face;if(interface_face(data,q,face))function(face);}const long long boundary_count=static_cast<long long>(data.boundary_count)*data.block_cells*data.block_cells;for(long long q=0;q<boundary_count;++q){Face face;if(boundary_face(data,q,face))function(face);}for(int q=0;q<data.cut_count;++q)function(data.cut_faces[q]);for(int q=0;q<data.wall_count;++q)function(data.wall_faces[q]);
}

struct WallReport {
 double pressure[3]={},viscous[3]={},pressure_moment[3]={},viscous_moment[3]={};
};

void wall_load_components(const AmrConfig& config,const std::vector<AmrControl>& controls,
 const std::vector<double>& w,const std::vector<double>& gradient,const std::vector<double>& molecular,
 const Face& face,double* pressure,double* viscous){
 const int cell=face.left;const double* primitive=w.data()+6*cell;double wall_state[6],corrected[18];reconstruct_face_state(config,controls.data(),w.data(),gradient.data(),face,cell,-1,wall_state);std::copy(gradient.data()+18*cell,gradient.data()+18*cell+18,corrected);
 const double distance=controls[cell].dwall;for(int v=1;v<=3;++v)for(int j=0;j<3;++j)corrected[3*v+j]=-primitive[v]*face.n[j]/distance;
 const double divergence=corrected[3]+corrected[7]+corrected[11];double stress[3]={};for(int i=0;i<3;++i)for(int j=0;j<3;++j)stress[i]+=molecular[cell]*(corrected[3*(i+1)+j]+corrected[3*(j+1)+i]-(i==j?2*divergence/3:0))*face.n[j];
 for(int j=0;j<3;++j){pressure[j]=wall_state[4]*face.n[j]*face.area;viscous[j]=-stress[j]*face.area;}
}

WallReport wall_report(const AmrConfig& config,const std::vector<AmrControl>& controls,
 const std::vector<Face>& faces,const std::vector<double>& w,const std::vector<double>& gradient,
 const std::vector<double>& molecular,const double* moment_reference){
 WallReport result;for(const Face& face:faces){double pressure[3],viscous[3],arm[3];wall_load_components(config,controls,w,gradient,molecular,face,pressure,viscous);for(int j=0;j<3;++j){result.pressure[j]+=pressure[j];result.viscous[j]+=viscous[j];arm[j]=face.x[j]-moment_reference[j];}
  for(int j=0;j<3;++j){result.pressure_moment[j]+=arm[(j+1)%3]*pressure[(j+2)%3]-arm[(j+2)%3]*pressure[(j+1)%3];result.viscous_moment[j]+=arm[(j+1)%3]*viscous[(j+2)%3]-arm[(j+2)%3]*viscous[(j+1)%3];}}
 return result;
}

double projected_drag(const double* force,double angle){return force[0]*cos(angle)+force[2]*sin(angle);}
double projected_lift(const double* force,double angle){return-force[0]*sin(angle)+force[2]*cos(angle);}

int main(int argc,char** argv)try{
 const auto process_start=std::chrono::steady_clock::now();std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("all options require values");args[argv[i]]=argv[i+1];}if(!args.count("--mesh")||!args.count("--out"))throw std::runtime_error("usage: amr3d --mesh DIRECTORY --out DIRECTORY [--backend cpu|cuda --geometry-audit 1 --uniform-audit 1 --restart state.bin --iterations 10]");
 const auto number=[&](const std::string& key,double value){return args.count(key)?std::stod(args[key]):value;};const std::filesystem::path root=args["--mesh"],output=args["--out"];std::filesystem::create_directories(output);const bool gpu=args.count("--backend")&&args["--backend"]=="cuda";const std::string integrator=args.count("--integrator")?args["--integrator"]:"rk3";if(integrator!="rk3"&&integrator!="dplur")throw std::runtime_error("integrator must be rk3 or dplur");const int dplur_sweeps=static_cast<int>(number("--dplur-sweeps",5));const double dplur_omega=number("--dplur-omega",.8);if(dplur_sweeps<1||dplur_sweeps>50||!(dplur_omega>0&&dplur_omega<=1))throw std::runtime_error("invalid DP-LUR controls");
 AmrMeshHeader header;std::ifstream hs(root/"header.bin",std::ios::binary);hs.read(reinterpret_cast<char*>(&header),sizeof(header));if(!hs||hs.peek()!=std::ifstream::traits_type::eof()||std::string(header.magic,7)!="QCFDAM1"||header.version!=1||header.endian!=0x01020304)throw std::runtime_error("invalid QCFDAM1 header");
 if(header.control_count>static_cast<uint64_t>(INT32_MAX)||header.procedural_cell_count>static_cast<uint64_t>(INT32_MAX))throw std::runtime_error("AMR table exceeds 32-bit limits");
 auto blocks=read_exact<AmrBlock>(root/"blocks.bin",header.block_count);auto interfaces=read_exact<AmrBlockInterface>(root/"interfaces.bin",header.interface_count);auto boundaries=read_exact<AmrBoundarySide>(root/"boundaries.bin",header.boundary_side_count);auto controls=read_exact<AmrControl>(root/"controls.bin",header.control_count);auto procedural_control=read_exact<int>(root/"procedural_control.bin",header.procedural_cell_count);auto procedural_status=read_exact<uint8_t>(root/"procedural_status.bin",header.procedural_cell_count);auto cut_faces=read_exact<Face>(root/"cut_faces.bin",header.cut_face_count);auto wall_faces=read_exact<Face>(root/"wall_faces.bin",header.wall_face_count);auto wall_triangle=read_exact<int32_t>(root/"wall_triangle.bin",header.wall_face_count);auto wall_distance=read_exact<double>(root/"wall_distance.bin",header.control_count);for(size_t i=0;i<controls.size();++i)controls[i].dwall=wall_distance[i];
 size_t wall_point_count=0,wall_triangle_count=0;std::ifstream(root/"wall_counts.txt")>>wall_point_count>>wall_triangle_count;if(!wall_point_count||wall_triangle_count==0)throw std::runtime_error("invalid wall surface counts");auto wall_points=read_exact<std::array<double,3>>(root/"wall_points.bin",wall_point_count);auto wall_surface=read_exact<std::array<int32_t,3>>(root/"wall_triangles.bin",wall_triangle_count);for(size_t i=0;i<wall_triangle.size();++i)if(wall_triangle[i]<0||static_cast<size_t>(wall_triangle[i])>=wall_triangle_count)throw std::runtime_error("invalid wall fragment triangle mapping");
 const int n=header.control_count;std::vector<double> u(6ull*n),u1(6ull*n),u2(6ull*n),w(6ull*n),gradient(18ull*n),balance(6ull*n),raw(6ull*n),dt(n),mu(n),eddy(n),diffusion(n),source(n),source_rate(n),spectral(n),viscous_rate(n),sa_rate(n);int bad=0;
 AmrData data{static_cast<int>(header.block_count),static_cast<int>(header.procedural_cell_count),n,static_cast<int>(header.interface_count),static_cast<int>(header.boundary_side_count),static_cast<int>(header.cut_face_count),static_cast<int>(header.wall_face_count),static_cast<int>(header.block_cells),blocks.data(),interfaces.data(),boundaries.data(),controls.data(),procedural_control.data(),procedural_status.data(),cut_faces.data(),wall_faces.data(),u.data(),u1.data(),u2.data(),w.data(),gradient.data(),balance.data(),raw.data(),dt.data(),mu.data(),eddy.data(),diffusion.data(),source.data(),source_rate.data(),spectral.data(),viscous_rate.data(),sa_rate.data(),&bad};
 AmrConfig config;config.mach=number("--mach",.85);const double aoa=number("--aoa",2.5)*acos(-1.)/180,re=number("--re",5e6),reference_area=number("--reference-area",7.818510433883157);const double moment_reference[3]={number("--moment-x",4.807469180565627),number("--moment-y",0),number("--moment-z",.6451776649746193)};if(!(reference_area>0))throw std::runtime_error("reference area must be positive");config.mu=number("--mu-star",config.mach/re);config.cfl=number("--cfl",integrator=="dplur"?1.:.05);config.S=110.4/(config.gamma*216.65);config.reference[0]=1;config.reference[1]=config.mach*cos(aoa);config.reference[2]=0;config.reference[3]=config.mach*sin(aoa);config.reference[4]=1/config.gamma;config.reference[5]=3*config.mu;
 config.spatial_order=static_cast<int>(number("--spatial-order",1));config.reconstruction_blend=number("--reconstruction-blend",config.spatial_order==2?1:0);if((config.spatial_order!=1&&config.spatial_order!=2)||config.reconstruction_blend<0||config.reconstruction_blend>1)throw std::runtime_error("invalid spatial reconstruction controls");
 double residual_low[3]={1e100,1e100,1e100},residual_high[3]={-1e100,-1e100,-1e100};const double residual_padding=number("--residual-padding",2);for(const Face& face:wall_faces)for(int j=0;j<3;++j){residual_low[j]=std::min(residual_low[j],face.x[j]-residual_padding);residual_high[j]=std::max(residual_high[j],face.x[j]+residual_padding);}
 if(args.count("--restart")){const auto saved=read_exact<double>(args["--restart"],6ull*n);std::copy(saved.begin(),saved.end(),u.begin());}
 else{double initial[6];conservative(config.reference,config.gamma,initial);for(int k=0;k<n;++k)std::copy(initial,initial+6,u.data()+6*k);}
 std::vector<void*> allocations;
#ifdef __CUDACC__
 const auto checked=[](cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));};const auto upload=[&](auto*& pointer,size_t count){using T=std::remove_pointer_t<std::decay_t<decltype(pointer)>>;T* device=nullptr;checked(cudaMalloc(&device,count*sizeof(T)));if(pointer)checked(cudaMemcpy(device,pointer,count*sizeof(T),cudaMemcpyHostToDevice));pointer=device;allocations.push_back(device);};
 if(gpu){upload(data.blocks,header.block_count);upload(data.interfaces,header.interface_count);upload(data.boundaries,header.boundary_side_count);upload(data.controls,header.control_count);upload(data.procedural_control,header.procedural_cell_count);upload(data.procedural_status,header.procedural_cell_count);upload(data.cut_faces,header.cut_face_count);upload(data.wall_faces,header.wall_face_count);for(double** pointer:{&data.u,&data.u1,&data.u2,&data.w,&data.gradient,&data.balance,&data.raw})upload(*pointer,6ull*n*(pointer==&data.gradient?3:1));for(double** pointer:{&data.dt,&data.mu,&data.eddy,&data.diffusion,&data.source,&data.source_rate,&data.spectral,&data.viscous_rate,&data.sa_rate})upload(*pointer,n);upload(data.bad,1);}
#else
 if(gpu)throw std::runtime_error("this build has no CUDA backend");
#endif
 const auto set_bad=[&](){bad=0;
#ifdef __CUDACC__
  if(gpu)checked(cudaMemset(data.bad,0,sizeof(int)));
#endif
 };
 const auto get_bad=[&](){
#ifdef __CUDACC__
  if(gpu){checked(cudaGetLastError());checked(cudaMemcpy(&bad,data.bad,sizeof(int),cudaMemcpyDeviceToHost));}
#endif
  return bad;
 };
 const auto zero=[&](double* pointer,size_t count){
#ifdef __CUDACC__
  if(gpu){checked(cudaMemset(pointer,0,count*sizeof(double)));return;}
#endif
  std::fill(pointer,pointer+count,0.);
 };
 const auto copy_to_host=[&](std::vector<double>& host,double* pointer,size_t count){
#ifdef __CUDACC__
  if(gpu){checked(cudaMemcpy(host.data(),pointer,count*sizeof(double),cudaMemcpyDeviceToHost));return;}
#endif
  if(pointer!=host.data())std::copy(pointer,pointer+count,host.data());
 };
 const long long internal_count=static_cast<long long>(data.blocks_count)*3*(data.block_cells-1)*data.block_cells*data.block_cells,interface_count=static_cast<long long>(data.interface_count)*data.block_cells*data.block_cells,boundary_count=static_cast<long long>(data.boundary_count)*data.block_cells*data.block_cells;const auto blocks_for=[](long long count){return static_cast<unsigned>((count+255)/256);};
 if(number("--geometry-audit",0)!=0){set_bad();zero(data.gradient,18ull*n);zero(data.spectral,n);
#ifdef __CUDACC__
  if(gpu){internal_geometry_kernel<<<blocks_for(internal_count),256>>>(data);interface_geometry_kernel<<<blocks_for(interface_count),256>>>(data);boundary_geometry_kernel<<<blocks_for(boundary_count),256>>>(data);stored_geometry_kernel<<<blocks_for(data.cut_count),256>>>(data,data.cut_faces,data.cut_count);stored_geometry_kernel<<<blocks_for(data.wall_count),256>>>(data,data.wall_faces,data.wall_count);checked(cudaDeviceSynchronize());}
  else
#endif
  host_faces(data,[&](const Face& face){geometry_face(data,face);});if(get_bad())throw std::runtime_error("geometry audit face generation failed");copy_to_host(gradient,data.gradient,18ull*n);copy_to_host(spectral,data.spectral,n);double closure_max=0,affine_max=0,global_closure[3]={},weighted_gradient[3]={};
  for(int k=0;k<n;++k){double closure=0,error=0;for(int j=0;j<3;++j){closure+=gradient[18*k+j]*gradient[18*k+j];const double expected=j==0?1:j==1?2:-.5,value=gradient[18*k+3+j]/controls[k].volume;error+=(value-expected)*(value-expected);global_closure[j]+=gradient[18*k+j];weighted_gradient[j]+=controls[k].volume*value;}closure_max=std::max(closure_max,sqrt(closure)/spectral[k]);affine_max=std::max(affine_max,sqrt(error));}
  std::ofstream report(output/"geometry_audit.json");report<<std::setprecision(17)<<"{\"backend\":\""<<(gpu?"cuda":"cpu")<<"\",\"controls\":"<<n<<",\"maximum_relative_area_vector_closure\":"<<closure_max<<",\"maximum_affine_green_gauss_error\":"<<affine_max<<",\"global_area_vector\":["<<global_closure[0]<<','<<global_closure[1]<<','<<global_closure[2]<<"],\"volume_weighted_gradient\":["<<weighted_gradient[0]<<','<<weighted_gradient[1]<<','<<weighted_gradient[2]<<"]}\n";std::cout<<"geometry closure="<<closure_max<<" affine_error="<<affine_max<<'\n';
 }
 const auto primitive_all=[&](const double* state){set_bad();
#ifdef __CUDACC__
  if(gpu){primitive_kernel<<<blocks_for(n),256>>>(config,data,state);return;}
#endif
  for(int k=0;k<n;++k)primitive_point(config,data,state,k);
 };
 const auto gradient_all=[&](){zero(data.gradient,18ull*n);
#ifdef __CUDACC__
  if(gpu){internal_gradient_kernel<<<blocks_for(internal_count),256>>>(config,data);interface_gradient_kernel<<<blocks_for(interface_count),256>>>(config,data);boundary_gradient_kernel<<<blocks_for(boundary_count),256>>>(config,data);stored_gradient_kernel<<<blocks_for(data.cut_count),256>>>(config,data,data.cut_faces,data.cut_count);stored_gradient_kernel<<<blocks_for(data.wall_count),256>>>(config,data,data.wall_faces,data.wall_count);finalize_gradient_kernel<<<blocks_for(n),256>>>(config,data);return;}
#endif
  host_faces(data,[&](const Face& face){gradient_face(config,data,face);});for(int k=0;k<n;++k)finalize_gradient(config,data,k);
 };
 const auto flux_all=[&](bool base){zero(data.balance,6ull*n);zero(data.spectral,n);zero(data.viscous_rate,n);zero(data.sa_rate,n);
#ifdef __CUDACC__
  if(gpu){internal_flux_kernel<<<blocks_for(internal_count),256>>>(config,data);interface_flux_kernel<<<blocks_for(interface_count),256>>>(config,data);boundary_flux_kernel<<<blocks_for(boundary_count),256>>>(config,data);stored_flux_kernel<<<blocks_for(data.cut_count),256>>>(config,data,data.cut_faces,data.cut_count);stored_flux_kernel<<<blocks_for(data.wall_count),256>>>(config,data,data.wall_faces,data.wall_count);finalize_residual_kernel<<<blocks_for(n),256>>>(config,data,base);return;}
#endif
  host_faces(data,[&](const Face& face){flux_face(config,data,face);});for(int k=0;k<n;++k)finalize_residual(config,data,k,base);
 };
 const auto rhs=[&](const double* state,bool base){primitive_all(state);if(config.uniform_transmissive){zero(data.gradient,18ull*n);zero(data.mu,n);zero(data.eddy,n);zero(data.diffusion,n);zero(data.source,n);zero(data.source_rate,n);}else gradient_all();flux_all(base);
#ifdef __CUDACC__
  if(gpu)checked(cudaDeviceSynchronize());
#endif
  if(get_bad())throw std::runtime_error("invalid AMR state or generated face");
 };
 const auto update=[&](int stage){
#ifdef __CUDACC__
  if(gpu){update_kernel<<<blocks_for(n),256>>>(data,stage);return;}
#endif
  for(int k=0;k<n;++k)update_point(data,k,stage);
 };
 const auto dplur_sum=[&](){zero(data.balance,6ull*n);
#ifdef __CUDACC__
  if(gpu){internal_dplur_kernel<<<blocks_for(internal_count),256>>>(config,data);interface_dplur_kernel<<<blocks_for(interface_count),256>>>(config,data);stored_dplur_kernel<<<blocks_for(data.cut_count),256>>>(config,data,data.cut_faces,data.cut_count);return;}
#endif
  host_faces(data,[&](const Face& face){dplur_face(config,data,face);});
 };
 const auto implicit_update=[&](){
#ifdef __CUDACC__
  if(gpu){dplur_initialize_kernel<<<blocks_for(n),256>>>(data);for(int sweep=1;sweep<dplur_sweeps;++sweep){dplur_sum();dplur_relax_kernel<<<blocks_for(n),256>>>(data,dplur_omega);std::swap(data.u1,data.u2);}dplur_apply_kernel<<<blocks_for(n),256>>>(data);return;}
#endif
  for(int k=0;k<n;++k)dplur_initialize_point(data,k);for(int sweep=1;sweep<dplur_sweeps;++sweep){dplur_sum();for(int k=0;k<n;++k)dplur_relax_point(data,k,dplur_omega);std::swap(data.u1,data.u2);}for(int k=0;k<n;++k)dplur_apply_point(data,k);
 };
 const auto norms=[&](){copy_to_host(raw,data.raw,6ull*n);double result[6]={},volume=0;for(int k=0;k<n;++k){volume+=controls[k].volume;for(int v=0;v<6;++v)result[v]+=controls[k].volume*fabs(raw[6*k+v]);}std::vector<double> values(6);for(int v=0;v<6;++v)values[v]=result[v]/volume/(v==5?config.reference[5]:1);return values;};
 const auto near_norms=[&](){double result[6]={},volume=0;for(int k=0;k<n;++k){bool selected=true;for(int j=0;j<3;++j)selected=selected&&controls[k].x[j]>=residual_low[j]&&controls[k].x[j]<=residual_high[j];if(selected){volume+=controls[k].volume;for(int v=0;v<6;++v)result[v]+=controls[k].volume*fabs(raw[6*k+v]);}}if(!(volume>0))throw std::runtime_error("empty near-field residual region");std::vector<double> values(6);for(int v=0;v<6;++v)values[v]=result[v]/volume/(v==5?config.reference[5]:1);return values;};
 if(number("--uniform-audit",0)!=0){config.uniform_transmissive=true;rhs(data.u,false);auto value=norms();double balance_error[6]={};for(int k=0;k<n;++k)for(int v=0;v<6;++v)balance_error[v]+=controls[k].volume*raw[6*k+v];std::ofstream report(output/"uniform_audit.json");report<<std::setprecision(17)<<"{\"backend\":\""<<(gpu?"cuda":"cpu")<<"\",\"raw_residual\":[";for(int v=0;v<6;++v)report<<(v?",":"")<<value[v];report<<"],\"global_conservation\":[";for(int v=0;v<6;++v)report<<(v?",":"")<<balance_error[v];report<<"]}\n";std::cout<<"uniform residual="<<*std::max_element(value.begin(),value.end())<<" conservation="<<*std::max_element(balance_error,balance_error+6)<<'\n';config.uniform_transmissive=false;}
 struct FlowReport {std::vector<double> residual,near_residual;WallReport wall;double force[3]={},cd=0,cl=0,cm=0,cdp=0,cdv=0,clp=0,clv=0;};
 const double qarea=.5*config.mach*config.mach*reference_area;
 const auto flow_report=[&](){rhs(data.u,false);FlowReport value;value.residual=norms();value.near_residual=near_norms();copy_to_host(w,data.w,6ull*n);copy_to_host(gradient,data.gradient,18ull*n);copy_to_host(mu,data.mu,n);value.wall=wall_report(config,controls,wall_faces,w,gradient,mu,moment_reference);double pressure[3],viscous[3];for(int j=0;j<3;++j){pressure[j]=value.wall.pressure[j];viscous[j]=value.wall.viscous[j];value.force[j]=pressure[j]+viscous[j];}value.cdp=projected_drag(pressure,aoa)/qarea;value.cdv=projected_drag(viscous,aoa)/qarea;value.cd=value.cdp+value.cdv;value.clp=projected_lift(pressure,aoa)/qarea;value.clv=projected_lift(viscous,aoa)/qarea;value.cl=value.clp+value.clv;value.cm=(value.wall.pressure_moment[1]+value.wall.viscous_moment[1])/qarea;return value;};
 const int iterations=static_cast<int>(number("--iterations",0)),check_period=static_cast<int>(number("--check-period",100));const double budget=number("--budget",30);if(iterations<0||check_period<1||!(budget>0))throw std::runtime_error("invalid iteration controls");const auto solve_start=std::chrono::steady_clock::now();const auto solve_seconds=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-solve_start).count();};int completed=0;
 if(iterations>0){
  std::ofstream history(output/"history.csv");history<<"iteration,seconds,res_rho,res_mx,res_my,res_mz,res_E,res_SA,res_mean_max,res_total_max,near_res_rho,near_res_mx,near_res_my,near_res_mz,near_res_E,near_res_SA,near_res_mean_max,near_res_total_max,CD,CL,CM_y,CD_pressure,CD_viscous,CL_pressure,CL_viscous,force_x,force_y,force_z\n";int last_report=-1;FlowReport latest;
  const auto record=[&](int iteration){latest=flow_report();const double mean=*std::max_element(latest.residual.begin(),latest.residual.begin()+5),total=std::max(mean,latest.residual[5]),near_mean=*std::max_element(latest.near_residual.begin(),latest.near_residual.begin()+5),near_total=std::max(near_mean,latest.near_residual[5]);history<<std::setprecision(17)<<iteration<<','<<solve_seconds();for(double value:latest.residual)history<<','<<value;history<<','<<mean<<','<<total;for(double value:latest.near_residual)history<<','<<value;history<<','<<near_mean<<','<<near_total<<','<<latest.cd<<','<<latest.cl<<','<<latest.cm<<','<<latest.cdp<<','<<latest.cdv<<','<<latest.clp<<','<<latest.clv<<','<<latest.force[0]<<','<<latest.force[1]<<','<<latest.force[2]<<'\n';history.flush();std::cout<<"iteration="<<iteration<<" global="<<total<<" near="<<near_total<<" CD="<<latest.cd<<" CL="<<latest.cl<<" CM="<<latest.cm<<'\n'<<std::flush;last_report=iteration;};
  record(0);for(;completed<iterations;){if(solve_seconds()>=budget)break;rhs(data.u,true);if(integrator=="dplur"){implicit_update();primitive_all(data.u2);if(get_bad())throw std::runtime_error("DP-LUR update produced an invalid state");std::swap(data.u,data.u2);}else{update(0);rhs(data.u1,false);update(1);rhs(data.u2,false);update(2);std::swap(data.u,data.u1);}++completed;if(completed%check_period==0)record(completed);}if(last_report!=completed)record(completed);const double compute_seconds=solve_seconds();
  copy_to_host(u,data.u,6ull*n);std::ofstream state(output/"state.bin",std::ios::binary);state.write(reinterpret_cast<const char*>(u.data()),static_cast<std::streamsize>(u.size()*sizeof(double)));state.close();if(!state)throw std::runtime_error("state output failed");
  std::vector<double> covered_area(wall_triangle_count),pressure_area(wall_triangle_count),triangle_pressure_force(3*wall_triangle_count),triangle_viscous_force(3*wall_triangle_count);for(size_t fi=0;fi<wall_faces.size();++fi){const size_t triangle=wall_triangle[fi];double pressure[3],viscous[3],wall_state[6];wall_load_components(config,controls,w,gradient,mu,wall_faces[fi],pressure,viscous);reconstruct_face_state(config,controls.data(),w.data(),gradient.data(),wall_faces[fi],wall_faces[fi].left,-1,wall_state);covered_area[triangle]+=wall_faces[fi].area;pressure_area[triangle]+=wall_state[4]*wall_faces[fi].area;for(int j=0;j<3;++j){triangle_pressure_force[3*triangle+j]+=pressure[j];triangle_viscous_force[3*triangle+j]+=viscous[j];}}
  double maximum_coverage_error=0;std::ofstream surface(output/"surface.csv");surface<<std::setprecision(17)<<"triangle,x,y,z,nx,ny,nz,area,covered_area,p,Cp,pressure_force_x,pressure_force_y,pressure_force_z,viscous_force_x,viscous_force_y,viscous_force_z,total_force_x,total_force_y,total_force_z\n";for(size_t triangle=0;triangle<wall_triangle_count;++triangle){const auto ids=wall_surface[triangle];const auto& a=wall_points[ids[0]];const auto& b=wall_points[ids[1]];const auto& c=wall_points[ids[2]];double ab[3],ac[3],cross[3],center[3];for(int j=0;j<3;++j){ab[j]=b[j]-a[j];ac[j]=c[j]-a[j];center[j]=(a[j]+b[j]+c[j])/3;}cross[0]=ab[1]*ac[2]-ab[2]*ac[1];cross[1]=ab[2]*ac[0]-ab[0]*ac[2];cross[2]=ab[0]*ac[1]-ab[1]*ac[0];const double twice_area=sqrt(dot3(cross,cross)),area=.5*twice_area;if(!(area>0)||!(covered_area[triangle]>0))throw std::runtime_error("uncovered or degenerate source wall triangle");maximum_coverage_error=std::max(maximum_coverage_error,fabs(covered_area[triangle]-area)/area);const double pressure=pressure_area[triangle]/covered_area[triangle],cp=(pressure-1/config.gamma)/(.5*config.mach*config.mach);surface<<triangle;for(double value:center)surface<<','<<value;for(double value:cross)surface<<','<<-value/twice_area;surface<<','<<area<<','<<covered_area[triangle]<<','<<pressure<<','<<cp;for(int j=0;j<3;++j)surface<<','<<triangle_pressure_force[3*triangle+j];for(int j=0;j<3;++j)surface<<','<<triangle_viscous_force[3*triangle+j];for(int j=0;j<3;++j)surface<<','<<triangle_pressure_force[3*triangle+j]+triangle_viscous_force[3*triangle+j];surface<<'\n';}surface.close();if(!surface)throw std::runtime_error("surface output failed");
  std::ofstream method(output/"method.json");method<<std::setprecision(17)<<"{\"integrator\":\""<<integrator<<"\",\"dplur_sweeps\":"<<(integrator=="dplur"?dplur_sweeps:0)<<",\"dplur_omega\":"<<(integrator=="dplur"?dplur_omega:0)<<",\"spatial_order\":"<<config.spatial_order<<",\"reconstruction_blend\":"<<config.reconstruction_blend<<"}\n";
  std::ofstream surface_summary(output/"surface_summary.json");surface_summary<<std::setprecision(17)<<"{\"source_triangles\":"<<wall_triangle_count<<",\"wall_fragments\":"<<wall_faces.size()<<",\"maximum_relative_fragment_area_coverage_error\":"<<maximum_coverage_error<<",\"Cp_definition\":\"(p-p_inf)/(0.5*rho_inf*V_inf^2)\"}\n";
  std::ofstream report(output/"result.json");report<<std::setprecision(17)<<"{\"backend\":\""<<(gpu?"cuda":"cpu")<<"\",\"mesh_format\":\"QCFDAM1 procedural\",\"restarted\":"<<(args.count("--restart")?"true":"false")<<",\"cells\":"<<n<<",\"iterations\":"<<completed<<",\"compute_seconds\":"<<compute_seconds<<",\"Mach\":"<<config.mach<<",\"AOA_degrees\":"<<number("--aoa",2.5)<<",\"Re_reference\":"<<config.mach/config.mu<<",\"CFL\":"<<config.cfl<<",\"reference_area\":"<<reference_area<<",\"moment_reference\":["<<moment_reference[0]<<','<<moment_reference[1]<<','<<moment_reference[2]<<"],\"raw_residual\":[";for(int v=0;v<6;++v)report<<(v?",":"")<<latest.residual[v];report<<"],\"near_field_residual\":[";for(int v=0;v<6;++v)report<<(v?",":"")<<latest.near_residual[v];report<<"],\"near_field_min\":["<<residual_low[0]<<','<<residual_low[1]<<','<<residual_low[2]<<"],\"near_field_max\":["<<residual_high[0]<<','<<residual_high[1]<<','<<residual_high[2]<<"],\"force\":["<<latest.force[0]<<','<<latest.force[1]<<','<<latest.force[2]<<"],\"pressure_force\":["<<latest.wall.pressure[0]<<','<<latest.wall.pressure[1]<<','<<latest.wall.pressure[2]<<"],\"viscous_force\":["<<latest.wall.viscous[0]<<','<<latest.wall.viscous[1]<<','<<latest.wall.viscous[2]<<"],\"CD\":"<<latest.cd<<",\"CL\":"<<latest.cl<<",\"CM_y\":"<<latest.cm<<",\"CD_pressure\":"<<latest.cdp<<",\"CD_viscous\":"<<latest.cdv<<",\"CL_pressure\":"<<latest.clp<<",\"CL_viscous\":"<<latest.clv<<",\"rejected\":0,\"converged\":false,\"surface_triangles\":"<<wall_triangle_count<<",\"maximum_surface_coverage_error\":"<<maximum_coverage_error<<",\"scope\":\"procedural AMR development run; aerodynamic accuracy not certified\"}\n";
 }
#ifdef __CUDACC__
 if(gpu)for(void* pointer:allocations)cudaFree(pointer);
#endif
 std::cout<<"total_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-process_start).count()<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<"ERROR: "<<error.what()<<'\n';return 2;}
