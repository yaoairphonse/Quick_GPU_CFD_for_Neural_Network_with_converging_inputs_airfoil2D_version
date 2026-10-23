// Reusable bounded-time static cut-cell benchmark; physical scaling lives in runner manifest.
#include "static_amr2d_binary.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
using namespace cfd;
int main(int argc,char**argv){
 std::string root=".",name="G0001",prefix="results/test10_m165_a15/G0001/flow",mode="steady",backend="cuda",mesh_preset="coarse",viscosity="constant",turbulence="none",sa_layout="reference";
 double stop_time=100;double budget=118,mu=1.5975760554438918e-7;bool events=true;
 for(int i=1;i<argc;++i){std::string k=argv[i];if(i+1==argc)return 2;std::string v=argv[++i];
  if(k=="--root")root=v;else if(k=="--case")name=v;else if(k=="--out")prefix=v;else if(k=="--mode")mode=v;
  else if(k=="--backend")backend=v;else if(k=="--mesh")mesh_preset=v;else if(k=="--viscosity")viscosity=v;else if(k=="--turbulence")turbulence=v;else if(k=="--budget")budget=std::stod(v);else if(k=="--stop-time")stop_time=std::stod(v);else if(k=="--mu-star")mu=std::stod(v);else if(k=="--events")events=v=="on";else if(k=="--sa-layout")sa_layout=v;else return 2;}
 if((mode!="steady"&&mode!="mesh"&&mode!="smoke"&&mode!="rest")||(backend!="cpu"&&backend!="cuda")||(mesh_preset!="coarse"&&mesh_preset!="fine")||(viscosity!="constant"&&viscosity!="sutherland")||(turbulence!="none"&&turbulence!="sa-neg")||!std::isfinite(stop_time)||stop_time<=0||stop_time>100||(sa_layout!="reference"&&sa_layout!="fused")||budget<2||budget>120||!std::isfinite(mu)||mu<=0)return 2;
 std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
 const auto begin=std::chrono::steady_clock::now();const auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();};
 std::vector<double>x,y;std::vector<int>patches;std::string line,error;
 std::ifstream geo(root+"/geometry/test10/"+name+".csv");while(std::getline(geo,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);double a,b;if(row>>a>>b){x.push_back(a);y.push_back(b);}}
 std::ifstream pat(root+"/geometry/test10/"+name+".patches.csv");while(std::getline(pat,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int a,b;if(row>>a>>b){if(a!=static_cast<int>(patches.size()))return 2;patches.push_back(b);}}
 if(x.size()<3||x.size()!=patches.size())return 2;
 StaticAmrOptions2D options;options.nx=42;options.ny=40;options.levels=5;options.xmin=-7;options.xmax=14;options.ymin=-10;options.ymax=10;options.agglomerate_small=true;
 options.boxes={{-.5,2,-.75,.75,3},{1,4,-.35,.65,3},{-2,6,-3,3,1}};
 if(mesh_preset=="fine"){options.levels=6;options.boxes[0].level=4;options.boxes.push_back({2,3.25,-.35,.65,4});}
 const double wall_dx=.5/(1<<options.levels);
 Params2D p=params2d_default();p.cfl=.15;p.limiter=LIM_FIRST_ORDER;p.time_integrator=TIME_SSP_RK3;p.flux=FLUX_HLLC;
 const double alpha=1.5*std::acos(-1.)/180,cs=std::cos(alpha),sn=std::sin(alpha),mach=1.65,qinf=.5*mach*mach;
 const std::array<double,4> reference={1,mode=="rest"?0:mach*cs,mode=="rest"?0:mach*sn,1/p.gamma};
 StaticAmrMesh2D mesh;
 if(static_amr2d_build(options,x,y,patches,&mesh,&error)){std::cerr<<"geometry: "<<error<<'\n';return 1;}
 for(auto&b:mesh.boundaries){b.kind=AMR_BC_FARFIELD;std::copy(reference.begin(),reference.end(),b.reference);}
 mesh.transport.model=AMR_MODEL_LAMINAR;mesh.transport.sa_model=turbulence=="sa-neg";mesh.transport.sa_nu_inf=3*mu;mesh.transport.mu_ref=mu;
 if(viscosity=="sutherland"){mesh.transport.law=AMR_MU_SUTHERLAND;mesh.transport.temperature_ref=1/p.gamma;mesh.transport.sutherland=110.4/(p.gamma*216.65);}mesh.transport.wall_probe_min_h=.25;mesh.transport.stabilization=AMR_VISCOUS_VOLUME;
 if(static_amr2d_prepare_viscous(&mesh,p,&error)){std::cerr<<error;return 1;}
 // Audit exact geometric integration, before any flow execution.
 long double solid=0,perimeter=0,V=0,W=0;const size_t n=mesh.cells.size();std::vector<std::array<double,2>>closure(n);std::vector<double>surface(n);
 for(size_t i=0;i<x.size();++i){size_t j=(i+1)%x.size();solid+=(static_cast<long double>(x[i])*y[j]-static_cast<long double>(y[i])*x[j])*.5;perimeter+=std::hypot(x[i]-x[j],y[i]-y[j]);}
 for(const auto&c:mesh.cells){if(!(c.volume>0))return 1;V+=c.volume;}
 for(const auto&f:mesh.faces){if(f.left>=0){closure[f.left][f.axis]+=f.length;surface[f.left]+=f.length;}if(f.right>=0){closure[f.right][f.axis]-=f.length;surface[f.right]+=f.length;}}
 for(const auto&w:mesh.walls){W+=w.length;closure[w.owner_cell][0]+=w.nx*w.length;closure[w.owner_cell][1]+=w.ny*w.length;surface[w.owner_cell]+=w.length;}
 double maxclosure=0;for(size_t k=0;k<n;++k)maxclosure=std::max(maxclosure,std::hypot(closure[k][0],closure[k][1])/surface[k]);
 const double areaerror=std::fabs(static_cast<double>(V-(420-std::fabs(solid)))),wallerror=std::fabs(static_cast<double>(W-perimeter));
 if(areaerror>1e-10||wallerror>1e-11||maxclosure>1e-9){std::cerr<<"geometry audit failed "<<areaerror<<' '<<wallerror<<' '<<maxclosure;return 1;}
 StaticAmrState2D state;if(static_amr2d_initialize(mesh,p,reference,&state))return 1;const auto initial=state.u;
 const double mesh_seconds=elapsed();
 StaticAmrGpu*raw=nullptr;std::unique_ptr<StaticAmrGpu,void(*)(StaticAmrGpu*)>gpu(nullptr,static_amr_gpu_destroy);
 if(backend=="cuda"&&mode!="mesh"){AmrGpuOptions o;o.fused=true;o.event_timing=events;o.sa_fused=sa_layout=="fused";if(static_amr_gpu_create(mesh,p,state,o,&raw,&error)){std::cerr<<error;return 1;}gpu.reset(raw);}
 double io=0;const auto write_mesh=[&](){auto t=std::chrono::steady_clock::now();const int z=static_amr_binary_mesh(mesh,p,prefix,&error);io+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return z==0;};
 const auto save=[&](std::string tag){auto t=std::chrono::steady_clock::now();const int z=static_amr_binary_snapshot(mesh,p,state,prefix+"."+tag,backend.c_str(),&error);io+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return z==0;};
 if(!write_mesh()||!save("initial"))return 1;
 bool converged=false,previous=false,failed=false;int hold=0,checks=0;double prevcl=0,prevcd=0,prevcm=0,cl=0,cd=0,cm=0,target=std::min(.01,stop_time);
 std::array<double,5>res={};std::string reason=mode=="mesh"?"mesh-only":"time-budget";
 std::ofstream history(prefix+".history.csv");history<<std::setprecision(17)<<"time,steps,rejected,elapsed,res_rho,res_mx,res_my,res_E,res_sa,Cl,Cd,Cm,delta_Cl,delta_Cd,delta_Cm,hold\n";
 double cpu_advance=0;
 while(mode!="mesh"&&elapsed()<budget-1&&state.time<stop_time){
  int status=0;
  if(raw){status=static_amr_gpu_advance(raw,target,50000000,&state,&error,std::max(.001,budget-1-elapsed()));if(static_amr_gpu_download(raw,&state,&error))return 1;}
  else{auto t=std::chrono::steady_clock::now();status=static_amr2d_run(mesh,p,target,50000000,&state,&error);cpu_advance+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
  if(status==1||state.rejected){std::cerr<<"flow: "<<error<<" rejected="<<state.rejected<<'\n';failed=true;reason="admissibility-or-advance-failure";break;}
  std::vector<double>rhs;if(static_amr2d_rhs(mesh,p,state.u,&rhs))return 1;
  std::array<long double,5>sum={};long double volume=0;bool positive=true;
  for(size_t k=0;k<n;++k){const double rho=state.u[k],pressure=(p.gamma-1)*(state.u[3*n+k]-.5*(state.u[n+k]*state.u[n+k]+state.u[2*n+k]*state.u[2*n+k])/rho);positive=positive&&std::isfinite(rho)&&rho>0&&std::isfinite(pressure)&&pressure>0;const auto&c=mesh.cells[k];if(c.cx< -1||c.cx>3||std::fabs(c.cy)>1)continue;volume+=c.volume;for(int v=0;v<(mesh.transport.sa_model?5:4);++v)sum[v]+=c.volume*std::fabs(rhs[v*n+k]);}
  if(!positive){failed=true;reason="nonpositive-state";break;}
  for(int v=0;v<(mesh.transport.sa_model?5:4);++v)res[v]=static_cast<double>(sum[v]/volume);
  if(mesh.transport.sa_model)res[4]/=mesh.transport.sa_nu_inf;
  AmrViscousFields2D visc;if(static_amr2d_viscous_diagnostics(mesh,p,state.u,&visc))return 1;
  double fx=0,fy=0,mz=0;
  for(size_t i=0;i<mesh.walls.size();++i){const auto&w=mesh.walls[i];size_t k=w.owner_cell;double pressure=(p.gamma-1)*(state.u[3*n+k]-.5*(state.u[n+k]*state.u[n+k]+state.u[2*n+k]*state.u[2*n+k])/state.u[k]);double a=pressure*w.nx*w.length-visc.wall_flux[mesh.walls.size()+i],b=pressure*w.ny*w.length-visc.wall_flux[2*mesh.walls.size()+i];fx+=a;fy+=b;mz+=((w.x0+w.x1)*.5-.25)*b-(w.y0+w.y1)*.5*a;}
  cl=(-fx*sn+fy*cs)/qinf;cd=(fx*cs+fy*sn)/qinf;cm=mz/qinf;
  const double dcl=previous?std::fabs(cl-prevcl)/std::max(.1,std::fabs(cl)):INFINITY,dcd=previous?std::fabs(cd-prevcd)/std::max(.01,std::fabs(cd)):INFINITY,dcm=previous?std::fabs(cm-prevcm)/std::max(.01,std::fabs(cm)):INFINITY;
  if(state.time>=3&&*std::max_element(res.begin(),res.end())<=.001&&dcl<=.005&&dcd<=.005&&dcm<=.005)++hold;else hold=0;
  history<<state.time<<','<<state.steps<<','<<state.rejected<<','<<elapsed();for(double v:res)history<<','<<v;history<<','<<cl<<','<<cd<<','<<cm<<','<<dcl<<','<<dcd<<','<<dcm<<','<<hold<<'\n';history.flush();
  std::cout<<"check "<<name<<" t="<<state.time<<" steps="<<state.steps<<" seconds="<<elapsed()<<" raw="<<*std::max_element(res.begin(),res.end())<<" Cl="<<cl<<" Cd="<<cd<<" Cm="<<cm<<" hold="<<hold<<'\n'<<std::flush;
  prevcl=cl;prevcd=cd;prevcm=cm;previous=true;++checks;
  if(mode=="smoke"||mode=="rest"){reason="short-validation";break;}
  if(hold>=5){converged=true;reason="raw-residual-and-three-coefficients";break;}
  if(status==2)break;
  if(state.time>=stop_time){reason="physical-time-cap";break;}target=state.time==.01?std::min(stop_time,.5):std::min(stop_time,target+.5);
 }
 if(!save(failed?"failed_last":"final")){std::cerr<<error;return 1;}
 auto stats=static_amr_gpu_stats(raw);if(static_amr_gpu_write_stats(stats,prefix+".gpu.json",io,&error))return 1;
 double rest_error=0;if(mode=="rest")for(size_t i=0;i<4*n;++i)rest_error=std::max(rest_error,std::fabs(initial[i]-state.u[i]));
 std::ofstream out(prefix+".result.json");out<<std::setprecision(17)<<"{\"case\":"<<std::quoted(name)<<",\"mode\":"<<std::quoted(mode)<<",\"converged\":"<<(converged?"true":"false")<<",\"failed\":"<<(failed?"true":"false")<<",\"stop_reason\":"<<std::quoted(reason)<<",\"elapsed_seconds\":"<<elapsed()<<",\"mesh_setup_seconds\":"<<mesh_seconds<<",\"cpu_advance_seconds\":"<<cpu_advance<<",\"physical_time\":"<<state.time<<",\"steps\":"<<state.steps<<",\"rejected\":"<<state.rejected<<",\"fluid_control_volumes\":"<<n<<",\"background_leaves\":"<<mesh.background_leaves<<",\"split_leaves\":"<<mesh.split_leaves<<",\"sa_layout\":"<<std::quoted(sa_layout)<<",\"stop_time_cap\":"<<stop_time<<",\"turbulence_model\":"<<std::quoted(turbulence)<<",\"mesh_preset\":"<<std::quoted(mesh_preset)<<",\"viscosity_law\":"<<std::quoted(viscosity)<<",\"wall_dx\":"<<wall_dx<<",\"Mach\":1.65,\"AOA_degrees\":1.5,\"mu_star\":"<<mu<<",\"kappa\":0,\"Cl\":"<<cl<<",\"Cd\":"<<cd<<",\"Cm\":"<<cm<<",\"moment_reference\":[0.25,0],\"moment_sign\":\"CCW-positive\",\"raw_residual\":[";
 for(int v=0;v<(mesh.transport.sa_model?5:4);++v){if(v)out<<',';out<<res[v];}out<<"],\"checks\":"<<checks<<",\"hold\":"<<hold<<",\"rest_linf\":"<<rest_error<<",\"area_error\":"<<areaerror<<",\"wall_length_error\":"<<wallerror<<",\"closure_relative_linf\":"<<maxclosure<<"}\n";
 return failed||(mode=="rest"&&rest_error>2e-11)?1:0;
}
