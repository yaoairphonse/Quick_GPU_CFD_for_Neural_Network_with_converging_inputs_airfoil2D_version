// Bounded-time exploratory RAE2822 driver. All steady criteria are explicit.
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
int main(int argc,char** argv) {
  std::string prefix="results/rae2822_m135_a5/steady",root=".";
  double budget=118;
  bool agglomerated=false;
  for(int i=1;i<argc;++i) {
    const std::string key=argv[i];
    if(i+1>=argc)return 2;
    if(key=="--out")prefix=argv[++i];
    else if(key=="--root")root=argv[++i];
    else if(key=="--budget")budget=std::stod(argv[++i]);
    else if(key=="--agglomerate"){const std::string value=argv[++i];if(value!="on"&&value!="off")return 2;agglomerated=value=="on";}
    else return 2;
  }
  if(!std::isfinite(budget)||budget<2||budget>120)return 2;
  std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
  const auto start=std::chrono::steady_clock::now();
  const auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
  std::vector<double> x,y;std::vector<int> patches;
  std::ifstream geometry(root+"/geometry/rae2822_uiuc_c1.csv");std::string line;
  while(std::getline(geometry,line)) {
    std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);double a,b;
    if(row>>a>>b){x.push_back(a);y.push_back(b);}
  }
  if(x.size()!=128)return 2;
  for(size_t i=0;i<x.size();++i)patches.push_back(i<64?1:2);
  StaticAmrOptions2D options;options.nx=42;options.ny=40;options.levels=5;
  options.xmin=-7;options.xmax=14;options.ymin=-10;options.ymax=10;
  options.agglomerate_small=agglomerated;
  options.boxes={{-.5,2,-.75,.75,3},{1,4,-.35,.65,3},{-2,6,-3,3,1}};
  Params2D p=params2d_default();p.cfl=.15;p.limiter=LIM_FIRST_ORDER;p.time_integrator=TIME_SSP_RK3;p.flux=FLUX_HLLC;
  const double alpha=5*std::acos(-1.)/180,cs=std::cos(alpha),sn=std::sin(alpha),qinf=.5*1.35*1.35;
  const std::array<double,4> reference={1,1.35*cs,1.35*sn,1/p.gamma};
  StaticAmrMesh2D mesh;std::string error;
  if(static_amr2d_build(options,x,y,patches,&mesh,&error)){std::cerr<<error<<'\n';return 1;}
  for(auto& b:mesh.boundaries){b.kind=AMR_BC_FARFIELD;std::copy(reference.begin(),reference.end(),b.reference);}
  mesh.transport.model=AMR_MODEL_LAMINAR;mesh.transport.mu_ref=.00075;
  mesh.transport.stabilization=agglomerated?AMR_VISCOUS_VOLUME:AMR_VISCOUS_SRD;mesh.transport.wall_probe_min_h=.25;
  if(static_amr2d_prepare_viscous(&mesh,p,&error)){std::cerr<<error<<'\n';return 1;}
  StaticAmrState2D state;
  if(static_amr2d_initialize(mesh,p,reference,&state))return 1;
  StaticAmrGpu* raw=nullptr;AmrGpuOptions gpu_options;gpu_options.fused=true;
  if(static_amr_gpu_create(mesh,p,state,gpu_options,&raw,&error)){std::cerr<<error<<'\n';return 1;}
  std::unique_ptr<StaticAmrGpu,void(*)(StaticAmrGpu*)> gpu(raw,static_amr_gpu_destroy);
  const auto mesh_io_start=std::chrono::steady_clock::now();
  if(static_amr_binary_mesh(mesh,p,prefix,&error)){std::cerr<<error;return 1;}
  double io_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-mesh_io_start).count();
  std::vector<std::pair<double,std::string>> snapshots;
  const auto save=[&](const std::string& tag){
    const auto io_start=std::chrono::steady_clock::now();
    const std::string path=prefix+"."+tag;
    if(static_amr_binary_snapshot(mesh,p,state,path,"cuda",&error))return false;
    snapshots.push_back({state.time,std::filesystem::path(path).filename().string()});
    std::ofstream index(prefix+".index.json");index<<std::setprecision(17)<<"{\"snapshots\":[";
    for(size_t i=0;i<snapshots.size();++i){if(i)index<<',';index<<"{\"time\":"<<snapshots[i].first<<",\"prefix\":"<<std::quoted(snapshots[i].second)<<'}';}
    index<<"]}\n";io_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-io_start).count();return static_cast<bool>(index);
  };
  if(!save("initial"))return 1;
  std::ofstream history(prefix+".history.csv");history<<std::setprecision(17)<<"time,steps,rejected,elapsed,res_rho,res_mx,res_my,res_E,Cl,Cd,delta_Cl,delta_Cd,hold\n";
  double previous_cl=0,previous_cd=0;bool previous=false,converged=false;int hold=0;
  std::array<double,4> final_residual={};double final_cl=0,final_cd=0;
  std::string reason="time-budget";
  size_t checks=0;double target=.01;
  while(elapsed()<budget-1&&state.time<100) {
    const int status=static_amr_gpu_advance(raw,target,50000000,&state,&error,std::max(.001,budget-1-elapsed()));
    if(static_amr_gpu_download(raw,&state,&error)){std::cerr<<error;return 1;}
    if(status==1){std::cerr<<error;save("failed_last");return 1;}
    if(state.rejected){std::cerr<<"rejection gate failed\n";save("failed_last");return 1;}
    const size_t n=mesh.cells.size();std::vector<double> rhs;
    if(static_amr2d_rhs(mesh,p,state.u,&rhs))return 1;
    std::array<long double,4> sum={};long double volume=0;
    for(size_t k=0;k<n;++k){const auto& c=mesh.cells[k];if(c.cx< -1||c.cx>3||std::fabs(c.cy)>1)continue;
      volume+=c.volume;for(int v=0;v<4;++v)sum[v]+=c.volume*std::fabs(rhs[v*n+k]);}
    for(int v=0;v<4;++v)final_residual[v]=static_cast<double>(sum[v]/volume);
    AmrViscousFields2D visc;
    if(static_amr2d_viscous_diagnostics(mesh,p,state.u,&visc))return 1;
    double fx=0,fy=0;
    for(size_t i=0;i<mesh.walls.size();++i){const auto& wall=mesh.walls[i];const size_t k=wall.owner_cell;
      const double pressure=(p.gamma-1)*(state.u[3*n+k]-.5*(state.u[n+k]*state.u[n+k]+state.u[2*n+k]*state.u[2*n+k])/state.u[k]);
      fx+=pressure*wall.nx*wall.length-visc.wall_flux[mesh.walls.size()+i];
      fy+=pressure*wall.ny*wall.length-visc.wall_flux[2*mesh.walls.size()+i];}
    final_cl=(-fx*sn+fy*cs)/qinf;final_cd=(fx*cs+fy*sn)/qinf;
    const double dcl=previous?std::fabs(final_cl-previous_cl)/std::max(.1,std::fabs(final_cl)):INFINITY;
    const double dcd=previous?std::fabs(final_cd-previous_cd)/std::max(.01,std::fabs(final_cd)):INFINITY;
    const bool small_residual=*std::max_element(final_residual.begin(),final_residual.end())<=1e-3;
    if(state.time>=3&&small_residual&&dcl<=.005&&dcd<=.005)++hold;else hold=0;
    history<<state.time<<','<<state.steps<<','<<state.rejected<<','<<elapsed();
    for(double value:final_residual)history<<','<<value;
    history<<','<<final_cl<<','<<final_cd<<','<<dcl<<','<<dcd<<','<<hold<<'\n';history.flush();
    std::cout<<"check t="<<state.time<<" steps="<<state.steps<<" elapsed="<<elapsed()<<" raw_res="<<*std::max_element(final_residual.begin(),final_residual.end())<<" Cl="<<final_cl<<" Cd="<<final_cd<<" hold="<<hold<<'\n'<<std::flush;
    previous_cl=final_cl;previous_cd=final_cd;previous=true;++checks;
    if(state.time==.01||state.time==.5||state.time==1||state.time==2||state.time==5||state.time==10){std::ostringstream name;name<<"s"<<std::setfill('0')<<std::setw(4)<<checks;if(!save(name.str()))return 1;}
    if(hold>=5){converged=true;reason="raw-residual-and-forces";break;}
    if(status==2)break;
    if(state.time>=100){reason="physical-time-cap";break;}
    target=state.time==.01?.5:std::min(100.,target+.5);
  }
  if(!save("final")){std::cerr<<error;return 1;}
  static_amr_gpu_write_stats(static_amr_gpu_stats(raw),prefix+".gpu.json",io_seconds,&error);
  std::ofstream result(prefix+".result.json");
  result<<std::setprecision(17)<<"{\"redistribution\":"<<std::quoted(agglomerated?"agglomerated-no-SRD":"state-SRD")<<",\"converged\":"<<(converged?"true":"false")<<",\"stop_reason\":"<<std::quoted(reason)<<",\"budget_seconds\":"<<budget<<",\"elapsed_seconds\":"<<elapsed()<<",\"physical_time\":"<<state.time<<",\"steps\":"<<state.steps<<",\"rejected\":"<<state.rejected<<",\"fluid_control_volumes\":"<<mesh.cells.size()<<",\"background_leaves\":"<<mesh.background_leaves<<",\"split_leaves\":"<<mesh.split_leaves<<",\"wall_dx\":0.015625,\"Mach\":1.35,\"AOA_degrees\":5,\"Re_chord\":1800,\"mu\":0.00075,\"kappa\":0,\"wall_probe_min_h\":0.25,\"Cl\":"<<final_cl<<",\"Cd\":"<<final_cd<<",\"nearfield_raw_residual\":[";
  for(int v=0;v<4;++v){if(v)result<<',';result<<final_residual[v];}
  result<<"],\"steady_rtol\":0.001,\"force_relative_tolerance\":0.005,\"required_hold\":5,\"checks\":"<<checks<<",\"physical_time_cap\":100}\n";
  std::cout<<"result "<<prefix<<".result.json converged="<<converged<<" elapsed="<<elapsed()<<'\n';
  return 0;
}
