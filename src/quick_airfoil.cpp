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
#include <map>
#include <sstream>
using namespace cfd;
int main(int argc,char**argv)try{
 if(argc!=9){std::cerr<<"usage: quick_airfoil --settings FILE --geometry FILE --patches FILE --out PREFIX\n";return 2;}
 std::map<std::string,std::string> args;
 for(int i=1;i<argc;i+=2)args[argv[i]]=argv[i+1];
 if(!args.count("--settings")||!args.count("--geometry")||!args.count("--patches")||!args.count("--out"))return 2;
 std::map<std::string,std::string> cfg;std::vector<AmrBox2D> boxes;std::string setting;
 std::ifstream settings(args["--settings"]);if(!settings)return 2;
 while(std::getline(settings,setting)){std::istringstream row(setting);std::string key,value;if(!(row>>key))continue;
  if(key=="box"){AmrBox2D box;if(!(row>>box.x0>>box.x1>>box.y0>>box.y1>>box.level))return 2;boxes.push_back(box);}
  else{if(!(row>>value))return 2;cfg[key]=value;}}
 const auto number=[&](const char*key){return std::stod(cfg.at(key));};
 const std::string name=cfg.at("name"),prefix=args["--out"],mode=cfg.at("mode"),backend=cfg.at("backend"),mesh_preset=cfg.at("mesh_preset"),viscosity=cfg.at("viscosity"),turbulence=cfg.at("turbulence"),sa_layout=cfg.at("sa_layout");
 const double stop_time=number("stop_time"),budget=number("budget"),mu=number("mu_star"),mach=number("mach"),aoa=number("aoa"),gamma=number("gamma");
 const bool events=number("events")!=0;
 if((mode!="steady"&&mode!="time"&&mode!="mesh"&&mode!="smoke"&&mode!="rest")||(backend!="cpu"&&backend!="cuda")||(viscosity!="constant"&&viscosity!="sutherland")||(turbulence!="none"&&turbulence!="sa-neg")||!std::isfinite(stop_time)||stop_time<=0||budget<2||!std::isfinite(mu)||mu<=0||!(mach>0)||!(gamma>1))return 2;
 std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
 const auto begin=std::chrono::steady_clock::now();const auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();};
 std::vector<double>x,y;std::vector<int>patches;std::string line,error;
 std::ifstream geo(args["--geometry"]);while(std::getline(geo,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);double a,b;if(row>>a>>b){x.push_back(a);y.push_back(b);}}
 std::ifstream pat(args["--patches"]);while(std::getline(pat,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int a,b;if(row>>a>>b){if(a!=static_cast<int>(patches.size()))return 2;patches.push_back(b);}}
 if(x.size()<3||x.size()!=patches.size())return 2;
 StaticAmrOptions2D options;options.nx=static_cast<int>(number("nx"));options.ny=static_cast<int>(number("ny"));options.levels=static_cast<int>(number("levels"));options.buffer=static_cast<int>(number("buffer"));options.xmin=number("xmin");options.xmax=number("xmax");options.ymin=number("ymin");options.ymax=number("ymax");options.max_cells=static_cast<size_t>(number("max_cells"));options.agglomerate_small=true;options.boxes=boxes;
 const double wall_dx=(options.xmax-options.xmin)/options.nx/(1<<options.levels),wall_dy=(options.ymax-options.ymin)/options.ny/(1<<options.levels);
 Params2D p=params2d_default();p.gamma=gamma;p.cfl=number("cfl");p.dt_fixed=number("fixed_dt");p.limiter=LIM_FIRST_ORDER;p.amr_spatial_order=cfg.count("spatial_order")?static_cast<int>(number("spatial_order")):1;p.time_integrator=TIME_SSP_RK3;p.flux=cfg.at("flux")=="hllc"?FLUX_HLLC:FLUX_AUSMPUP;
 p.low_mach_preconditioning=cfg.count("preconditioning")&&cfg.at("preconditioning")=="turkel";p.low_mach_reference=mach;
 p.local_pseudo_time=cfg.count("local_pseudo_time")&&number("local_pseudo_time")!=0;
 const bool adaptive=cfg.count("adaptive_cfl")&&number("adaptive_cfl")!=0;
 const double cflmin=adaptive?number("cfl_min"):p.cfl,cflmax=adaptive?number("cfl_max"):p.cfl;
 if((p.local_pseudo_time||adaptive)&&(mode=="time"||p.dt_fixed>0)){std::cerr<<"local/adaptive stepping requires steady pseudo-time\n";return 2;}
 if(cfg.count("mach_floor"))p.low_mach_floor=number("mach_floor");
 if(p.low_mach_preconditioning&&(mode=="time"||p.dt_fixed>0||!(p.low_mach_floor>0&&p.low_mach_floor<=1))){std::cerr<<"Turkel preconditioning requires steady pseudo-time and automatic CFL\n";return 2;}
 const double alpha=aoa*std::acos(-1.)/180,cs=std::cos(alpha),sn=std::sin(alpha),qinf=.5*mach*mach;
 const double mxref=number("moment_x"),myref=number("moment_y"),rawtol=number("raw_tol"),loadtol=number("load_tol"),tmin=number("tmin"),period=number("period");
 const std::string convergence_metric=cfg.count("convergence_metric")?cfg.at("convergence_metric"):"all_loads";
 const double cp_tol=cfg.count("cp_tol")?number("cp_tol"):.005;
 if((convergence_metric!="all_loads"&&convergence_metric!="cl_cm_cp")||!(cp_tol>0))return 2;
 const double resxmin=number("res_xmin"),resxmax=number("res_xmax"),resymin=number("res_ymin"),resymax=number("res_ymax");
 const int required_hold=static_cast<int>(number("hold"));
 const std::array<double,4> reference={1,mode=="rest"?0:mach*cs,mode=="rest"?0:mach*sn,1/p.gamma};
 StaticAmrMesh2D mesh;
 if(static_amr2d_build(options,x,y,patches,&mesh,&error)){std::cerr<<"geometry: "<<error<<'\n';return 1;}
 for(int side=0;side<4;++side){auto&b=mesh.boundaries[side];std::copy(reference.begin(),reference.end(),b.reference);const std::string kind=cfg.at("bc"+std::to_string(side));
  b.kind=kind=="pressure-outlet"?AMR_BC_PRESSURE_OUTLET:kind=="transmissive"?AMR_BC_TRANSMISSIVE:AMR_BC_FARFIELD;b.outlet_pressure=number("outlet_p_star");}
 mesh.transport.model=AMR_MODEL_LAMINAR;mesh.transport.sa_model=turbulence=="sa-neg";mesh.transport.sa_nu_inf=number("sa_nu_inf");mesh.transport.mu_ref=mu;
 if(viscosity=="sutherland"){mesh.transport.law=AMR_MU_SUTHERLAND;mesh.transport.temperature_ref=1/p.gamma;mesh.transport.sutherland=number("sutherland_star");}mesh.transport.viscous_cfl=number("viscous_cfl");mesh.transport.sa_cfl=number("sa_cfl");mesh.transport.wall_probe_min_h=number("wall_probe_min_h");mesh.transport.stabilization=AMR_VISCOUS_VOLUME;
 if(static_amr2d_prepare_viscous(&mesh,p,&error)){std::cerr<<error;return 1;}
 // Audit exact geometric integration, before any flow execution.
 long double solid=0,perimeter=0,V=0,W=0;const size_t n=mesh.cells.size();std::vector<std::array<double,2>>closure(n);std::vector<double>surface(n);
 for(size_t i=0;i<x.size();++i){size_t j=(i+1)%x.size();solid+=(static_cast<long double>(x[i])*y[j]-static_cast<long double>(y[i])*x[j])*.5;perimeter+=std::hypot(x[i]-x[j],y[i]-y[j]);}
 for(const auto&c:mesh.cells){if(!(c.volume>0))return 1;V+=c.volume;}
 for(const auto&f:mesh.faces){if(f.left>=0){closure[f.left][f.axis]+=f.length;surface[f.left]+=f.length;}if(f.right>=0){closure[f.right][f.axis]-=f.length;surface[f.right]+=f.length;}}
 for(const auto&w:mesh.walls){W+=w.length;closure[w.owner_cell][0]+=w.nx*w.length;closure[w.owner_cell][1]+=w.ny*w.length;surface[w.owner_cell]+=w.length;}
 double maxclosure=0;for(size_t k=0;k<n;++k)maxclosure=std::max(maxclosure,std::hypot(closure[k][0],closure[k][1])/surface[k]);
 const double areaerror=std::fabs(static_cast<double>(V-((options.xmax-options.xmin)*(options.ymax-options.ymin)-std::fabs(solid)))),wallerror=std::fabs(static_cast<double>(W-perimeter));
 if(areaerror>1e-10||wallerror>1e-11||maxclosure>1e-9){std::cerr<<"geometry audit failed "<<areaerror<<' '<<wallerror<<' '<<maxclosure;return 1;}
 StaticAmrState2D state;if(static_amr2d_initialize(mesh,p,reference,&state))return 1;const auto initial=state.u;
 const double mesh_seconds=elapsed();
 StaticAmrGpu*raw=nullptr;std::unique_ptr<StaticAmrGpu,void(*)(StaticAmrGpu*)>gpu(nullptr,static_amr_gpu_destroy);
 if(backend=="cuda"&&mode!="mesh"){AmrGpuOptions o;o.fused=true;o.event_timing=events;o.sa_fused=sa_layout=="fused";if(static_amr_gpu_create(mesh,p,state,o,&raw,&error)){std::cerr<<error;return 1;}gpu.reset(raw);}
 double io=0;const auto write_mesh=[&](){auto t=std::chrono::steady_clock::now();const int z=static_amr_binary_mesh(mesh,p,prefix,&error);io+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return z==0;};
 const auto save=[&](std::string tag){auto t=std::chrono::steady_clock::now();const int z=static_amr_binary_snapshot(mesh,p,state,prefix+"."+tag,backend.c_str(),&error);io+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return z==0;};
 if(!write_mesh()||!save("initial"))return 1;
 bool converged=false,previous=false,failed=false;int hold=0,checks=0;double prevcl=0,prevcd=0,prevcm=0,cl=0,cd=0,cm=0,target=std::min(.01,stop_time);std::vector<double> previous_wall_pressure;
 std::array<double,5>res={};std::string reason=mode=="mesh"?"mesh-only":"time-budget";
 std::ofstream history(prefix+".history.csv");history<<std::setprecision(17)<<"time,steps,rejected,elapsed,res_rho,res_mx,res_my,res_E,res_sa,Cl,Cd,Cm,delta_Cl,delta_Cd,delta_Cm,delta_Cp_rms,hold\n";
 double cpu_advance=0,previous_raw=INFINITY;long previous_rejected=0;
 std::ofstream cflhistory(prefix+".cfl.csv");cflhistory<<"pseudo_time,steps,cfl,raw_max,rejected\n";
 while(mode!="mesh"&&elapsed()<budget-1&&state.time<stop_time){
  int status=0;
  if(raw){status=static_amr_gpu_advance(raw,target,50000000,&state,&error,std::max(.001,budget-1-elapsed()));if(static_amr_gpu_download(raw,&state,&error))return 1;}
  else{auto t=std::chrono::steady_clock::now();status=static_amr2d_run(mesh,p,target,50000000,&state,&error);cpu_advance+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}
  if(status==1||(state.rejected&&!p.local_pseudo_time)){std::cerr<<"flow: "<<error<<" rejected="<<state.rejected<<'\n';failed=true;reason="admissibility-or-advance-failure";break;}
  std::vector<double>rhs;if(static_amr2d_rhs(mesh,p,state.u,&rhs))return 1;
  std::array<long double,5>sum={};long double volume=0;bool positive=true;
  for(size_t k=0;k<n;++k){const double rho=state.u[k],pressure=(p.gamma-1)*(state.u[3*n+k]-.5*(state.u[n+k]*state.u[n+k]+state.u[2*n+k]*state.u[2*n+k])/rho);positive=positive&&std::isfinite(rho)&&rho>0&&std::isfinite(pressure)&&pressure>0;const auto&c=mesh.cells[k];if(c.cx<resxmin||c.cx>resxmax||c.cy<resymin||c.cy>resymax)continue;volume+=c.volume;for(int v=0;v<(mesh.transport.sa_model?5:4);++v)sum[v]+=c.volume*std::fabs(rhs[v*n+k]);}
  if(!positive){failed=true;reason="nonpositive-state";break;}
  if(!(volume>0)){std::cerr<<"empty residual region";return 2;}
  for(int v=0;v<(mesh.transport.sa_model?5:4);++v)res[v]=static_cast<double>(sum[v]/volume);
  if(mesh.transport.sa_model)res[4]/=mesh.transport.sa_nu_inf;
  AmrViscousFields2D visc;if(static_amr2d_viscous_diagnostics(mesh,p,state.u,&visc))return 1;
  std::vector<double> wall_pressure;if(static_amr2d_wall_pressure(mesh,p,state.u,&wall_pressure))return 1;
  double fx=0,fy=0,mz=0;
  for(size_t i=0;i<mesh.walls.size();++i){const auto&w=mesh.walls[i];double pressure=wall_pressure[i];double a=pressure*w.nx*w.length-visc.wall_flux[mesh.walls.size()+i],b=pressure*w.ny*w.length-visc.wall_flux[2*mesh.walls.size()+i];fx+=a;fy+=b;mz+=((w.x0+w.x1)*.5-mxref)*b-((w.y0+w.y1)*.5-myref)*a;}
  cl=(-fx*sn+fy*cs)/qinf;cd=(fx*cs+fy*sn)/qinf;cm=mz/qinf;
  const double dcl=previous?std::fabs(cl-prevcl)/std::max(.1,std::fabs(cl)):INFINITY,dcd=previous?std::fabs(cd-prevcd)/std::max(.01,std::fabs(cd)):INFINITY,dcm=previous?std::fabs(cm-prevcm)/std::max(.01,std::fabs(cm)):INFINITY;
  double dcp=INFINITY;if(previous){double sum=0,length=0;for(size_t i=0;i<mesh.walls.size();++i){const double delta=(wall_pressure[i]-previous_wall_pressure[i])/qinf;sum+=mesh.walls[i].length*delta*delta;length+=mesh.walls[i].length;}dcp=std::sqrt(sum/length);}
  const bool loads_stable=convergence_metric=="cl_cm_cp"?dcl<=loadtol&&dcm<=loadtol&&dcp<=cp_tol:dcl<=loadtol&&dcd<=loadtol&&dcm<=loadtol;
  if(state.time>=tmin&&*std::max_element(res.begin(),res.end())<=rawtol&&loads_stable)++hold;else hold=0;
  history<<state.time<<','<<state.steps<<','<<state.rejected<<','<<elapsed();for(double v:res)history<<','<<v;history<<','<<cl<<','<<cd<<','<<cm<<','<<dcl<<','<<dcd<<','<<dcm<<','<<dcp<<','<<hold<<'\n';history.flush();
  std::cout<<"check "<<name<<" t="<<state.time<<" steps="<<state.steps<<" seconds="<<elapsed()<<" raw="<<*std::max_element(res.begin(),res.end())<<" Cl="<<cl<<" Cd="<<cd<<" Cm="<<cm<<" hold="<<hold<<'\n'<<std::flush;
  prevcl=cl;prevcd=cd;prevcm=cm;previous_wall_pressure=std::move(wall_pressure);previous=true;++checks;
  const double current_raw=*std::max_element(res.begin(),res.end());
  cflhistory<<state.time<<','<<state.steps<<','<<p.cfl<<','<<current_raw<<','<<state.rejected<<'\n';cflhistory.flush();
  if(adaptive){
    if(current_raw>1.15*previous_raw||state.rejected>previous_rejected)p.cfl=std::max(cflmin,p.cfl*number("cfl_shrink"));
    else if(current_raw<.98*previous_raw)p.cfl=std::min(cflmax,p.cfl*number("cfl_growth"));
    if(raw&&static_amr_gpu_set_cfl(raw,p.cfl,&error)){std::cerr<<error;return 1;}
  }
  previous_raw=current_raw;previous_rejected=state.rejected;
  if(mode=="smoke"||mode=="rest"){reason="short-validation";break;}
  if(mode!="time"&&hold>=required_hold){converged=true;reason=convergence_metric=="cl_cm_cp"?"raw-residual-cl-cm-cp":"raw-residual-and-three-coefficients";break;}
  if(status==2)break;
  if(state.time>=stop_time){reason="physical-time-cap";break;}target=checks==1?std::min(stop_time,period):std::min(stop_time,target+period);
 }
 if(!save(failed?"failed_last":"final")){std::cerr<<error;return 1;}
 auto stats=static_amr_gpu_stats(raw);if(static_amr_gpu_write_stats(stats,prefix+".gpu.json",io,&error))return 1;
 double rest_error=0;if(mode=="rest")for(size_t i=0;i<4*n;++i)rest_error=std::max(rest_error,std::fabs(initial[i]-state.u[i]));
 std::ofstream out(prefix+".result.json");out<<std::setprecision(17)<<"{\"case\":"<<std::quoted(name)<<",\"mode\":"<<std::quoted(mode)<<",\"converged\":"<<(converged?"true":"false")<<",\"failed\":"<<(failed?"true":"false")<<",\"stop_reason\":"<<std::quoted(reason)<<",\"elapsed_seconds\":"<<elapsed()<<",\"mesh_setup_seconds\":"<<mesh_seconds<<",\"cpu_advance_seconds\":"<<cpu_advance<<",\"physical_time\":"<<state.time<<",\"steps\":"<<state.steps<<",\"rejected\":"<<state.rejected<<",\"fluid_control_volumes\":"<<n<<",\"background_leaves\":"<<mesh.background_leaves<<",\"split_leaves\":"<<mesh.split_leaves<<",\"sa_layout\":"<<std::quoted(sa_layout)<<",\"stop_time_cap\":"<<stop_time<<",\"turbulence_model\":"<<std::quoted(turbulence)<<",\"mesh_preset\":"<<std::quoted(mesh_preset)<<",\"viscosity_law\":"<<std::quoted(viscosity)<<",\"wall_dx\":"<<wall_dx<<",\"wall_dy\":"<<wall_dy<<",\"Mach\":"<<mach<<",\"AOA_degrees\":"<<aoa<<",\"mu_star\":"<<mu<<",\"kappa\":0,\"Cl\":"<<cl<<",\"Cd\":"<<cd<<",\"Cm\":"<<cm<<",\"moment_reference\":["<<mxref<<","<<myref<<"],\"moment_sign\":\"CCW-positive\",\"raw_residual\":[";
 for(int v=0;v<(mesh.transport.sa_model?5:4);++v){if(v)out<<',';out<<res[v];}out<<"],\"spatial_order\":"<<p.amr_spatial_order<<",\"convergence_metric\":"<<std::quoted(convergence_metric)<<",\"cp_change_tolerance\":"<<cp_tol<<",\"checks\":"<<checks<<",\"hold\":"<<hold<<",\"rest_linf\":"<<rest_error<<",\"area_error\":"<<areaerror<<",\"wall_length_error\":"<<wallerror<<",\"closure_relative_linf\":"<<maxclosure<<"}\n";
 return failed||(mode=="rest"&&rest_error>2e-11)?1:0;
} catch(const std::exception& e){std::cerr<<"configuration/runtime error: "<<e.what()<<"\n";return 2;}
