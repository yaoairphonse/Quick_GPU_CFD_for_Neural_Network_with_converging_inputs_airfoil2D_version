#include "static_amr2d.h"
#include "static_amr2d_binary.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <filesystem>
#include <iomanip>

using namespace cfd;
namespace {
void usage() {
  std::cerr<<"usage: euler2d_amr_cli --out PREFIX [--body polygon:path.csv]\n"
    "  --base-nx N --base-ny N --levels L --wall-buffer 4\n"
    "  --xmin X --xmax X --ymin Y --ymax Y --max-cells N\n"
    "  --refine-box xmin,xmax,ymin,ymax,level  (repeatable)\n"
    "  --surface-markers edges.csv --freestream rho,u,v,p\n"
    "  --tend T --maxsteps N --cfl C --dt DT --gamma G --mco M\n"
    "  --flux hllc|ausm --bcx transmissive|reflect --bcy transmissive|reflect\n"
    "  --bc-xmin/--bc-xmax/--bc-ymin/--bc-ymax transmissive|reflect|farfield|total-inlet|pressure-outlet\n"
    "  --bc-reference-SIDE rho,u,v,p --bc-total-SIDE Pt,Tt,R,dx,dy --bc-pressure-SIDE p\n"
    "  --mesh-only --backend cpu|cuda --limiter first --time-integrator rk3\n"
    "  --binary [--snapshot-times .1,.5,1,2.6,3.9] --gpu-batch 32 --gpu-graph on|off\n"
    "  --model euler|laminar --mu MU | --re RE --reference-length L (default 1)\n"
    "  --viscosity constant|sutherland --sutherland-Tref T --sutherland-S S --gas-constant R\n"
    "  --viscous-cfl .1 --bc-SIDE no-slip --wall-velocity-SIDE u,v\n"
    "  --viscous-stabilization volume|srd (default volume; srd experimental)\n"
    "  --wall-probe-min-h H (default 0; select visible donor with actual dn>=H*h)\n"
    "  --agglomerate on|off (default off; merge small connected control volumes)\n"
    "  --gpu-kernels reference|fused (default reference)\n"
    "  Laminar: no heat conduction (kappa=0), viscous work retained.\n"
    "  Static first-order shared-dt RK3; no implicit/restart/subcycling.\n";
}
int boundary_side(const std::string& key,const std::string& prefix) {
  const char* names[]={"xmin","xmax","ymin","ymax"};
  for(int i=0;i<4;++i)if(key==prefix+names[i])return i;
  return -1;
}
bool number(const std::string& text,double* value) {
  char* end=nullptr;errno=0;*value=std::strtod(text.c_str(),&end);
  return end!=text.c_str()&&*end=='\0'&&errno==0&&std::isfinite(*value);
}
bool integer(const std::string& text,int* value) {
  char* end=nullptr;errno=0;const long result=std::strtol(text.c_str(),&end,10);
  if(end==text.c_str()||*end!='\0'||errno||result<INT_MIN||result>INT_MAX)return false;
  *value=static_cast<int>(result);return true;
}
bool numbers(std::string text,size_t count,std::vector<double>* result) {
  std::replace(text.begin(),text.end(),',',' ');std::istringstream in(text);std::string token;
  result->clear();while(in>>token){double v;if(!number(token,&v))return false;result->push_back(v);}
  return count==0?!result->empty():result->size()==count;
}
bool polygon(const std::string& path,std::vector<double>* px,std::vector<double>* py) {
  std::ifstream in(path);if(!in)return false;std::string line;
  while(std::getline(in,line)) {
    std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);std::string a,b,extra;
    if(!(row>>a)||a[0]=='#')continue;
    if(!(row>>b)||(row>>extra))return false;
    if(a=="x"&&b=="y")continue;
    double x,y;if(!number(a,&x)||!number(b,&y))return false;px->push_back(x);py->push_back(y);
  }
  if(in.bad())return false;
  if(px->size()>1&&px->front()==px->back()&&py->front()==py->back()){px->pop_back();py->pop_back();}
  return px->size()>=3;
}
bool markers(const std::string& path,size_t n,std::vector<int>* patches) {
  std::ifstream in(path);if(!in)return false;std::string line;patches->assign(n,-1);
  while(std::getline(in,line)) {
    std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);std::string a,b,extra;
    if(!(row>>a)||a[0]=='#')continue;
    if(!(row>>b)||(row>>extra))return false;
    if(a=="edge_id"&&b=="patch_id")continue;
    int edge,patch;if(!integer(a,&edge)||!integer(b,&patch)||edge<0||static_cast<size_t>(edge)>=n||patch<0||(*patches)[edge]>=0)return false;
    (*patches)[edge]=patch;
  }
  return !in.bad()&&std::find(patches->begin(),patches->end(),-1)==patches->end();
}
void json_string(std::ostream& out,const char* value) {
  out<<'"';for(const unsigned char* p=reinterpret_cast<const unsigned char*>(value);*p;++p) {
    if(*p=='"'||*p=='\\')out<<'\\'<<*p;
    else if(*p<32){char escaped[7];std::snprintf(escaped,sizeof(escaped),"\\u%04x",static_cast<unsigned>(*p));out<<escaped;}
    else out<<*p;
  }out<<'"';
}
}
int main(int argc,char** argv) {
  StaticAmrOptions2D options;Params2D params=params2d_default();
  params.limiter=LIM_FIRST_ORDER;params.cfl=0.25;
  std::array<double,4> freestream={1,1.8,0,1.0/1.4};
  std::string body,marker_path,prefix,backend="cpu";double tend=0.01;int maxsteps=1000000;bool mesh_only=false,binary=false;
  AmrTransport2D transport;double reynolds=0,reference_length=1;bool mu_set=false,re_set=false,tref_set=false,s_set=false;
  AmrGpuOptions gpu_options;std::vector<double> snapshot_times;
  AmrBoundary2D boundaries[4];bool reference_set[4]={};
  for(int a=1;a<argc;++a) {
    const std::string key=argv[a];
    if(key=="--help"){usage();return 0;}
    if(key=="--mesh-only"){mesh_only=true;continue;}
    if(key=="--binary"){binary=true;continue;}
    if(a+1==argc){usage();return 2;}
    const std::string value=argv[++a];bool ok=true;
    if(key=="--out")prefix=value;
    else if(key=="--body")body=value.rfind("polygon:",0)==0?value.substr(8):value;
    else if(key=="--surface-markers")marker_path=value;
    else if(key=="--model"){ok=value=="euler"||value=="laminar";transport.model=value=="laminar"?AMR_MODEL_LAMINAR:AMR_MODEL_EULER;}
    else if(key=="--viscosity"){ok=value=="constant"||value=="sutherland";transport.law=value=="sutherland"?AMR_MU_SUTHERLAND:AMR_MU_CONSTANT;}
    else if(key=="--viscous-stabilization"){ok=value=="volume"||value=="srd";transport.stabilization=value=="srd"?AMR_VISCOUS_SRD:AMR_VISCOUS_VOLUME;}
    else if(key=="--gpu-kernels"){ok=value=="reference"||value=="fused";gpu_options.fused=value=="fused";}
    else if(key=="--mu"){ok=number(value,&transport.mu_ref)&&transport.mu_ref>0;mu_set=true;}
    else if(key=="--wall-probe-min-h"){ok=number(value,&transport.wall_probe_min_h)&&transport.wall_probe_min_h>=0&&transport.wall_probe_min_h<=2;}
    else if(key=="--re"){ok=number(value,&reynolds)&&reynolds>0;re_set=true;}
    else if(key=="--reference-length")ok=number(value,&reference_length)&&reference_length>0;
    else if(key=="--gas-constant")ok=number(value,&transport.gas_constant)&&transport.gas_constant>0;
    else if(key=="--sutherland-Tref"){ok=number(value,&transport.temperature_ref)&&transport.temperature_ref>0;tref_set=true;}
    else if(key=="--sutherland-S"){ok=number(value,&transport.sutherland)&&transport.sutherland>=0;s_set=true;}
    else if(key=="--viscous-cfl")ok=number(value,&transport.viscous_cfl)&&transport.viscous_cfl>0&&transport.viscous_cfl<=.2;
    else if(key=="--base-nx")ok=integer(value,&options.nx);
    else if(key=="--base-ny")ok=integer(value,&options.ny);
    else if(key=="--levels")ok=integer(value,&options.levels);
    else if(key=="--agglomerate"){ok=value=="on"||value=="off";options.agglomerate_small=value=="on";}
    else if(key=="--wall-buffer")ok=integer(value,&options.buffer);
    else if(key=="--max-cells"){int n=0;ok=integer(value,&n)&&n>0;if(ok)options.max_cells=static_cast<size_t>(n);}
    else if(key=="--maxsteps")ok=integer(value,&maxsteps);
    else if(key=="--xmin")ok=number(value,&options.xmin);
    else if(key=="--xmax")ok=number(value,&options.xmax);
    else if(key=="--ymin")ok=number(value,&options.ymin);
    else if(key=="--ymax")ok=number(value,&options.ymax);
    else if(key=="--tend")ok=number(value,&tend)&&tend>=0;
    else if(key=="--cfl")ok=number(value,&params.cfl)&&params.cfl>0&&params.cfl<=0.5;
    else if(key=="--dt")ok=number(value,&params.dt_fixed)&&params.dt_fixed>=0;
    else if(key=="--gamma")ok=number(value,&params.gamma)&&params.gamma>1;
    else if(key=="--mco")ok=number(value,&params.ausm_mco)&&params.ausm_mco>0;
    else if(key=="--backend"){ok=value=="cpu"||value=="cuda";backend=value;}
    else if(key=="--gpu-batch")ok=integer(value,&gpu_options.batch_trials)&&gpu_options.batch_trials>0&&gpu_options.batch_trials<=1024;
    else if(key=="--gpu-graph"){ok=value=="on"||value=="off";gpu_options.graph=value=="on";}
    else if(key=="--snapshot-times")ok=numbers(value,0,&snapshot_times);
    else if(key=="--limiter")ok=value=="first";
    else if(key=="--time-integrator")ok=value=="rk3";
    else if(key=="--flux"){ok=value=="hllc"||value=="ausm";params.flux=value=="hllc"?FLUX_HLLC:FLUX_AUSMPUP;}
    else if(key=="--bcx"||key=="--bcy") {
      ok=value=="reflect"||value=="transmissive";const int i=key=="--bcx"?0:2;
      params.bc[i]=params.bc[i+1]=value=="reflect"?BC_REFLECT:BC_TRANSMISSIVE;
    }else if(boundary_side(key,"--bc-")>=0) {
      const int side=boundary_side(key,"--bc-");
      if(value=="transmissive")boundaries[side].kind=AMR_BC_TRANSMISSIVE;
      else if(value=="reflect")boundaries[side].kind=AMR_BC_REFLECT;
      else if(value=="farfield")boundaries[side].kind=AMR_BC_FARFIELD;
      else if(value=="total-inlet")boundaries[side].kind=AMR_BC_TOTAL_INLET;
      else if(value=="pressure-outlet")boundaries[side].kind=AMR_BC_PRESSURE_OUTLET;
      else if(value=="no-slip")boundaries[side].kind=AMR_BC_NOSLIP;
      else ok=false;
    }else if(boundary_side(key,"--wall-velocity-")>=0){
      const int side=boundary_side(key,"--wall-velocity-");std::vector<double> values;ok=numbers(value,2,&values);
      if(ok)std::copy(values.begin(),values.end(),boundaries[side].wall_velocity);
    }else if(boundary_side(key,"--bc-reference-")>=0) {
      const int side=boundary_side(key,"--bc-reference-");std::vector<double> values;ok=numbers(value,4,&values);
      if(ok){std::copy(values.begin(),values.end(),boundaries[side].reference);reference_set[side]=true;}
    }else if(boundary_side(key,"--bc-total-")>=0) {
      const int side=boundary_side(key,"--bc-total-");std::vector<double> values;ok=numbers(value,5,&values);
      if(ok){auto& b=boundaries[side];b.total_pressure=values[0];b.total_temperature=values[1];b.gas_constant=values[2];
        const double norm=std::hypot(values[3],values[4]);ok=norm>0&&std::isfinite(norm);
        if(ok){b.direction[0]=values[3]/norm;b.direction[1]=values[4]/norm;}}
    }else if(boundary_side(key,"--bc-pressure-")>=0) {
      ok=number(value,&boundaries[boundary_side(key,"--bc-pressure-")].outlet_pressure);
    }else if(key=="--freestream") {
      std::vector<double> values;ok=numbers(value,4,&values);
      if(ok)std::copy(values.begin(),values.end(),freestream.begin());
    }else if(key=="--refine-box") {
      std::vector<double> values;ok=numbers(value,5,&values);
      if(ok){ok=values[4]>=0&&values[4]<=16&&values[4]==std::floor(values[4]);if(ok)options.boxes.push_back({values[0],values[1],values[2],values[3],static_cast<int>(values[4])});}
    }else ok=false;
    if(!ok){std::cerr<<"unsupported or invalid option: "<<key<<" "<<value<<"\n";return 2;}
  }
  if(prefix.empty()||maxsteps<1||(!marker_path.empty()&&body.empty())){usage();return 2;}
  if(!binary&&!snapshot_times.empty()){std::cerr<<"snapshot-times requires --binary\n";return 2;}
  for(double time:snapshot_times)if(time<0||time>tend){std::cerr<<"snapshot time outside [0,tend]\n";return 2;}
  if(mu_set&&re_set){std::cerr<<"specify mu or Re, not both\n";return 2;}
  if(transport.model==AMR_MODEL_EULER&&(mu_set||re_set||tref_set||s_set||transport.stabilization!=AMR_VISCOUS_VOLUME||transport.wall_probe_min_h!=0)){std::cerr<<"transport parameters require --model laminar\n";return 2;}
  if(re_set)transport.mu_ref=freestream[0]*std::hypot(freestream[1],freestream[2])*reference_length/reynolds;
  if(transport.model==AMR_MODEL_LAMINAR&&(!mu_set&&!re_set)){std::cerr<<"laminar requires explicit --mu or --re\n";return 2;}
  if(transport.model==AMR_MODEL_LAMINAR&&transport.law==AMR_MU_SUTHERLAND&&(!tref_set||!s_set)){std::cerr<<"Sutherland requires explicit Tref and S in solver units\n";return 2;}
  if(!amr_transport_valid(transport)){std::cerr<<"invalid transport parameters\n";return 2;}
  std::vector<double> px,py;std::vector<int> patches;
  if(!body.empty()&&!polygon(body,&px,&py)){std::cerr<<"invalid polygon: "<<body<<"\n";return 2;}
  if(!marker_path.empty()&&!markers(marker_path,px.size(),&patches)){std::cerr<<"invalid markers: "<<marker_path<<"\n";return 2;}
  for(int side=0;side<4;++side){if(!reference_set[side])std::copy(freestream.begin(),freestream.end(),boundaries[side].reference);
    if(!amr_bc_valid(boundaries[side],params.bc[side],side)){std::cerr<<"invalid boundary configuration at side "<<side<<"\n";return 2;}}
  StaticAmrMesh2D mesh;StaticAmrState2D state;std::string error;
  const auto start=std::chrono::steady_clock::now();
  if(static_amr2d_build(options,px,py,patches,&mesh,&error)){std::cerr<<"mesh failed: "<<error<<"\n";return 3;}
  const auto built=std::chrono::steady_clock::now();
  std::copy(std::begin(boundaries),std::end(boundaries),mesh.boundaries);
  mesh.transport=transport;
  if(static_amr2d_prepare_viscous(&mesh,params,&error)){std::cerr<<"viscous setup failed: "<<error<<"\n";return 2;}
  if(static_amr2d_initialize(mesh,params,freestream,&state)){std::cerr<<"invalid freestream\n";return 2;}
  std::unique_ptr<StaticAmrGpu,void(*)(StaticAmrGpu*)> gpu(nullptr,static_amr_gpu_destroy);
  if(!mesh_only&&backend=="cuda") {
    StaticAmrGpu* raw=nullptr;
    if(static_amr_gpu_create(mesh,params,state,gpu_options,&raw,&error)){std::cerr<<"GPU setup failed: "<<error<<"\n";return 4;}
    gpu.reset(raw);
  }
  double io_seconds=0;
  const auto advance=[&](double target) {
    int status=gpu?static_amr_gpu_advance(gpu.get(),target,maxsteps,&state,&error):static_amr2d_run(mesh,params,target,maxsteps,&state,&error);
    if(gpu&&static_amr_gpu_download(gpu.get(),&state,&error))status=1;
    return status;
  };
  if(binary) {
    auto io_start=std::chrono::steady_clock::now();
    if(static_amr_binary_mesh(mesh,params,prefix,&error)){std::cerr<<error<<"\n";return 5;}
    io_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-io_start).count();
    if(mesh_only)snapshot_times={0};else {snapshot_times.push_back(0);snapshot_times.push_back(tend);}
    std::sort(snapshot_times.begin(),snapshot_times.end());snapshot_times.erase(std::unique(snapshot_times.begin(),snapshot_times.end()),snapshot_times.end());
    std::vector<std::string> names;std::vector<double> written_times;
    for(size_t i=0;i<snapshot_times.size();++i) {
      if(snapshot_times[i]>state.time&&advance(snapshot_times[i])) {
        std::cerr<<"solve failed: "<<error<<" time="<<state.time<<" steps="<<state.steps<<"\n";
        std::string ignored;static_amr_binary_snapshot(mesh,params,state,prefix+".failed_last",backend.c_str(),&ignored);
        if(gpu)static_amr_gpu_write_stats(static_amr_gpu_stats(gpu.get()),prefix+".gpu.json",io_seconds,&ignored);
        return 4;
      }
      std::ostringstream suffix;suffix<<".s"<<std::setfill('0')<<std::setw(4)<<i;const std::string snapshot=prefix+suffix.str();
      io_start=std::chrono::steady_clock::now();
      if(static_amr_binary_snapshot(mesh,params,state,snapshot,backend.c_str(),&error)){std::cerr<<error<<"\n";return 5;}
      names.push_back(std::filesystem::path(snapshot).filename().string());written_times.push_back(state.time);
      std::ofstream index(prefix+".index.json");index<<std::setprecision(17)<<"{\"backend\":\""<<backend<<"\",\"target\":"<<tend<<",\"complete\":"<<((mesh_only||state.time==tend)?"true":"false")<<",\"snapshots\":[";
      for(size_t q=0;q<names.size();++q){if(q)index<<',';index<<"{\"time\":"<<written_times[q]<<",\"prefix\":";json_string(index,names[q].c_str());index<<'}';}index<<"]}\n";
      index.close();if(!index){std::cerr<<"index write failed\n";return 5;}
      io_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-io_start).count();
      std::cerr<<"snapshot: "<<snapshot<<" t="<<state.time<<" steps="<<state.steps<<" rejected="<<state.rejected<<"\n";
    }
  }else {
    if(!mesh_only&&advance(tend)){std::cerr<<"solve failed: "<<error<<" time="<<state.time<<" steps="<<state.steps<<"\n";return 4;}
    const auto io_start=std::chrono::steady_clock::now();
    if(static_amr2d_write(mesh,params,state,prefix,&error,backend.c_str())){std::cerr<<"output failed: "<<error<<"\n";return 5;}
    io_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-io_start).count();
  }
  const auto solved=std::chrono::steady_clock::now();
  if(gpu&&static_amr_gpu_write_stats(static_amr_gpu_stats(gpu.get()),prefix+".gpu.json",io_seconds,&error)){std::cerr<<error<<"\n";return 5;}
  std::ofstream command(prefix+".command.json");command<<"{\"argv\":[";
  for(int a=0;a<argc;++a){if(a)command<<',';json_string(command,argv[a]);}command<<"]}\n";command.close();
  if(!command){std::cerr<<"command manifest write failed\n";return 5;}
  if(!px.empty()) {
    std::ofstream geometry(prefix+".geometry.csv");geometry.precision(17);geometry<<"x,y\n";
    for(size_t a=0;a<px.size();++a)geometry<<px[a]<<','<<py[a]<<'\n';
    geometry.close();if(!geometry){std::cerr<<"geometry snapshot write failed\n";return 5;}
    std::ofstream patch_file(prefix+".patches.csv");patch_file<<"edge_id,patch_id\n";
    for(size_t a=0;a<px.size();++a)patch_file<<a<<','<<(patches.empty()?0:patches[a])<<'\n';
    patch_file.close();if(!patch_file){std::cerr<<"patch snapshot write failed\n";return 5;}
  }
  std::cerr<<"static-amr: "<<backend<<" "<<(transport.model==AMR_MODEL_EULER?"euler/slip":"laminar/no-slip kappa=0")<<" first-order RK3, shared dt; fluid_leaves="<<mesh.cells.size()
    <<" background_leaves="<<mesh.background_leaves<<" faces="<<mesh.faces.size()
    <<" coarse_fine_subfaces="<<mesh.coarse_fine_faces<<" walls="<<mesh.walls.size()<<" small="<<mesh.small_cells
    <<" time="<<state.time<<" steps="<<state.steps<<" rejected="<<state.rejected
    <<" mesh_seconds="<<std::chrono::duration<double>(built-start).count()
    <<" solve_and_output_seconds="<<std::chrono::duration<double>(solved-built).count()<<" io_seconds="<<io_seconds<<"\n";
  return 0;
}
