#include "static_amr2d_binary.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace cfd {
namespace {
int failure(std::string* error,const std::string& text){if(error)*error=text;return 1;}
}
int static_amr_write_npy(const std::string& path,const double* values,size_t rows,size_t columns,std::string* error) {
  const uint16_t endian=1;if(*reinterpret_cast<const unsigned char*>(&endian)!=1)return failure(error,"NPY writer requires little-endian host");
  std::ostringstream shape;shape<<"{'descr': '<f8', 'fortran_order': False, 'shape': ("<<rows<<", "<<columns<<"), }";
  std::string header=shape.str();while((10+header.size()+1)%64)header+=' ';header+='\n';
  if(header.size()>65535)return failure(error,"NPY v1 header too large");
  FILE* f=std::fopen(path.c_str(),"wb");if(!f)return failure(error,"cannot open "+path);
  const unsigned char magic[]={0x93,'N','U','M','P','Y',1,0,static_cast<unsigned char>(header.size()&255),static_cast<unsigned char>(header.size()>>8)};
  bool ok=std::fwrite(magic,1,sizeof(magic),f)==sizeof(magic)&&std::fwrite(header.data(),1,header.size(),f)==header.size();
  if(rows&&columns)ok=ok&&std::fwrite(values,sizeof(double),rows*columns,f)==rows*columns;
  const bool closed=std::fclose(f)==0;return ok&&closed?0:failure(error,"NPY write failed: "+path);
}
int static_amr_binary_mesh(const StaticAmrMesh2D& m,const Params2D& p,const std::string& prefix,std::string* error) {
  std::error_code ec;const auto parent=std::filesystem::path(prefix).parent_path();if(!parent.empty())std::filesystem::create_directories(parent,ec);
  if(ec)return failure(error,"cannot create output folder: "+ec.message());
  if(static_amr2d_write_boundaries(m,p,prefix,error))return 1;
  std::vector<double> cells,faces,walls;cells.reserve(13*m.cells.size());faces.reserve(9*m.faces.size());walls.reserve(11*m.walls.size());
  for(const auto& c:m.cells)for(double v:{static_cast<double>(c.ix),static_cast<double>(c.iy),static_cast<double>(c.span),static_cast<double>(c.level),c.x0,c.x1,c.y0,c.y1,c.volume,c.cx,c.cy,static_cast<double>(c.wall_begin),static_cast<double>(c.wall_count)})cells.push_back(v);
  for(const auto& f:m.faces)for(double v:{static_cast<double>(f.left),static_cast<double>(f.right),static_cast<double>(f.axis),static_cast<double>(f.boundary),f.coordinate,f.lo,f.hi,f.length,f.centroid})faces.push_back(v);
  for(const auto& w:m.walls)for(double v:{static_cast<double>(w.owner_cell),static_cast<double>(w.source_edge_id),static_cast<double>(w.body_id),static_cast<double>(w.patch_id),w.x0,w.y0,w.x1,w.y1,w.nx,w.ny,w.length})walls.push_back(v);
  if(static_amr_write_npy(prefix+".cells.npy",cells.data(),m.cells.size(),13,error)||
     static_amr_write_npy(prefix+".faces.npy",faces.data(),m.faces.size(),9,error)||
     static_amr_write_npy(prefix+".walls.npy",walls.data(),m.walls.size(),11,error))return 1;
  if(m.split_leaves){std::vector<double> mapping;for(const auto& pair:m.leaf_components){mapping.push_back(static_cast<double>(pair[0]));mapping.push_back(static_cast<double>(pair[1]));}
    if(static_amr_write_npy(prefix+".leaf_components.npy",mapping.data(),m.leaf_components.size(),2,error))return 1;}
  if(!m.agglomeration.empty()){std::vector<double> members;for(const auto& row:m.agglomeration)members.insert(members.end(),row.begin(),row.end());
    if(static_amr_write_npy(prefix+".agglomeration.npy",members.data(),m.agglomeration.size(),9,error))return 1;}
  std::ofstream out(prefix+".mesh.json");const auto& o=m.options;
  out<<std::setprecision(17)<<"{\n\"format\":\"NPY v1 <f8, C-order\",\n\"model\":\""<<(m.transport.model==AMR_MODEL_EULER?"2D Euler, static slip wall":(m.transport.sa_model?"2D SA-neg mechanical viscous, kappa=0, stationary no-slip wall":"2D laminar mechanical viscous, kappa=0, stationary no-slip wall"))<<(o.agglomerate_small?", agglomerated FV, no SRD, SSP-RK3\",\n":", zero-degree SRD, SSP-RK3\",\n")
    <<"\"cells_columns\":[\"ix\",\"iy\",\"span\",\"level\",\"x0\",\"x1\",\"y0\",\"y1\",\"fluid_area\",\"cx\",\"cy\",\"wall_begin\",\"wall_count\"],\n"
    <<"\"faces_columns\":[\"left\",\"right\",\"axis\",\"boundary\",\"coordinate\",\"lo\",\"hi\",\"open_length\",\"open_centroid\"],\n"
    <<"\"walls_columns\":[\"owner\",\"source_edge\",\"body\",\"patch\",\"x0\",\"y0\",\"x1\",\"y1\",\"nx\",\"ny\",\"length\"],\n"
    <<"\"state_shape\":["<<(m.transport.sa_model?5:4)<<","<<m.cells.size()<<"],\n\"state_rows\":[\"rho\",\"rho_u\",\"rho_v\",\"rho_E\""<<(m.transport.sa_model?",\"rho_nutilde\"":"")<<"],\n"
    <<"\"bounds\":["<<o.xmin<<','<<o.xmax<<','<<o.ymin<<','<<o.ymax<<"],\n\"base_shape\":["<<o.nx<<','<<o.ny<<"],\n\"levels\":"<<o.levels
    <<",\n\"buffer\":"<<o.buffer<<",\n\"gamma\":"<<p.gamma<<",\n\"cfl\":"<<p.cfl<<",\n\"fixed_dt\":"<<p.dt_fixed<<",\n\"flux\":\""<<(p.flux==FLUX_HLLC?"hllc":"ausm")<<"\",\n\"bc\":["<<(m.boundaries[0].kind<0?p.bc[0]:m.boundaries[0].kind)<<','<<(m.boundaries[1].kind<0?p.bc[1]:m.boundaries[1].kind)<<','<<(m.boundaries[2].kind<0?p.bc[2]:m.boundaries[2].kind)<<','<<(m.boundaries[3].kind<0?p.bc[3]:m.boundaries[3].kind)<<"],\n\"coarse_fine_subfaces\":"<<m.coarse_fine_faces<<",\n\"small_cells\":"<<m.small_cells<<",\n\"split_background_leaves\":"<<m.split_leaves<<",\n\"redistribution_mode\":\""<<(o.agglomerate_small?"agglomerated-no-SRD":"state-SRD")<<"\"\n}\n";
  out.close();return out?0:failure(error,"mesh JSON write failed");
}
int static_amr_binary_snapshot(const StaticAmrMesh2D& m,const Params2D& p,const StaticAmrState2D& s,
                               const std::string& prefix,const char* backend,std::string* error) {
  const size_t n=m.cells.size();const size_t nv=m.transport.sa_model?5:4;if(s.u.size()!=nv*n)return failure(error,"snapshot shape mismatch");
  std::vector<double> wall_pressure;if(static_amr2d_wall_pressure(m,p,s.u,&wall_pressure))return failure(error,"snapshot wall pressure failed");
  double rhomin=INFINITY,rhomax=0,pmin=INFINITY,pmax=0,fx=0,fy=0,mz=0;long double totals[4]={};
  for(size_t k=0;k<n;++k){const double rho=s.u[k],vx=s.u[n+k]/rho,vy=s.u[2*n+k]/rho;
    const double pressure=(p.gamma-1)*(s.u[3*n+k]-.5*rho*(vx*vx+vy*vy));
    if(!std::isfinite(rho)||!(rho>0)||!std::isfinite(pressure)||!(pressure>0))return failure(error,"nonphysical snapshot; no file certified");
    rhomin=std::min(rhomin,rho);rhomax=std::max(rhomax,rho);pmin=std::min(pmin,pressure);pmax=std::max(pmax,pressure);
    for(int v=0;v<4;++v)totals[v]+=static_cast<long double>(m.cells[k].volume)*s.u[v*n+k];
    const auto& c=m.cells[k];for(size_t q=c.wall_begin;q<c.wall_begin+c.wall_count;++q){const auto& w=m.walls[q];const double x=wall_pressure[q]*w.nx*w.length,y=wall_pressure[q]*w.ny*w.length;fx+=x;fy+=y;mz+=.5*(w.x0+w.x1)*y-.5*(w.y0+w.y1)*x;}
  }
  if(static_amr_write_npy(prefix+".state.npy",s.u.data(),nv,n,error))return 1;
  std::ofstream out(prefix+".json");out<<std::setprecision(17)<<"{\"backend\":\""<<backend<<"\",\"time\":"<<s.time<<",\"steps\":"<<s.steps<<",\"rejected\":"<<s.rejected
    <<",\"rho_min\":"<<rhomin<<",\"rho_max\":"<<rhomax<<",\"p_min\":"<<pmin<<",\"p_max\":"<<pmax<<",\"fx_pressure\":"<<fx<<",\"fy_pressure\":"<<fy<<",\"mz_pressure_about_origin\":"<<mz
    <<",\"integrated_conserved\":["<<totals[0]<<','<<totals[1]<<','<<totals[2]<<','<<totals[3]<<"],\"state_bytes\":"<<nv*n*sizeof(double)<<",\"dimensional_units\":false}\n";
  out.close();if(!out)return failure(error,"snapshot JSON write failed");
  return static_amr2d_write_viscous_snapshot(m,p,s,prefix,error);
}
int static_amr_gpu_write_stats(const AmrGpuStats& s,const std::string& path,double io,std::string* error) {
  std::ofstream out(path);out<<std::setprecision(17)<<"{\n\"allocated_device_bytes\":"<<s.allocated_bytes<<",\n\"pinned_host_bytes\":"<<s.pinned_bytes
    <<",\n\"geometry_h2d_bytes\":"<<s.geometry_h2d_bytes<<",\n\"state_h2d_bytes\":"<<s.state_h2d_bytes<<",\n\"control_h2d_bytes\":"<<s.control_h2d_bytes
    <<",\n\"control_d2h_bytes\":"<<s.control_d2h_bytes<<",\n\"state_d2h_bytes\":"<<s.state_d2h_bytes<<",\n\"status_polls\":"<<s.status_polls<<",\n\"full_downloads\":"<<s.full_downloads
    <<",\n\"setup_seconds\":"<<s.setup_seconds<<",\n\"advance_wall_seconds\":"<<s.advance_seconds<<",\n\"download_wall_seconds\":"<<s.download_seconds<<",\n\"binary_io_seconds\":"<<io
    <<",\n\"advance_cuda_event_seconds\":"<<s.advance_event_seconds<<",\n\"event_batches\":"<<s.event_batches<<",\n\"graph_nodes_per_trial\":"<<s.graph_nodes_per_trial
    <<",\n\"event_scope\":\"sum of timed trial-batch stream intervals; excludes control/state transfers; may include launch starvation idle; not kernel busy sum\""
    <<",\n\"transfer_counter_scope\":\"explicit cudaMemcpy only; kernel arguments excluded\",\n\"overlap_claim\":false\n}\n";
  out.close();return out?0:failure(error,"GPU statistics JSON write failed");
}
} // namespace cfd
