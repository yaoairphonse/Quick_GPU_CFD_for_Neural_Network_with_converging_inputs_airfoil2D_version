#include "static_amr2d.h"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>

namespace cfd {
namespace {
bool close_file(FILE* f) {const bool bad=std::ferror(f)!=0;return std::fclose(f)==0&&!bad;}
void array_start(FILE* f,const char* name,int components=1,const char* type="Float64") {
  std::fprintf(f,"<DataArray type=\"%s\" Name=\"%s\" NumberOfComponents=\"%d\" format=\"ascii\">\n",type,name,components);
}
void array_end(FILE* f){std::fprintf(f,"</DataArray>\n");}
}
int static_amr2d_write(const StaticAmrMesh2D& m,const Params2D& p,const StaticAmrState2D& s,
                       const std::string& prefix,std::string* error,const char* backend) {
  const auto failed=[&](const std::string& message){if(error)*error=message;return 1;};
  const size_t n=m.cells.size();
  if(prefix.empty()||s.u.size()!=(m.transport.sa_model?5:4)*n)return failed("invalid output prefix or state");
  std::error_code ec;const auto parent=std::filesystem::path(prefix).parent_path();
  if(!parent.empty())std::filesystem::create_directories(parent,ec);
  if(ec)return failed("cannot create output directory: "+ec.message());
  std::vector<double> rho(n),vx(n),vy(n),pressure(n);
  double totals[4]={};
  for(size_t k=0;k<n;++k) {
    rho[k]=s.u[k];vx[k]=s.u[n+k]/rho[k];vy[k]=s.u[2*n+k]/rho[k];
    pressure[k]=(p.gamma-1)*(s.u[3*n+k]-0.5*rho[k]*(vx[k]*vx[k]+vy[k]*vy[k]));
    if(!std::isfinite(rho[k])||!(rho[k]>0)||!std::isfinite(pressure[k])||!(pressure[k]>0))return failed("nonphysical output state");
    for(int v=0;v<4;++v)totals[v]+=m.cells[k].volume*s.u[v*n+k];
  }
  std::vector<double> wall_pressure;if(static_amr2d_wall_pressure(m,p,s.u,&wall_pressure))return failed("wall pressure reconstruction failed");
  FILE* f=std::fopen((prefix+".cells.csv").c_str(),"w");
  if(!f)return failed("cannot open cells CSV");
  std::fprintf(f,"cell_id,level,ix,iy,span,x0,x1,y0,y1,cx,cy,fluid_area,alpha,rho,u,v,p,wall_begin,wall_count\n");
  for(size_t k=0;k<n;++k) {
    const auto& c=m.cells[k];
    std::fprintf(f,"%zu,%d,%d,%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%zu,%zu\n",
      k,c.level,c.ix,c.iy,c.span,c.x0,c.x1,c.y0,c.y1,c.cx,c.cy,c.volume,c.volume/((c.x1-c.x0)*(c.y1-c.y0)),rho[k],vx[k],vy[k],pressure[k],c.wall_begin,c.wall_count);
  }
  if(!close_file(f))return failed("cells CSV write failed");
  f=std::fopen((prefix+".faces.csv").c_str(),"w");if(!f)return failed("cannot open faces CSV");
  std::fprintf(f,"face_id,left_cell,right_cell,axis,boundary,coordinate,lo,hi,open_length,open_centroid\n");
  for(size_t q=0;q<m.faces.size();++q) {
    const auto& a=m.faces[q];
    std::fprintf(f,"%zu,%d,%d,%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e\n",q,a.left,a.right,a.axis,a.boundary,a.coordinate,a.lo,a.hi,a.length,a.centroid);
  }
  if(!close_file(f))return failed("faces CSV write failed");
  f=std::fopen((prefix+".mesh.vtu").c_str(),"w");if(!f)return failed("cannot open mesh VTU");
  std::fprintf(f,"<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\"><UnstructuredGrid>\n");
  std::fprintf(f,"<!-- Leaf background rectangles, not fluid polygons. alpha gives actual fluid fraction; overlay surface.vtp. -->\n");
  std::fprintf(f,"<FieldData><DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">%.17e</DataArray></FieldData>\n",s.time);
  std::fprintf(f,"<Piece NumberOfPoints=\"%zu\" NumberOfCells=\"%zu\"><Points>",4*n,n);array_start(f,"Points",3);
  for(const auto& c:m.cells)std::fprintf(f,"%.17e %.17e 0 %.17e %.17e 0 %.17e %.17e 0 %.17e %.17e 0\n",c.x0,c.y0,c.x1,c.y0,c.x1,c.y1,c.x0,c.y1);
  array_end(f);std::fprintf(f,"</Points><Cells>");array_start(f,"connectivity",1,"Int64");
  for(size_t k=0;k<n;++k)std::fprintf(f,"%zu %zu %zu %zu\n",4*k,4*k+1,4*k+2,4*k+3);
  array_end(f);array_start(f,"offsets",1,"Int64");for(size_t k=0;k<n;++k)std::fprintf(f,"%zu\n",4*(k+1));array_end(f);
  array_start(f,"types",1,"UInt8");for(size_t k=0;k<n;++k)std::fprintf(f,"9\n");array_end(f);std::fprintf(f,"</Cells><CellData>\n");
  array_start(f,"cell_id",1,"Int64");for(size_t k=0;k<n;++k)std::fprintf(f,"%zu\n",k);array_end(f);
  array_start(f,"level",1,"Int32");for(const auto& c:m.cells)std::fprintf(f,"%d\n",c.level);array_end(f);
  array_start(f,"alpha");for(const auto& c:m.cells)std::fprintf(f,"%.17e\n",c.volume/((c.x1-c.x0)*(c.y1-c.y0)));array_end(f);
  array_start(f,"fluid_area");for(const auto& c:m.cells)std::fprintf(f,"%.17e\n",c.volume);array_end(f);
  const char* names[]={"rho","u","v","p"};const std::vector<double>* values[]={&rho,&vx,&vy,&pressure};
  for(int v=0;v<4;++v){array_start(f,names[v]);for(double value:*values[v])std::fprintf(f,"%.17e\n",value);array_end(f);}
  std::fprintf(f,"</CellData></Piece></UnstructuredGrid></VTKFile>\n");
  if(!close_file(f))return failed("mesh VTU write failed");
  f=std::fopen((prefix+".surface.vtp").c_str(),"w");if(!f)return failed("cannot open surface VTP");
  const size_t nw=m.walls.size();
  std::fprintf(f,"<?xml version=\"1.0\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\"><PolyData>\n");
  std::fprintf(f,"<FieldData><DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">%.17e</DataArray></FieldData>\n",s.time);
  std::fprintf(f,"<Piece NumberOfPoints=\"%zu\" NumberOfVerts=\"0\" NumberOfLines=\"%zu\" NumberOfStrips=\"0\" NumberOfPolys=\"0\"><Points>",2*nw,nw);array_start(f,"Points",3);
  for(const auto& a:m.walls)std::fprintf(f,"%.17e %.17e 0 %.17e %.17e 0\n",a.x0,a.y0,a.x1,a.y1);
  array_end(f);std::fprintf(f,"</Points><Lines>");array_start(f,"connectivity",1,"Int64");
  for(size_t q=0;q<nw;++q)std::fprintf(f,"%zu %zu\n",2*q,2*q+1);
  array_end(f);
  array_start(f,"offsets",1,"Int64");for(size_t q=0;q<nw;++q)std::fprintf(f,"%zu\n",2*(q+1));array_end(f);std::fprintf(f,"</Lines><CellData>\n");
  const char* ids[]={"fragment_id","owner_cell","body_id","patch_id","source_edge_id"};
  for(int v=0;v<5;++v){array_start(f,ids[v],1,"Int64");for(size_t q=0;q<nw;++q){const auto& a=m.walls[q];const size_t value=v==0?q:v==1?a.owner_cell:static_cast<size_t>(v==2?a.body_id:v==3?a.patch_id:a.source_edge_id);std::fprintf(f,"%zu\n",value);}array_end(f);}
  array_start(f,"normal_fluid_to_solid",3);for(const auto& a:m.walls)std::fprintf(f,"%.17e %.17e 0\n",a.nx,a.ny);array_end(f);
  array_start(f,"length");for(const auto& a:m.walls)std::fprintf(f,"%.17e\n",a.length);array_end(f);
  array_start(f,"p_wall");for(double value:wall_pressure)std::fprintf(f,"%.17e\n",value);array_end(f);
  std::fprintf(f,"</CellData></Piece></PolyData></VTKFile>\n");if(!close_file(f))return failed("surface VTP write failed");
  std::map<int,SurfaceIntegral2D> loads;SurfaceIntegral2D total={};
  for(size_t q=0;q<m.walls.size();++q) {const auto& a=m.walls[q];
    const double fx=wall_pressure[q]*a.nx*a.length,fy=wall_pressure[q]*a.ny*a.length;
    const double mz=0.5*(a.x0+a.x1)*fy-0.5*(a.y0+a.y1)*fx;
    for(SurfaceIntegral2D* r:{&loads[a.patch_id],&total}){r->length+=a.length;r->fx+=fx;r->fy+=fy;r->mz+=mz;}
  }
  loads[-1]=total;f=std::fopen((prefix+".loads.csv").c_str(),"w");if(!f)return failed("cannot open loads CSV");
  std::fprintf(f,"# pressure-only per unit span; normal fluid-to-solid; moment reference=(0,0)\n");
  std::fprintf(f,"body_id,patch_id,length,fx_pressure,fy_pressure,mz_pressure\n");
  for(const auto& entry:loads)std::fprintf(f,"0,%d,%.17e,%.17e,%.17e,%.17e\n",entry.first,entry.second.length,entry.second.fx,entry.second.fy,entry.second.mz);
  if(!close_file(f))return failed("loads CSV write failed");
  f=std::fopen((prefix+".json").c_str(),"w");if(!f)return failed("cannot open manifest");
  const auto& o=m.options;const unsigned long long uniform=static_cast<unsigned long long>(o.nx)*(1ULL<<o.levels)*o.ny*(1ULL<<o.levels);
  std::fprintf(f,"{\n  \"method\": \"static 2:1 quadtree leaves, shared subface flux, SSP-RK3\",\n  \"backend\": \"%s\",\n  \"spatial_order\": %d,\n  \"srd_polynomial_degree\": 0,\n  \"time_subcycling\": false,\n  \"bounds\": [%.17e,%.17e,%.17e,%.17e],\n  \"base_shape\": [%d,%d],\n  \"max_level\": %d,\n  \"wall_buffer\": %d,\n  \"background_leaves\": %zu,\n  \"fluid_leaves\": %zu,\n  \"uniform_finest_background_cells\": %llu,\n  \"active_subfaces\": %zu,\n  \"coarse_fine_subfaces\": %zu,\n  \"wall_fragments\": %zu,\n  \"small_cells\": %zu,\n  \"gamma\": %.17e,\n  \"cfl\": %.17e,\n  \"dt_fixed\": %.17e,\n  \"flux\": \"%s\",\n  \"bc\": [%d,%d,%d,%d],\n  \"time\": %.17e,\n  \"steps\": %ld,\n  \"rejected_trials\": %ld,\n  \"integrated_conserved\": [%.17e,%.17e,%.17e,%.17e],\n  \"mesh_visualization\": \"background rectangles, not clipped fluid polygons\",\n  \"validation\": \"not run for this implementation\"\n}\n",
    backend,p.amr_spatial_order,o.xmin,o.xmax,o.ymin,o.ymax,o.nx,o.ny,o.levels,o.buffer,m.background_leaves,n,uniform,m.faces.size(),m.coarse_fine_faces,nw,m.small_cells,p.gamma,p.cfl,p.dt_fixed,p.flux==FLUX_HLLC?"hllc":"ausm",m.boundaries[0].kind<0?p.bc[0]:m.boundaries[0].kind,m.boundaries[1].kind<0?p.bc[1]:m.boundaries[1].kind,m.boundaries[2].kind<0?p.bc[2]:m.boundaries[2].kind,m.boundaries[3].kind<0?p.bc[3]:m.boundaries[3].kind,s.time,s.steps,s.rejected,totals[0],totals[1],totals[2],totals[3]);
  if(!close_file(f))return failed("manifest write failed");
  if(static_amr2d_write_boundaries(m,p,prefix,error))return 1;
  return static_amr2d_write_viscous_snapshot(m,p,s,prefix,error);
}
} // namespace cfd
