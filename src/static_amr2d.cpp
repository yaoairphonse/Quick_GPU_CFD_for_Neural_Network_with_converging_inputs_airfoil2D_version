// Static, face-balanced leaf mesh; one shared flux per physical subface.
#include "static_amr2d.h"
#include "static_amr2d_reconstruction.h"
#include "static_amr2d_sa.h"
#include "flux.h"
#include "low_mach.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>

namespace cfd {
namespace {
struct Side { int lo,hi,cell; bool positive; };
struct Connection { int left,right,axis,coordinate,lo,hi; };
bool fail(std::string* error,const std::string& message) { if(error)*error=message;return false; }

std::vector<Connection> connections(const std::vector<AmrCell2D>& cells) {
  std::map<std::pair<int,int>,std::vector<Side>> lines;
  for(size_t k=0;k<cells.size();++k) {
    const auto& c=cells[k];const int id=static_cast<int>(k);
    lines[{0,c.ix}].push_back({c.iy,c.iy+c.span,id,false});
    lines[{0,c.ix+c.span}].push_back({c.iy,c.iy+c.span,id,true});
    lines[{1,c.iy}].push_back({c.ix,c.ix+c.span,id,false});
    lines[{1,c.iy+c.span}].push_back({c.ix,c.ix+c.span,id,true});
  }
  std::vector<Connection> result;
  for(auto& entry:lines) {
    std::vector<Side> positive,negative;
    for(const auto& s:entry.second) (s.positive?positive:negative).push_back(s);
    const auto order=[](const Side& a,const Side& b){return a.lo<b.lo;};
    std::sort(positive.begin(),positive.end(),order);std::sort(negative.begin(),negative.end(),order);
    const int axis=entry.first.first,coordinate=entry.first.second;
    if(positive.empty()||negative.empty()) {
      for(const auto& s:entry.second)
        result.push_back({s.positive?s.cell:-1,s.positive?-1:s.cell,axis,coordinate,s.lo,s.hi});
      continue;
    }
    size_t i=0,j=0;
    while(i<positive.size()&&j<negative.size()) {
      const auto& a=positive[i];const auto& b=negative[j];
      const int lo=std::max(a.lo,b.lo),hi=std::min(a.hi,b.hi);
      if(hi>lo) result.push_back({a.cell,b.cell,axis,coordinate,lo,hi});
      const int end=std::min(a.hi,b.hi);
      if(a.hi==end)++i;
      if(b.hi==end)++j;
    }
  }
  return result;
}

void geometry_bounds(AmrCell2D* c,const StaticAmrOptions2D& o) {
  const double dx=(o.xmax-o.xmin)/(o.nx*(1<<o.levels));
  const double dy=(o.ymax-o.ymin)/(o.ny*(1<<o.levels));
  c->x0=o.xmin+c->ix*dx;c->x1=o.xmin+(c->ix+c->span)*dx;
  c->y0=o.ymin+c->iy*dy;c->y1=o.ymin+(c->iy+c->span)*dy;
}
void children(const AmrCell2D& c,const StaticAmrOptions2D& o,std::vector<AmrCell2D>* out) {
  const int h=c.span/2;
  for(int y=0;y<2;++y)for(int x=0;x<2;++x) {
    AmrCell2D child={};child.ix=c.ix+x*h;child.iy=c.iy+y*h;child.span=h;child.level=c.level+1;
    geometry_bounds(&child,o);out->push_back(child);
  }
}

bool make_groups(StaticAmrMesh2D* m,std::string* error) {
  const size_t n=m->cells.size();
  m->group_offsets={0};m->group_members.clear();m->small_cells=0;
  std::vector<int> visited(n,-1);
  for(size_t k=0;k<n;++k) {
    const auto& cell=m->cells[k];
    const double target=0.5*(cell.x1-cell.x0)*(cell.y1-cell.y0);
    std::vector<size_t> group={k};double volume=cell.volume;visited[k]=static_cast<int>(k);
    if(volume<target&&!m->options.agglomerate_small) {
      ++m->small_cells;
      for(size_t pos=0;pos<group.size()&&volume<target;++pos) {
        const size_t a=group[pos];
        for(size_t q=m->face_offsets[a];q<m->face_offsets[a+1]&&volume<target;++q) {
          const auto& f=m->faces[m->cell_faces[q]];
          const int neighbor=f.left==static_cast<int>(a)?f.right:f.left;
          if(neighbor<0||visited[neighbor]==static_cast<int>(k))continue;
          visited[neighbor]=static_cast<int>(k);group.push_back(static_cast<size_t>(neighbor));
          volume+=m->cells[neighbor].volume;
        }
      }
      if(volume<target)return fail(error,"small-cell neighborhood has insufficient connected fluid volume");
    }
    m->group_members.insert(m->group_members.end(),group.begin(),group.end());
    m->group_offsets.push_back(m->group_members.size());
  }
  m->member_offsets.assign(n+1,0);
  for(size_t cell:m->group_members)++m->member_offsets[cell+1];
  for(size_t k=0;k<n;++k)m->member_offsets[k+1]+=m->member_offsets[k];
  auto cursor=m->member_offsets;m->member_groups.resize(m->group_members.size());
  m->group_volume.assign(n,0);
  for(size_t g=0;g<n;++g)for(size_t q=m->group_offsets[g];q<m->group_offsets[g+1];++q) {
    const size_t k=m->group_members[q];
    m->member_groups[cursor[k]++]=g;
    m->group_volume[g]+=m->cells[k].volume/(m->member_offsets[k+1]-m->member_offsets[k]);
  }
  return true;
}

bool agglomerate(StaticAmrMesh2D* m,std::string* error) {
  const size_t n=m->cells.size();std::vector<size_t> parent(n),order(n),root_id(n,n),count;
  std::iota(parent.begin(),parent.end(),0);std::iota(order.begin(),order.end(),0);
  std::vector<double> volume(n),target(n);
  for(size_t k=0;k<n;++k){const auto& c=m->cells[k];volume[k]=c.volume;target[k]=.5*(c.x1-c.x0)*(c.y1-c.y0);}
  const auto find=[&](size_t k){while(parent[k]!=k){parent[k]=parent[parent[k]];k=parent[k];}return k;};
  std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return volume[a]/target[a]<volume[b]/target[b];});
  for(size_t seed:order){size_t r=find(seed);
    while(volume[r]<target[r]){size_t other=n;double best=0;
      for(const auto& f:m->faces){if(f.left<0||f.right<0||m->cells[f.left].level!=m->cells[f.right].level)continue;
        const size_t a=find(static_cast<size_t>(f.left)),b=find(static_cast<size_t>(f.right));
        if(a==b)continue;
        const size_t neighbor=a==r?b:b==r?a:n;
        if(neighbor<n&&f.length>best){other=neighbor;best=f.length;}}
      if(other==n)return fail(error,"small-cell agglomeration has no connected same-level neighbor");
      parent[r]=other;volume[other]+=volume[r];target[other]=std::max(target[other],target[r]);r=other;
    }
  }
  std::vector<AmrCell2D> cells;std::vector<size_t> id(n);std::vector<long double> mx,my;
  for(size_t k=0;k<n;++k){const size_t r=find(k);if(root_id[r]==n){root_id[r]=cells.size();auto c=m->cells[k];c.volume=0;c.wall_begin=c.wall_count=0;cells.push_back(c);count.push_back(0);mx.push_back(0);my.push_back(0);}
    const size_t g=root_id[r];id[k]=g;auto& c=cells[g];const auto& old=m->cells[k];
    c.x0=std::min(c.x0,old.x0);c.x1=std::max(c.x1,old.x1);c.y0=std::min(c.y0,old.y0);c.y1=std::max(c.y1,old.y1);
    c.ix=std::min(c.ix,old.ix);c.iy=std::min(c.iy,old.iy);c.volume+=old.volume;
    mx[g]+=static_cast<long double>(old.volume)*old.cx;my[g]+=static_cast<long double>(old.volume)*old.cy;++count[g];
    m->agglomeration.push_back({static_cast<double>(k),static_cast<double>(g),old.x0,old.x1,old.y0,old.y1,old.volume,old.cx,old.cy});
  }
  for(size_t g=0;g<cells.size();++g){auto& c=cells[g];if(count[g]>1){c.cx=static_cast<double>(mx[g]/c.volume);c.cy=static_cast<double>(my[g]/c.volume);}}
  std::vector<std::vector<WallFragment2D>> walls(cells.size());
  for(auto w:m->walls){w.owner_cell=id[w.owner_cell];walls[w.owner_cell].push_back(w);}
  m->walls.clear();for(size_t g=0;g<cells.size();++g){cells[g].wall_begin=m->walls.size();cells[g].wall_count=walls[g].size();m->walls.insert(m->walls.end(),walls[g].begin(),walls[g].end());}
  std::vector<AmrFace2D> faces;
  for(auto f:m->faces){if(f.left>=0)f.left=static_cast<int>(id[f.left]);if(f.right>=0)f.right=static_cast<int>(id[f.right]);if(f.left>=0&&f.left==f.right)continue;faces.push_back(f);}
  for(auto& pair:m->leaf_components)pair[1]=id[pair[1]];
  for(auto& leaf:m->partition)if(leaf[4]>=0)leaf[4]=static_cast<int>(id[leaf[4]]);
  m->cells.swap(cells);m->faces.swap(faces);return true;
}

bool primitive(const std::vector<double>& u,size_t n,double gamma,std::vector<double>* w) {
  w->resize(4*n);int bad=0;
  #pragma omp parallel for reduction(|:bad)
  for(size_t k=0;k<n;++k) {
    const double rho=u[k],mx=u[n+k],my=u[2*n+k],energy=u[3*n+k];
    if(!std::isfinite(rho)||!(rho>0)||!std::isfinite(mx)||!std::isfinite(my)||!std::isfinite(energy)){bad=1;continue;}
    const double vx=mx/rho,vy=my/rho,p=(gamma-1)*(energy-0.5*(mx*vx+my*vy));
    if(!std::isfinite(p)||!(p>0)){bad=1;continue;}
    (*w)[k]=rho;(*w)[n+k]=vx;(*w)[2*n+k]=vy;(*w)[3*n+k]=p;
  }
  return bad==0;
}

bool residual(const StaticAmrMesh2D& m,const Params2D& p,const std::vector<double>& u,
              std::vector<double>* r,std::vector<double>* w,std::vector<double>* flux,
              std::vector<double>* speed,std::vector<double>* viscous_dt=nullptr) {
  const size_t n=m.cells.size(),nf=m.faces.size();
  if(!primitive(u,n,p.gamma,w))return false;
  std::vector<double> slopes;
  if(p.amr_spatial_order==2)static_amr2d_reconstruct(m,*w,&slopes);
  flux->resize(4*nf);speed->resize(nf);r->resize(4*n);int bad=0;
  #pragma omp parallel for reduction(|:bad)
  for(size_t q=0;q<nf;++q) {
    const auto& f=m.faces[q];const int inside=f.left>=0?f.left:f.right;
    double a[4],b[4];
    for(int v=0;v<4;++v) {
      a[v]=p.amr_spatial_order==2&&f.left>=0?amr_point_value(w->data(),slopes.data(),n,v,f.left,m.cells[f.left],amr_face_x(f),amr_face_y(f)):(*w)[v*n+(f.left>=0?f.left:inside)];
      b[v]=p.amr_spatial_order==2&&f.right>=0?amr_point_value(w->data(),slopes.data(),n,v,f.right,m.cells[f.right],amr_face_x(f),amr_face_y(f)):(*w)[v*n+(f.right>=0?f.right:inside)];
    }
    if(f.boundary>=0) {
      double* outside=f.left<0?a:b;const double* interior=f.left<0?b:a;
      if(!amr_bc_state(m.boundaries[f.boundary],p.bc[f.boundary],p.gamma,f.boundary,interior,outside)){bad=1;continue;}
    }
    const int normal=1+f.axis,tangent=2-f.axis;double local[4];
    const int bc=f.boundary<0?-1:(m.boundaries[f.boundary].kind<0?p.bc[f.boundary]:m.boundaries[f.boundary].kind);
    if(bc==AMR_BC_NOSLIP){local[0]=0;local[1]=(*w)[3*n+inside];local[2]=0;local[3]=0;}
    else if(p.low_mach_preconditioning){
      const double al[5]={a[0],a[normal],a[tangent],0,a[3]},br[5]={b[0],b[normal],b[tangent],0,b[3]},nn[3]={1,0,0};double f5[5];
      lm_flux(al,br,nn,p.gamma,p.low_mach_reference,p.low_mach_floor,f5);
      local[0]=f5[0];local[1]=f5[1];local[2]=f5[2];local[3]=f5[4];
    }
    else if(p.flux==FLUX_HLLC)flux_hllc_t(a[0],a[normal],a[tangent],a[3],b[0],b[normal],b[tangent],b[3],p.gamma,local);
    else flux_ausmpup_t(a[0],a[normal],a[tangent],a[3],b[0],b[normal],b[tangent],b[3],p.gamma,p.ausm_mco,local);
    (*flux)[q]=local[0]*f.length;(*flux)[normal*nf+q]=local[1]*f.length;
    (*flux)[tangent*nf+q]=local[2]*f.length;(*flux)[3*nf+q]=local[3]*f.length;
    (*speed)[q]=std::max(std::fabs(a[normal])+std::sqrt(p.gamma*a[3]/a[0]),
                         std::fabs(b[normal])+std::sqrt(p.gamma*b[3]/b[0]));
    if(p.low_mach_preconditioning)(*speed)[q]=lm_face_speed(a,b,normal,3,p.gamma,p.low_mach_reference,p.low_mach_floor);
    for(double value:local)if(!std::isfinite(value))bad=1;
  }
  if(bad)return false;
  #pragma omp parallel for
  for(size_t k=0;k<n;++k) {
    const auto& c=m.cells[k];double sum[4]={};
    for(size_t q=m.face_offsets[k];q<m.face_offsets[k+1];++q) {
      const size_t face=m.cell_faces[q];const double sign=m.faces[face].left==static_cast<int>(k)?1.0:-1.0;
      for(int v=0;v<4;++v)sum[v]+=sign*(*flux)[v*nf+face];
    }
    for(size_t q=c.wall_begin;q<c.wall_begin+c.wall_count;++q) {
      const auto& wall=m.walls[q];const double pressure=p.amr_spatial_order==2?amr_point_value(w->data(),slopes.data(),n,3,k,c,amr_wall_x(wall),amr_wall_y(wall)):(*w)[3*n+k];
      sum[1]+=pressure*wall.nx*wall.length;sum[2]+=pressure*wall.ny*wall.length;
    }
    for(int v=0;v<4;++v)(*r)[v*n+k]=-sum[v]/c.volume;
  }
  if(m.transport.model==AMR_MODEL_LAMINAR){AmrViscousFields2D fields;if(static_amr2d_viscous_evaluate(m,p,*w,&fields,&u))return false;
    for(size_t q=0;q<r->size();++q)(*r)[q]+=fields.residual[q];
    for(size_t q=0;q<flux->size();++q)(*flux)[q]-=fields.face_flux[q];
    if(m.transport.sa_model){SaFields2D sa;if(static_amr2d_sa_evaluate(m,p,u,*w,fields,*flux,&sa))return false;
      r->resize(5*n);for(size_t k=0;k<n;++k){(*r)[4*n+k]=sa.residual[k];fields.cell_dt[k]=std::min(fields.cell_dt[k],sa.cell_dt[k]);}}
    if(viscous_dt)*viscous_dt=std::move(fields.cell_dt);
  }
  return true;
}

void redistribute(const StaticAmrMesh2D& m,std::vector<double>* u,std::vector<double>* groups,std::vector<double>* out) {
  if(m.small_cells==0)return;
  const size_t n=m.cells.size();groups->assign(4*n,0);out->resize(4*n);
  #pragma omp parallel for
  for(size_t g=0;g<n;++g)for(int v=0;v<4;++v) {
    double sum=0;
    for(size_t q=m.group_offsets[g];q<m.group_offsets[g+1];++q) {
      const size_t k=m.group_members[q];
      sum+=m.cells[k].volume*(*u)[v*n+k]/(m.member_offsets[k+1]-m.member_offsets[k]);
    }
    (*groups)[v*n+g]=sum/m.group_volume[g];
  }
  #pragma omp parallel for
  for(size_t k=0;k<n;++k)for(int v=0;v<4;++v) {
    double sum=0;for(size_t q=m.member_offsets[k];q<m.member_offsets[k+1];++q)sum+=(*groups)[v*n+m.member_groups[q]];
    (*out)[v*n+k]=sum/(m.member_offsets[k+1]-m.member_offsets[k]);
  }
  u->swap(*out);
}
} // namespace

void static_amr2d_reconstruct(const StaticAmrMesh2D& m,const std::vector<double>& w,std::vector<double>* slopes) {
  const size_t n=m.cells.size();slopes->assign(8*n,0);
  #pragma omp parallel for
  for(size_t k=0;k<n;++k){const auto& c=m.cells[k];
    for(int v=0;v<4;++v){const double center=w[v*n+k];double gx=0,gy=0,lower=center,upper=center;
      for(size_t pos=m.reconstruction_offsets[k];pos<m.reconstruction_offsets[k+1];++pos){const auto& t=m.reconstruction_terms[pos];const double other=w[v*n+t.neighbor],diff=other-center;
        gx+=t.x*diff;gy+=t.y*diff;lower=std::min(lower,other);upper=std::max(upper,other);}
      if(v==0||v==3)lower=std::max(lower,.1*center);
      double theta=1;
      for(size_t pos=m.face_offsets[k];pos<m.face_offsets[k+1];++pos){const auto& f=m.faces[m.cell_faces[pos]];
        theta=std::min(theta,amr_limiter(center,lower,upper,gx*(amr_face_x(f)-c.cx)+gy*(amr_face_y(f)-c.cy)));}
      for(size_t pos=c.wall_begin;pos<c.wall_begin+c.wall_count;++pos){const auto& wall=m.walls[pos];
        theta=std::min(theta,amr_limiter(center,lower,upper,gx*(amr_wall_x(wall)-c.cx)+gy*(amr_wall_y(wall)-c.cy)));}
      (*slopes)[2*v*n+k]=theta*gx;(*slopes)[(2*v+1)*n+k]=theta*gy;
    }
  }
}

int static_amr2d_wall_pressure(const StaticAmrMesh2D& m,const Params2D& p,const std::vector<double>& u,std::vector<double>* pressure) {
  const size_t n=m.cells.size();if(!pressure||u.size()<(m.transport.sa_model?5:4)*n)return 1;
  std::vector<double> w,slopes;if(!primitive(u,n,p.gamma,&w))return 1;
  if(p.amr_spatial_order==2)static_amr2d_reconstruct(m,w,&slopes);
  pressure->resize(m.walls.size());
  for(size_t q=0;q<m.walls.size();++q){const auto& wall=m.walls[q];const size_t k=wall.owner_cell;
    (*pressure)[q]=p.amr_spatial_order==2?amr_point_value(w.data(),slopes.data(),n,3,k,m.cells[k],amr_wall_x(wall),amr_wall_y(wall)):w[3*n+k];}
  return 0;
}

int static_amr2d_build(const StaticAmrOptions2D& o,const std::vector<double>& px,
                       const std::vector<double>& py,const std::vector<int>& patches,
                       StaticAmrMesh2D* m,std::string* error) {
  const auto invalid=[&](const std::string& message){fail(error,message);return 1;};
  if(!m||o.nx<1||o.ny<1||o.levels<0||o.levels>16||o.buffer<2||o.max_cells<1||o.max_cells>INT_MAX/8||
     !std::isfinite(o.xmin)||!std::isfinite(o.xmax)||!std::isfinite(o.ymin)||!std::isfinite(o.ymax)||
     !(o.xmax>o.xmin)||!(o.ymax>o.ymin)||
     static_cast<long long>(o.nx)*(1<<o.levels)>INT_MAX/2||static_cast<long long>(o.ny)*(1<<o.levels)>INT_MAX/2||
     px.size()!=py.size()||(!px.empty()&&px.size()<3)||px.size()>INT_MAX||
     (!patches.empty()&&patches.size()!=px.size()))return invalid("invalid mesh options or polygon arrays");
  for(const auto& box:o.boxes)if(box.level<0||box.level>o.levels||!std::isfinite(box.x0)||!std::isfinite(box.x1)||
      !std::isfinite(box.y0)||!std::isfinite(box.y1)||!(box.x1>box.x0)||!(box.y1>box.y0))return invalid("invalid refinement box");
  const int scale=1<<o.levels,npx=static_cast<int>(px.size());
  const double dx=(o.xmax-o.xmin)/(o.nx*scale),dy=(o.ymax-o.ymin)/(o.ny*scale);
  for(size_t a=0;a<px.size();++a) {
    const size_t b=(a+1)%px.size();
    if(!std::isfinite(px[a])||!std::isfinite(py[a])||(px[a]==px[b]&&py[a]==py[b])||(!patches.empty()&&patches[a]<0))return invalid("invalid polygon vertex or patch");
    if(px[a]<=o.xmin+3*dx||px[a]>=o.xmax-3*dx||py[a]<=o.ymin+3*dy||py[a]>=o.ymax-3*dy)return invalid("body needs three finest cells of domain clearance");

  }
  *m=StaticAmrMesh2D{};m->options=o;m->body_x=px;m->body_y=py;
  std::vector<AmrCell2D> leaves;bool too_many=false;
  std::function<void(AmrCell2D)> visit=[&](AmrCell2D c) {
    if(too_many)return;
    geometry_bounds(&c,o);bool refine=false;
    if(c.level<o.levels) {
      if(npx>0)refine=cut2d_rect_wall_distance(px.data(),py.data(),npx,c.x0,c.x1,c.y0,c.y1)<=o.buffer*0.5*std::min(c.x1-c.x0,c.y1-c.y0);
      for(const auto& box:o.boxes)if(c.level<box.level&&c.x0<box.x1&&c.x1>box.x0&&c.y0<box.y1&&c.y1>box.y0)refine=true;
    }
    if(refine){std::vector<AmrCell2D> sub;children(c,o,&sub);for(const auto& child:sub)visit(child);}
    else {leaves.push_back(c);if(leaves.size()>o.max_cells)too_many=true;}
  };
  for(int j=0;j<o.ny&&!too_many;++j)for(int i=0;i<o.nx&&!too_many;++i) {
    AmrCell2D c={};c.ix=i*scale;c.iy=j*scale;c.span=scale;visit(c);
  }
  if(too_many)return invalid("max-cells exceeded during refinement");
  std::vector<Connection> links;
  for(;;) {
    links=connections(leaves);std::vector<bool> split(leaves.size(),false);size_t count=0;
    for(const auto& f:links)if(f.left>=0&&f.right>=0) {
      const int difference=leaves[f.left].level-leaves[f.right].level;
      if(std::abs(difference)>1){const int k=difference<0?f.left:f.right;if(!split[k]){split[k]=true;++count;}}
    }
    if(count==0)break;
    if(count>(o.max_cells-leaves.size())/3)return invalid("max-cells exceeded during 2:1 balance");
    std::vector<AmrCell2D> next;next.reserve(leaves.size()+3*count);
    for(size_t k=0;k<leaves.size();++k)if(split[k])children(leaves[k],o,&next);else next.push_back(leaves[k]);
    leaves.swap(next);
  }
  m->background_leaves=leaves.size();std::vector<int> active(leaves.size(),-1);
  std::vector<std::vector<int>> components(leaves.size());
  std::vector<std::vector<std::array<double,2>>> fluid_boundaries;
  for(size_t k=0;k<leaves.size();++k) {
    auto c=leaves[k];std::vector<RectMetrics2D> regions;
    const int geometry_status=cut2d_rect_components(px.data(),py.data(),npx,c.x0,c.x1,c.y0,c.y1,patches.empty()?nullptr:patches.data(),&regions);
    if(geometry_status!=0) {
      std::ostringstream message;message.precision(17);
      message<<"cut geometry status="<<geometry_status<<" at leaf ("<<c.ix<<","<<c.iy<<") level="<<c.level
             <<" bounds=["<<c.x0<<","<<c.x1<<","<<c.y0<<","<<c.y1<<"]";
      return invalid(message.str());
    }
    if(regions.size()>1)++m->split_leaves;
    for(auto& r:regions){
    if(r.volume==0)continue;
    if(!r.walls.empty()&&c.level!=o.levels)return invalid("cut cell is not on finest level");
    c.volume=r.volume;c.cx=r.cx;c.cy=r.cy;c.wall_begin=m->walls.size();c.wall_count=r.walls.size();
    if(active[k]<0)active[k]=static_cast<int>(m->cells.size());
    components[k].push_back(static_cast<int>(m->cells.size()));m->leaf_components.push_back({k,m->cells.size()});
    for(auto wall:r.walls){wall.owner_cell=m->cells.size();m->walls.push_back(wall);}
    m->cells.push_back(c);
    fluid_boundaries.push_back(std::move(r.boundary));
    }
  }
  if(m->cells.empty())return invalid("no active fluid cells");
  m->partition.reserve(leaves.size());
  for(size_t k=0;k<leaves.size();++k) {
    const auto& c=leaves[k];m->partition.push_back({c.ix,c.iy,c.span,c.level,active[k]});
  }
  for(const auto& link:links) {
    const int left=link.left<0?-1:active[link.left],right=link.right<0?-1:active[link.right];
    if(left<0&&right<0)continue;
    // A solid neighbor is not a physical exterior boundary.
    if((link.left>=0&&left<0)||(link.right>=0&&right<0))continue;
    const double origin=link.axis==0?o.xmin:o.ymin,h=link.axis==0?dx:dy;
    const double tangent_origin=link.axis==0?o.ymin:o.xmin,tangent_h=link.axis==0?dy:dx;
    AmrFace2D face={};face.left=left;face.right=right;face.axis=link.axis;
    face.coordinate=origin+link.coordinate*h;face.lo=tangent_origin+link.lo*tangent_h;face.hi=tangent_origin+link.hi*tangent_h;
    face.boundary=link.left<0?2*link.axis:link.right<0?2*link.axis+1:-1;
    if(npx>0)cut2d_open_segment(px.data(),py.data(),npx,link.axis,face.coordinate,face.lo,face.hi,&face.length,&face.centroid);
    else {face.length=face.hi-face.lo;face.centroid=0.5*(face.lo+face.hi);}
    if(face.length==0)continue;
    const bool complex=(link.left>=0&&!components[link.left].empty()&&!fluid_boundaries[components[link.left][0]].empty())||
                       (link.right>=0&&!components[link.right].empty()&&!fluid_boundaries[components[link.right][0]].empty());
    if(complex){
      const auto locate=[&](int background,double midpoint){if(background<0)return -1;const auto& ids=components[background];if(ids.size()==1)return ids[0];
        const double x=link.axis==0?face.coordinate:midpoint,y=link.axis==1?face.coordinate:midpoint;
        const auto& cell=leaves[background];const double tol=1e-10*std::min(cell.x1-cell.x0,cell.y1-cell.y0);
        for(int id:ids){const auto& polygon=fluid_boundaries[id];for(size_t i=0;i<polygon.size();++i){const auto a=polygon[i],b=polygon[(i+1)%polygon.size()];const double dx=b[0]-a[0],dy=b[1]-a[1],d2=dx*dx+dy*dy;
          const double t=std::max(0.,std::min(1.,((x-a[0])*dx+(y-a[1])*dy)/d2));
          if(std::hypot(x-a[0]-t*dx,y-a[1]-t*dy)<=tol)return id;}}
        return -2;};
      for(const auto interval:cut2d_open_intervals(px.data(),py.data(),npx,link.axis,face.coordinate,face.lo,face.hi)){
        auto sub=face;sub.lo=interval[0];sub.hi=interval[1];sub.length=sub.hi-sub.lo;sub.centroid=.5*(sub.lo+sub.hi);
        sub.left=locate(link.left,sub.centroid);sub.right=locate(link.right,sub.centroid);
        if(sub.left==-2||sub.right==-2)return invalid("cannot associate open aperture with a connected fluid component");
        if(sub.left>=0&&sub.right>=0&&m->cells[sub.left].level!=m->cells[sub.right].level)return invalid("coarse/fine face touches a complex cut cell; increase buffer");
        m->faces.push_back(sub);
      }
      continue;
    }
    if(left>=0&&right>=0&&m->cells[left].level!=m->cells[right].level) {
      if(m->cells[left].wall_count||m->cells[right].wall_count)return invalid("coarse/fine face touches cut cell; increase buffer");
      ++m->coarse_fine_faces;
    }
    m->faces.push_back(face);
  }
  if(o.agglomerate_small&&!agglomerate(m,error))return 1;
  const size_t n=m->cells.size();m->face_offsets.assign(n+1,0);
  for(const auto& f:m->faces){if(f.left>=0)++m->face_offsets[f.left+1];if(f.right>=0)++m->face_offsets[f.right+1];}
  for(size_t k=0;k<n;++k)m->face_offsets[k+1]+=m->face_offsets[k];
  m->cell_faces.resize(m->face_offsets.back());auto cursor=m->face_offsets;
  for(size_t q=0;q<m->faces.size();++q){const auto& f=m->faces[q];if(f.left>=0)m->cell_faces[cursor[f.left]++]=q;if(f.right>=0)m->cell_faces[cursor[f.right]++]=q;}
  m->reconstruction_offsets.reserve(n+1);m->reconstruction_offsets.push_back(0);
  for(size_t k=0;k<n;++k){
    const auto& c=m->cells[k];std::vector<int> neighbors;
    for(size_t pos=m->face_offsets[k];pos<m->face_offsets[k+1];++pos){const auto& f=m->faces[m->cell_faces[pos]];
      const int other=f.left==static_cast<int>(k)?f.right:f.left;if(other>=0)neighbors.push_back(other);}
    std::sort(neighbors.begin(),neighbors.end());neighbors.erase(std::unique(neighbors.begin(),neighbors.end()),neighbors.end());
    double xx=0,xy=0,yy=0;
    for(int other:neighbors){const double dx=m->cells[other].cx-c.cx,dy=m->cells[other].cy-c.cy,d2=dx*dx+dy*dy;
      if(!(d2>0))return invalid("coincident reconstruction centers");xx+=dx*dx/d2;xy+=dx*dy/d2;yy+=dy*dy/d2;}
    const double det=xx*yy-xy*xy;
    if(det>1e-12*(xx+yy)*(xx+yy))for(int other:neighbors){const double dx=m->cells[other].cx-c.cx,dy=m->cells[other].cy-c.cy,d2=dx*dx+dy*dy;
      m->reconstruction_terms.push_back({other,(yy*dx-xy*dy)/(det*d2),(xx*dy-xy*dx)/(det*d2)});}
    m->reconstruction_offsets.push_back(m->reconstruction_terms.size());
  }
  for(size_t k=0;k<n;++k) {
    const auto& c=m->cells[k];double closure[2]={};
    for(size_t q=m->face_offsets[k];q<m->face_offsets[k+1];++q){const auto& f=m->faces[m->cell_faces[q]];closure[f.axis]+=(f.left==static_cast<int>(k)?1:-1)*f.length;}
    for(size_t q=c.wall_begin;q<c.wall_begin+c.wall_count;++q){closure[0]+=m->walls[q].nx*m->walls[q].length;closure[1]+=m->walls[q].ny*m->walls[q].length;}
    if(std::hypot(closure[0],closure[1])>1e-9*std::max(c.x1-c.x0,c.y1-c.y0))return invalid("cell geometry is not closed");
  }
  return make_groups(m,error)?0:1;
}

int static_amr2d_initialize(const StaticAmrMesh2D& m,const Params2D& p,const std::array<double,4>& w,StaticAmrState2D* s) {
  if(!s||!(p.gamma>1)||!std::isfinite(p.gamma)||!std::isfinite(w[0])||!std::isfinite(w[1])||!std::isfinite(w[2])||!std::isfinite(w[3])||!(w[0]>0)||!(w[3]>0))return 1;
  *s=StaticAmrState2D{};const size_t n=m.cells.size();s->u.resize((m.transport.sa_model?5:4)*n);
  const double values[4]={w[0],w[0]*w[1],w[0]*w[2],w[3]/(p.gamma-1)+0.5*w[0]*(w[1]*w[1]+w[2]*w[2])};
  for(int v=0;v<4;++v)std::fill(s->u.begin()+v*n,s->u.begin()+(v+1)*n,values[v]);
  if(m.transport.sa_model)std::fill(s->u.begin()+4*n,s->u.end(),w[0]*m.transport.sa_nu_inf);
  return 0;
}

int static_amr2d_rhs(const StaticAmrMesh2D& m,const Params2D& p,const std::vector<double>& u,
                     std::vector<double>* result,std::vector<double>* face_flux) {
  if(!static_amr2d_viscous_ready(m)||!result||m.cells.empty()||u.size()!=(m.transport.sa_model?5:4)*m.cells.size()||!(p.gamma>1))return 1;
  for(int side=0;side<4;++side)if(!amr_bc_valid(m.boundaries[side],p.bc[side],side))return 1;
  std::vector<double> w,flux,speed;
  if(!residual(m,p,u,result,&w,&flux,&speed))return 1;
  if(face_flux)face_flux->swap(flux);
  return 0;
}

int static_amr2d_redistribute(const StaticAmrMesh2D& m,std::vector<double>* u) {
  if(!u||u->size()!=(m.transport.sa_model?5:4)*m.cells.size()||m.group_offsets.size()!=m.cells.size()+1)return 1;
  std::vector<double> groups,out;redistribute(m,u,&groups,&out);return 0;
}

int static_amr2d_run(const StaticAmrMesh2D& m,const Params2D& p,double tend,long maxsteps,StaticAmrState2D* s,std::string* error) {
  const auto invalid=[&](const std::string& message){fail(error,message);return 1;};
  if(!static_amr2d_viscous_ready(m)||!s||m.cells.empty()||s->u.size()!=(m.transport.sa_model?5:4)*m.cells.size()||!std::isfinite(s->time)||s->time<0||!std::isfinite(tend)||tend<s->time||maxsteps<1||
     !(p.gamma>1)||!std::isfinite(p.gamma)||!(p.cfl>0)||p.cfl>0.5||!std::isfinite(p.dt_fixed)||p.dt_fixed<0||
     !std::isfinite(p.ausm_mco)||p.ausm_mco<=0||p.time_integrator!=TIME_SSP_RK3||(p.amr_spatial_order!=1&&p.amr_spatial_order!=2)||
     (p.flux!=FLUX_HLLC&&p.flux!=FLUX_AUSMPUP))return invalid("static AMR requires first-order RK3, valid state/options and CFL <= 0.5");
  for(int side=0;side<4;++side)if(!amr_bc_valid(m.boundaries[side],p.bc[side],side))return invalid("invalid AMR boundary configuration");
  const size_t n=m.cells.size();std::vector<double> r,w,flux,speed,u1((m.transport.sa_model?5:4)*n),u2((m.transport.sa_model?5:4)*n),groups,out,base_r,viscous_dt,local_dt(n);
  const auto precondition=[&](const std::vector<double>& state){if(!p.low_mach_preconditioning)return;
    for(size_t k=0;k<n;++k){double rr[5],ww[5]={w[k],w[n+k],w[2*n+k],w[3*n+k],m.transport.sa_model?state[4*n+k]/w[k]:0};
      for(int v=0;v<(m.transport.sa_model?5:4);++v)rr[v]=r[v*n+k];
      lm_apply(rr,ww,p.gamma,p.low_mach_reference,p.low_mach_floor,2,m.transport.sa_model);
      for(int v=0;v<(m.transport.sa_model?5:4);++v)r[v*n+k]=rr[v];}};
  while(s->time<tend) {
    if(s->steps>=maxsteps)return invalid("maxsteps reached before tend");
    if(!residual(m,p,s->u,&r,&w,&flux,&speed,&viscous_dt))return invalid("nonphysical base state or nonfinite face flux");
    double dt=p.dt_fixed>0?p.dt_fixed:std::numeric_limits<double>::infinity();
    std::fill(local_dt.begin(),local_dt.end(),dt);
    if(p.dt_fixed==0)for(size_t k=0;k<n;++k) {
      const auto& c=m.cells[k];double spectral=0;
      for(size_t q=m.face_offsets[k];q<m.face_offsets[k+1];++q){const size_t f=m.cell_faces[q];spectral+=speed[f]*m.faces[f].length;}
      const double a=std::sqrt(p.gamma*w[3*n+k]/w[k]);
      const double wall_a=p.low_mach_preconditioning?a*std::sqrt(lm_beta2(w[n+k]*w[n+k]+w[2*n+k]*w[2*n+k],a*a,p.low_mach_reference,p.low_mach_floor)):a;
      for(size_t q=c.wall_begin;q<c.wall_begin+c.wall_count;++q)spectral+=wall_a*m.walls[q].length;
      if(spectral>0)local_dt[k]=p.cfl*(m.options.agglomerate_small?c.volume:(c.x1-c.x0)*(c.y1-c.y0))/spectral;
      if(m.transport.model==AMR_MODEL_LAMINAR)local_dt[k]=std::min(local_dt[k],viscous_dt[k]);
      dt=std::min(dt,local_dt[k]);
    }
    if(p.dt_fixed==0&&m.transport.model==AMR_MODEL_LAMINAR)for(double value:viscous_dt)dt=std::min(dt,value);
    const double base_dt=dt;
    const double remaining=tend-s->time;
    dt=std::min(dt,remaining);
    // A roundoff-sized final remainder must not trigger another SRD projection.
    const double time_tol=32*std::numeric_limits<double>::epsilon()*std::max({std::fabs(tend),std::fabs(s->time),dt});
    if(remaining-dt<=time_tol)dt=remaining;
    precondition(s->u);base_r=r;bool accepted=false;
    for(int attempt=0;attempt<21;++attempt) {
      if(!std::isfinite(dt)||!(dt>0)||s->time+dt==s->time)return invalid("physical time step underflow");
      for(size_t q=0;q<s->u.size();++q)u1[q]=s->u[q]+dt*(p.local_pseudo_time?local_dt[q%n]/base_dt:1)*base_r[q];
      redistribute(m,&u1,&groups,&out);
      bool ok=residual(m,p,u1,&r,&w,&flux,&speed);
      if(ok) {
        precondition(u1);
        for(size_t q=0;q<s->u.size();++q)u2[q]=0.75*s->u[q]+0.25*(u1[q]+dt*(p.local_pseudo_time?local_dt[q%n]/base_dt:1)*r[q]);
        redistribute(m,&u2,&groups,&out);
        ok=residual(m,p,u2,&r,&w,&flux,&speed);
      }
      if(ok) {
        precondition(u2);
        for(size_t q=0;q<s->u.size();++q)u1[q]=s->u[q]/3+(2.0/3)*(u2[q]+dt*(p.local_pseudo_time?local_dt[q%n]/base_dt:1)*r[q]);
        redistribute(m,&u1,&groups,&out);
        ok=primitive(u1,n,p.gamma,&w);
        if(m.transport.sa_model)for(size_t k=0;k<n;++k)ok=ok&&std::isfinite(u1[4*n+k]);
      }
      if(ok){s->u.swap(u1);s->time=dt==remaining?tend:s->time+dt;++s->steps;accepted=true;break;}
      ++s->rejected;dt*=0.5;
    }
    if(!accepted)return invalid("RK candidates failed admissibility after 21 trials; physical step not committed");
  }
  return 0;
}
} // namespace cfd
