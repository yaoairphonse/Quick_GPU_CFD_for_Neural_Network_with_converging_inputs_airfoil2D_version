// Convert a tetrahedral CRM mesh into reusable binary face/CSR/gradient tables.
// Exact point-to-triangle wall distance is computed with an AABB hierarchy.
#include "mesh_format.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <parallel/algorithm>
#include <stdexcept>
#include <vector>
#include <omp.h>
using V=std::array<double,3>;
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V add_scaled(V a,V b,double t){for(int j=0;j<3;++j)a[j]+=t*b[j];return a;}
struct Key {int a,b,c,owner;bool operator<(const Key& q)const{return a!=q.a?a<q.a:b!=q.b?b<q.b:c<q.c;}};
bool same(const Key& a,const Key& b){return a.a==b.a&&a.b==b.b&&a.c==b.c;}
using Boundary=ConnectivityBoundary;
struct Triangle {V a,b,c,lo,hi,center;};
struct Node {V lo,hi;int first,last,left=-1,right=-1;};
double triangle_distance2(V p,const Triangle& t){
 const V ab=sub(t.b,t.a),ac=sub(t.c,t.a),ap=sub(p,t.a);const double d1=dot(ab,ap),d2=dot(ac,ap);
 if(d1<=0&&d2<=0)return dot(ap,ap);const V bp=sub(p,t.b);const double d3=dot(ab,bp),d4=dot(ac,bp);
 if(d3>=0&&d4<=d3)return dot(bp,bp);const double vc=d1*d4-d3*d2;
 if(vc<=0&&d1>=0&&d3<=0){V e=sub(p,add_scaled(t.a,ab,d1/(d1-d3)));return dot(e,e);}
 const V cp=sub(p,t.c);const double d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0&&d5<=d6)return dot(cp,cp);
 const double vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0){V e=sub(p,add_scaled(t.a,ac,d2/(d2-d6)));return dot(e,e);}
 const double va=d3*d6-d5*d4;if(va<=0&&d4-d3>=0&&d5-d6>=0){V e=sub(p,add_scaled(t.b,sub(t.c,t.b),(d4-d3)/((d4-d3)+(d5-d6))));return dot(e,e);}
 const double inv=1/(va+vb+vc);V q=add_scaled(add_scaled(t.a,ab,vb*inv),ac,vc*inv);V e=sub(p,q);return dot(e,e);
}
struct BVH {
 std::vector<Triangle> tri;std::vector<int> indices;std::vector<Node> nodes;
 int build(int first,int last){
  Node node;node.first=first;node.last=last;for(int j=0;j<3;++j){node.lo[j]=1e100;node.hi[j]=-1e100;}
  for(int i=first;i<last;++i)for(int j=0;j<3;++j){node.lo[j]=std::min(node.lo[j],tri[indices[i]].lo[j]);node.hi[j]=std::max(node.hi[j],tri[indices[i]].hi[j]);}
  int id=nodes.size();nodes.push_back(node);if(last-first<=8)return id;
  int axis=0;for(int j=1;j<3;++j)if(node.hi[j]-node.lo[j]>node.hi[axis]-node.lo[axis])axis=j;
  int mid=(first+last)/2;std::nth_element(indices.begin()+first,indices.begin()+mid,indices.begin()+last,[&](int a,int b){return tri[a].center[axis]<tri[b].center[axis];});
  int l=build(first,mid),r=build(mid,last);nodes[id].left=l;nodes[id].right=r;return id;
 }
 double box_distance2(V p,const Node& n)const{double d=0;for(int j=0;j<3;++j){double q=std::max(n.lo[j]-p[j],std::max(0.,p[j]-n.hi[j]));d+=q*q;}return d;}
 double nearest(V p,int id,double best)const{
  const Node& n=nodes[id];if(box_distance2(p,n)>=best)return best;
  if(n.left<0){for(int i=n.first;i<n.last;++i)best=std::min(best,triangle_distance2(p,tri[indices[i]]));return best;}
  double dl=box_distance2(p,nodes[n.left]),dr=box_distance2(p,nodes[n.right]);
  if(dl<dr){best=nearest(p,n.left,best);return nearest(p,n.right,best);}best=nearest(p,n.right,best);return nearest(p,n.left,best);
 }
};
template<class T>std::vector<T> load(const std::filesystem::path& file,size_t n){std::vector<T>a(n);std::ifstream f(file,std::ios::binary);f.read(reinterpret_cast<char*>(a.data()),n*sizeof(T));if(!f)throw std::runtime_error("cannot read "+file.string());return a;}
int main(int argc,char** argv)try{
 if(argc<2)throw std::runtime_error("usage: prepare_tetra DIRECTORY [threads] [gauss|qr]");const std::filesystem::path root=argv[1];omp_set_num_threads(argc>2?std::stoi(argv[2]):16);const bool gauss=argc>3&&std::string(argv[3])=="gauss";
 const auto start=std::chrono::steady_clock::now();auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
 size_t np,n,nb;std::ifstream(root/"input_counts.txt")>>np>>n>>nb;if(n<1||4*n>2000000000ull)throw std::runtime_error("invalid counts");
 auto xyz=load<V>(root/"points.bin",np);auto tetra=load<std::array<int,4>>(root/"tetra.bin",n);auto boundary=load<Boundary>(root/"boundary.bin",nb);
 std::vector<Cell> cells(n);std::vector<Key> keys(4*n),bkeys(nb);BVH bvh;
 for(size_t i=0;i<nb;++i){int v[3]={boundary[i].node[0],boundary[i].node[1],boundary[i].node[2]};std::sort(v,v+3);bkeys[i]={v[0],v[1],v[2],static_cast<int>(i)};
  if(boundary[i].bc==1){Triangle t;t.a=xyz[v[0]];t.b=xyz[v[1]];t.c=xyz[v[2]];for(int j=0;j<3;++j){t.lo[j]=std::min({t.a[j],t.b[j],t.c[j]});t.hi[j]=std::max({t.a[j],t.b[j],t.c[j]});t.center[j]=(t.a[j]+t.b[j]+t.c[j])/3;}bvh.tri.push_back(t);}}
 std::sort(bkeys.begin(),bkeys.end());
 #pragma omp parallel for
 for(size_t k=0;k<n;++k){const auto t=tetra[k];Cell c{};for(int j=0;j<3;++j)c.x[j]=.25*(xyz[t[0]][j]+xyz[t[1]][j]+xyz[t[2]][j]+xyz[t[3]][j]);
  c.volume=fabs(dot(sub(xyz[t[1]],xyz[t[0]]),cross(sub(xyz[t[2]],xyz[t[0]]),sub(xyz[t[3]],xyz[t[0]]))))/6;c.count=c.term_count=4;c.first=c.term_first=4*k;cells[k]=c;
  for(int omit=0;omit<4;++omit){int v[3],q=0;for(int j=0;j<4;++j)if(j!=omit)v[q++]=t[j];std::sort(v,v+3);keys[4*k+omit]={v[0],v[1],v[2],static_cast<int>(k)};}}
 // This companion file retains import topology for NNLCI point search and
 // interpolation. It is deliberately separate from the hot solver tables.
 std::vector<uint64_t> node_offsets(np+1,0);
 for(const auto& t:tetra)for(int v:t){if(v<0||static_cast<size_t>(v)>=np)throw std::runtime_error("cell references an invalid point");++node_offsets[v+1];}
 std::partial_sum(node_offsets.begin(),node_offsets.end(),node_offsets.begin());
 if(node_offsets.back()!=4*n)throw std::runtime_error("node incidence count mismatch");
 std::vector<uint64_t> node_cursor(node_offsets.begin(),node_offsets.end()-1);std::vector<int32_t> node_cells(4*n);
 for(size_t k=0;k<n;++k)for(int v:tetra[k])node_cells[node_cursor[v]++]=static_cast<int32_t>(k);
 ConnectivityHeader connectivity;connectivity.point_count=np;connectivity.cell_count=n;connectivity.boundary_face_count=nb;connectivity.incidence_count=4*n;
 connectivity.points_offset=sizeof(connectivity);
 connectivity.cell_nodes_offset=connectivity.points_offset+np*sizeof(V);
 connectivity.node_offsets_offset=connectivity.cell_nodes_offset+n*sizeof(tetra[0]);
 connectivity.node_cells_offset=connectivity.node_offsets_offset+(np+1)*sizeof(uint64_t);
 connectivity.boundary_offset=connectivity.node_cells_offset+4*n*sizeof(int32_t);
 connectivity.file_bytes=connectivity.boundary_offset+nb*sizeof(Boundary);
 std::ofstream connectivity_output(root/"connectivity.bin",std::ios::binary);connectivity_output.write(reinterpret_cast<const char*>(&connectivity),sizeof(connectivity));
 const auto write_connectivity=[&](const auto& values){connectivity_output.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));};
 write_connectivity(xyz);write_connectivity(tetra);write_connectivity(node_offsets);write_connectivity(node_cells);write_connectivity(boundary);connectivity_output.close();
 if(!connectivity_output)throw std::runtime_error("connectivity table write failed");node_cursor.clear();node_cursor.shrink_to_fit();node_cells.clear();node_cells.shrink_to_fit();node_offsets.clear();node_offsets.shrink_to_fit();
 tetra.clear();tetra.shrink_to_fit();std::cout<<"Sorting "<<keys.size()<<" half faces at "<<elapsed()<<" s\n"<<std::flush;
 __gnu_parallel::sort(keys.begin(),keys.end());std::vector<Face> faces;faces.reserve((4*n+nb)/2);size_t exterior=0;
 for(size_t i=0;i<keys.size();){const Key a=keys[i];const bool shared=i+1<keys.size()&&same(a,keys[i+1]);if(shared&&i+2<keys.size()&&same(a,keys[i+2]))throw std::runtime_error("nonmanifold face");
  Face f{};f.left=a.owner;f.right=shared?keys[i+1].owner:-1;f.bc=0;
  if(!shared){auto it=std::lower_bound(bkeys.begin(),bkeys.end(),a);if(it==bkeys.end()||!same(*it,a))throw std::runtime_error("unmarked exterior face");f.bc=boundary[it->owner].bc;++exterior;}
  const V p=xyz[a.a],q=xyz[a.b],r=xyz[a.c];V av=cross(sub(q,p),sub(r,p));for(int j=0;j<3;++j){av[j]*=.5;f.x[j]=(p[j]+q[j]+r[j])/3;}
  const V center={cells[f.left].x[0],cells[f.left].x[1],cells[f.left].x[2]};V delta={f.x[0]-center[0],f.x[1]-center[1],f.x[2]-center[2]};if(dot(av,delta)<0)for(double& x:av)x=-x;
  f.area=sqrt(dot(av,av));if(!(f.area>0))throw std::runtime_error("zero face area");for(int j=0;j<3;++j)f.n[j]=av[j]/f.area;
  if(shared)for(int j=0;j<3;++j)delta[j]=cells[f.right].x[j]-center[j];f.distance=dot(delta,{f.n[0],f.n[1],f.n[2]});if(!(f.distance>0))throw std::runtime_error("nonpositive face distance");faces.push_back(f);i+=shared?2:1;
 }
 if(exterior!=nb)throw std::runtime_error("exterior count mismatch");keys.clear();keys.shrink_to_fit();boundary.clear();boundary.shrink_to_fit();bkeys.clear();bkeys.shrink_to_fit();xyz.clear();xyz.shrink_to_fit();
 const size_t nf=faces.size();std::vector<int> ids(4*n),cursor(n);for(size_t k=0;k<n;++k)cursor[k]=cells[k].first;
 for(size_t q=0;q<nf;++q){const auto f=faces[q];ids[cursor[f.left]++]=q;if(f.right>=0)ids[cursor[f.right]++]=q;}
 for(size_t k=0;k<n;++k)if(cursor[k]!=cells[k].first+4)throw std::runtime_error("tetrahedron does not have four faces");cursor.clear();cursor.shrink_to_fit();
 bvh.indices.resize(bvh.tri.size());std::iota(bvh.indices.begin(),bvh.indices.end(),0);bvh.build(0,bvh.tri.size());std::cout<<"Faces="<<nf<<", wall triangles="<<bvh.tri.size()<<"; computing wall distances at "<<elapsed()<<" s\n"<<std::flush;
 std::vector<Term> terms(4*n);double closure_max=0,minvolume=1e100,maxcondition=0;int bad=0;
 #pragma omp parallel for reduction(max:closure_max,maxcondition) reduction(min:minvolume) reduction(|:bad) schedule(dynamic,1024)
 for(size_t k=0;k<n;++k){Cell& c=cells[k];V p={c.x[0],c.x[1],c.x[2]},closure={};double surface=0,m[3][3]={};V delta[4];int neighbor[4];
  c.dwall=sqrt(bvh.nearest(p,0,1e100));if(!(c.volume>0)||!(c.dwall>0)||!std::isfinite(c.dwall)){bad|=1;continue;}minvolume=std::min(minvolume,c.volume);
  for(int j=0;j<4;++j){const auto f=faces[ids[c.first+j]];const double sign=f.left==static_cast<int>(k)?1:-1;for(int a=0;a<3;++a)closure[a]+=sign*f.area*f.n[a];surface+=f.area;
   const int other=f.left==static_cast<int>(k)?f.right:f.left;neighbor[j]=other>=0?other:-f.bc;
   for(int a=0;a<3;++a)delta[j][a]=(other>=0?cells[other].x[a]:f.x[a])-p[a];const double d2=dot(delta[j],delta[j]);
   for(int a=0;a<3;++a)for(int b=0;b<3;++b)m[a][b]+=delta[j][a]*delta[j][b]/d2;
  }
  closure_max=std::max(closure_max,sqrt(dot(closure,closure))/surface);
  // Reorthogonalized QR avoids squaring the condition number on thin RANS tets.
  double Q[4][3]={},R[3][3]={},length[4];for(int i=0;i<4;++i)length[i]=sqrt(dot(delta[i],delta[i]));
  bool rank_ok=true;
  for(int j=0;j<3;++j){double v[4];for(int i=0;i<4;++i)v[i]=delta[i][j]/length[i];
   for(int pass=0;pass<2;++pass)for(int a=0;a<j;++a){double coefficient=0;for(int i=0;i<4;++i)coefficient+=Q[i][a]*v[i];R[a][j]+=coefficient;for(int i=0;i<4;++i)v[i]-=coefficient*Q[i][a];}
   double norm=0;for(double x:v)norm+=x*x;norm=sqrt(norm);if(!(norm>1e-14)){rank_ok=false;break;}R[j][j]=norm;for(int i=0;i<4;++i)Q[i][j]=v[i]/norm;
  }
  if(!rank_ok&&!gauss){bad|=2;continue;}
  double inverse_norm=0;
  for(int i=0;i<4;++i){Term t{};t.source=neighbor[i];t.face=ids[c.first+i];
   for(int j=2;j>=0;--j){double value=Q[i][j]/length[i];for(int a=j+1;a<3;++a)value-=R[j][a]*t.coefficient[a];t.coefficient[j]=value/R[j][j];}
   terms[c.term_first+i]=t;
  }
  for(int col=0;col<3;++col){double z[3]={};for(int j=2;j>=0;--j){double value=j==col?1:0;for(int a=j+1;a<3;++a)value-=R[j][a]*z[a];z[j]=value/R[j][j];}for(double x:z)inverse_norm+=x*x;}
  maxcondition=std::max(maxcondition,4*inverse_norm);
  if(gauss)for(int j=0;j<4;++j){const auto f=faces[ids[c.first+j]];const double sign=f.left==static_cast<int>(k)?1:-1;double fraction=1;
   if(neighbor[j]>=0){double dn=0;for(int a=0;a<3;++a)dn+=(f.x[a]-c.x[a])*sign*f.n[a];fraction=dn/f.distance;if(!(fraction>0&&fraction<1)){bad|=4;continue;}}
   for(int a=0;a<3;++a)terms[c.term_first+j].coefficient[a]=fraction*sign*f.area*f.n[a]/c.volume;
  }
 }
 if(bad||closure_max>1e-10){std::cerr<<"audit bits="<<bad<<" closure="<<closure_max<<" minV="<<minvolume<<" condition="<<maxcondition<<" elapsed="<<elapsed()<<'\n';throw std::runtime_error("geometry/gradient audit failed");}
 std::cout<<"Writing tables at "<<elapsed()<<" s\n"<<std::flush;std::ofstream output(root/"mesh.bin",std::ios::binary);MeshHeader header;header.n=n;header.nf=nf;header.ni=header.nt=4*n;output.write(reinterpret_cast<char*>(&header),sizeof(header));
 const auto write=[&](const auto& v){output.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(v[0]));};write(cells);write(faces);write(ids);write(terms);output.close();if(!output)throw std::runtime_error("binary table write failed");
 const double bytes=n*sizeof(Cell)+nf*sizeof(Face)+4*n*(sizeof(int)+sizeof(Term))+60*n*sizeof(double)+9*nf*sizeof(double)
                    +7*((n+127)/128)*sizeof(double)+6*((nf+127)/128)*sizeof(double)+13*sizeof(double)+sizeof(int);
 std::ofstream audit(root/"geometry_audit.json");audit<<std::setprecision(17)<<"{\"cells\":"<<n<<",\"faces\":"<<nf<<",\"closure_relative_linf\":"<<closure_max<<",\"min_volume\":"<<minvolume<<",\"gradient_trace_condition_max\":"<<maxcondition<<",\"gradient\":\""<<(gauss?"distance-weighted Green-Gauss":"reorthogonalized weighted least-squares QR")<<"\",\"GPU_bytes_estimate\":"<<bytes<<",\"prepare_seconds\":"<<elapsed()<<",\"wall_distance\":\"exact nearest triangulated no-slip surface via AABB tree\"}\n";
 std::cout<<"Finished in "<<elapsed()<<" s; estimated GPU bytes="<<bytes<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 2;}
