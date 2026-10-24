// Exact control-centroid distance to the triangulated CRM wall via an AABB BVH.
#include "amr_mesh_format.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <omp.h>

using V=std::array<double,3>;
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V add_scaled(V a,V b,double t){for(int j=0;j<3;++j)a[j]+=t*b[j];return a;}
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
 const double inv=1/(va+vb+vc);V q=add_scaled(add_scaled(t.a,ab,vb*inv),ac,vc*inv),e=sub(p,q);return dot(e,e);
}

struct BVH {
 std::vector<Triangle> triangles;std::vector<int> indices;std::vector<Node> nodes;
 int build(int first,int last){
  Node node;node.first=first;node.last=last;for(int j=0;j<3;++j){node.lo[j]=1e100;node.hi[j]=-1e100;}
  for(int i=first;i<last;++i)for(int j=0;j<3;++j){node.lo[j]=std::min(node.lo[j],triangles[indices[i]].lo[j]);node.hi[j]=std::max(node.hi[j],triangles[indices[i]].hi[j]);}
  const int id=nodes.size();nodes.push_back(node);if(last-first<=8)return id;
  int axis=0;for(int j=1;j<3;++j)if(node.hi[j]-node.lo[j]>node.hi[axis]-node.lo[axis])axis=j;
  const int middle=(first+last)/2;std::nth_element(indices.begin()+first,indices.begin()+middle,indices.begin()+last,[&](int a,int b){return triangles[a].center[axis]<triangles[b].center[axis];});
  nodes[id].left=build(first,middle);nodes[id].right=build(middle,last);return id;
 }
 double box_distance2(V p,const Node& node)const{double value=0;for(int j=0;j<3;++j){const double d=std::max(node.lo[j]-p[j],std::max(0.,p[j]-node.hi[j]));value+=d*d;}return value;}
 double nearest(V p,int node_id,double best)const{
  const Node& node=nodes[node_id];if(box_distance2(p,node)>=best)return best;
  if(node.left<0){for(int i=node.first;i<node.last;++i)best=std::min(best,triangle_distance2(p,triangles[indices[i]]));return best;}
  const double left=box_distance2(p,nodes[node.left]),right=box_distance2(p,nodes[node.right]);
  if(left<right){best=nearest(p,node.left,best);return nearest(p,node.right,best);}best=nearest(p,node.right,best);return nearest(p,node.left,best);
 }
};

template<class T>std::vector<T> read_exact(const std::filesystem::path& path,size_t count){
 std::vector<T> result(count);std::ifstream stream(path,std::ios::binary);stream.read(reinterpret_cast<char*>(result.data()),static_cast<std::streamsize>(count*sizeof(T)));
 if(!stream||stream.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("invalid file size: "+path.string());return result;
}

int main(int argc,char** argv)try{
 if(argc<2||argc>3)throw std::runtime_error("usage: prepare_amr_wall_distance AMR_DIRECTORY [threads]");
 const std::filesystem::path root=argv[1];omp_set_num_threads(argc==3?std::stoi(argv[2]):16);
 const auto start=std::chrono::steady_clock::now();const auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
 AmrMeshHeader header;std::ifstream hs(root/"header.bin",std::ios::binary);hs.read(reinterpret_cast<char*>(&header),sizeof(header));
 if(!hs||hs.peek()!=std::ifstream::traits_type::eof()||std::string(header.magic,7)!="QCFDAM1"||header.version!=1||header.endian!=0x01020304)throw std::runtime_error("invalid AMR header");
 size_t point_count=0,triangle_count=0;std::ifstream(root/"wall_counts.txt")>>point_count>>triangle_count;
 if(!point_count||!triangle_count)throw std::runtime_error("invalid wall counts");
 const auto points=read_exact<V>(root/"wall_points.bin",point_count);
 const auto ids=read_exact<std::array<int32_t,3>>(root/"wall_triangles.bin",triangle_count);
 const auto controls=read_exact<AmrControl>(root/"controls.bin",header.control_count);
 BVH bvh;bvh.triangles.resize(triangle_count);
 for(size_t i=0;i<triangle_count;++i){auto& t=bvh.triangles[i];t.a=points[ids[i][0]];t.b=points[ids[i][1]];t.c=points[ids[i][2]];for(int j=0;j<3;++j){t.lo[j]=std::min({t.a[j],t.b[j],t.c[j]});t.hi[j]=std::max({t.a[j],t.b[j],t.c[j]});t.center[j]=(t.a[j]+t.b[j]+t.c[j])/3;}}
 bvh.indices.resize(triangle_count);std::iota(bvh.indices.begin(),bvh.indices.end(),0);bvh.build(0,triangle_count);
 std::cout<<"BVH nodes="<<bvh.nodes.size()<<" controls="<<controls.size()<<" build_seconds="<<elapsed()<<'\n'<<std::flush;
 std::vector<double> distance(controls.size());double minimum=1e100,maximum=0;int bad=0;
 #pragma omp parallel for reduction(min:minimum) reduction(max:maximum) reduction(|:bad) schedule(dynamic,1024)
 for(size_t i=0;i<controls.size();++i){const V point={controls[i].x[0],controls[i].x[1],controls[i].x[2]};const double value=std::sqrt(bvh.nearest(point,0,1e300));distance[i]=value;if(!(value>0)||!std::isfinite(value))bad|=1;minimum=std::min(minimum,value);maximum=std::max(maximum,value);}
 if(bad)throw std::runtime_error("non-positive or non-finite wall distance");
 std::ofstream output(root/"wall_distance.bin",std::ios::binary);output.write(reinterpret_cast<const char*>(distance.data()),static_cast<std::streamsize>(distance.size()*sizeof(double)));output.close();if(!output)throw std::runtime_error("wall-distance write failed");
 std::ofstream report(root/"wall_distance.json");report<<std::setprecision(17)<<"{\"method\":\"exact nearest triangulated wall via AABB BVH\",\"controls\":"<<controls.size()<<",\"triangles\":"<<triangle_count<<",\"minimum\":"<<minimum<<",\"maximum\":"<<maximum<<",\"seconds\":"<<elapsed()<<"}\n";
 std::cout<<std::setprecision(17)<<"minimum="<<minimum<<" maximum="<<maximum<<" seconds="<<elapsed()<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<"ERROR: "<<error.what()<<'\n';return 2;}
