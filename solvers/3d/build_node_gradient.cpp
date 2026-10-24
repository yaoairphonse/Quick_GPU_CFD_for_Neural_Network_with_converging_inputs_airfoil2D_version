// Build bounded pseudo-Laplacian nodal interpolation weights and exact
// tetrahedral shape gradients for the node-based Green--Gauss solver path.
#include "mesh_format.h"
#include "spatial_operators.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

template<class T>
void read_at(std::ifstream& input,uint64_t offset,std::vector<T>& values){
 input.seekg(static_cast<std::streamoff>(offset));
 input.read(reinterpret_cast<char*>(values.data()),static_cast<std::streamsize>(values.size()*sizeof(T)));
 if(!input)throw std::runtime_error("truncated input table");
}

int main(int argc,char** argv)try{
 if(argc<4)throw std::runtime_error("usage: build_node_gradient CONNECTIVITY MESH OUTPUT [threads] [report.json] [bounded|unbounded|hybrid]");
 const std::filesystem::path connectivity_path=argv[1],mesh_path=argv[2],output_path=argv[3];
 const int threads=argc>4?std::stoi(argv[4]):16;if(threads<1)throw std::runtime_error("threads must be positive");omp_set_num_threads(threads);
 const std::filesystem::path report_path=argc>5?std::filesystem::path(argv[5]):std::filesystem::path(output_path.string()+".json");
 const std::string weight_mode=argc>6?argv[6]:"hybrid";if(weight_mode!="bounded"&&weight_mode!="unbounded"&&weight_mode!="hybrid")throw std::runtime_error("weight mode must be bounded, unbounded, or hybrid");const bool limit_weights=weight_mode=="bounded";
 const auto start=std::chrono::steady_clock::now();const auto elapsed=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};

 std::ifstream connectivity(connectivity_path,std::ios::binary);ConnectivityHeader ch;connectivity.read(reinterpret_cast<char*>(&ch),sizeof(ch));
 if(!connectivity||std::string(ch.magic,7)!="QCFDNC1"||ch.version!=1||ch.endian!=0x01020304||ch.incidence_count!=4*ch.cell_count||std::filesystem::file_size(connectivity_path)!=ch.file_bytes)throw std::runtime_error("incompatible connectivity file");
 if(ch.cell_count>static_cast<uint64_t>(INT32_MAX)||ch.point_count>static_cast<uint64_t>(INT32_MAX))throw std::runtime_error("connectivity exceeds index limits");
 std::vector<double> points(3*ch.point_count);std::vector<int32_t> cell_nodes(4*ch.cell_count),node_cells(ch.incidence_count);std::vector<uint64_t> node_offsets(ch.point_count+1);std::vector<ConnectivityBoundary> boundary(ch.boundary_face_count);
 read_at(connectivity,ch.points_offset,points);read_at(connectivity,ch.cell_nodes_offset,cell_nodes);read_at(connectivity,ch.node_offsets_offset,node_offsets);read_at(connectivity,ch.node_cells_offset,node_cells);read_at(connectivity,ch.boundary_offset,boundary);connectivity.close();
 std::vector<uint8_t> symmetry_node(ch.point_count,0);for(const auto& face:boundary)if(face.bc==3)for(int node:face.node)symmetry_node[node]=1;boundary.clear();boundary.shrink_to_fit();

 std::ifstream mesh(mesh_path,std::ios::binary);MeshHeader mh;mesh.read(reinterpret_cast<char*>(&mh),sizeof(mh));
 if(!mesh||std::string(mh.magic,7)!="QCFD3D1"||mh.n!=ch.cell_count)throw std::runtime_error("mesh and connectivity cell tables do not match");
 std::vector<Cell> cells(mh.n);mesh.read(reinterpret_cast<char*>(cells.data()),static_cast<std::streamsize>(cells.size()*sizeof(Cell)));if(!mesh)throw std::runtime_error("truncated mesh cell table");
 std::vector<int32_t> face_nodes(3*mh.nf);const size_t face_chunk_size=250000;std::vector<Face> face_chunk(face_chunk_size);uint64_t invalid_faces=0;
 for(uint64_t first=0;first<mh.nf;first+=face_chunk_size){const size_t count=static_cast<size_t>(std::min<uint64_t>(face_chunk_size,mh.nf-first));mesh.read(reinterpret_cast<char*>(face_chunk.data()),static_cast<std::streamsize>(count*sizeof(Face)));if(!mesh)throw std::runtime_error("truncated mesh face table");
  #pragma omp parallel for reduction(+:invalid_faces) schedule(static)
  for(size_t local=0;local<count;++local){const Face& face=face_chunk[local];int32_t* result=face_nodes.data()+3*(first+local);if(face.left<0||static_cast<uint64_t>(face.left)>=mh.n){++invalid_faces;continue;}
   const int32_t* left=cell_nodes.data()+4*face.left;if(face.right>=0){if(static_cast<uint64_t>(face.right)>=mh.n){++invalid_faces;continue;}const int32_t* right=cell_nodes.data()+4*face.right;int found=0;for(int a=0;a<4;++a)for(int b=0;b<4;++b)if(left[a]==right[b]&&found<3)result[found++]=left[a];if(found!=3)++invalid_faces;}
   else{int best_omit=0;double best_error=1e300;for(int omit=0;omit<4;++omit){double center[3]={};for(int a=0;a<4;++a)if(a!=omit)for(int j=0;j<3;++j)center[j]+=points[3*left[a]+j]/3;double error=0;for(int j=0;j<3;++j){const double difference=center[j]-face.x[j];error+=difference*difference;}if(error<best_error){best_error=error;best_omit=omit;}}
    int found=0;for(int a=0;a<4;++a)if(a!=best_omit)result[found++]=left[a];if(!std::isfinite(best_error))++invalid_faces;}
  }
 }
 mesh.close();if(invalid_faces)throw std::runtime_error("could not recover all face-node triples");
 const double dimensional_epsilon=.9*std::min_element(cells.begin(),cells.end(),[](const Cell& left,const Cell& right){return left.dwall<right.dwall;})->dwall;

 std::vector<double> node_weights(ch.incidence_count),cell_gradients(12*ch.cell_count);
 uint64_t bounded_nodes=0,rank_deficient_nodes=0,dimension0=0,dimension1=0,dimension2=0,dimension3=0;double affine_moment_error=0,max_weight=-1e300,min_weight=1e300;
 #pragma omp parallel for reduction(+:bounded_nodes,rank_deficient_nodes,dimension0,dimension1,dimension2,dimension3) reduction(max:affine_moment_error,max_weight) reduction(min:min_weight) schedule(dynamic,2048)
 for(uint64_t node_index=0;node_index<ch.point_count;++node_index){
  const uint64_t first=node_offsets[node_index],last=node_offsets[node_index+1];const int count=static_cast<int>(last-first);std::vector<double> centers(3*count);
  for(int i=0;i<count;++i){const int cell=node_cells[first+i];if(cell<0||static_cast<uint64_t>(cell)>=ch.cell_count)throw std::runtime_error("node-to-cell index outside mesh");for(int j=0;j<3;++j)centers[3*i+j]=cells[cell].x[j];}
  bool bounded=false;int dimension=0;const int maximum_dimension=symmetry_node[node_index]?2:3;bool valid=qcfd::pseudo_laplacian_weights(centers.data(),count,points.data()+3*node_index,node_weights.data()+first,bounded,limit_weights,dimensional_epsilon,maximum_dimension,&dimension);
  if(weight_mode=="hybrid"){double largest=0;for(int i=0;i<count;++i)largest=std::max(largest,std::fabs(node_weights[first+i]));if(!valid||largest>2){bool clipped=false;valid=qcfd::pseudo_laplacian_weights(centers.data(),count,points.data()+3*node_index,node_weights.data()+first,clipped,true,dimensional_epsilon,maximum_dimension,&dimension);bounded=true;}}
  bounded_nodes+=bounded;rank_deficient_nodes+=!valid||dimension<maximum_dimension;dimension0+=dimension==0;dimension1+=dimension==1;dimension2+=dimension==2;dimension3+=dimension==3;double moment[3]={},radius=0;
  for(int i=0;i<count;++i){max_weight=std::max(max_weight,node_weights[first+i]);min_weight=std::min(min_weight,node_weights[first+i]);double distance2=0;for(int j=0;j<3;++j){const double d=centers[3*i+j]-points[3*node_index+j];moment[j]+=node_weights[first+i]*d;distance2+=d*d;}radius=std::max(radius,std::sqrt(distance2));}
  const double error=std::sqrt(qcfd::dot3(moment,moment))/std::max(radius,1e-300);affine_moment_error=std::max(affine_moment_error,error);
 }
 std::cout<<"Nodal weights completed at "<<elapsed()<<" s; bounded nodes="<<bounded_nodes<<"\n"<<std::flush;

 uint64_t invalid_tetrahedra=0;double volume_relative_error=0,max_shape_gradient=0;
 #pragma omp parallel for reduction(+:invalid_tetrahedra) reduction(max:volume_relative_error,max_shape_gradient) schedule(static)
 for(uint64_t cell=0;cell<ch.cell_count;++cell){
  double vertices[12];bool valid=true;for(int i=0;i<4;++i){const int node=cell_nodes[4*cell+i];if(node<0||static_cast<uint64_t>(node)>=ch.point_count){valid=false;break;}for(int j=0;j<3;++j)vertices[3*i+j]=points[3*node+j];}if(!valid){++invalid_tetrahedra;continue;}
  double* gradient=cell_gradients.data()+12*cell;if(!qcfd::tetra_shape_gradients(vertices,gradient)){++invalid_tetrahedra;continue;}
  double e1[3],e2[3],e3[3],cross[3];for(int j=0;j<3;++j){e1[j]=vertices[3+j]-vertices[j];e2[j]=vertices[6+j]-vertices[j];e3[j]=vertices[9+j]-vertices[j];}
  qcfd::cross3(e2,e3,cross);const double volume=std::fabs(qcfd::dot3(e1,cross))/6;
  volume_relative_error=std::max(volume_relative_error,std::fabs(volume-cells[cell].volume)/cells[cell].volume);
  for(int i=0;i<12;++i)max_shape_gradient=std::max(max_shape_gradient,std::fabs(gradient[i]));
 }
 if(invalid_tetrahedra||volume_relative_error>1e-8)throw std::runtime_error("tetrahedral shape-gradient audit failed");

 // End-to-end affine audit: interpolate the three cell-center coordinates to
 // nodes, apply the tetrahedral gradient, and compare with the identity.
 std::vector<double> interpolated_position(3*ch.point_count,0);
 #pragma omp parallel for schedule(static)
 for(uint64_t node=0;node<ch.point_count;++node)for(uint64_t index=node_offsets[node];index<node_offsets[node+1];++index){const int cell=node_cells[index];const double weight=node_weights[index];for(int j=0;j<3;++j)interpolated_position[3*node+j]+=weight*cells[cell].x[j];}
 uint64_t affine_cells_gt_1e3=0,affine_cells_gt_1e2=0,affine_cells_gt_1e1=0,nonsym_affine_cells_gt_1e3=0;double affine_cell_linf=0,affine_volume_error2=0,total_volume=0,nonsym_affine_cell_linf=0,nonsym_affine_volume_error2=0,nonsym_total_volume=0;
 #pragma omp parallel for reduction(+:affine_cells_gt_1e3,affine_cells_gt_1e2,affine_cells_gt_1e1,nonsym_affine_cells_gt_1e3,affine_volume_error2,total_volume,nonsym_affine_volume_error2,nonsym_total_volume) reduction(max:affine_cell_linf,nonsym_affine_cell_linf) schedule(static)
 for(uint64_t cell=0;cell<ch.cell_count;++cell){double error2=0;for(int variable=0;variable<3;++variable)for(int direction=0;direction<3;++direction){double recovered=0;for(int vertex=0;vertex<4;++vertex)recovered+=interpolated_position[3*cell_nodes[4*cell+vertex]+variable]*cell_gradients[12*cell+3*vertex+direction];const double error=recovered-(variable==direction?1.:0.);error2+=error*error;}
  const double error=std::sqrt(error2/3);affine_cell_linf=std::max(affine_cell_linf,error);affine_volume_error2+=cells[cell].volume*error2/3;total_volume+=cells[cell].volume;affine_cells_gt_1e3+=error>1e-3;affine_cells_gt_1e2+=error>1e-2;affine_cells_gt_1e1+=error>1e-1;bool symmetry=false;for(int vertex=0;vertex<4;++vertex)symmetry=symmetry||symmetry_node[cell_nodes[4*cell+vertex]];if(!symmetry){nonsym_affine_cell_linf=std::max(nonsym_affine_cell_linf,error);nonsym_affine_volume_error2+=cells[cell].volume*error2/3;nonsym_total_volume+=cells[cell].volume;nonsym_affine_cells_gt_1e3+=error>1e-3;}}
 const double affine_volume_l2=std::sqrt(affine_volume_error2/total_volume),nonsym_affine_volume_l2=std::sqrt(nonsym_affine_volume_error2/nonsym_total_volume);

 NodeGradientHeader header;header.point_count=ch.point_count;header.cell_count=ch.cell_count;header.face_count=mh.nf;header.incidence_count=ch.incidence_count;
 header.node_weights_offset=sizeof(header);header.cell_gradients_offset=header.node_weights_offset+node_weights.size()*sizeof(double);header.face_nodes_offset=header.cell_gradients_offset+cell_gradients.size()*sizeof(double);header.file_bytes=header.face_nodes_offset+face_nodes.size()*sizeof(int32_t);
 std::filesystem::create_directories(output_path.parent_path());std::ofstream output(output_path,std::ios::binary);output.write(reinterpret_cast<const char*>(&header),sizeof(header));
 output.write(reinterpret_cast<const char*>(node_weights.data()),static_cast<std::streamsize>(node_weights.size()*sizeof(double)));
 output.write(reinterpret_cast<const char*>(cell_gradients.data()),static_cast<std::streamsize>(cell_gradients.size()*sizeof(double)));
 output.write(reinterpret_cast<const char*>(face_nodes.data()),static_cast<std::streamsize>(face_nodes.size()*sizeof(int32_t)));output.close();
 if(!output||std::filesystem::file_size(output_path)!=header.file_bytes)throw std::runtime_error("node-gradient table write failed");
 std::ofstream report(report_path);report<<std::setprecision(17)
  <<"{\"format\":\"QCFDNG1\",\"version\":2,\"points\":"<<ch.point_count<<",\"cells\":"<<ch.cell_count<<",\"faces\":"<<mh.nf<<",\"incidences\":"<<ch.incidence_count
  <<",\"bounded_nodes\":"<<bounded_nodes<<",\"rank_deficient_nodes\":"<<rank_deficient_nodes<<",\"adaptive_dimension_counts\":["<<dimension0<<','<<dimension1<<','<<dimension2<<','<<dimension3<<"],\"dimensional_epsilon\":"<<dimensional_epsilon<<",\"affine_moment_relative_linf\":"<<affine_moment_error
  <<",\"affine_cell_gradient_linf\":"<<affine_cell_linf<<",\"affine_cell_gradient_volume_L2\":"<<affine_volume_l2<<",\"affine_cells_error_gt_1e-3\":"<<affine_cells_gt_1e3<<",\"affine_cells_error_gt_1e-2\":"<<affine_cells_gt_1e2<<",\"affine_cells_error_gt_1e-1\":"<<affine_cells_gt_1e1
  <<",\"nonsym_affine_cell_gradient_linf\":"<<nonsym_affine_cell_linf<<",\"nonsym_affine_cell_gradient_volume_L2\":"<<nonsym_affine_volume_l2<<",\"nonsym_affine_cells_error_gt_1e-3\":"<<nonsym_affine_cells_gt_1e3
  <<",\"minimum_normalized_weight\":"<<min_weight<<",\"maximum_normalized_weight\":"<<max_weight<<",\"tetra_volume_relative_linf\":"<<volume_relative_error<<",\"maximum_shape_gradient\":"<<max_shape_gradient
  <<",\"file_bytes\":"<<header.file_bytes<<",\"build_seconds\":"<<elapsed()<<",\"method\":\""<<weight_mode<<" pseudo-Laplacian nodal interpolation plus exact tetrahedral shape gradients\"}\n";report.close();
 if(!report)throw std::runtime_error("node-gradient report write failed");
 std::cout<<"Wrote "<<header.file_bytes<<" bytes in "<<elapsed()<<" s\n";return 0;
}catch(const std::exception& error){std::cerr<<"ERROR: "<<error.what()<<'\n';return 2;}
