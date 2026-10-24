// Build a deterministic face-connected aggregation hierarchy for 3-D multigrid.
// The output stores parent maps only; coarse operators and FAS transfer kernels
// are separate solver work. Boundary classes are never merged with a different
// nonzero boundary class, while interior cells may join a boundary aggregate.
#include "mesh_format.h"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct MatchResult {
 std::vector<int32_t> parent;
 std::vector<uint8_t> boundary_mask;
 uint64_t pairs=0;
};

template<class Function>
void visit_faces(const std::filesystem::path& mesh_path,uint64_t face_offset,uint64_t face_count,Function function){
 std::ifstream input(mesh_path,std::ios::binary);if(!input)throw std::runtime_error("cannot open mesh");
 input.seekg(static_cast<std::streamoff>(face_offset));if(!input)throw std::runtime_error("cannot seek to face table");
 constexpr size_t chunk_size=1u<<18;std::vector<Face> chunk(chunk_size);uint64_t consumed=0;
 while(consumed<face_count){const size_t count=static_cast<size_t>(std::min<uint64_t>(chunk_size,face_count-consumed));
  input.read(reinterpret_cast<char*>(chunk.data()),static_cast<std::streamsize>(count*sizeof(Face)));if(!input)throw std::runtime_error("truncated face table");
  for(size_t i=0;i<count;++i)function(chunk[i]);consumed+=count;
 }
}

MatchResult match_once(const std::filesystem::path& mesh_path,uint64_t face_offset,uint64_t face_count,
                       const std::vector<int32_t>& fine_to_current,const std::vector<uint8_t>& current_mask){
 MatchResult result;result.parent.assign(current_mask.size(),-1);int32_t next=0;
 visit_faces(mesh_path,face_offset,face_count,[&](const Face& face){
  if(face.right<0)return;const int32_t a=fine_to_current[face.left],b=fine_to_current[face.right];if(a==b||result.parent[a]>=0||result.parent[b]>=0)return;
  const uint8_t ma=current_mask[a],mb=current_mask[b];if(ma&&mb&&ma!=mb)return;
  result.parent[a]=result.parent[b]=next++;++result.pairs;
 });
 for(int32_t& value:result.parent)if(value<0)value=next++;
 result.boundary_mask.assign(static_cast<size_t>(next),0);
 for(size_t i=0;i<result.parent.size();++i)result.boundary_mask[result.parent[i]]|=current_mask[i];
 return result;
}

int main(int argc,char** argv)try{
 if(argc<5||std::string(argv[1])!="--mesh"||std::string(argv[3])!="--out")throw std::runtime_error("usage: build_agglomeration --mesh mesh.bin --out hierarchy.bin [--levels 3] [--merges-per-level 2]");
 const std::filesystem::path mesh_path=argv[2],output_path=argv[4];int level_count=3,merges_per_level=2;
 for(int i=5;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("all options need values");const std::string key=argv[i];if(key=="--levels")level_count=std::stoi(argv[i+1]);else if(key=="--merges-per-level")merges_per_level=std::stoi(argv[i+1]);else throw std::runtime_error("unknown option: "+key);}
 if(level_count<2||level_count>8||merges_per_level<1||merges_per_level>4)throw std::runtime_error("invalid hierarchy dimensions");
 std::ifstream mesh(mesh_path,std::ios::binary);MeshHeader mesh_header;mesh.read(reinterpret_cast<char*>(&mesh_header),sizeof(mesh_header));
 if(!mesh||std::string(mesh_header.magic,7)!="QCFD3D1"||mesh_header.n>static_cast<uint64_t>(INT32_MAX))throw std::runtime_error("unsupported mesh header");
 const uint64_t face_offset=sizeof(MeshHeader)+mesh_header.n*sizeof(Cell);mesh.close();
 std::vector<uint8_t> boundary_mask(mesh_header.n,0);
 visit_faces(mesh_path,face_offset,mesh_header.nf,[&](const Face& face){if(face.right<0&&face.bc>0&&face.bc<8)boundary_mask[face.left]|=static_cast<uint8_t>(1u<<face.bc);});
 std::vector<int32_t> fine_to_current(mesh_header.n);std::iota(fine_to_current.begin(),fine_to_current.end(),0);
 std::vector<std::vector<int32_t>> level_maps;std::vector<uint64_t> counts{mesh_header.n};std::vector<uint8_t> current_mask=boundary_mask;
 for(int level=1;level<level_count;++level){
  const size_t child_count=current_mask.size();std::vector<int32_t> child_to_current(child_count);std::iota(child_to_current.begin(),child_to_current.end(),0);
  for(int pass=0;pass<merges_per_level;++pass){
   MatchResult match=match_once(mesh_path,face_offset,mesh_header.nf,fine_to_current,current_mask);
   for(int32_t& value:fine_to_current)value=match.parent[value];
   for(int32_t& value:child_to_current)value=match.parent[value];
   current_mask=std::move(match.boundary_mask);
   std::cout<<"level="<<level<<" pass="<<pass+1<<" pairs="<<match.pairs<<" aggregates="<<current_mask.size()<<'\n';
  }
  std::vector<uint64_t> children(current_mask.size(),0);for(int32_t parent:child_to_current)++children[parent];
  if(*std::min_element(children.begin(),children.end())==0)throw std::runtime_error("empty aggregate generated");
  level_maps.push_back(std::move(child_to_current));counts.push_back(current_mask.size());
 }
 AgglomerationHeader output_header;output_header.level_count=static_cast<uint32_t>(level_count);output_header.merges_per_level=static_cast<uint32_t>(merges_per_level);output_header.fine_cell_count=mesh_header.n;
 output_header.file_bytes=sizeof(output_header);for(size_t level=0;level<level_maps.size();++level)output_header.file_bytes+=sizeof(AgglomerationLevelHeader)+level_maps[level].size()*sizeof(int32_t);
 std::filesystem::create_directories(output_path.parent_path());std::ofstream output(output_path,std::ios::binary);if(!output)throw std::runtime_error("cannot create hierarchy output");
 output.write(reinterpret_cast<const char*>(&output_header),sizeof(output_header));
 for(size_t level=0;level<level_maps.size();++level){const AgglomerationLevelHeader header{counts[level],counts[level+1]};output.write(reinterpret_cast<const char*>(&header),sizeof(header));output.write(reinterpret_cast<const char*>(level_maps[level].data()),static_cast<std::streamsize>(level_maps[level].size()*sizeof(int32_t)));}
 output.close();if(!output||std::filesystem::file_size(output_path)!=output_header.file_bytes)throw std::runtime_error("incomplete hierarchy output");
 const auto manifest_path=output_path.string()+".json";std::ofstream manifest(manifest_path);manifest<<std::setprecision(17)<<"{\n  \"format\": \"QCFDMG1\",\n  \"version\": 1,\n  \"source_mesh\": \""<<mesh_path.string()<<"\",\n  \"levels\": "<<level_count<<",\n  \"merges_per_level\": "<<merges_per_level<<",\n  \"cell_counts\": [";
 for(size_t i=0;i<counts.size();++i)manifest<<(i?", ":"")<<counts[i];manifest<<"],\n  \"file_bytes\": "<<output_header.file_bytes<<",\n  \"boundary_policy\": \"different nonzero boundary-class masks are not merged\",\n  \"status\": \"validated parent hierarchy; coarse finite-volume operators and FAS cycles are not included\"\n}\n";
 std::cout<<"wrote="<<output_path<<" bytes="<<output_header.file_bytes<<" levels="<<level_count<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<"ERROR: "<<error.what()<<'\n';return 2;}
