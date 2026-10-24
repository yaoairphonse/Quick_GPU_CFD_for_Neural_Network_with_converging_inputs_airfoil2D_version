// Native geometry format shared by the preprocessor and CPU/CUDA solver.
#ifndef QUICK_CFD_LOCAL_MESH_FORMAT_H
#define QUICK_CFD_LOCAL_MESH_FORMAT_H
#include <cstdint>
struct Cell {double x[3],volume,dwall;int first,count,term_first,term_count;};
struct Face {int left,right,bc;double x[3],n[3],area,distance;};
struct Term {int source,face;double coefficient[3];};
struct MeshHeader {char magic[8]={'Q','C','F','D','3','D','1','\0'};uint64_t n,nf,ni,nt;};
struct ConnectivityBoundary {int32_t node[3],bc,patch;};
struct ConnectivityHeader {
 char magic[8]={'Q','C','F','D','N','C','1','\0'};
 uint32_t version=1,endian=0x01020304;
 uint64_t point_count=0,cell_count=0,boundary_face_count=0,incidence_count=0;
 uint64_t points_offset=0,cell_nodes_offset=0,node_offsets_offset=0,node_cells_offset=0,boundary_offset=0,file_bytes=0;
};
struct NodeGradientHeader {
 char magic[8]={'Q','C','F','D','N','G','1','\0'};
 uint32_t version=2,endian=0x01020304;
 uint64_t point_count=0,cell_count=0,face_count=0,incidence_count=0;
 uint64_t node_weights_offset=0,cell_gradients_offset=0,face_nodes_offset=0,file_bytes=0;
};
struct AgglomerationHeader {
 char magic[8]={'Q','C','F','D','M','G','1','\0'};
 uint32_t version=1,endian=0x01020304,level_count=0,merges_per_level=0;
 uint64_t fine_cell_count=0,file_bytes=0;
};
struct AgglomerationLevelHeader {uint64_t child_count=0,parent_count=0;};
static_assert(sizeof(ConnectivityBoundary)==20,"connectivity boundary layout changed");
static_assert(sizeof(ConnectivityHeader)==96,"connectivity header layout changed");
static_assert(sizeof(NodeGradientHeader)==80,"node-gradient header layout changed");
static_assert(sizeof(AgglomerationHeader)==40,"agglomeration header layout changed");
static_assert(sizeof(AgglomerationLevelHeader)==16,"agglomeration level header layout changed");
#endif
