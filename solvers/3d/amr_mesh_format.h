// Factorized Cartesian-AMR cut-cell geometry used by the procedural backend.
#ifndef QUICK_CFD_LOCAL_AMR_MESH_FORMAT_H
#define QUICK_CFD_LOCAL_AMR_MESH_FORMAT_H
#include <cstdint>

struct AmrMeshHeader {
 char magic[8]={'Q','C','F','D','A','M','1','\0'};
 uint32_t version=1,endian=0x01020304,block_cells=8,reserved=0;
 uint64_t block_count=0,procedural_cell_count=0,control_count=0;
 uint64_t interface_count=0,boundary_side_count=0,cut_face_count=0,wall_face_count=0;
 double root_low[3]={},root_size=0;
};

struct AmrBlock {
 double low[3],spacing;
 int32_t fine_coordinate[3],fine_span,first_procedural;
 int16_t level;
 uint8_t kind,padding;
};

struct AmrBlockInterface {
 int32_t first_block,second_block;
 uint8_t first_side;
 int8_t level_difference;
 uint8_t padding[2];
};

struct AmrBoundarySide {
 int32_t block;
 uint8_t side,padding[3];
};

struct AmrControl {
 double x[3],volume,dwall;
};

static_assert(sizeof(AmrMeshHeader)==112,"AMR header layout changed");
static_assert(sizeof(AmrBlock)==56,"AMR block layout changed");
static_assert(sizeof(AmrBlockInterface)==12,"AMR interface layout changed");
static_assert(sizeof(AmrBoundarySide)==8,"AMR boundary layout changed");
static_assert(sizeof(AmrControl)==40,"AMR control layout changed");
#endif
