#ifndef CFD_STATIC_AMR2D_BINARY_H
#define CFD_STATIC_AMR2D_BINARY_H
#include "static_amr2d_gpu.h"
namespace cfd {
int static_amr_write_npy(const std::string& path,const double* values,size_t rows,size_t columns,std::string* error);
int static_amr_binary_mesh(const StaticAmrMesh2D&,const Params2D&,const std::string&,std::string*);
int static_amr_binary_snapshot(const StaticAmrMesh2D&,const Params2D&,const StaticAmrState2D&,
                               const std::string&,const char* backend,std::string*);
int static_amr_gpu_write_stats(const AmrGpuStats&,const std::string&,double io_seconds,std::string*);
}
#endif
