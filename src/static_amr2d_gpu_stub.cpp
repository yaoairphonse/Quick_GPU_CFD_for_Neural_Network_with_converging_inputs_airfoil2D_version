#include "static_amr2d_gpu.h"
#ifndef CFD_HAVE_CUDA
namespace cfd {
int static_amr_gpu_create(const StaticAmrMesh2D&,const Params2D&,const StaticAmrState2D&,
                          const AmrGpuOptions&,StaticAmrGpu** session,std::string* error) {
  if(session)*session=nullptr;
  if(error)*error="AMR CUDA backend was not compiled";
  return 1;
}
int static_amr_gpu_advance(StaticAmrGpu*,double,long,StaticAmrState2D*,std::string* error,double){if(error)*error="AMR CUDA backend was not compiled";return 1;}
int static_amr_gpu_download(StaticAmrGpu*,StaticAmrState2D*,std::string* error){if(error)*error="AMR CUDA backend was not compiled";return 1;}
AmrGpuStats static_amr_gpu_stats(const StaticAmrGpu*){return {};}
int static_amr_gpu_set_cfl(StaticAmrGpu*,double,std::string* error){if(error)*error="AMR CUDA backend was not compiled";return 1;}
void static_amr_gpu_destroy(StaticAmrGpu*){}
}
#endif
