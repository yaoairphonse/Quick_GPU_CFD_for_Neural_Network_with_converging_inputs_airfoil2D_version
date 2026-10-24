#ifndef CFD_STATIC_AMR2D_GPU_H
#define CFD_STATIC_AMR2D_GPU_H
#include "static_amr2d.h"
#include <cstdint>

namespace cfd {
struct StaticAmrGpu;
struct AmrGpuOptions {int batch_trials=32;bool graph=true;bool fused=false;bool event_timing=false;bool sa_fused=false;};
struct AmrGpuStats {
  uint64_t allocated_bytes=0,pinned_bytes=0,geometry_h2d_bytes=0,state_h2d_bytes=0;
  uint64_t control_h2d_bytes=0;
  uint64_t control_d2h_bytes=0,state_d2h_bytes=0,status_polls=0,full_downloads=0;
  double setup_seconds=0,advance_seconds=0,download_seconds=0;
  double advance_event_seconds=0;
  uint64_t event_batches=0,graph_nodes_per_trial=0;
};
int static_amr_gpu_create(const StaticAmrMesh2D&,const Params2D&,const StaticAmrState2D&,
                          const AmrGpuOptions&,StaticAmrGpu**,std::string*);
// advance updates host metadata, not host U; call download when a full snapshot is needed.
// Returns 2 on optional wall-clock budget expiry, preserving a resumable accepted state.
int static_amr_gpu_advance(StaticAmrGpu*,double tend,long maxsteps,StaticAmrState2D*,std::string*,double wall_budget_seconds=0);
int static_amr_gpu_download(StaticAmrGpu*,StaticAmrState2D*,std::string*);
int static_amr_gpu_set_cfl(StaticAmrGpu*,double,std::string*);
AmrGpuStats static_amr_gpu_stats(const StaticAmrGpu*);
void static_amr_gpu_destroy(StaticAmrGpu*);
}
#endif
