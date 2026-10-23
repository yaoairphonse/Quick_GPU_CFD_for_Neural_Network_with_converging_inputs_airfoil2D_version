#ifndef CFD_STATIC_AMR2D_SA_H
#define CFD_STATIC_AMR2D_SA_H
#include "static_amr2d.h"
#include "sa_model.h"
namespace cfd {
struct SaFields2D {std::vector<double> gradient,density_gradient,diffusion,source,source_rate,face_flux,wall_flux,residual,cell_dt;};
int static_amr2d_prepare_sa(StaticAmrMesh2D*,const Params2D&,std::string*);
int static_amr2d_sa_evaluate(const StaticAmrMesh2D&,const Params2D&,const std::vector<double>& conserved,
 const std::vector<double>& primitive,const AmrViscousFields2D&,const std::vector<double>& mean_face_flux,SaFields2D*);
}
#endif
