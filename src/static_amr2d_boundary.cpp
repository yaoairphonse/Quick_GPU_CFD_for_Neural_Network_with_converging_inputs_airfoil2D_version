#include "static_amr2d.h"
#include <fstream>
#include <iomanip>

namespace cfd {
int static_amr2d_write_boundaries(const StaticAmrMesh2D& m,const Params2D& p,
                                  const std::string& prefix,std::string* error) {
  std::ofstream out(prefix+".bc.json");out<<std::setprecision(17);
  const char* names[]={"xmin","xmax","ymin","ymax"};
  const char* kinds[]={"unsupported","transmissive","reflect","farfield","total-inlet","pressure-outlet","no-slip"};
  out<<"{\"model\":\""<<(m.transport.model==AMR_MODEL_EULER?"perfect-gas Euler; slip":"laminar mechanical viscous; kappa=0; no-slip")<<"\",\"characteristic_classification\":\"interior outward normal velocity and sound speed\",\"boundaries\":[";
  for(int side=0;side<4;++side){const auto& b=m.boundaries[side];const int kind=b.kind<0?p.bc[side]:b.kind;
    if(!amr_bc_valid(b,p.bc[side],side)){if(error)*error="invalid boundary metadata";return 1;}
    if(side)out<<',';
    out<<"{\"id\":"<<side<<",\"marker\":\""<<names[side]<<"\",\"kind\":\""<<kinds[kind]<<"\",\"normal\":["
      <<(side<2?(side%2?1:-1):0)<<','<<(side>=2?(side%2?1:-1):0)<<"],\"reference_rho_u_v_p\":["
      <<b.reference[0]<<','<<b.reference[1]<<','<<b.reference[2]<<','<<b.reference[3]<<']';
    if(kind==AMR_BC_TOTAL_INLET)out<<",\"total_pressure\":"<<b.total_pressure<<",\"total_temperature\":"<<b.total_temperature<<",\"gas_constant\":"<<b.gas_constant<<",\"direction\":["<<b.direction[0]<<','<<b.direction[1]<<']';
    if(kind==AMR_BC_PRESSURE_OUTLET)out<<",\"static_pressure\":"<<b.outlet_pressure<<",\"backflow\":\"characteristic reference; fixed pressure not enforced during reversal\"";
    if(kind==AMR_BC_NOSLIP)out<<",\"wall_velocity\":["<<b.wall_velocity[0]<<','<<b.wall_velocity[1]<<']';
    out<<'}';
  }
  out<<"],\"units\":\"consistent solver units; T and R must match p/rho=RT\"}\n";
  out.close();if(!out){if(error)*error="cannot write boundary metadata";return 1;}return static_amr2d_write_viscous(m,p,prefix,error);
}
}
