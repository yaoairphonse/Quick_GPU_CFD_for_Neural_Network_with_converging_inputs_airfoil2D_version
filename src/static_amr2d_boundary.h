#ifndef CFD_STATIC_AMR2D_BOUNDARY_H
#define CFD_STATIC_AMR2D_BOUNDARY_H
#include <cmath>

#ifdef __CUDACC__
#define AMR_BC_HD __host__ __device__
#else
#define AMR_BC_HD
#endif
namespace cfd {
// -1 inherits the existing Params2D bc; the legacy solver is unchanged.
enum AmrBoundaryKind { AMR_BC_INHERIT=-1, AMR_BC_TRANSMISSIVE=1, AMR_BC_REFLECT=2,
                       AMR_BC_FARFIELD=3, AMR_BC_TOTAL_INLET=4, AMR_BC_PRESSURE_OUTLET=5, AMR_BC_NOSLIP=6 };
struct AmrBoundary2D {
  int kind=AMR_BC_INHERIT;
  double reference[4]={1,1.8,0,1.0/1.4}; // rho,u,v,p; also the outlet backflow reservoir
  double total_pressure=0,total_temperature=0,gas_constant=1;
  double direction[2]={1,0};
  double outlet_pressure=0;
  double wall_velocity[2]={0,0};
};
AMR_BC_HD inline bool amr_bc_finite(double x) {
#ifdef __CUDA_ARCH__
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}
AMR_BC_HD inline bool amr_bc_primitive_valid(const double* w) {
  return amr_bc_finite(w[0])&&w[0]>0&&amr_bc_finite(w[1])&&amr_bc_finite(w[2])&&amr_bc_finite(w[3])&&w[3]>0;
}
AMR_BC_HD inline bool amr_bc_valid(const AmrBoundary2D& b,int legacy,int side) {
  if(b.kind<AMR_BC_INHERIT)return false;
  const int kind=b.kind<0?legacy:b.kind;
  if(kind==AMR_BC_TRANSMISSIVE||kind==AMR_BC_REFLECT)return true;
  if(kind==AMR_BC_NOSLIP)return amr_bc_finite(b.wall_velocity[0])&&amr_bc_finite(b.wall_velocity[1])&&b.wall_velocity[side/2]==0;
  if(kind<AMR_BC_FARFIELD||kind>AMR_BC_PRESSURE_OUTLET||!amr_bc_primitive_valid(b.reference))return false;
  if(kind==AMR_BC_PRESSURE_OUTLET)return amr_bc_finite(b.outlet_pressure)&&b.outlet_pressure>0;
  if(kind==AMR_BC_TOTAL_INLET) {
    const double d2=b.direction[0]*b.direction[0]+b.direction[1]*b.direction[1];
    const double dn=(side%2?1.0:-1.0)*b.direction[side/2];
    return amr_bc_finite(d2)&&fabs(d2-1)<1e-12&&dn<0&&
      amr_bc_finite(b.total_pressure)&&b.total_pressure>0&&amr_bc_finite(b.total_temperature)&&b.total_temperature>0&&
      amr_bc_finite(b.gas_constant)&&b.gas_constant>0;
  }
  return true;
}
// Reconstruct an exterior Riemann state. All velocities are in global coordinates.
// false means a physical/configuration failure; no state clipping or fallback.
AMR_BC_HD inline bool amr_bc_state(const AmrBoundary2D& b,int legacy,double gamma,
                                  int side,const double* wi,double* wb) {
  const int axis=side/2,normal=axis+1,tangent=2-axis;
  const double sign=side%2?1.0:-1.0,gm=gamma-1;
  const int kind=b.kind<0?legacy:b.kind;
  for(int v=0;v<4;++v)wb[v]=wi[v];
  if(kind==AMR_BC_TRANSMISSIVE)return true;
  if(kind==AMR_BC_NOSLIP){wb[1]=2*b.wall_velocity[0]-wi[1];wb[2]=2*b.wall_velocity[1]-wi[2];return true;}
  if(kind==AMR_BC_REFLECT){wb[normal]=-wi[normal];return true;}
  if(!amr_bc_primitive_valid(wi)||!(gamma>1))return false;
  const double ai=sqrt(gamma*wi[3]/wi[0]),un=sign*wi[normal];
  if(kind==AMR_BC_TOTAL_INLET) {
    if(!(un<0&&un>-ai))return false; // requires one outgoing acoustic characteristic
    const double dn=sign*b.direction[axis],A=gamma*b.gas_constant*b.total_temperature;
    if(!(dn<0)||!amr_bc_finite(A))return false;
    const double rp=un+2*ai/gm,upper=sqrt(2*A/(gm+2*dn*dn));
    const double f0=2*sqrt(A)/gm-rp;
    const double f1=dn*upper+2*sqrt(A-.5*gm*upper*upper)/gm-rp;
    if(!(f0>=0&&f1<=0))return false;
    double lo=0,hi=upper;
    for(int it=0;it<64;++it){const double v=.5*(lo+hi),a=sqrt(A-.5*gm*v*v);
      if(dn*v+2*a/gm>rp)lo=v;else hi=v;
    }
    const double v=.5*(lo+hi),temp=(A-.5*gm*v*v)/(gamma*b.gas_constant);
    const double pressure=b.total_pressure*pow(temp/b.total_temperature,gamma/gm);
    wb[0]=pressure/(b.gas_constant*temp);wb[1]=v*b.direction[0];wb[2]=v*b.direction[1];wb[3]=pressure;
    return amr_bc_primitive_valid(wb);
  }
  if(kind==AMR_BC_PRESSURE_OUTLET&&un>=0) {
    if(un>=ai)return true; // all characteristics leave; do not impose p_out
    const double density=wi[0]*pow(b.outlet_pressure/wi[3],1/gamma);
    const double ab=sqrt(gamma*b.outlet_pressure/density),ub=un+2*(ai-ab)/gm;
    if(ub>=0) {
      wb[0]=density;wb[normal]=sign*ub;wb[3]=b.outlet_pressure;
      return amr_bc_primitive_valid(wb);
    } // impending reversal uses the explicit backflow reference below
  }
  if(kind!=AMR_BC_FARFIELD&&kind!=AMR_BC_PRESSURE_OUTLET)return false;
  if(un>=ai)return true;
  if(un<=-ai){for(int v=0;v<4;++v)wb[v]=b.reference[v];return amr_bc_primitive_valid(wb);}
  const double* wr=b.reference;
  const double ar=sqrt(gamma*wr[3]/wr[0]);
  const double rp=un+2*ai/gm,rm=sign*wr[normal]-2*ar/gm;
  const double ab=.25*gm*(rp-rm),ub=.5*(rp+rm);
  if(!(ab>0)||!amr_bc_finite(ab))return false;
  // Select scalars explicitly across CUDA local and parameter address spaces.
  const double advrho=un>=0?wi[0]:wr[0],advp=un>=0?wi[3]:wr[3];
  const double advt=un>=0?wi[tangent]:wr[tangent];
  const double entropy=advp/pow(advrho,gamma);
  wb[0]=pow(ab*ab/(gamma*entropy),1/gm);
  wb[3]=entropy*pow(wb[0],gamma);wb[normal]=sign*ub;wb[tangent]=advt;
  return amr_bc_primitive_valid(wb);
}
} // namespace cfd
#undef AMR_BC_HD
#endif
