#include "low_mach.h"
#include <algorithm>
#include <cmath>
#include <iostream>

int main() {
  constexpr double gamma=1.4;
  const double normal[3]={1,0,0};
  const double left[6]={1,.01,0,0,1/gamma,0};
  const double fast_right[6]={1,.8,0,0,1/gamma,0};
  const double fast_bound=cfd::lm_spectral(fast_right[1],1,
      cfd::lm_beta2(.8*.8,1,.2,.02));
  const double fast_face=cfd::lm_face_speed_normal(left,fast_right,normal,3,4,gamma,.2,.02);
  if(!std::isfinite(fast_face)||fast_face<fast_bound) {
    std::cerr<<"right-state spectral radius was not bounded\n";
    return 1;
  }

  const double pressure_right[6]={1,.01,0,0,4/gamma,0};
  const double pressure_face=cfd::lm_face_speed_normal(left,pressure_right,normal,3,4,gamma,.2,.02);
  const double acoustic_bound=std::max(std::sqrt(gamma*left[4]/left[0]),
                                       std::sqrt(gamma*pressure_right[4]/pressure_right[0]));
  if(!std::isfinite(pressure_face)||pressure_face<acoustic_bound) {
    std::cerr<<"pressure-jump acoustic radius was not bounded\n";
    return 1;
  }

  const double reverse_normal[3]={-1,0,0};
  const double reverse=cfd::lm_face_speed_normal(pressure_right,left,reverse_normal,3,4,gamma,.2,.02);
  if(std::fabs(reverse-pressure_face)>1e-14*pressure_face) {
    std::cerr<<"face spectral radius depends on owner orientation\n";
    return 1;
  }
  return 0;
}
