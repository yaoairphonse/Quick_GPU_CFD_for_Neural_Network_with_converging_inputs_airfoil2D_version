// types.h -- 1D 與 2D 求解器共用的選項列舉。
#ifndef CFD_TYPES_H
#define CFD_TYPES_H

namespace cfd {

enum FluxType { FLUX_HLLC = 0, FLUX_AUSMPUP = 1 };

enum LimiterType { LIM_NONE = 0, LIM_MINMOD = 1, LIM_VENKAT = 2, LIM_FIRST_ORDER = 3 };

// 邊界條件：
//   PERIODIC     週期
//   TRANSMISSIVE 穿透（零梯度外插），只適合出流或波還沒到邊界的情況
//   REFLECT      反射（滑移壁／對稱面）：鬼格子是內部格的鏡像，法向動量變號
enum BCType { BC_PERIODIC = 0, BC_TRANSMISSIVE = 1, BC_REFLECT = 2 };

}  // namespace cfd

#endif  // CFD_TYPES_H
