// limiter.h -- 斜率與限制器（1D 與 2D 共用的單一來源）。
//
// 對一條格線上的三個連續格心值 (ql, qc, qr) 與其間距，回傳限制後的斜率。
// 2D 求解器在 x 與 y 兩個方向各自呼叫一次（逐方向限制）。
#ifndef CFD_LIMITER_H
#define CFD_LIMITER_H

#include <algorithm>
#include <cmath>

#include "types.h"

namespace cfd {

inline double minmod2(double a, double b) {
  if (a * b <= 0.0) return 0.0;
  return (std::fabs(a) < std::fabs(b)) ? a : b;
}

// Venkatakrishnan (1993) 的平滑限制函數。
//   d2 = 面外插值與格心值的差（未限制），d1 = 鄰居極值與格心值的差（同號）。
//   回傳 phi（可能略大於 1，呼叫端取 min）；eps2 = (K*h)^3。
inline double venkat_phi(double d1, double d2, double eps2) {
  if (std::fabs(d2) < 1e-300) return 1.0;
  const double num = (d1 * d1 + eps2) * d2 + 2.0 * d2 * d2 * d1;
  const double den = d2 * (d1 * d1 + 2.0 * d2 * d2 + d1 * d2 + eps2);
  return num / den;
}

// 非均勻網格上的二階中央斜率（格心距 hl = xc - xl，hr = xr - xc）：
//   s = ( hl^2 (qr - qc) + hr^2 (qc - ql) ) / ( hl*hr*(hl+hr) )
inline double central_slope(double ql, double qc, double qr, double hl,
                            double hr) {
  return (hl * hl * (qr - qc) + hr * hr * (qc - ql)) / (hl * hr * (hl + hr));
}

// 限制後的斜率。h 是中間格的格寬（用於 Venkat 的 eps^2 = (K h)^3）。
inline double limited_slope(double ql, double qc, double qr, double hl,
                            double hr, double h, int limiter, double K) {
  if (limiter == LIM_FIRST_ORDER) return 0.0;
  if (limiter == LIM_NONE) return central_slope(ql, qc, qr, hl, hr);
  if (limiter == LIM_MINMOD) return minmod2((qc - ql) / hl, (qr - qc) / hr);

  // LIM_VENKAT
  const double s0 = central_slope(ql, qc, qr, hl, hr);
  const double qmax = std::max(qc, std::max(ql, qr));
  const double qmin = std::min(qc, std::min(ql, qr));
  const double eps2 = std::pow(K * h, 3.0);
  double phi = 1.0;
  const double d2R = +0.5 * h * s0;  // 右面的外插量
  const double d2L = -0.5 * h * s0;  // 左面的外插量
  if (d2R > 0.0)
    phi = std::min(phi, venkat_phi(qmax - qc, d2R, eps2));
  else if (d2R < 0.0)
    phi = std::min(phi, venkat_phi(qmin - qc, d2R, eps2));
  if (d2L > 0.0)
    phi = std::min(phi, venkat_phi(qmax - qc, d2L, eps2));
  else if (d2L < 0.0)
    phi = std::min(phi, venkat_phi(qmin - qc, d2L, eps2));
  return phi * s0;
}

}  // namespace cfd

#endif  // CFD_LIMITER_H
