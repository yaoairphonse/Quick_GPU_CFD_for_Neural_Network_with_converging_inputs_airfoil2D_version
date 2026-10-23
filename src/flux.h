// flux.h -- Euler 方程的數值通量（純函式，沒有狀態）。
//
// 守恆變數（單位體積）：rho, rho*un, rho*ut, E，其中 E = p/(gamma-1) + rho*(un^2+ut^2)/2。
//   un = 面法向速度，ut = 面切向速度（被動量，只被對流）。
// 通量核心 flux_hllc_t / flux_ausmpup_t 吃左右兩側的 (rho, un, ut, p)，
// 吐出界面通量 F[4] = (質量, 法向動量, 切向動量, 能量)。
// 1D 版 flux_hllc / flux_ausmpup 就是 ut = 0 的特例（輸出只取 3 個分量），
// 所以 1D 與 2D 只有一份通量程式碼；2D 沿一軸的問題必須與 1D 逐位元相同。
// 全部無因次；左右兩側必須用同一組參考量。
#ifndef CFD_FLUX_H
#define CFD_FLUX_H

#include <algorithm>
#include <cmath>

namespace cfd {

// 精確的 Euler 通量 F(U)（含切向速度）。用來做一致性檢查（F(U,U) 必須等於它）。
inline void flux_euler_exact_t(double rho, double un, double ut, double p,
                               double gamma, double F[4]) {
  const double E = p / (gamma - 1.0) + 0.5 * rho * un * un + 0.5 * rho * ut * ut;
  F[0] = rho * un;
  F[1] = rho * un * un + p;
  F[2] = rho * un * ut;
  F[3] = un * (E + p);
}

inline void flux_euler_exact(double rho, double u, double p, double gamma,
                             double F[3]) {
  double G[4];
  flux_euler_exact_t(rho, u, 0.0, p, gamma, G);
  F[0] = G[0];
  F[1] = G[1];
  F[2] = G[3];
}

// ---------------------------------------------------------------------------
// HLLC（Toro, Spruce, Speares 1994）。波速估計用 Einfeldt（Roe 平均）。
// 能精確解析接觸間斷與靜止的接觸；在強稀疏波附近比 AUSM 型更保守。
// 切向速度是被動量：中間區域的切向速度等於各側的切向速度。
// ---------------------------------------------------------------------------
inline void flux_hllc_t(double rhoL, double uL, double vL, double pL,
                        double rhoR, double uR, double vR, double pR,
                        double gamma, double F[4]) {
  const double aL = std::sqrt(gamma * pL / rhoL);
  const double aR = std::sqrt(gamma * pR / rhoR);
  const double EL = pL / (gamma - 1.0) + 0.5 * rhoL * uL * uL + 0.5 * rhoL * vL * vL;
  const double ER = pR / (gamma - 1.0) + 0.5 * rhoR * uR * uR + 0.5 * rhoR * vR * vR;

  // Roe 平均
  const double sL = std::sqrt(rhoL), sR = std::sqrt(rhoR);
  const double HL = (EL + pL) / rhoL, HR = (ER + pR) / rhoR;
  const double uT = (sL * uL + sR * uR) / (sL + sR);
  const double vT = (sL * vL + sR * vR) / (sL + sR);
  const double HT = (sL * HL + sR * HR) / (sL + sR);
  const double aT = std::sqrt(
      std::max((gamma - 1.0) * (HT - 0.5 * uT * uT - 0.5 * vT * vT), 1e-300));

  const double SL = std::min(uL - aL, uT - aT);
  const double SR = std::max(uR + aR, uT + aT);

  double FL[4], FR[4];
  flux_euler_exact_t(rhoL, uL, vL, pL, gamma, FL);
  flux_euler_exact_t(rhoR, uR, vR, pR, gamma, FR);

  if (SL >= 0.0) {
    for (int k = 0; k < 4; ++k) F[k] = FL[k];
    return;
  }
  if (SR <= 0.0) {
    for (int k = 0; k < 4; ++k) F[k] = FR[k];
    return;
  }

  // 接觸波速 S*
  const double Sstar =
      (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) /
      (rhoL * (SL - uL) - rhoR * (SR - uR));

  if (Sstar >= 0.0) {
    const double c = rhoL * (SL - uL) / (SL - Sstar);
    const double Us0 = c;
    const double Us1 = c * Sstar;
    const double Us2 = c * vL;
    const double Us3 =
        c * (EL / rhoL + (Sstar - uL) * (Sstar + pL / (rhoL * (SL - uL))));
    F[0] = FL[0] + SL * (Us0 - rhoL);
    F[1] = FL[1] + SL * (Us1 - rhoL * uL);
    F[2] = FL[2] + SL * (Us2 - rhoL * vL);
    F[3] = FL[3] + SL * (Us3 - EL);
  } else {
    const double c = rhoR * (SR - uR) / (SR - Sstar);
    const double Us0 = c;
    const double Us1 = c * Sstar;
    const double Us2 = c * vR;
    const double Us3 =
        c * (ER / rhoR + (Sstar - uR) * (Sstar + pR / (rhoR * (SR - uR))));
    F[0] = FR[0] + SR * (Us0 - rhoR);
    F[1] = FR[1] + SR * (Us1 - rhoR * uR);
    F[2] = FR[2] + SR * (Us2 - rhoR * vR);
    F[3] = FR[3] + SR * (Us3 - ER);
  }
}

inline void flux_hllc(double rhoL, double uL, double pL, double rhoR, double uR,
                      double pR, double gamma, double F[3]) {
  double G[4];
  flux_hllc_t(rhoL, uL, 0.0, pL, rhoR, uR, 0.0, pR, gamma, G);
  F[0] = G[0];
  F[1] = G[1];
  F[2] = G[3];
}

// ---------------------------------------------------------------------------
// AUSM+-up（Liou 2006, JCP 214:137-170）。
// 把界面通量拆成「對流（質量通量）」與「壓力」兩部分，各自用 Mach 數分裂。
// 參數取論文建議值：Kp=0.25, Ku=0.75, sigma=1, beta=1/8, alpha 由 fa 決定。
// Mco 是截斷 Mach 數 = 該問題的參考（自由流）Mach 數，不是調參數：
// 可壓縮跨音速/超音速問題取 1；取小值會讓壓力擴散項放大 1/fa 倍。
// ---------------------------------------------------------------------------
namespace ausm_detail {
inline double M1(double M, int s) { return 0.5 * (M + s * std::fabs(M)); }
inline double M2(double M, int s) { return s * 0.25 * (M + s) * (M + s); }
inline double M4(double M, int s, double beta) {
  if (std::fabs(M) >= 1.0) return M1(M, s);
  return M2(M, s) * (1.0 - s * 16.0 * beta * M2(M, -s));
}
inline double P5(double M, int s, double alpha) {
  if (std::fabs(M) >= 1.0)
    return 0.5 * (1.0 + s * (M > 0.0 ? 1.0 : (M < 0.0 ? -1.0 : 0.0)));
  return M2(M, s) * ((s * 2.0 - M) - s * 16.0 * alpha * M * M2(M, -s));
}
}  // namespace ausm_detail

inline void flux_ausmpup_t(double rhoL, double uL, double vL, double pL,
                           double rhoR, double uR, double vR, double pR,
                           double gamma, double Mco, double F[4]) {
  using namespace ausm_detail;
  const double Kp = 0.25, Ku = 0.75, sigma = 1.0, beta = 0.125;

  const double aL = std::sqrt(gamma * pL / rhoL);
  const double aR = std::sqrt(gamma * pR / rhoR);
  const double a12 = 0.5 * (aL + aR);
  const double rho12 = 0.5 * (rhoL + rhoR);

  const double ML = uL / a12, MR = uR / a12;
  const double Mbar2 = 0.5 * (uL * uL + uR * uR) / (a12 * a12);
  const double M02 = std::min(1.0, std::max(Mbar2, Mco * Mco));
  const double M0 = std::sqrt(M02);
  const double fa = M0 * (2.0 - M0);
  const double alpha = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa);

  // 界面 Mach 數（含壓力擴散項）
  const double M12 = M4(ML, +1, beta) + M4(MR, -1, beta) -
                     (Kp / fa) * std::max(1.0 - sigma * Mbar2, 0.0) * (pR - pL) /
                         (rho12 * a12 * a12);

  // 界面壓力（含速度擴散項）
  const double Pp = P5(ML, +1, alpha), Pm = P5(MR, -1, alpha);
  const double p12 = Pp * pL + Pm * pR -
                     Ku * Pp * Pm * (rhoL + rhoR) * (fa * a12) * (uR - uL);

  // 質量通量，依 M12 的正負迎風
  const double mdot = a12 * M12 * (M12 > 0.0 ? rhoL : rhoR);
  const double rho = (M12 > 0.0) ? rhoL : rhoR;
  const double u = (M12 > 0.0) ? uL : uR;
  const double v = (M12 > 0.0) ? vL : vR;
  const double p = (M12 > 0.0) ? pL : pR;
  const double E = p / (gamma - 1.0) + 0.5 * rho * u * u + 0.5 * rho * v * v;
  const double H = (E + p) / rho;

  F[0] = mdot;
  F[1] = mdot * u + p12;
  F[2] = mdot * v;
  F[3] = mdot * H;
}

inline void flux_ausmpup(double rhoL, double uL, double pL, double rhoR,
                         double uR, double pR, double gamma, double Mco,
                         double F[3]) {
  double G[4];
  flux_ausmpup_t(rhoL, uL, 0.0, pL, rhoR, uR, 0.0, pR, gamma, Mco, G);
  F[0] = G[0];
  F[1] = G[1];
  F[2] = G[3];
}

}  // namespace cfd

#endif  // CFD_FLUX_H
