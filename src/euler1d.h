// euler1d.h -- 1D 可壓縮 Euler 求解器（cell-centered 有限體積，守恆形式）。
//
// 這是第 1 週的實驗載體：先把「通量、重構、限制器、時間積分、批次維度」在 1D
// 驗證好，再搬到 2D。資料結構已經是最後的樣子：
//   U[(case * NVAR + var) * m + (i + ng)]，m = n + 2*ng，ng = 鬼格子層數。
// 每個 case 有自己的時間與時間步，所以「批次結果」必須與「單 case 結果」逐位元相同。
#ifndef CFD_EULER1D_H
#define CFD_EULER1D_H

#include "array.h"
#include "types.h"

namespace cfd {

enum { NVAR1D = 3 };  // rho, rho*u, rho*E

struct Params1D {
  double gamma;
  double cfl;
  int flux;       // FluxType
  int limiter;    // LimiterType
  double venkat_k;  // Venkatakrishnan 的 K；eps^2 = (K*dx)^3。K 越大越不限制
  double ausm_mco;  // AUSM+-up 的截斷 Mach 數
  int bc;         // BCType
  double dt_fixed;  // > 0 時以此值當時間步（測試用，例如與 2D 逐位元比對）；0 = 用 CFL
};

struct Grid1D {
  int n;    // 內部格數
  int ng;   // 鬼格子層數（目前固定 2）
  int m;    // n + 2*ng
  double xmin, xmax;
  Array xc;  // 格心座標（含鬼格子），長度 m
  Array dx;  // 格寬（含鬼格子），長度 m
};

struct Solver1D {
  Grid1D grid;
  Params1D par;
  int ncase;
  Array U, U1, U2;  // 守恆變數，三組給 SSP-RK3
  Array R;          // 殘差 dU/dt
  Array W;          // 原始變數 (rho,u,p)
  Array S;          // 限制後的斜率 (rho,u,p)
  Array t;          // 每個 case 的時間
  Array dt;         // 每個 case 這一步的時間步
  long nsteps;      // 已推進的步數（每個 case 的最大值）
};

Params1D params1d_default();

// 建立均勻網格上的求解器。ncase >= 1。
void solver1d_init(Solver1D* s, int n, double xmin, double xmax, int ncase,
                   const Params1D& par);

// 寫入某個 case 某格的原始變數 (rho, u, p)（內部格 i = 0..n-1）。
void solver1d_set_primitive(Solver1D* s, int c, int i, double rho, double u,
                            double p);

// 在 x = x0 處設定 Riemann 初值：x < x0 用左態，否則用右態。
void solver1d_set_riemann(Solver1D* s, int c, double x0, double rhoL,
                          double uL, double pL, double rhoR, double uR,
                          double pR);

// 推進到 tend（每個 case 各自推進，最後一步會縮短以剛好落在 tend）。
// 回傳 0 表示成功；非 0 表示超過 maxsteps 或出現非物理狀態。
int solver1d_run(Solver1D* s, double tend, long maxsteps);

// 讀出內部格的原始變數。
void solver1d_get_primitive(const Solver1D* s, int c, int i, double* rho,
                            double* u, double* p);

// 某個守恆量在內部格上的積分（sum U * dx），用來檢查守恆。
double solver1d_integral(const Solver1D* s, int c, int var);

// 殘差計算（對外公開，單元測試用）。U 必須已經套用過邊界條件。
void solver1d_apply_bc(const Solver1D* s, double* U);
void solver1d_residual(const Solver1D* s, const double* U, double* R,
                       double* W, double* S);

}  // namespace cfd

#endif  // CFD_EULER1D_H
