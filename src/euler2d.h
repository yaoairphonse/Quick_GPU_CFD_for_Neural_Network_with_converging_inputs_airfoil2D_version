// euler2d.h -- 2D 可壓縮 Euler 求解器（拉伸矩形網格上的 cell-centered 有限體積）。
//
// 設計重點：
//  * 網格是 x、y 各自的一維拉伸（tensor product），仍是規則結構：每格 4 個鄰居，
//    面通量迴圈不需要原子加。同一組網格被一批 case 共用。
//  * 場量存成 U[((case*NVAR + var) * my + (j+ng)) * mx + (i+ng)]，mx = nx+2ng，my = ny+2ng。
//  * 重構逐方向做（x 方向用 x 斜率、y 方向用 y 斜率，各自套 limiter.h 的限制器），
//    通量用 flux.h 的 HLLC 或 AUSM+-up。沿單一軸的問題因此與 1D 求解器逐位元相同。
//  * 每個 case 有自己的時間、時間步與 AUSM+-up 的參考 Mach 數（mco）。
#ifndef CFD_EULER2D_H
#define CFD_EULER2D_H

#include <cstddef>
#include <vector>

#include "array.h"
#include "types.h"

namespace cfd {

enum { NVAR2D = 4 };  // rho, rho*u, rho*v, rho*E

enum SolverBackend { BACKEND_CPU = 0, BACKEND_CUDA = 1 };
enum TimeIntegrator2D { TIME_SSP_RK3 = 0, TIME_BACKWARD_EULER = 1 };

// 一維網格拉伸方式（作用在 x 或 y 其中一個方向）
enum StretchType {
  STRETCH_UNIFORM = 0,
  STRETCH_GEOMETRIC = 1,     // 相鄰格寬比固定 = param（>1 表示往 max 端變寬，群聚在 min 端）
  STRETCH_TANH_ENDS = 2,     // 兩端加密，param = delta > 0（越大越密）
  STRETCH_SINH_CENTER = 3    // 中央加密，param = delta > 0（越大越密）
};
struct Stretch {
  int type;
  double param;
};

struct Params2D {
  double gamma;
  double cfl;
  int flux;         // FluxType
  int limiter;      // LimiterType
  double venkat_k;  // Venkatakrishnan 的 K
  double ausm_mco;  // AUSM+-up 的參考 Mach 數（每個 case 之後可個別覆寫）
  int bc[4];        // xmin, xmax, ymin, ymax 的 BCType；週期必須成對
  double dt_fixed;  // > 0 specifies physical dt; 0 = use full-cell CFL
  int time_integrator; // TimeIntegrator2D; explicit SSP-RK3 remains the default
  double pseudo_cfl;   // local pseudo-time relaxation, independent of physical CFL
  int implicit_maxiter;
  double implicit_rtol, implicit_atol; // tolerances on the actual BE defect
  Stretch sx, sy;
};

struct Grid2D {
  int nx, ny, ng, mx, my;
  double xmin, xmax, ymin, ymax;
  Array xf, yf;  // 面座標，長度 nx+1、ny+1
  Array xc, dx;  // 格心與格寬（含鬼格子），長度 mx
  Array yc, dy;  // 長度 my
};

// 立即邊界法（IB）資料：鬼格子加影像點的滑移壁。由 ib2d_build_polygon 產生。
// 整數（索引）以 double 存在 Array 裡（< 2^53 時精確），維持「只有一種陣列類別」。
enum { IB_NS = 36 };  // 影像點內插模板最多的格子數（6x6）

struct IB2D {
  int nghost;
  int nfallback;  // 影像點周圍 4 格不全是流體、改用最小平方法（線性重現）的鬼格子數
  Array fluid;    // 長度 mx*my；1 = 流體（也包含計算域邊界鬼格子），0 = IB 鬼格子或固體
  Array gidx;     // nghost：鬼格子的線性索引 (j+ng)*mx + (i+ng)
  Array sidx;     // nghost*IB_NS：影像點內插用的格子索引
  Array sw;       // nghost*IB_NS：權重（和為 1，不用的格權重 0）
  Array nrm;      // nghost*2：壁面法向（指向流體）
  // 壓力沿法向的線性外插：第二個探針在影像點再往流體內 dprobe 處（沿同一法向）。
  // p_ghost = p_I - 2d * (p_2 - p_I)/dprobe；密度由同熵關係 rho_g = rho_I (p_g/p_I)^(1/gamma)。
  // 這修正了「鬼格子壓力 = 影像壓力」在曲面上的 O(h) 誤差（法向壓力梯度 = rho*u_t^2*curvature）。
  Array sidx2;    // nghost*IB_NS：第二探針的模板索引
  Array sw2;      // nghost*IB_NS：第二探針的權重
  Array d2;       // nghost：2d（鬼格子到影像點的距離）
  double dprobe;  // 第二探針與影像點的距離
  int use_p_extrap;           // 1 = 用上面的壓力外插；0 = 壓力直接抄影像點（一階，舊行為）
  mutable long nguard;        // 外插壓力不是正值而退回影像壓力的次數（累計）
};

// Cut-cell（嵌入邊界）資料：由 cut2d_build_polygon 產生。見 cutcell2d.cpp 的說明。
// 面向量約定：nx = 單位外法向，面 x 在格東側為 +x。壁面向量 Aw = (指向固體的單位法向) * 壁面長度。
struct WallFragment2D {
  size_t owner_cell;
  int source_edge_id, body_id, patch_id;
  double x0, y0, x1, y1, nx, ny, length; // normal: fluid -> solid
};
struct WallCell2D {
  size_t owner_cell, begin, count;
};
struct SurfaceIntegral2D {
  double length, fx, fy, mz; // pressure only, per unit span
};
struct Cut2D {
  std::vector<WallFragment2D> wall_fragments;
  std::vector<WallCell2D> wall_cells; // sparse cell ranges, not a full-grid allocation
  int nsmall;       // 需要 SRD 合併鄰域的小格子數
  int nbad_nbhd;    // 找不到體積足夠的鄰域的小格子數（應為 0）
  int nsolid, ncut; // 統計
  Array vol;        // mx*my：流體面積（0 = 固體）
  Array cx, cy;     // mx*my：流體形心
  Array flag;       // mx*my：0 = 完整流體格，1 = 被切到，2 = 固體
  Array axl, axy;   // (nx+1)*ny：x 面的開口長度與開口形心 y（索引 j*(nx+1)+i）
  Array ayl, ayx;   // nx*(ny+1)：y 面的開口長度與開口形心 x（索引 j*nx+i）
  Array wax, way;   // mx*my：壁面向量（指向固體）
  Array wcx, wcy;   // mx*my：壁面形心
  Array nbrcount;   // mx*my：N_rs（包含該格的鄰域數）
  Array smallpos;   // mx*my：在小格子清單中的位置，-1 = 不是小格子
  Array sidx;       // nsmall：小格子的線性索引
  Array scnt;       // nsmall：鄰域格數
  Array smem;       // nsmall*25：鄰域成員的線性索引
  Array svhat, sxhat, syhat;  // nsmall：加權體積與加權形心
};

struct ImplicitStats2D {
  long updates, retries; // pseudo updates/trial retries, separate from physical steps
  double initial_defect, final_defect, tolerance; // latest physical step, scaled Linf
};

struct Solver2D {
  Grid2D grid;
  Params2D par;
  int ncase;
  Array U, U1, U2;  // 守恆變數（SSP-RK3 三組）
  Array R;          // 殘差 dU/dt
  Array W;          // 原始變數 (rho,u,v,p)
  Array SX, SY;     // 限制後的 x、y 斜率
  Array FX, FY;     // x 面與 y 面的通量
  Array t, dt, mco; // 每個 case 的時間、時間步、參考 Mach 數
  long nsteps;
  long nreject;       // rejected RK candidates; state is never clipped or committed
  ImplicitStats2D implicit;
  const IB2D* ib;  // nullptr = 沒有物體
  const Cut2D* cut;  // nullptr = 沒有 cut-cell 物體
};

Params2D params2d_default();

// 在 [a,b] 上產生 n 個格子的 n+1 個面座標（兩端點精確等於 a、b）。
void grid2d_stretch_faces(int n, double a, double b, Stretch s, double* xf);

// 場量索引（i、j 為內部索引，可為負＝鬼格子）
inline size_t u2idx(const Grid2D& g, int c, int v, int i, int j) {
  return ((static_cast<size_t>(c) * NVAR2D + v) * g.my + (j + g.ng)) * g.mx +
         (i + g.ng);
}

void solver2d_init(Solver2D* s, int nx, int ny, double xmin, double xmax,
                   double ymin, double ymax, int ncase, const Params2D& par);

void solver2d_set_primitive(Solver2D* s, int c, int i, int j, double rho,
                            double u, double v, double p);
void solver2d_get_primitive(const Solver2D* s, int c, int i, int j, double* rho,
                            double* u, double* v, double* p);
void solver2d_set_mco(Solver2D* s, int c, double mco);

// 掛上 IB 資料（不複製；呼叫者要讓它活得比求解器久）。
void solver2d_set_ib(Solver2D* s, const IB2D* ib);

// 掛上 cut-cell 資料（不複製）。與 IB 不可同時使用。
void solver2d_set_cut(Solver2D* s, const Cut2D* cut);

// 由閉合多邊形（固體內部）建立 cut-cell 幾何與 SRD 鄰域。成功回傳 0。
// 限制：多邊形邊不可與格線完全重合（請位移 1e-9）；物體要離計算域邊界至少 3 格。
int cut2d_build_polygon(const Solver2D* s, const double* px, const double* py, int n, Cut2D* cut,
                        const int* edge_patch_ids = nullptr);
// Surface pressure follows the existing residual's one-pressure-per-cut-cell rule.
// patch_id=-1 selects all patches; xref/yref are the moment reference point.
int cut2d_surface_integral(Solver2D* s, int c, int patch_id, double xref, double yref,
                           SurfaceIntegral2D* result);
int cut2d_write_surface_vtp(Solver2D* s, int c, const char* path);
int cut2d_write_surface_integrals(Solver2D* s, int c, const char* path,
                                 double xref, double yref);

// cut-cell 版殘差（solver2d_residual 在 s->cut != nullptr 時呼叫）。
void cut2d_residual(const Solver2D* s, const double* U, double* R, double* W, double* SX,
                    double* SY, double* FX, double* FY);

// 狀態重新分配（Berger & Giuliani 2020）：對一個 RK 階段的結果 U（含所有 case）就地套用。
void cut2d_srd(const Solver2D* s, double* U);

// 壁面壓力力：作用在物體上的力 = sum p_w * Aw，Aw 指向固體。
void cut2d_wall_force(const Solver2D* s, int c, double* fx, double* fy);

// 由閉合多邊形（固體內部）建立 IB 資料：SDF 分類、鬼格子、影像點內插模板。
// 成功回傳 0。全部在 CPU 上做一次，之後每個 RK 階段只做模板內插。
int ib2d_build_polygon(const Solver2D* s, const double* px, const double* py, int n, IB2D* ib);

// 壁面上某點 (wx,wy)（法向 nx,ny 指向流體）的密度與壓力：沿法向取距壁 1.5h 與 2.5h 兩個探針，
// 用流體格內插後線性外插回壁面。回傳 0 成功。
int ib2d_wall_state(const Solver2D* s, int c, double wx, double wy, double nx, double ny,
                    double* rho, double* p);

// 流體格中、壁面的壓力積分（力）：沿多邊形的每條線段取樣，用周圍流體格內插壓力。
// 回傳作用在物體上的力 (fx, fy)（單位：壓力×長度）。case c。
void ib2d_wall_force(const Solver2D* s, int c, const double* px, const double* py, int n, double* fx, double* fy);

// 推進到 tend。0 success; 1 maxsteps; 2 nonphysical; 3 unsupported/invalid;
// 4 implicit inner iteration did not converge (unaccepted physical step is rolled back).
int solver2d_run(Solver2D* s, double tend, long maxsteps);
int solver2d_run_implicit(Solver2D* s, double tend, long maxsteps);

// CUDA backend is optional at build time.  It currently implements the fp64,
// first-order regular-grid and cut-cell+SRD paths used by the aircraft run.
// Return codes match solver2d_run; 3 means the CUDA backend/runtime is unavailable
// or the requested configuration is not implemented.
int solver2d_run_cuda(Solver2D* s, double tend, long maxsteps);
int solver2d_run_backend(Solver2D* s, double tend, long maxsteps, int backend);
bool solver2d_cuda_available(char* reason, size_t reason_size);

// 某個守恆量的積分（sum U * dx * dy），用來檢查守恆。
double solver2d_integral(const Solver2D* s, int c, int var);

// 對外公開（單元測試用）：U 要先套用邊界條件。
void solver2d_apply_bc(const Solver2D* s, double* U);
void solver2d_residual(const Solver2D* s, const double* U, double* R, double* W,
                       double* SX, double* SY, double* FX, double* FY);

}  // namespace cfd

#endif  // CFD_EULER2D_H
