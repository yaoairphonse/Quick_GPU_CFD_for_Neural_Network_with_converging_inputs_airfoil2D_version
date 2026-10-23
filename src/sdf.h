// sdf.h -- 2D 有號距離函數（SDF）與立即邊界法（IB）的格子分類。
//
// 約定：固體內部 SDF < 0，流體 > 0；|SDF| 是到壁面（多邊形）的歐氏距離。
// 多邊形是閉合折線（最後一點接回第一點），頂點順序不拘。
// 全部是 header-only 的小函數，沒有 new/delete；陣列由呼叫者提供。
#ifndef CFD_SDF_H
#define CFD_SDF_H

#include <cmath>
#include <cstddef>

namespace cfd {

// 點 (x,y) 到線段 (ax,ay)-(bx,by) 的最短距離平方
inline double seg_dist2(double x, double y, double ax, double ay, double bx, double by) {
  const double dx = bx - ax, dy = by - ay;
  const double l2 = dx * dx + dy * dy;
  double t = 0.0;
  if (l2 > 0.0) {
    t = ((x - ax) * dx + (y - ay) * dy) / l2;
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
  }
  const double px = ax + t * dx - x, py = ay + t * dy - y;
  return px * px + py * py;
}

// 同 sdf_polygon，另外回傳壁面上最近點 (*cx,*cy) 與該線段編號 *seg。
inline double sdf_polygon_closest(const double* px, const double* py, int n, double x,
                                  double y, double* cx, double* cy, int* seg);

// 射線法（奇偶交點數）：點是否在閉合多邊形內。
inline bool point_in_polygon(const double* px, const double* py, int n, double x, double y) {
  bool in = false;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const bool straddle = (py[i] > y) != (py[j] > y);
    if (straddle) {
      const double xc = px[j] + (y - py[j]) * (px[i] - px[j]) / (py[i] - py[j]);
      if (x < xc) in = !in;
    }
  }
  return in;
}

// 多邊形的有號距離（內部為負）。O(n) 暴力法，對 LF 網格的規模足夠。
inline double sdf_polygon(const double* px, const double* py, int n, double x, double y) {
  double d2 = 1e300;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const double s = seg_dist2(x, y, px[j], py[j], px[i], py[i]);
    if (s < d2) d2 = s;
  }
  const double d = std::sqrt(d2);
  return point_in_polygon(px, py, n, x, y) ? -d : d;
}

inline double sdf_polygon_closest(const double* px, const double* py, int n, double x,
                                  double y, double* cx, double* cy, int* seg) {
  double best = 1e300;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const double dx = px[i] - px[j], dy = py[i] - py[j];
    const double l2 = dx * dx + dy * dy;
    double t = 0.0;
    if (l2 > 0.0) {
      t = ((x - px[j]) * dx + (y - py[j]) * dy) / l2;
      t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    }
    const double qx = px[j] + t * dx, qy = py[j] + t * dy;
    const double d2 = (qx - x) * (qx - x) + (qy - y) * (qy - y);
    if (d2 < best) {
      best = d2;
      *cx = qx;
      *cy = qy;
      *seg = j;  // 線段 j -> i
    }
  }
  const double d = std::sqrt(best);
  return point_in_polygon(px, py, n, x, y) ? -d : d;
}

// 多邊形面積（鞋帶公式，取絕對值）
inline double polygon_area(const double* px, const double* py, int n) {
  double a = 0.0;
  for (int i = 0, j = n - 1; i < n; j = i++) a += px[j] * py[i] - px[i] * py[j];
  return 0.5 * std::fabs(a);
}

// NACA 對稱翼型 00tt 的半厚度（弦長 1，後緣閉合係數 -0.1036）。x in [0,1]。
inline double naca_halfthickness(double x, double t) {
  const double s = std::sqrt(x > 0.0 ? x : 0.0);
  return 5.0 * t * (0.2969 * s - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x -
                    0.1036 * x * x * x * x);
}

// 產生 NACA 對稱翼型的閉合多邊形：從後緣沿上表面到前緣再沿下表面回來，
// 頂點用餘弦間距（前緣加密）。npan 是每個表面的段數；總頂點數 = 2*npan+1
// （上表面 npan 點、下表面含前緣與後緣共 npan+1 點）。呼叫者提供長度 2*npan+1 的 px、py，函數回傳頂點數。攻角 aoa_deg 繞 (0.25,0) 旋轉幾何
// （只給壁面幾何測試用；規劃書的 AOA 預設是旋轉自由流而不是幾何）。
inline int naca_polygon(double t, int npan, double aoa_deg, double* px, double* py) {
  const double pi = 3.14159265358979323846;
  int k = 0;
  for (int i = 0; i < npan; ++i) {  // 上表面：x 由 1 到接近 0（不含前緣點）
    const double th = pi * (1.0 - static_cast<double>(i) / npan);
    const double x = 0.5 * (1.0 - std::cos(th));
    px[k] = x; py[k] = naca_halfthickness(x, t); ++k;
  }
  for (int i = 0; i <= npan; ++i) {  // 下表面：x 由 0（前緣）到 1（後緣）
    const double th = pi * static_cast<double>(i) / npan;
    const double x = 0.5 * (1.0 - std::cos(th));
    px[k] = x; py[k] = -naca_halfthickness(x, t); ++k;
  }
  if (aoa_deg != 0.0) {
    const double a = -aoa_deg * pi / 180.0, c = std::cos(a), s = std::sin(a);
    for (int i = 0; i < k; ++i) {
      const double x = px[i] - 0.25, y = py[i];
      px[i] = 0.25 + c * x - s * y;
      py[i] = s * x + c * y;
    }
  }
  return k;
}

// 圓的內接多邊形（m 個頂點）。測試用：與解析 SDF r-R 比較。
inline void circle_polygon(double cx, double cy, double R, int m, double* px, double* py,
                           double phase = 0.0) {
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < m; ++i) {
    px[i] = cx + R * std::cos(2.0 * pi * i / m + phase);
    py[i] = cy + R * std::sin(2.0 * pi * i / m + phase);
  }
}

// 格子類型
enum CellType { CELL_FLUID = 0, CELL_GHOST = 1, CELL_SOLID = 2 };

// 以格心 SDF 分類（phi 與 type 長度 nx*ny，i 為快軸）：
//   phi > 0               -> 流體
//   phi <= 0 且 4 個鄰格中有流體 -> 鬼格子（壁面內側的一層）
//   其他 phi <= 0          -> 固體（深處，不參與計算）
// 回傳鬼格子數。邊界外的鄰格視為不存在。
inline int ib_classify(const double* phi, int nx, int ny, unsigned char* type) {
  int nghost = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const int id = j * nx + i;
      if (phi[id] > 0.0) { type[id] = CELL_FLUID; continue; }
      bool touch = false;
      if (i > 0 && phi[id - 1] > 0.0) touch = true;
      if (i < nx - 1 && phi[id + 1] > 0.0) touch = true;
      if (j > 0 && phi[id - nx] > 0.0) touch = true;
      if (j < ny - 1 && phi[id + nx] > 0.0) touch = true;
      type[id] = touch ? CELL_GHOST : CELL_SOLID;
      nghost += touch ? 1 : 0;
    }
  return nghost;
}

}  // namespace cfd

#endif  // CFD_SDF_H
