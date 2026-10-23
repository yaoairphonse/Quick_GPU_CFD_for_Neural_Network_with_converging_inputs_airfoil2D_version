// Optional CUDA backend for the fp64 first-order 2-D Euler solver.
// Geometry construction and file I/O remain on the CPU.  A complete time-step,
// including cut-cell wall fluxes, rejection checks, and overlapping SRD
// neighborhoods, stays resident on the GPU.
#include "euler2d.h"
#include "implicit2d.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace cfd {
namespace {

constexpr int TPB = 256;

struct DeviceData {
  double *U = nullptr, *U1 = nullptr, *U2 = nullptr, *R = nullptr;
  double *W = nullptr, *FX = nullptr, *FY = nullptr, *dt = nullptr, *mco = nullptr;
  double *dx = nullptr, *dy = nullptr;
  double *vol = nullptr, *flag = nullptr, *axl = nullptr, *ayl = nullptr;
  double *wax = nullptr, *way = nullptr, *nbrcount = nullptr, *smallpos = nullptr;
  double *sidx = nullptr, *scnt = nullptr, *smem = nullptr, *svhat = nullptr;
  double *Qh = nullptr, *acc = nullptr, *smax = nullptr;
  int* bad = nullptr;

  ~DeviceData() {
    cudaFree(U); cudaFree(U1); cudaFree(U2); cudaFree(R); cudaFree(W);
    cudaFree(FX); cudaFree(FY); cudaFree(dt); cudaFree(mco);
    cudaFree(dx); cudaFree(dy); cudaFree(vol); cudaFree(flag);
    cudaFree(axl); cudaFree(ayl); cudaFree(wax); cudaFree(way);
    cudaFree(nbrcount); cudaFree(smallpos); cudaFree(sidx); cudaFree(scnt);
    cudaFree(smem); cudaFree(svhat); cudaFree(Qh); cudaFree(acc);
    cudaFree(smax); cudaFree(bad);
  }
};

struct DeviceConfig {
  int nx, ny, ng, mx, my, ncase;
  int flux, bc0, bc1, bc2, bc3, has_cut, nsmall;
  double gamma;
};

bool cuda_call(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return true;
  std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
  return false;
}

template <class T>
bool device_alloc(T** p, size_t n, size_t* bytes) {
  if (n == 0) return true;
  if (!cuda_call(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(T)), "cudaMalloc")) return false;
  *bytes += n * sizeof(T);
  return true;
}

bool copy_h2d(double* dst, const Array& src) {
  if (src.size() == 0) return true;
  return cuda_call(cudaMemcpy(dst, src.data(), src.size() * sizeof(double), cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
}

__device__ inline size_t didx(const DeviceConfig& c, int icase, int v, int i, int j) {
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  return (static_cast<size_t>(icase) * NVAR2D + v) * cells +
         static_cast<size_t>(j + c.ng) * c.mx + (i + c.ng);
}

__device__ inline double d_m1(double m, int s) { return 0.5 * (m + s * fabs(m)); }
__device__ inline double d_m2(double m, int s) { return s * 0.25 * (m + s) * (m + s); }
__device__ inline double d_m4(double m, int s) {
  if (fabs(m) >= 1.0) return d_m1(m, s);
  return d_m2(m, s) * (1.0 - s * 2.0 * d_m2(m, -s));
}
__device__ inline double d_p5(double m, int s, double alpha) {
  if (fabs(m) >= 1.0) return 0.5 * (1.0 + s * (m > 0.0 ? 1.0 : (m < 0.0 ? -1.0 : 0.0)));
  return d_m2(m, s) * ((s * 2.0 - m) - s * 16.0 * alpha * m * d_m2(m, -s));
}

__device__ void flux_exact(double r, double u, double v, double p, double gm, double* f) {
  const double E = p / (gm - 1.0) + 0.5 * r * (u * u + v * v);
  f[0] = r * u; f[1] = r * u * u + p; f[2] = r * u * v; f[3] = u * (E + p);
}

__device__ void flux_hllc(double rL, double uL, double vL, double pL,
                          double rR, double uR, double vR, double pR,
                          double gm, double* f) {
  const double aL = sqrt(gm * pL / rL), aR = sqrt(gm * pR / rR);
  const double EL = pL / (gm - 1.0) + 0.5 * rL * (uL * uL + vL * vL);
  const double ER = pR / (gm - 1.0) + 0.5 * rR * (uR * uR + vR * vR);
  const double qL = sqrt(rL), qR = sqrt(rR);
  const double HL = (EL + pL) / rL, HR = (ER + pR) / rR;
  const double uT = (qL * uL + qR * uR) / (qL + qR);
  const double vT = (qL * vL + qR * vR) / (qL + qR);
  const double HT = (qL * HL + qR * HR) / (qL + qR);
  const double aT = sqrt(fmax((gm - 1.0) * (HT - 0.5 * (uT * uT + vT * vT)), 1e-300));
  const double SL = fmin(uL - aL, uT - aT), SR = fmax(uR + aR, uT + aT);
  double fL[4], fR[4];
  flux_exact(rL, uL, vL, pL, gm, fL); flux_exact(rR, uR, vR, pR, gm, fR);
  if (SL >= 0.0) { for (int v = 0; v < 4; ++v) f[v] = fL[v]; return; }
  if (SR <= 0.0) { for (int v = 0; v < 4; ++v) f[v] = fR[v]; return; }
  const double Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) /
                    (rL * (SL - uL) - rR * (SR - uR));
  if (Ss >= 0.0) {
    const double a = rL * (SL - uL) / (SL - Ss);
    const double us[4] = {a, a * Ss, a * vL,
      a * (EL / rL + (Ss - uL) * (Ss + pL / (rL * (SL - uL))))};
    const double ul[4] = {rL, rL * uL, rL * vL, EL};
    for (int v = 0; v < 4; ++v) f[v] = fL[v] + SL * (us[v] - ul[v]);
  } else {
    const double a = rR * (SR - uR) / (SR - Ss);
    const double us[4] = {a, a * Ss, a * vR,
      a * (ER / rR + (Ss - uR) * (Ss + pR / (rR * (SR - uR))))};
    const double ur[4] = {rR, rR * uR, rR * vR, ER};
    for (int v = 0; v < 4; ++v) f[v] = fR[v] + SR * (us[v] - ur[v]);
  }
}

__device__ void flux_ausm(double rL, double uL, double vL, double pL,
                          double rR, double uR, double vR, double pR,
                          double gm, double mco, double* f) {
  const double a12 = 0.5 * (sqrt(gm * pL / rL) + sqrt(gm * pR / rR));
  const double rho12 = 0.5 * (rL + rR), ML = uL / a12, MR = uR / a12;
  const double mb2 = 0.5 * (uL * uL + uR * uR) / (a12 * a12);
  const double m02 = fmin(1.0, fmax(mb2, mco * mco));
  const double fa = sqrt(m02) * (2.0 - sqrt(m02));
  const double alpha = 3.0 / 16.0 * (-4.0 + 5.0 * fa * fa);
  const double m12 = d_m4(ML, 1) + d_m4(MR, -1) -
    (0.25 / fa) * fmax(1.0 - mb2, 0.0) * (pR - pL) / (rho12 * a12 * a12);
  const double pp = d_p5(ML, 1, alpha), pm = d_p5(MR, -1, alpha);
  const double p12 = pp * pL + pm * pR - 0.75 * pp * pm * (rL + rR) * fa * a12 * (uR - uL);
  const bool left = m12 > 0.0;
  const double r = left ? rL : rR, u = left ? uL : uR;
  const double v = left ? vL : vR, p = left ? pL : pR;
  const double mdot = a12 * m12 * r;
  const double E = p / (gm - 1.0) + 0.5 * r * (u * u + v * v);
  f[0] = mdot; f[1] = mdot * u + p12; f[2] = mdot * v; f[3] = mdot * (E + p) / r;
}

__device__ void face_flux(const DeviceConfig& c, double mco,
                          double rL, double uL, double vL, double pL,
                          double rR, double uR, double vR, double pR, double* f) {
  if (c.flux == FLUX_AUSMPUP) flux_ausm(rL, uL, vL, pL, rR, uR, vR, pR, c.gamma, mco, f);
  else flux_hllc(rL, uL, vL, pL, rR, uR, vR, pR, c.gamma, f);
}

__global__ void apply_bc_kernel(DeviceConfig c, double* U) {
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * NVAR2D * cells) return;
  const size_t k = q % cells;
  const int j = static_cast<int>(k / c.mx) - c.ng, i = static_cast<int>(k % c.mx) - c.ng;
  if (i >= 0 && i < c.nx && j >= 0 && j < c.ny) return;
  const int v = static_cast<int>((q / cells) % NVAR2D);
  int ii = i, jj = j; double sign = 1.0;
  if (ii < 0) {
    if (c.bc0 == BC_PERIODIC) ii += c.nx;
    else if (c.bc0 == BC_REFLECT) { ii = -ii - 1; if (v == 1) sign = -sign; }
    else ii = 0;
  } else if (ii >= c.nx) {
    if (c.bc1 == BC_PERIODIC) ii -= c.nx;
    else if (c.bc1 == BC_REFLECT) { ii = 2 * c.nx - 1 - ii; if (v == 1) sign = -sign; }
    else ii = c.nx - 1;
  }
  if (jj < 0) {
    if (c.bc2 == BC_PERIODIC) jj += c.ny;
    else if (c.bc2 == BC_REFLECT) { jj = -jj - 1; if (v == 2) sign = -sign; }
    else jj = 0;
  } else if (jj >= c.ny) {
    if (c.bc3 == BC_PERIODIC) jj -= c.ny;
    else if (c.bc3 == BC_REFLECT) { jj = 2 * c.ny - 1 - jj; if (v == 2) sign = -sign; }
    else jj = c.ny - 1;
  }
  const int icase = static_cast<int>(q / (NVAR2D * cells));
  U[q] = sign * U[didx(c, icase, v, ii, jj)];
}

__global__ void primitive_kernel(DeviceConfig c, const double* U, const double* vol, double* W) {
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * cells) return;
  const size_t k = q % cells, base = (q / cells) * NVAR2D * cells;
  if (c.has_cut && vol[k] <= 0.0) {
    W[base + k] = 1.0; W[base + cells + k] = 0.0;
    W[base + 2 * cells + k] = 0.0; W[base + 3 * cells + k] = 1.0; return;
  }
  const double r = U[base + k], mu = U[base + cells + k];
  const double mv = U[base + 2 * cells + k], E = U[base + 3 * cells + k];
  W[base + k] = r; W[base + cells + k] = mu / r; W[base + 2 * cells + k] = mv / r;
  W[base + 3 * cells + k] = (c.gamma - 1.0) * (E - 0.5 * (mu * mu + mv * mv) / r);
}

__global__ void xflux_kernel(DeviceConfig c, const double* W, const double* mco,
                             const double* axl, double* FX) {
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t nfx = static_cast<size_t>(c.nx + 1) * c.ny;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * nfx) return;
  const int icase = static_cast<int>(q / nfx), f0 = static_cast<int>(q % nfx);
  const int j = f0 / (c.nx + 1), i = f0 % (c.nx + 1);
  const size_t L = static_cast<size_t>(j + c.ng) * c.mx + (i - 1 + c.ng);
  const size_t R = L + 1, base = static_cast<size_t>(icase) * NVAR2D * cells;
  double f[4] = {0.0, 0.0, 0.0, 0.0};
  if (!c.has_cut || axl[f0] > 0.0)
    face_flux(c, mco[icase], W[base + L], W[base + cells + L], W[base + 2 * cells + L], W[base + 3 * cells + L],
              W[base + R], W[base + cells + R], W[base + 2 * cells + R], W[base + 3 * cells + R], f);
  const size_t out = static_cast<size_t>(icase) * NVAR2D * nfx;
  for (int v = 0; v < 4; ++v) FX[out + static_cast<size_t>(v) * nfx + f0] = f[v];
}

__global__ void yflux_kernel(DeviceConfig c, const double* W, const double* mco,
                             const double* ayl, double* FY) {
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t nfy = static_cast<size_t>(c.nx) * (c.ny + 1);
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * nfy) return;
  const int icase = static_cast<int>(q / nfy), f0 = static_cast<int>(q % nfy);
  const int j = f0 / c.nx, i = f0 % c.nx;
  const size_t L = static_cast<size_t>(j - 1 + c.ng) * c.mx + (i + c.ng), R = L + c.mx;
  const size_t base = static_cast<size_t>(icase) * NVAR2D * cells;
  double f[4] = {0.0, 0.0, 0.0, 0.0};
  if (!c.has_cut || ayl[f0] > 0.0)
    face_flux(c, mco[icase], W[base + L], W[base + 2 * cells + L], W[base + cells + L], W[base + 3 * cells + L],
              W[base + R], W[base + 2 * cells + R], W[base + cells + R], W[base + 3 * cells + R], f);
  const size_t out = static_cast<size_t>(icase) * NVAR2D * nfy;
  FY[out + f0] = f[0]; FY[out + nfy + f0] = f[2];
  FY[out + 2 * nfy + f0] = f[1]; FY[out + 3 * nfy + f0] = f[3];
}

__global__ void collect_kernel(DeviceConfig c, const double* W, const double* FX, const double* FY,
                               const double* dx, const double* dy, const double* vol,
                               const double* flag, const double* axl, const double* ayl,
                               const double* wax, const double* way, double* R) {
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t inner = static_cast<size_t>(c.nx) * c.ny;
  if (q >= static_cast<size_t>(c.ncase) * inner) return;
  const int icase = static_cast<int>(q / inner), ij = static_cast<int>(q % inner);
  const int i = ij % c.nx, j = ij / c.nx;
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t k = static_cast<size_t>(j + c.ng) * c.mx + (i + c.ng);
  const size_t nfx = static_cast<size_t>(c.nx + 1) * c.ny, nfy = static_cast<size_t>(c.nx) * (c.ny + 1);
  const size_t fe = static_cast<size_t>(j) * (c.nx + 1) + i, fn = static_cast<size_t>(j) * c.nx + i;
  const size_t rb = static_cast<size_t>(icase) * NVAR2D * cells;
  if (c.has_cut && vol[k] <= 0.0) { for (int v = 0; v < 4; ++v) R[rb + static_cast<size_t>(v) * cells + k] = 0.0; return; }
  const size_t xb = static_cast<size_t>(icase) * NVAR2D * nfx;
  const size_t yb = static_cast<size_t>(icase) * NVAR2D * nfy;
  if (!c.has_cut || flag[k] == 0.0) {
    for (int v = 0; v < 4; ++v)
      R[rb + static_cast<size_t>(v) * cells + k] =
        -(FX[xb + static_cast<size_t>(v) * nfx + fe + 1] - FX[xb + static_cast<size_t>(v) * nfx + fe]) / dx[i + c.ng]
        -(FY[yb + static_cast<size_t>(v) * nfy + fn + c.nx] - FY[yb + static_cast<size_t>(v) * nfy + fn]) / dy[j + c.ng];
    return;
  }
  const double iv = 1.0 / vol[k], pw = W[rb + 3 * cells + k];
  for (int v = 0; v < 4; ++v) {
    double r = -iv * (axl[fe + 1] * FX[xb + static_cast<size_t>(v) * nfx + fe + 1]
      - axl[fe] * FX[xb + static_cast<size_t>(v) * nfx + fe]
      + ayl[fn + c.nx] * FY[yb + static_cast<size_t>(v) * nfy + fn + c.nx]
      - ayl[fn] * FY[yb + static_cast<size_t>(v) * nfy + fn]);
    if (v == 1) r -= iv * pw * wax[k];
    if (v == 2) r -= iv * pw * way[k];
    R[rb + static_cast<size_t>(v) * cells + k] = r;
  }
}

__global__ void smax_kernel(DeviceConfig c, const double* U, const double* dx, const double* dy,
                            const double* vol, double* smax) {
  const size_t inner = static_cast<size_t>(c.nx) * c.ny;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * inner) return;
  const int icase = static_cast<int>(q / inner), ij = static_cast<int>(q % inner);
  const int i = ij % c.nx, j = ij / c.nx;
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t k0 = static_cast<size_t>(j + c.ng) * c.mx + (i + c.ng);
  if (c.has_cut && vol[k0] <= 0.0) return;
  const size_t k = static_cast<size_t>(icase) * NVAR2D * cells + k0;
  const double r = U[k], mu = U[k + cells], mv = U[k + 2 * cells], E = U[k + 3 * cells];
  const double u = mu / r, v = mv / r;
  const double p = (c.gamma - 1.0) * (E - 0.5 * (mu * mu + mv * mv) / r);
  const double a = sqrt(c.gamma * p / r);
  const double val = (fabs(u) + a) / dx[i + c.ng] + (fabs(v) + a) / dy[j + c.ng];
  atomicMax(reinterpret_cast<unsigned long long*>(smax + icase), __double_as_longlong(val));
}

__global__ void rk_kernel(DeviceConfig c, double a, const double* A, double b, const double* B,
                          const double* R, const double* dt, double* out) {
  const size_t inner = static_cast<size_t>(c.nx) * c.ny;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * NVAR2D * inner) return;
  const int ij = static_cast<int>(q % inner), v = static_cast<int>((q / inner) % NVAR2D);
  const int icase = static_cast<int>(q / (NVAR2D * inner));
  const int i = ij % c.nx, j = ij / c.nx;
  const size_t k = didx(c, icase, v, i, j);
  out[k] = a * A[k] + b * (B[k] + dt[icase] * R[k]);
}

__global__ void bad_kernel(DeviceConfig c, const double* U, const double* vol, int* bad) {
  const size_t inner = static_cast<size_t>(c.nx) * c.ny;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * inner) return;
  const int icase = static_cast<int>(q / inner), ij = static_cast<int>(q % inner);
  const int i = ij % c.nx, j = ij / c.nx;
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t k0 = static_cast<size_t>(j + c.ng) * c.mx + (i + c.ng);
  if (c.has_cut && vol[k0] <= 0.0) return;
  const size_t k = static_cast<size_t>(icase) * NVAR2D * cells + k0;
  const double r = U[k], mu = U[k + cells], mv = U[k + 2 * cells], E = U[k + 3 * cells];
  const double p = (c.gamma - 1.0) * (E - 0.5 * (mu * mu + mv * mv) / r);
  if (!(r > 0.0) || !(p > 0.0) || !isfinite(r) || !isfinite(p)) atomicExch(bad, 1);
}

__global__ void srd_qhat_kernel(DeviceConfig c, const double* U, const double* vol,
                                const double* nbrcount, const double* scnt,
                                const double* smem, const double* svhat, double* Qh) {
  const size_t q0 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(c.ncase) * c.nsmall * NVAR2D;
  if (q0 >= count) return;
  const int v = static_cast<int>(q0 % NVAR2D), q = static_cast<int>((q0 / NVAR2D) % c.nsmall);
  const int icase = static_cast<int>(q0 / (static_cast<size_t>(c.nsmall) * NVAR2D));
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  double sum = 0.0;
  for (int m = 0; m < static_cast<int>(scnt[q]); ++m) {
    const size_t r = static_cast<size_t>(smem[25 * static_cast<size_t>(q) + m]);
    sum += vol[r] / nbrcount[r] * U[(static_cast<size_t>(icase) * NVAR2D + v) * cells + r];
  }
  Qh[(static_cast<size_t>(icase) * c.nsmall + q) * NVAR2D + v] = sum / svhat[q];
}

__global__ void srd_init_kernel(DeviceConfig c, const double* U, const double* vol,
                                const double* smallpos, double* acc) {
  const size_t inner = static_cast<size_t>(c.nx) * c.ny;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * NVAR2D * inner) return;
  const int ij = static_cast<int>(q % inner), v = static_cast<int>((q / inner) % NVAR2D);
  const int icase = static_cast<int>(q / (NVAR2D * inner));
  const int i = ij % c.nx, j = ij / c.nx;
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t k0 = static_cast<size_t>(j + c.ng) * c.mx + (i + c.ng);
  const size_t k = (static_cast<size_t>(icase) * NVAR2D + v) * cells + k0;
  acc[k] = (vol[k0] > 0.0 && smallpos[k0] < 0.0) ? U[k] : 0.0;
}

__global__ void srd_scatter_kernel(DeviceConfig c, const double* scnt, const double* smem,
                                   const double* Qh, double* acc) {
  const size_t count = static_cast<size_t>(c.ncase) * c.nsmall * 25 * NVAR2D;
  const size_t z = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (z >= count) return;
  const int v = static_cast<int>(z % NVAR2D), m = static_cast<int>((z / NVAR2D) % 25);
  const int q = static_cast<int>((z / (25 * NVAR2D)) % c.nsmall);
  const int icase = static_cast<int>(z / (static_cast<size_t>(c.nsmall) * 25 * NVAR2D));
  if (m >= static_cast<int>(scnt[q])) return;
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t r = static_cast<size_t>(smem[25 * static_cast<size_t>(q) + m]);
  atomicAdd(acc + (static_cast<size_t>(icase) * NVAR2D + v) * cells + r,
            Qh[(static_cast<size_t>(icase) * c.nsmall + q) * NVAR2D + v]);
}

__global__ void srd_final_kernel(DeviceConfig c, double* U, const double* vol,
                                 const double* smallpos, const double* nbrcount, const double* acc) {
  const size_t inner = static_cast<size_t>(c.nx) * c.ny;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * NVAR2D * inner) return;
  const int ij = static_cast<int>(q % inner), v = static_cast<int>((q / inner) % NVAR2D);
  const int icase = static_cast<int>(q / (NVAR2D * inner));
  const int i = ij % c.nx, j = ij / c.nx;
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t k0 = static_cast<size_t>(j + c.ng) * c.mx + (i + c.ng);
  if (vol[k0] <= 0.0 || (smallpos[k0] < 0.0 && nbrcount[k0] == 1.0)) return;
  const size_t k = (static_cast<size_t>(icase) * NVAR2D + v) * cells + k0;
  U[k] = acc[k] / nbrcount[k0];
}

__global__ void copy_inner_kernel(DeviceConfig c, const double* src, double* dst) {
  const size_t inner = static_cast<size_t>(c.nx) * c.ny;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * NVAR2D * inner) return;
  const int ij = static_cast<int>(q % inner), v = static_cast<int>((q / inner) % NVAR2D);
  const int icase = static_cast<int>(q / (NVAR2D * inner));
  const size_t k = didx(c, icase, v, ij % c.nx, ij / c.nx);
  dst[k] = src[k];
}

int blocks(size_t n) { return static_cast<int>((n + TPB - 1) / TPB); }

bool launch_residual(const DeviceConfig& c, DeviceData& d, const double* input = nullptr) {
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t nfx = static_cast<size_t>(c.nx + 1) * c.ny;
  const size_t nfy = static_cast<size_t>(c.nx) * (c.ny + 1);
  primitive_kernel<<<blocks(static_cast<size_t>(c.ncase) * cells), TPB>>>(c, input ? input : d.U, d.vol, d.W);
  xflux_kernel<<<blocks(static_cast<size_t>(c.ncase) * nfx), TPB>>>(c, d.W, d.mco, d.axl, d.FX);
  yflux_kernel<<<blocks(static_cast<size_t>(c.ncase) * nfy), TPB>>>(c, d.W, d.mco, d.ayl, d.FY);
  collect_kernel<<<blocks(static_cast<size_t>(c.ncase) * c.nx * c.ny), TPB>>>
    (c, d.W, d.FX, d.FY, d.dx, d.dy, d.vol, d.flag, d.axl, d.ayl, d.wax, d.way, d.R);
  return cuda_call(cudaGetLastError(), "residual kernels");
}

bool check_bad(const DeviceConfig& c, DeviceData& d, const double* U, int* bad) {
  if (!cuda_call(cudaMemset(d.bad, 0, sizeof(int)), "clear bad flag")) return false;
  bad_kernel<<<blocks(static_cast<size_t>(c.ncase) * c.nx * c.ny), TPB>>>(c, U, d.vol, d.bad);
  if (!cuda_call(cudaGetLastError(), "bad-state kernel")) return false;
  return cuda_call(cudaMemcpy(bad, d.bad, sizeof(int), cudaMemcpyDeviceToHost), "copy bad flag");
}

bool apply_srd(const DeviceConfig& c, DeviceData& d, double* U) {
  if (!c.has_cut || c.nsmall == 0) return true;
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  srd_qhat_kernel<<<blocks(static_cast<size_t>(c.ncase) * c.nsmall * NVAR2D), TPB>>>
    (c, U, d.vol, d.nbrcount, d.scnt, d.smem, d.svhat, d.Qh);
  if (!cuda_call(cudaMemset(d.acc, 0, static_cast<size_t>(c.ncase) * NVAR2D * cells * sizeof(double)),
                 "clear SRD accumulator")) return false;
  srd_init_kernel<<<blocks(static_cast<size_t>(c.ncase) * NVAR2D * c.nx * c.ny), TPB>>>
    (c, U, d.vol, d.smallpos, d.acc);
  srd_scatter_kernel<<<blocks(static_cast<size_t>(c.ncase) * c.nsmall * 25 * NVAR2D), TPB>>>
    (c, d.scnt, d.smem, d.Qh, d.acc);
  srd_final_kernel<<<blocks(static_cast<size_t>(c.ncase) * NVAR2D * c.nx * c.ny), TPB>>>
    (c, U, d.vol, d.smallpos, d.nbrcount, d.acc);
  return cuda_call(cudaGetLastError(), "SRD kernels");
}

__global__ void implicit_norm_kernel(DeviceConfig c, const double* old, const double* Q,
                                     const double* R, const double* dt, const double* vol,
                                     double* norms) {
  const size_t inner = static_cast<size_t>(c.nx) * c.ny;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * inner) return;
  const int icase = static_cast<int>(q / inner), ij = static_cast<int>(q % inner);
  if (dt[icase] <= 0.0) return;
  const int i = ij % c.nx, j = ij / c.nx;
  const size_t k0 = static_cast<size_t>(j+c.ng) * c.mx + i+c.ng;
  if (c.has_cut && vol[k0] <= 0.0) return;
  double norm = 0.0;
  for (int v = 0; v < NVAR2D; ++v) {
    const size_t k = didx(c, icase, v, i, j);
    const double defect = implicit_defect(old[k], Q[k], R[k], dt[icase]);
    if (!isfinite(defect)) { norm = CUDART_INF; break; }
    norm = fmax(norm, implicit_scaled_defect(old[k], defect));
  }
  atomicMax(reinterpret_cast<unsigned long long*>(norms+icase), __double_as_longlong(norm));
}

__global__ void implicit_trial_kernel(DeviceConfig c, const double* old, const double* Q,
                                      const double* R, const double* dt, const double* dx,
                                      const double* dy, const double* vol, const double* axl,
                                      const double* ayl, const double* wax, const double* way,
                                      double pseudo_cfl, double* trial) {
  const size_t inner = static_cast<size_t>(c.nx) * c.ny;
  const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (q >= static_cast<size_t>(c.ncase) * inner) return;
  const int icase = static_cast<int>(q / inner), ij = static_cast<int>(q % inner);
  if (dt[icase] <= 0.0) return;
  const int i = ij % c.nx, j = ij / c.nx;
  const size_t cells = static_cast<size_t>(c.mx) * c.my;
  const size_t k0 = static_cast<size_t>(j+c.ng) * c.mx + i+c.ng;
  if (c.has_cut && vol[k0] <= 0.0) return;
  const size_t b = didx(c, icase, 0, i, j);
  const size_t fe = static_cast<size_t>(j) * (c.nx+1) + i;
  const size_t fn = static_cast<size_t>(j) * c.nx + i;
  const double rate = implicit_spectral_rate(Q[b], Q[b+cells], Q[b+2*cells], Q[b+3*cells],
    c.gamma, dx[i+c.ng], dy[j+c.ng], c.has_cut != 0,
    c.has_cut ? vol[k0] : 0.0, c.has_cut ? axl[fe+1] : 0.0, c.has_cut ? axl[fe] : 0.0,
    c.has_cut ? ayl[fn+c.nx] : 0.0, c.has_cut ? ayl[fn] : 0.0,
    c.has_cut ? wax[k0] : 0.0, c.has_cut ? way[k0] : 0.0);
  const double omega = pseudo_cfl / (pseudo_cfl + dt[icase] * rate);
  for (int v = 0; v < NVAR2D; ++v) {
    const size_t k = b + static_cast<size_t>(v) * cells;
    trial[k] = Q[k] + omega * implicit_defect(old[k], Q[k], R[k], dt[icase]);
  }
}

int run_implicit_cuda(const DeviceConfig& c, DeviceData& d, Solver2D* s,
                       double tend, long maxsteps, size_t states) {
  const size_t inner = static_cast<size_t>(c.ncase) * c.nx * c.ny;
  const size_t bytes = static_cast<size_t>(c.ncase) * sizeof(double);
  std::vector<double> norms(c.ncase), initial(c.ncase), hdt(c.ncase);
  int bad = 0;
  if (!check_bad(c, d, d.U, &bad)) return 3;
  if (bad) { std::fprintf(stderr, "CUDA backward Euler: invalid initial fluid state\n"); return 2; }
  for (long step = 0; ; ++step) {
    bool done = true;
    for (int ic = 0; ic < c.ncase; ++ic) if (s->t[ic] < tend) done = false;
    if (done) return 0;
    if (step >= maxsteps) return 1;
    if (!cuda_call(cudaMemset(d.smax, 0, bytes), "clear implicit smax")) return 3;
    smax_kernel<<<blocks(inner), TPB>>>(c, d.U, d.dx, d.dy, d.vol, d.smax);
    if (!cuda_call(cudaMemcpy(norms.data(), d.smax, bytes, cudaMemcpyDeviceToHost), "implicit smax")) return 3;
    for (int ic = 0; ic < c.ncase; ++ic) {
      if (!implicit2d_physical_dt(s,ic,norms[ic],tend)) return 3;
      hdt[ic] = s->dt[ic];
    }
    if (!cuda_call(cudaMemcpy(d.dt, hdt.data(), bytes, cudaMemcpyHostToDevice), "implicit physical dt") ||
        !cuda_call(cudaMemcpy(d.U1, d.U, states*sizeof(double), cudaMemcpyDeviceToDevice), "implicit initial guess") ||
        !cuda_call(cudaMemcpy(d.U2, d.U, states*sizeof(double), cudaMemcpyDeviceToDevice), "implicit initial trial")) return 3;
    double* Q = d.U1;
    double* trial = d.U2;
    double pseudo_cfl = s->par.pseudo_cfl;
    for (int iter = 0; ; ++iter) {
      apply_bc_kernel<<<blocks(states), TPB>>>(c, Q);
      if (!launch_residual(c, d, Q) || !cuda_call(cudaMemset(d.smax, 0, bytes), "clear BE defect")) return 3;
      implicit_norm_kernel<<<blocks(inner), TPB>>>(c, d.U, Q, d.R, d.dt, d.vol, d.smax);
      if (!cuda_call(cudaMemcpy(norms.data(), d.smax, bytes, cudaMemcpyDeviceToHost), "copy BE defect")) return 3;
      for (double norm : norms) if (!std::isfinite(norm)) return 2;
      if (implicit2d_converged(s, norms.data(), initial.data(), iter == 0)) break;
      if (iter >= s->par.implicit_maxiter) {
        std::fprintf(stderr, "CUDA backward Euler not converged: step=%ld updates=%d initial=%.6e defect=%.6e tolerance=%.6e; physical step not accepted\n",
          s->nsteps, iter, s->implicit.initial_defect, s->implicit.final_defect, s->implicit.tolerance);
        return 4;
      }
      int retries = 0;
      for (;;) {
        implicit_trial_kernel<<<blocks(inner), TPB>>>(c, d.U, Q, d.R, d.dt, d.dx, d.dy,
          d.vol, d.axl, d.ayl, d.wax, d.way, pseudo_cfl, trial);
        if (!check_bad(c, d, trial, &bad)) return 3;
        if (!bad) break;
        ++s->implicit.retries;
        if (++retries > 12) {
          std::fprintf(stderr, "CUDA backward Euler: nonphysical pseudo candidate; physical step not accepted\n");
          return 2;
        }
        pseudo_cfl *= 0.5;
      }
      double* save = Q; Q = trial; trial = save;
      ++s->implicit.updates;
    }
    copy_inner_kernel<<<blocks(inner*NVAR2D), TPB>>>(c, Q, d.U);
    if (!cuda_call(cudaDeviceSynchronize(), "implicit commit")) return 3;
    implicit2d_advance_time(s,tend);
    ++s->nsteps;
  }
}

}  // namespace

bool solver2d_cuda_available(char* reason, size_t reason_size) {
  int count = 0;
  const cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess || count <= 0) {
    if (reason != nullptr && reason_size > 0)
      std::snprintf(reason, reason_size, "cudaGetDeviceCount: %s (devices=%d)", cudaGetErrorString(e), count);
    return false;
  }
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess || p.major < 6) {
    if (reason != nullptr && reason_size > 0)
      std::snprintf(reason, reason_size, "device 0 lacks fp64 atomicAdd support (compute capability < 6.0)");
    return false;
  }
  if (reason != nullptr && reason_size > 0)
    std::snprintf(reason, reason_size, "%s, compute capability %d.%d", p.name, p.major, p.minor);
  return true;
}

int solver2d_run_cuda(Solver2D* s, double tend, long maxsteps) {
  const bool implicit = s->par.time_integrator == TIME_BACKWARD_EULER;
  if (implicit && !implicit2d_validate(s, tend, maxsteps)) return 3;
  if (!implicit && s->par.time_integrator != TIME_SSP_RK3) return 3;
  char available[256];
  if (!solver2d_cuda_available(available, sizeof(available))) {
    std::fprintf(stderr, "CUDA backend unavailable: %s\n", available);
    return 3;
  }
  if (s->par.limiter != LIM_FIRST_ORDER) {
    std::fprintf(stderr, "CUDA backend currently requires --limiter first; refusing to change the method implicitly\n");
    return 3;
  }
  if (s->ib != nullptr) {
    std::fprintf(stderr, "CUDA backend supports regular grids and cut-cell+SRD, not the legacy IB path\n");
    return 3;
  }
  if (s->cut != nullptr && s->cut->nbad_nbhd != 0) {
    std::fprintf(stderr, "CUDA backend refuses %d invalid SRD neighborhoods\n", s->cut->nbad_nbhd);
    return 3;
  }

  const Grid2D& g = s->grid;
  DeviceConfig c{g.nx, g.ny, g.ng, g.mx, g.my, s->ncase, s->par.flux,
                 s->par.bc[0], s->par.bc[1], s->par.bc[2], s->par.bc[3],
                 s->cut != nullptr ? 1 : 0, s->cut != nullptr ? s->cut->nsmall : 0,
                 s->par.gamma};
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  const size_t states = static_cast<size_t>(s->ncase) * NVAR2D * cells;
  const size_t nfx = static_cast<size_t>(s->ncase) * NVAR2D * (g.nx + 1) * g.ny;
  const size_t nfy = static_cast<size_t>(s->ncase) * NVAR2D * g.nx * (g.ny + 1);
  DeviceData d;
  size_t allocated = 0, free0 = 0, total0 = 0, free1 = 0, total1 = 0;
  cudaMemGetInfo(&free0, &total0);
  const auto setup_start = std::chrono::steady_clock::now();
#define ALLOC(member, count) if (!device_alloc(&d.member, (count), &allocated)) return 3
  ALLOC(U, states); ALLOC(U1, states); ALLOC(U2, states); ALLOC(R, states); ALLOC(W, states);
  ALLOC(FX, nfx); ALLOC(FY, nfy); ALLOC(dt, s->ncase); ALLOC(mco, s->ncase);
  ALLOC(dx, g.dx.size()); ALLOC(dy, g.dy.size()); ALLOC(smax, s->ncase); ALLOC(bad, 1);
  if (c.has_cut) {
    const Cut2D& ct = *s->cut;
    ALLOC(vol, ct.vol.size()); ALLOC(flag, ct.flag.size()); ALLOC(axl, ct.axl.size()); ALLOC(ayl, ct.ayl.size());
    ALLOC(wax, ct.wax.size()); ALLOC(way, ct.way.size()); ALLOC(nbrcount, ct.nbrcount.size());
    ALLOC(smallpos, ct.smallpos.size()); ALLOC(sidx, ct.sidx.size()); ALLOC(scnt, ct.scnt.size());
    ALLOC(smem, ct.smem.size()); ALLOC(svhat, ct.svhat.size());
    ALLOC(Qh, static_cast<size_t>(s->ncase) * c.nsmall * NVAR2D); ALLOC(acc, states);
  }
#undef ALLOC
  if (!cuda_call(cudaMemcpy(d.U, s->U.data(), states * sizeof(double), cudaMemcpyHostToDevice), "copy initial U") ||
      !copy_h2d(d.dx, g.dx) || !copy_h2d(d.dy, g.dy) || !copy_h2d(d.mco, s->mco)) return 3;
  if (c.has_cut) {
    const Cut2D& ct = *s->cut;
    if (!copy_h2d(d.vol, ct.vol) || !copy_h2d(d.flag, ct.flag) || !copy_h2d(d.axl, ct.axl) ||
        !copy_h2d(d.ayl, ct.ayl) || !copy_h2d(d.wax, ct.wax) || !copy_h2d(d.way, ct.way) ||
        !copy_h2d(d.nbrcount, ct.nbrcount) || !copy_h2d(d.smallpos, ct.smallpos) ||
        !copy_h2d(d.sidx, ct.sidx) || !copy_h2d(d.scnt, ct.scnt) || !copy_h2d(d.smem, ct.smem) ||
        !copy_h2d(d.svhat, ct.svhat)) return 3;
  }
  if (!cuda_call(cudaDeviceSynchronize(), "CUDA setup synchronize")) return 3;
  const auto setup_end = std::chrono::steady_clock::now();
  cudaMemGetInfo(&free1, &total1);

  cudaEvent_t ev0{}, ev1{};
  if (!cuda_call(cudaEventCreate(&ev0), "create timing event") ||
      !cuda_call(cudaEventCreate(&ev1), "create timing event")) return 3;
  cudaEventRecord(ev0);
  std::vector<double> hsmax(s->ncase), hdt(s->ncase);
  const size_t inner = static_cast<size_t>(s->ncase) * g.nx * g.ny;
  const size_t rk_work = static_cast<size_t>(s->ncase) * NVAR2D * g.nx * g.ny;
  int result = 1;
  if (implicit) result = run_implicit_cuda(c, d, s, tend, maxsteps, states);
  else for (long step = 0; step < maxsteps; ++step) {
    bool done = true;
    for (int icase = 0; icase < s->ncase; ++icase) if (s->t[icase] < tend) done = false;
    if (done) { result = 0; break; }
    apply_bc_kernel<<<blocks(states), TPB>>>(c, d.U);
    if (!cuda_call(cudaMemset(d.smax, 0, static_cast<size_t>(s->ncase) * sizeof(double)), "clear smax")) { result = 3; break; }
    smax_kernel<<<blocks(inner), TPB>>>(c, d.U, d.dx, d.dy, d.vol, d.smax);
    if (!cuda_call(cudaMemcpy(hsmax.data(), d.smax, static_cast<size_t>(s->ncase) * sizeof(double), cudaMemcpyDeviceToHost), "copy smax")) { result = 3; break; }
    for (int icase = 0; icase < s->ncase; ++icase) {
      double dt = s->par.dt_fixed > 0.0 ? s->par.dt_fixed : s->par.cfl / hsmax[icase];
      const double remain = tend - s->t[icase];
      if (remain <= 0.0) dt = 0.0; else if (dt > remain) dt = remain;
      hdt[icase] = dt;
    }
    if (!cuda_call(cudaMemcpy(d.dt, hdt.data(), static_cast<size_t>(s->ncase) * sizeof(double), cudaMemcpyHostToDevice), "copy dt")) { result = 3; break; }

    int retries = 0;
    for (;;) {
      int bad = 0;
      if (!launch_residual(c, d)) { result = 3; break; }
      rk_kernel<<<blocks(rk_work), TPB>>>(c, 0.0, d.U, 1.0, d.U, d.R, d.dt, d.U1);
      if (!check_bad(c, d, d.U1, &bad)) { result = 3; break; }
      if (!bad && !apply_srd(c, d, d.U1)) { result = 3; break; }
      if (!bad && !check_bad(c, d, d.U1, &bad)) { result = 3; break; }
      if (!bad) {
        apply_bc_kernel<<<blocks(states), TPB>>>(c, d.U1); double* save = d.U; d.U = d.U1;
        if (!launch_residual(c, d)) { d.U = save; result = 3; break; } d.U = save;
        rk_kernel<<<blocks(rk_work), TPB>>>(c, 0.75, d.U, 0.25, d.U1, d.R, d.dt, d.U2);
        if (!check_bad(c, d, d.U2, &bad)) { result = 3; break; }
        if (!bad && !apply_srd(c, d, d.U2)) { result = 3; break; }
        if (!bad && !check_bad(c, d, d.U2, &bad)) { result = 3; break; }
      }
      if (!bad) {
        apply_bc_kernel<<<blocks(states), TPB>>>(c, d.U2); double* save = d.U; d.U = d.U2;
        if (!launch_residual(c, d)) { d.U = save; result = 3; break; } d.U = save;
        rk_kernel<<<blocks(rk_work), TPB>>>(c, 1.0 / 3.0, d.U, 2.0 / 3.0, d.U2, d.R, d.dt, d.U1);
        if (!check_bad(c, d, d.U1, &bad)) { result = 3; break; }
        if (!bad && !apply_srd(c, d, d.U1)) { result = 3; break; }
        if (!bad && !check_bad(c, d, d.U1, &bad)) { result = 3; break; }
      }
      if (result == 3 || !bad) break;
      ++s->nreject; ++retries;
      bool usable = retries <= 12;
      for (int icase = 0; icase < s->ncase; ++icase) {
        if (hdt[icase] > 0.0) hdt[icase] *= 0.5;
        if (hdt[icase] < 0.0 || !std::isfinite(hdt[icase])) usable = false;
      }
      if (!usable) {
        std::fprintf(stderr, "CUDA solver: non-physical RK candidate after %d retries at accepted step %ld\n", retries, s->nsteps);
        result = 2; break;
      }
      if (!cuda_call(cudaMemcpy(d.dt, hdt.data(), static_cast<size_t>(s->ncase) * sizeof(double), cudaMemcpyHostToDevice), "copy retry dt")) { result = 3; break; }
    }
    if (result == 2 || result == 3) break;
    copy_inner_kernel<<<blocks(rk_work), TPB>>>(c, d.U1, d.U);
    if (!cuda_call(cudaGetLastError(), "accept state kernel")) { result = 3; break; }
    for (int icase = 0; icase < s->ncase; ++icase) {
      s->dt[icase] = hdt[icase];
      if (hdt[icase] > 0.0) s->t[icase] += hdt[icase];
    }
    ++s->nsteps;
    if (s->nsteps % 1000 == 0) {
      const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_end).count();
      std::fprintf(stderr,
        "cuda progress: steps=%ld t=%.9g dt=%.3e rejected=%ld elapsed=%.1f s\n",
        s->nsteps, s->t[0], s->dt[0], s->nreject, elapsed);
    }
  }
  cudaEventRecord(ev1); cudaEventSynchronize(ev1);
  float solve_ms = 0.0F; cudaEventElapsedTime(&solve_ms, ev0, ev1);
  cudaEventDestroy(ev0); cudaEventDestroy(ev1);
  if (!cuda_call(cudaMemcpy(s->U.data(), d.U, states * sizeof(double), cudaMemcpyDeviceToHost), "copy final U")) result = 3;
  const double setup_s = std::chrono::duration<double>(setup_end - setup_start).count();
  std::fprintf(stderr,
    "cuda: device=%s allocated=%.3f GiB free_before=%.3f GiB free_after_setup=%.3f GiB setup_h2d=%.6f s solve=%.6f s\n",
    available, allocated / 1073741824.0, free0 / 1073741824.0, free1 / 1073741824.0,
    setup_s, static_cast<double>(solve_ms) * 1e-3);
  return result;
}

}  // namespace cfd
