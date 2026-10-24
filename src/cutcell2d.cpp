// cutcell2d.cpp -- 2D Cartesian embedded-boundary finite volumes: geometry, conservative wall flux, and SRD.
//
// Design follows Colella et al. (2006) for wall fluxes and Berger & Giuliani (2020) for SRD; see manual references.
//   * Integrate only fluid: area V, centroid, four face apertures/centroids, and wall vector/centroid.
//   * Multiply HLLC/AUSM+-up face fluxes by aperture length; wall flux is pressure only, (0,p*n,0),
//     so wall mass/energy fluxes are exactly zero and conservation holds algebraically.
//   * SRD addresses explicit small-cell stability after each RK stage.
// Geometry uses boundary integrals over face apertures and wall segments to compute area and centroid.
// Thus aperture vectors plus wall vectors form a closed boundary by construction.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "euler2d.h"
#include "flux.h"
#include "limiter.h"
#include "sdf.h"
#include "static_amr2d.h"

namespace cfd {

namespace {

// Intersect polygon edge a->b with x=c; use half-open convention (ax>=c)!=(bx>=c).
inline bool cross_vline(double ax, double ay, double bx, double by, double c, double* y) {
  if ((ax >= c) == (bx >= c)) return false;
  *y = ay + (c - ax) * (by - ay) / (bx - ax);
  return true;
}
inline bool cross_hline(double ax, double ay, double bx, double by, double c, double* x) {
  if ((ay >= c) == (by >= c)) return false;
  *x = ax + (c - ay) * (bx - ax) / (by - ay);
  return true;
}

// A wall coincident with a Cartesian face blocks that face on both sides.
bool on_aligned_wall(const double* px,const double* py,int n,bool vertical,double c,double m){
  for(int a=0;a<n;++a){const int b=(a+1)%n;const double ca=vertical?px[a]:py[a],cb=vertical?px[b]:py[b];
    const double ta=vertical?py[a]:px[a],tb=vertical?py[b]:px[b];
    if(ca==c&&cb==c&&m>=std::min(ta,tb)&&m<=std::max(ta,tb))return true;}
  return false;
}

// Return total fluid aperture length and its centroid along a vertical or horizontal segment.
// vertical=true: x=c and y ranges from lo to hi.
void aperture_line(const double* px, const double* py, int n, bool vertical, double c,
                   double lo, double hi, double* len, double* cen) {
  std::vector<double> t;
  t.push_back(lo);
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const double ca=vertical?px[j]:py[j],cb=vertical?px[i]:py[i];
    if(ca==c&&cb==c)for(double v:{vertical?py[j]:px[j],vertical?py[i]:px[i]})if(v>lo&&v<hi)t.push_back(v);
    double v;
    const bool hit = vertical ? cross_vline(px[j], py[j], px[i], py[i], c, &v)
                              : cross_hline(px[j], py[j], px[i], py[i], c, &v);
    if (hit && v > lo && v < hi) t.push_back(v);
  }
  t.push_back(hi);
  std::sort(t.begin(), t.end());
  double L = 0.0, M = 0.0;
  for (size_t k = 0; k + 1 < t.size(); ++k) {
    const double a = t[k], b = t[k + 1];
    if (b - a <= 0.0) continue;
    const double m = 0.5 * (a + b);
    const bool solid = vertical ? point_in_polygon(px, py, n, c, m) : point_in_polygon(px, py, n, m, c);
    if (!solid&&!on_aligned_wall(px,py,n,vertical,c,m)) {
      L += b - a;
      M += 0.5 * (b * b - a * a);
    }
  }
  *len = L;
  *cen = (L > 0.0) ? M / L : 0.5 * (lo + hi);
}

// Liang-Barsky intersection of P->Q with [x0,x1]x[y0,y1], using the same half-open convention.
bool clip_segment(double* x1, double* y1, double* x2, double* y2, double xa, double xb,
                  double ya, double yb) {
  double t0 = 0.0, t1 = 1.0;
  const double dx = *x2 - *x1, dy = *y2 - *y1;
  const double p[4] = {-dx, dx, -dy, dy};
  const double q[4] = {*x1 - xa, xb - *x1, *y1 - ya, yb - *y1};
  for (int k = 0; k < 4; ++k) {
    if (p[k] == 0.0) {
      if (q[k] < 0.0) return false;
    } else {
      const double r = q[k] / p[k];
      if (p[k] < 0.0) {
        if (r > t1) return false;
        if (r > t0) t0 = r;
      } else {
        if (r < t0) return false;
        if (r < t1) t1 = r;
      }
    }
  }
  const double nx1 = *x1 + t0 * dx, ny1 = *y1 + t0 * dy, nx2 = *x1 + t1 * dx, ny2 = *y1 + t1 * dy;
  *x1 = nx1; *y1 = ny1; *x2 = nx2; *y2 = ny2;
  return (t1 - t0) > 1e-14;
}

// Fluid-directed unit normal of a polygon edge; stepping along it leaves the polygon.
void edge_normal_to_fluid(const double* px, const double* py, int n, int a, double mx, double my,
                          double eps, double* nx, double* ny) {
  const int b = (a + 1) % n;
  const double tx = px[b] - px[a], ty = py[b] - py[a];
  const double L = std::sqrt(tx * tx + ty * ty);
  double ux = ty / L, uy = -tx / L;
  if (point_in_polygon(px, py, n, mx + eps * ux, my + eps * uy)) { ux = -ux; uy = -uy; }
  *nx = ux; *ny = uy;
}

}  // namespace

void cut2d_open_segment(const double* px, const double* py, int n, int axis,
                         double coordinate, double lo, double hi, double* length, double* centroid) {
  aperture_line(px, py, n, axis == 0, coordinate, lo, hi, length, centroid);
  if (std::fabs(*length-(hi-lo)) < 1e-13*(hi-lo)) *length = hi-lo;
  if (*length < 1e-13*(hi-lo)) *length = 0.0;
}

double cut2d_rect_wall_distance(const double* px, const double* py, int n,
                                double x0, double x1, double y0, double y1) {
  double d2 = std::numeric_limits<double>::infinity();
  for (int a=0; a<n; ++a) {
    const int b=(a+1)%n;
    double ax=px[a], ay=py[a], bx=px[b], by=py[b];
    if (clip_segment(&ax,&ay,&bx,&by,x0,x1,y0,y1)) return 0.0;
    for (double x : {x0,x1}) for (double y : {y0,y1})
      d2=std::min(d2,seg_dist2(x,y,px[a],py[a],px[b],py[b]));
    for (int k : {a,b}) {
      const double dx=std::max({x0-px[k],0.0,px[k]-x1});
      const double dy=std::max({y0-py[k],0.0,py[k]-y1});
      d2=std::min(d2,dx*dx+dy*dy);
    }
  }
  return std::sqrt(d2);
}

int cut2d_rect_metrics(const double* px, const double* py, int n,
                       double x0, double x1, double y0, double y1,
                       const int* patches, RectMetrics2D* r) {
  r->walls.clear();r->boundary.clear();
  r->volume=(x1-x0)*(y1-y0); r->cx=0.5*(x0+x1); r->cy=0.5*(y0+y1);
  if (n==0) return 0;
  const double h=std::min(x1-x0,y1-y0);
  double area=0, mx=0, my=0;
  std::vector<std::array<double,2>> crossings;
  for (int a=0; a<n; ++a) {
    const int b=(a+1)%n;
    double ax=px[a],ay=py[a],bx=px[b],by=py[b];
    if (!clip_segment(&ax,&ay,&bx,&by,x0,x1,y0,y1)) continue;
    const double length=std::hypot(bx-ax,by-ay);
    double nfx,nfy;
    edge_normal_to_fluid(px,py,n,a,0.5*(px[a]+px[b]),0.5*(py[a]+py[b]),1e-7*h,&nfx,&nfy);
    if ((bx-ax)*nfy-(by-ay)*nfx < 0) { std::swap(ax,bx);std::swap(ay,by); }
    // An aligned segment belongs only to the leaf on its fluid-normal side.
    if((ax==bx&&((ax==x0&&nfx<0)||(ax==x1&&nfx>0)))||
       (ay==by&&((ay==y0&&nfy<0)||(ay==y1&&nfy>0))))continue;
    // Local coordinates avoid subtracting large global-coordinate moments in tiny leaves.
    const double lax=ax-x0,lbx=bx-x0,lay=ay-y0,lby=by-y0;
    area+=0.5*(lax+lbx)*(lby-lay);
    mx+=(lax*lax+lax*lbx+lbx*lbx)*(lby-lay)/6;
    my-=(lay*lay+lay*lby+lby*lby)*(lbx-lax)/6;
    r->walls.push_back({0,a,0,patches?patches[a]:0,ax,ay,bx,by,-nfx,-nfy,length});
    for (const auto& p : {std::array<double,2>{ax,ay},std::array<double,2>{bx,by}}) {
      const double tol=1e-11*h;
      if (std::min({std::fabs(p[0]-x0),std::fabs(p[0]-x1),std::fabs(p[1]-y0),std::fabs(p[1]-y1)})>tol) continue;
      bool duplicate=false;
      for (const auto& q : crossings) if (std::hypot(p[0]-q[0],p[1]-q[1])<tol) duplicate=true;
      if (!duplicate) crossings.push_back(p);
    }
  }
  if (r->walls.empty()) {
    if (point_in_polygon(px,py,n,r->cx,r->cy)) r->volume=0;
    return 0;
  }
  // Restricted topology: a single wall arc through a leaf. No implicit gap sealing.
  if (crossings.size()!=2) return 2;
  double aw,ae,as,an,c;
  cut2d_open_segment(px,py,n,0,x0,y0,y1,&aw,&c);
  cut2d_open_segment(px,py,n,0,x1,y0,y1,&ae,&c);
  cut2d_open_segment(px,py,n,1,y0,x0,x1,&as,&c);
  cut2d_open_segment(px,py,n,1,y1,x0,x1,&an,&c);
  area+=(x1-x0)*ae;
  mx+=0.5*(x1-x0)*(x1-x0)*ae;
  my+=0.5*(y1-y0)*(y1-y0)*an;
  if (!std::isfinite(area) || area>r->volume*(1+1e-10) || area < -r->volume*1e-10) return 3;
  if (area<1e-13*r->volume) {r->volume=0;r->walls.clear();return 0;}
  r->volume=area;r->cx=x0+mx/area;r->cy=y0+my/area;
  return 0;
}

std::vector<std::array<double,2>> cut2d_open_intervals(const double* px,const double* py,int n,int axis,double coordinate,double lo,double hi){
  std::vector<double> t={lo,hi};for(int a=0;a<n;++a){const int b=(a+1)%n;double hit;
    const double ca=axis==0?px[a]:py[a],cb=axis==0?px[b]:py[b];
    if(ca==coordinate&&cb==coordinate)for(double v:{axis==0?py[a]:px[a],axis==0?py[b]:px[b]})if(v>lo&&v<hi)t.push_back(v);
    const bool crossed=axis==0?cross_vline(px[a],py[a],px[b],py[b],coordinate,&hit):cross_hline(px[a],py[a],px[b],py[b],coordinate,&hit);
    if(crossed&&hit>lo&&hit<hi)t.push_back(hit);}
  std::sort(t.begin(),t.end());std::vector<std::array<double,2>> result;
  for(size_t i=0;i+1<t.size();++i){if(!(t[i+1]>t[i]))continue;const double mid=.5*(t[i]+t[i+1]);
    if(!on_aligned_wall(px,py,n,axis==0,coordinate,mid)&&!(axis==0?point_in_polygon(px,py,n,coordinate,mid):point_in_polygon(px,py,n,mid,coordinate)))result.push_back({t[i],t[i+1]});}
  return result;
}

int cut2d_rect_components(const double* px,const double* py,int n,double x0,double x1,double y0,double y1,const int* patches,std::vector<RectMetrics2D>* output){
  output->clear();RectMetrics2D original;const int status=cut2d_rect_metrics(px,py,n,x0,x1,y0,y1,patches,&original);
  if(status==0){if(original.volume>0)output->push_back(std::move(original));return 0;}
  if(status!=2)return status;
  // Trace oriented fluid boundaries. Each loop becomes an independent control
  // volume; opposite sides of a thin solid sheet never share a state.
  struct Edge{int a,b,wall;bool used=false;};std::vector<Edge> edges;std::vector<std::array<double,2>> nodes;
  const double h=std::min(x1-x0,y1-y0),tol=1e-11*h;
  const auto node=[&](double x,double y){for(size_t k=0;k<nodes.size();++k)if(std::hypot(nodes[k][0]-x,nodes[k][1]-y)<=tol)return static_cast<int>(k);nodes.push_back({x,y});return static_cast<int>(nodes.size()-1);};
  const auto edge=[&](double ax,double ay,double bx,double by,int wall){const int a=node(ax,ay),b=node(bx,by);if(a!=b)edges.push_back({a,b,wall,false});};
  for(size_t q=0;q<original.walls.size();++q){const auto& w=original.walls[q];edge(w.x0,w.y0,w.x1,w.y1,static_cast<int>(q));}
  for(int side=0;side<4;++side){const bool vertical=side<2;const double c=side==0?x0:side==1?x1:side==2?y0:y1;
    for(const auto& interval:cut2d_open_intervals(px,py,n,vertical?0:1,c,vertical?y0:x0,vertical?y1:x1)){
      const double a=interval[0],b=interval[1];if(side==0)edge(c,b,c,a,-1);if(side==1)edge(c,a,c,b,-1);if(side==2)edge(a,c,b,c,-1);if(side==3)edge(b,c,a,c,-1);}}
  const double pi=std::acos(-1.);
  for(size_t start=0;start<edges.size();++start){if(edges[start].used)continue;RectMetrics2D r{};size_t current=start;const int first=edges[start].a;
    long double twice_area=0,mx=0,my=0;
    for(size_t count=0;;++count){if(count>edges.size()||edges[current].used)return 4;auto& e=edges[current];e.used=true;
      const auto a=nodes[e.a],b=nodes[e.b];r.boundary.push_back(a);
      const long double ax=a[0]-x0,ay=a[1]-y0,bx=b[0]-x0,by=b[1]-y0,cross=ax*by-bx*ay;
      twice_area+=cross;mx+=(ax+bx)*cross;my+=(ay+by)*cross;
      if(e.wall>=0)r.walls.push_back(original.walls[e.wall]);
      if(e.b==first)break;
      const double reverse=std::atan2(a[1]-b[1],a[0]-b[0]);double best=INFINITY;size_t next=edges.size();
      for(size_t q=0;q<edges.size();++q)if(!edges[q].used&&edges[q].a==e.b){const auto end=nodes[edges[q].b];double clockwise=reverse-std::atan2(end[1]-b[1],end[0]-b[0]);
        while(clockwise<0)clockwise+=2*pi;
        while(clockwise>=2*pi)clockwise-=2*pi;
        if(clockwise<best){best=clockwise;next=q;}}
      if(next==edges.size())return 4;
      current=next;
    }
    if(twice_area<=0)return 5; // unsupported hole/orientation; never silently fill
    r.volume=static_cast<double>(.5L*twice_area);r.cx=x0+static_cast<double>(mx/(3*twice_area));r.cy=y0+static_cast<double>(my/(3*twice_area));
    if(!std::isfinite(r.volume)||r.volume>(x1-x0)*(y1-y0)*(1+1e-10))return 5;
    output->push_back(std::move(r));
  }
  return output->empty()?5:0;
}

int cut2d_build_polygon(const Solver2D* s, const double* px, const double* py, int n, Cut2D* cut,
                        const int* edge_patch_ids) {
  cut->wall_fragments.clear();
  cut->wall_cells.clear();
  if (n < 3) return 1;
  for (int a = 0; a < n; ++a) {
    const int b = (a + 1) % n;
    if (!std::isfinite(px[a]) || !std::isfinite(py[a]) ||
        (px[a] == px[b] && py[a] == py[b]) ||
        (edge_patch_ids && edge_patch_ids[a] < 0)) return 1;
  }
  const Grid2D& g = s->grid;
  const int nx = g.nx, ny = g.ny, ng = g.ng, mx = g.mx;
  const size_t cells = static_cast<size_t>(mx) * g.my;
  // Grid-aligned polygon edges are outside the existing geometry contract.
  // Reject explicitly instead of assigning the same segment to two cells.
  for (int a = 0; a < n; ++a) {
    const int b = (a + 1) % n;
    if (px[a] == px[b]) {
      for (int i = 0; i <= nx; ++i)
        if (std::fabs(px[a]-g.xf[i]) <= 16*std::numeric_limits<double>::epsilon()*
            std::fmax(1.0, std::fabs(px[a]))) {
          std::fprintf(stderr, "cut2d_build_polygon: edge %d lies on an x grid line; unsupported geometry\n", a);
          return 3;
        }
    }
    if (py[a] == py[b]) {
      for (int j = 0; j <= ny; ++j)
        if (std::fabs(py[a]-g.yf[j]) <= 16*std::numeric_limits<double>::epsilon()*
            std::fmax(1.0, std::fabs(py[a]))) {
          std::fprintf(stderr, "cut2d_build_polygon: edge %d lies on a y grid line; unsupported geometry\n", a);
          return 3;
        }
    }
  }

  cut->vol.allocate(cells); cut->cx.allocate(cells); cut->cy.allocate(cells); cut->flag.allocate(cells);
  cut->axl.allocate(static_cast<size_t>(nx + 1) * ny); cut->axy.allocate(static_cast<size_t>(nx + 1) * ny);
  cut->ayl.allocate(static_cast<size_t>(nx) * (ny + 1)); cut->ayx.allocate(static_cast<size_t>(nx) * (ny + 1));
  cut->wax.allocate(cells); cut->way.allocate(cells); cut->wcx.allocate(cells); cut->wcy.allocate(cells);
  cut->nbrcount.allocate(cells); cut->smallpos.allocate(cells);
  for (size_t k = 0; k < cells; ++k) { cut->smallpos[k] = -1.0; cut->nbrcount[k] = 1.0; }

  // 1) Initialize all domain cells, including ghosts, as full fluid.
  for (int j = -ng; j < ny + ng; ++j)
    for (int i = -ng; i < nx + ng; ++i) {
      const size_t k = static_cast<size_t>(j + ng) * mx + (i + ng);
      cut->vol[k] = g.dx[i + ng] * g.dy[j + ng];
      cut->cx[k] = g.xc[i + ng];
      cut->cy[k] = g.yc[j + ng];
      cut->flag[k] = 0.0;
    }
  double hmin = 1e300;
  for (int i = 0; i < nx; ++i) hmin = std::fmin(hmin, g.dx[i + ng]);
  for (int j = 0; j < ny; ++j) hmin = std::fmin(hmin, g.dy[j + ng]);
  double pminx = px[0], pmaxx = px[0], pminy = py[0], pmaxy = py[0];
  for (int q = 1; q < n; ++q) {
    pminx = std::fmin(pminx, px[q]); pmaxx = std::fmax(pmaxx, px[q]);
    pminy = std::fmin(pminy, py[q]); pmaxy = std::fmax(pmaxy, py[q]);
  }

  // 2) Compute each global face aperture once.
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i <= nx; ++i) {
      const size_t f = static_cast<size_t>(j) * (nx + 1) + i;
      // Faces outside the polygon bounding box are fully open. Fill the same exact widths/centers
      // used by full-aperture normalization below, avoiding an all-edge polygon scan
      // for every distant face of a large grid.
      if (g.xf[i] < pminx || g.xf[i] > pmaxx ||
          g.yf[j + 1] < pminy || g.yf[j] > pmaxy) {
        cut->axl[f] = g.dy[j + ng]; cut->axy[f] = g.yc[j + ng];
        continue;
      }
      double L, C;
      aperture_line(px, py, n, true, g.xf[i], g.yf[j], g.yf[j + 1], &L, &C);
      cut->axl[f] = L; cut->axy[f] = C;
      // Use the exact width for a full aperture, avoiding ~1e-16 subtraction roundoff.
      if (std::fabs(L - g.dy[j + ng]) < 1e-13 * g.dy[j + ng]) { cut->axl[f] = g.dy[j + ng]; cut->axy[f] = g.yc[j + ng]; }
      if (L < 1e-13 * g.dy[j + ng]) { cut->axl[f] = 0.0; cut->axy[f] = g.yc[j + ng]; }
    }
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const size_t f = static_cast<size_t>(j) * nx + i;
      if (g.yf[j] < pminy || g.yf[j] > pmaxy ||
          g.xf[i + 1] < pminx || g.xf[i] > pmaxx) {
        cut->ayl[f] = g.dx[i + ng]; cut->ayx[f] = g.xc[i + ng];
        continue;
      }
      double L, C;
      aperture_line(px, py, n, false, g.yf[j], g.xf[i], g.xf[i + 1], &L, &C);
      cut->ayl[f] = L; cut->ayx[f] = C;
      if (std::fabs(L - g.dx[i + ng]) < 1e-13 * g.dx[i + ng]) { cut->ayl[f] = g.dx[i + ng]; cut->ayx[f] = g.xc[i + ng]; }
      if (L < 1e-13 * g.dx[i + ng]) { cut->ayl[f] = 0.0; cut->ayx[f] = g.xc[i + ng]; }
    }

  // 3) Per-cell wall segments, area, and centroid from boundary integrals.
  cut->nsolid = cut->ncut = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const size_t k = static_cast<size_t>(j + ng) * mx + (i + ng);
      const double x0 = g.xf[i], x1 = g.xf[i + 1], y0 = g.yf[j], y1 = g.yf[j + 1];
      if (x1 < pminx || x0 > pmaxx || y1 < pminy || y0 > pmaxy) continue;
      double Awx = 0.0, Awy = 0.0, wL = 0.0, wxm = 0.0, wym = 0.0;
      double A = 0.0, MX = 0.0, MY = 0.0;  // Boundary-integral area and first moments.
      bool has_wall = false;
      const size_t fragment_begin = cut->wall_fragments.size();
      for (int a = 0, b0 = n - 1; a < n; b0 = a++) {  // Edge b0 -> a.
        double sx1 = px[b0], sy1 = py[b0], sx2 = px[a], sy2 = py[a];
        if (std::fmax(sx1, sx2) < x0 || std::fmin(sx1, sx2) > x1 || std::fmax(sy1, sy2) < y0 || std::fmin(sy1, sy2) > y1) continue;
        if (!clip_segment(&sx1, &sy1, &sx2, &sy2, x0, x1, y0, y1)) continue;
        double nfx, nfy;
        edge_normal_to_fluid(px, py, n, b0, 0.5 * (px[b0] + px[a]), 0.5 * (py[b0] + py[a]), 1e-7 * hmin, &nfx, &nfy);
        const double len = std::sqrt((sx2 - sx1) * (sx2 - sx1) + (sy2 - sy1) * (sy2 - sy1));
        // Fluid on the left: tangent t = (nfy,-nfx).
        const double tx = nfy, ty = -nfx;
        if ((sx2 - sx1) * tx + (sy2 - sy1) * ty < 0.0) { std::swap(sx1, sx2); std::swap(sy1, sy2); }
        A += 0.5 * (sx1 + sx2) * (sy2 - sy1);
        MX += (sx1 * sx1 + sx1 * sx2 + sx2 * sx2) * (sy2 - sy1) / 6.0;
        MY += -(sy1 * sy1 + sy1 * sy2 + sy2 * sy2) * (sx2 - sx1) / 6.0;
        Awx += -nfx * len; Awy += -nfy * len;
        wL += len; wxm += len * 0.5 * (sx1 + sx2); wym += len * 0.5 * (sy1 + sy2);
        cut->wall_fragments.push_back({k, b0, 0, edge_patch_ids ? edge_patch_ids[b0] : 0,
                                      sx1, sy1, sx2, sy2, -nfx, -nfy, len});
        has_wall = true;
      }
      const double ae = cut->axl[static_cast<size_t>(j) * (nx + 1) + i + 1];
      const double aw = cut->axl[static_cast<size_t>(j) * (nx + 1) + i];
      const double an = cut->ayl[static_cast<size_t>(j + 1) * nx + i];
      const double as = cut->ayl[static_cast<size_t>(j) * nx + i];
      if (!has_wall) {
        const bool inside = point_in_polygon(px, py, n, g.xc[i + ng], g.yc[j + ng]);
        if (inside) {
          cut->vol[k] = 0.0; cut->flag[k] = 2.0; ++cut->nsolid;
        }
        continue;  // Full fluid cell: retain defaults.
      }
      // Face orientations: east +y, north -x, west -y, south +x.
      A += x1 * ae - x0 * aw;
      MX += 0.5 * x1 * x1 * ae - 0.5 * x0 * x0 * aw;
      MY += -0.5 * y1 * y1 * (-an) - 0.5 * y0 * y0 * as;
      cut->wax[k] = Awx; cut->way[k] = Awy;
      if (wL > 0.0) { cut->wcx[k] = wxm / wL; cut->wcy[k] = wym / wL; }
      if (A < 1e-13 * g.dx[i + ng] * g.dy[j + ng]) {
        cut->wall_fragments.resize(fragment_begin);
        cut->vol[k] = 0.0; cut->flag[k] = 2.0; ++cut->nsolid;
        continue;
      }
      cut->vol[k] = A;
      cut->cx[k] = MX / A;
      cut->cy[k] = MY / A;
      cut->flag[k] = 1.0;
      cut->wall_cells.push_back({k, fragment_begin, cut->wall_fragments.size() - fragment_begin});
      ++cut->ncut;
    }

  // 4) Close faces adjacent to solid cells; solid cells must have zero aperture.
  // A cell discarded for tiny area can leave a tiny opening; relative conservation error is <=1e-13.
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i <= nx; ++i) {
      const size_t f = static_cast<size_t>(j) * (nx + 1) + i;
      const bool lsol = (i > 0) && cut->vol[static_cast<size_t>(j + ng) * mx + (i - 1 + ng)] == 0.0;
      const bool rsol = (i < nx) && cut->vol[static_cast<size_t>(j + ng) * mx + (i + ng)] == 0.0;
      if (lsol || rsol) cut->axl[f] = 0.0;
    }
  for (int j = 0; j <= ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const size_t f = static_cast<size_t>(j) * nx + i;
      const bool bsol = (j > 0) && cut->vol[static_cast<size_t>(j - 1 + ng) * mx + (i + ng)] == 0.0;
      const bool tsol = (j < ny) && cut->vol[static_cast<size_t>(j + ng) * mx + (i + ng)] == 0.0;
      if (bsol || tsol) cut->ayl[f] = 0.0;
    }

  // 5) Keep the body at least three cells from the domain boundary.
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      if (cut->flag[static_cast<size_t>(j + ng) * mx + (i + ng)] == 0.0) continue;
      if (i < 3 || i >= nx - 3 || j < 3 || j >= ny - 3) {
        std::fprintf(stderr, "cut2d_build_polygon: body too close to the domain boundary (cell %d,%d)\n", i, j);
        return 2;
      }
    }

  // 6) SRD neighborhoods.
  std::vector<int> small_list;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      const size_t k = static_cast<size_t>(j + ng) * mx + (i + ng);
      if (cut->vol[k] > 0.0 && cut->vol[k] < 0.5 * g.dx[i + ng] * g.dy[j + ng]) small_list.push_back(static_cast<int>(k));
    }
  const int ns = static_cast<int>(small_list.size());
  cut->nsmall = ns;
  cut->nbad_nbhd = 0;
  cut->sidx.allocate(ns); cut->scnt.allocate(ns); cut->smem.allocate(25 * static_cast<size_t>(ns));
  cut->svhat.allocate(ns); cut->sxhat.allocate(ns); cut->syhat.allocate(ns);
  for (int q = 0; q < ns; ++q) {
    cut->sidx[q] = small_list[q];
    cut->smallpos[small_list[q]] = q;
  }
  for (int q = 0; q < ns; ++q) {
    const int k = small_list[q];
    const int i = k % mx - ng, j = k / mx - ng;
    const double half = 0.5 * g.dx[i + ng] * g.dy[j + ng];
    int cnt = 0;
    int mem[25];
    auto vol_of = [&](int ii, int jj) { return cut->vol[static_cast<size_t>(jj + ng) * mx + (ii + ng)]; };
    bool ok = false;
    // (1) Merge along the wall normal.
    const double awx = cut->wax[k], awy = cut->way[k];
    const double alen = std::sqrt(awx * awx + awy * awy);
    if (alen > 1e-14) {
      const double nfx = -awx / alen, nfy = -awy / alen;  // Points into fluid.
      int di = 0, dj = 0;
      if (std::fabs(nfx) >= std::fabs(nfy)) di = (nfx > 0) ? 1 : -1; else dj = (nfy > 0) ? 1 : -1;
      const int ii = i + di, jj = j + dj;
      if (ii >= 0 && ii < nx && jj >= 0 && jj < ny && vol_of(ii, jj) > 0.0 && vol_of(i, j) + vol_of(ii, jj) >= half) {
        mem[cnt++] = k;
        mem[cnt++] = (jj + ng) * mx + (ii + ng);
        ok = true;
      }
    }
    // (2) Centered 3x3 neighborhood, then (3) 5x5.
    for (int r = 1; r <= 2 && !ok; ++r) {
      cnt = 0;
      double vs = 0.0;
      for (int dj = -r; dj <= r; ++dj)
        for (int di = -r; di <= r; ++di) {
          const int ii = i + di, jj = j + dj;
          if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;
          const double v = vol_of(ii, jj);
          if (v <= 0.0) continue;
          mem[cnt++] = (jj + ng) * mx + (ii + ng);
          vs += v;
        }
      if (vs >= half) ok = true;
    }
    if (!ok) ++cut->nbad_nbhd;  // Even all fluid cells in 5x5 have insufficient volume; extreme geometry.
    cut->scnt[q] = cnt;
    for (int m = 0; m < cnt; ++m) {
      cut->smem[25 * static_cast<size_t>(q) + m] = mem[m];
      if (mem[m] != k) cut->nbrcount[mem[m]] += 1.0;
    }
  }
  for (int q = 0; q < ns; ++q) {
    double vh = 0.0, xh = 0.0, yh = 0.0;
    for (int m = 0; m < static_cast<int>(cut->scnt[q]); ++m) {
      const size_t r = static_cast<size_t>(cut->smem[25 * static_cast<size_t>(q) + m]);
      const double w = cut->vol[r] / cut->nbrcount[r];
      vh += w; xh += w * cut->cx[r]; yh += w * cut->cy[r];
    }
    cut->svhat[q] = vh; cut->sxhat[q] = xh / vh; cut->syhat[q] = yh / vh;
  }
  return 0;
}


// ---------------------------------------------------------------------------
// Residual assembly.
// ---------------------------------------------------------------------------
namespace {

inline void face_flux_cut(const Params2D& par, double mco, double rL, double unL, double utL,
                          double pL, double rR, double unR, double utR, double pR, double F[4]) {
  if (par.flux == FLUX_AUSMPUP)
    flux_ausmpup_t(rL, unL, utL, pL, rR, unR, utR, pR, par.gamma, mco, F);
  else
    flux_hllc_t(rL, unL, utL, pL, rR, unR, utR, pR, par.gamma, F);
}

inline void admissible_face_state(double rc, double unc, double utc, double pc,
                                  double* r, double* un, double* ut, double* p) {
  if (!(*r > 0.0) || !(*p > 0.0) || !std::isfinite(*r) || !std::isfinite(*un) ||
      !std::isfinite(*ut) || !std::isfinite(*p)) {
    *r = rc;
    *un = unc;
    *ut = utc;
    *p = pc;
  }
}

}  // namespace

void cut2d_residual(const Solver2D* s, const double* U, double* R, double* W, double* SX,
                    double* SY, double* FX, double* FY) {
  const Grid2D& g = s->grid;
  const Params2D& par = s->par;
  const Cut2D& ct = *s->cut;
  const double gm = par.gamma;
  const int nx = g.nx, ny = g.ny, ng = g.ng, mx = g.mx, my = g.my;
  const size_t cells = static_cast<size_t>(mx) * my;
  const size_t nfx = static_cast<size_t>(nx + 1) * ny;
  const size_t nfy = static_cast<size_t>(nx) * (ny + 1);

  // Regular cell and four face neighbors are full fluid: use the body-free directional slope formula.
  std::vector<unsigned char> regular(cells, 0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(cells >= 4096)
#endif
  for (int j = -ng + 1; j < ny + ng - 1; ++j)
    for (int i = -ng + 1; i < nx + ng - 1; ++i) {
      const size_t k = static_cast<size_t>(j + ng) * mx + (i + ng);
      regular[k] = (ct.flag[k] == 0.0 && ct.flag[k - 1] == 0.0 && ct.flag[k + 1] == 0.0 &&
                    ct.flag[k - mx] == 0.0 && ct.flag[k + mx] == 0.0) ? 1 : 0;
    }

  for (int c = 0; c < s->ncase; ++c) {
    const size_t base = static_cast<size_t>(c) * NVAR2D * cells;
    const double* Uc = U + base;
    double* Rc = R + base;
    double* Wc = W + base;
    double* SXc = SX + base;
    double* SYc = SY + base;
    double* FXc = FX + static_cast<size_t>(c) * NVAR2D * nfx;
    double* FYc = FY + static_cast<size_t>(c) * NVAR2D * nfy;
    const double mco = s->mco[c];

    // 1) Primitive variables, excluding solid.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (size_t k = 0; k < cells; ++k) {
      if (ct.vol[k] <= 0.0) {
        Wc[k] = 1.0; Wc[cells + k] = 0.0; Wc[2 * cells + k] = 0.0; Wc[3 * cells + k] = 1.0;
        continue;
      }
      const double r = Uc[k], mu = Uc[cells + k], mv = Uc[2 * cells + k], E = Uc[3 * cells + k];
      Wc[k] = r;
      Wc[cells + k] = mu / r;
      Wc[2 * cells + k] = mv / r;
      Wc[3 * cells + k] = (gm - 1.0) * (E - 0.5 * mu * mu / r - 0.5 * mv * mv / r);
    }

    // 2) Slopes.
    // 2a) Regular region: original directional limiting, including i=-1..nx and j=-1..ny in ghost layers.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int task = 0; task < NVAR2D * ny; ++task) {
      const int v = task / ny, j = task % ny;
      const double* q = Wc + static_cast<size_t>(v) * cells;
      double* sx = SXc + static_cast<size_t>(v) * cells;
        const size_t row = static_cast<size_t>(j + ng) * mx;
        for (int i = -1; i <= nx; ++i) {
          const int a = i + ng;
          if (!regular[row + a]) continue;
          const double hl = g.xc[a] - g.xc[a - 1], hr = g.xc[a + 1] - g.xc[a];
          sx[row + a] = limited_slope(q[row + a - 1], q[row + a], q[row + a + 1], hl, hr, g.dx[a], par.limiter, par.venkat_k);
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int task = 0; task < NVAR2D * nx; ++task) {
      const int v = task / nx, i = task % nx;
      const double* q = Wc + static_cast<size_t>(v) * cells;
      double* sy = SYc + static_cast<size_t>(v) * cells;
        for (int j = -1; j <= ny; ++j) {
          const int b = j + ng;
          const size_t id = static_cast<size_t>(b) * mx + (i + ng);
          if (!regular[id]) continue;
          const double hl = g.yc[b] - g.yc[b - 1], hr = g.yc[b + 1] - g.yc[b];
          sy[id] = limited_slope(q[id - mx], q[id], q[id + mx], hl, hr, g.dy[b], par.limiter, par.venkat_k);
        }
    }
    // 2b) Irregular region: least-squares gradients and multidimensional limiting for cut cells and face neighbors.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(cells >= 4096)
#endif
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const size_t k = static_cast<size_t>(j + ng) * mx + (i + ng);
        if (regular[k] || ct.vol[k] <= 0.0) continue;
        const double xk = ct.cx[k], yk = ct.cy[k];
        // Fluid neighbors in 3x3, expanded to 5x5 if necessary.
        int nb[24], nn = 0;
        double Sxx = 0, Sxy = 0, Syy = 0;
        for (int r = 1; r <= 2; ++r) {
          nn = 0; Sxx = Sxy = Syy = 0.0;
          for (int dj = -r; dj <= r; ++dj)
            for (int di = -r; di <= r; ++di) {
              if (di == 0 && dj == 0) continue;
              const int ii = i + di, jj = j + dj;
              if (ii < -ng || ii >= nx + ng || jj < -ng || jj >= ny + ng) continue;
              const size_t t = static_cast<size_t>(jj + ng) * mx + (ii + ng);
              if (ct.vol[t] <= 0.0) continue;
              const double dx = ct.cx[t] - xk, dy = ct.cy[t] - yk;
              nb[nn++] = static_cast<int>(t);
              Sxx += dx * dx; Sxy += dx * dy; Syy += dy * dy;
            }
          const double det = Sxx * Syy - Sxy * Sxy;
          if (nn >= 2 && det > 1e-8 * Sxx * Syy && det > 0.0) break;
          if (r == 2) nn = -1;  // Degenerate stencil: use zero slope.
        }
        const double det = Sxx * Syy - Sxy * Sxy;
        // Evaluation points: aperture centroids and wall centroid.
        double ex[5], ey[5];
        int ne = 0;
        const double xfE = g.xf[i + 1], xfW = g.xf[i], yfN = g.yf[j + 1], yfS = g.yf[j];
        if (ct.axl[static_cast<size_t>(j) * (nx + 1) + i + 1] > 0.0) { ex[ne] = xfE - xk; ey[ne] = ct.axy[static_cast<size_t>(j) * (nx + 1) + i + 1] - yk; ++ne; }
        if (ct.axl[static_cast<size_t>(j) * (nx + 1) + i] > 0.0) { ex[ne] = xfW - xk; ey[ne] = ct.axy[static_cast<size_t>(j) * (nx + 1) + i] - yk; ++ne; }
        if (ct.ayl[static_cast<size_t>(j + 1) * nx + i] > 0.0) { ex[ne] = ct.ayx[static_cast<size_t>(j + 1) * nx + i] - xk; ey[ne] = yfN - yk; ++ne; }
        if (ct.ayl[static_cast<size_t>(j) * nx + i] > 0.0) { ex[ne] = ct.ayx[static_cast<size_t>(j) * nx + i] - xk; ey[ne] = yfS - yk; ++ne; }
        if (ct.flag[k] == 1.0) { ex[ne] = ct.wcx[k] - xk; ey[ne] = ct.wcy[k] - yk; ++ne; }
        const double h = std::sqrt(g.dx[i + ng] * g.dy[j + ng]);
        const double eps2 = std::pow(par.venkat_k * h, 3.0);
        for (int v = 0; v < NVAR2D; ++v) {
          if (par.limiter == LIM_FIRST_ORDER) {
            SXc[static_cast<size_t>(v) * cells + k] = 0.0;
            SYc[static_cast<size_t>(v) * cells + k] = 0.0;
            continue;
          }
          const double* q = Wc + static_cast<size_t>(v) * cells;
          double gx = 0.0, gy = 0.0;
          if (nn > 0) {
            double bx = 0.0, by = 0.0;
            for (int m = 0; m < nn; ++m) {
              const size_t t = static_cast<size_t>(nb[m]);
              const double dq = q[t] - q[k];
              bx += (ct.cx[t] - xk) * dq;
              by += (ct.cy[t] - yk) * dq;
            }
            gx = (bx * Syy - by * Sxy) / det;
            gy = (by * Sxx - bx * Sxy) / det;
            if (par.limiter != LIM_NONE) {
              double qmax = q[k], qmin = q[k];
              for (int m = 0; m < nn; ++m) { qmax = std::max(qmax, q[nb[m]]); qmin = std::min(qmin, q[nb[m]]); }
              double phi = 1.0;
              for (int m = 0; m < ne; ++m) {
                const double d2 = gx * ex[m] + gy * ey[m];
                if (d2 > 0.0) {
                  phi = std::min(phi, par.limiter == LIM_VENKAT ? venkat_phi(qmax - q[k], d2, eps2) : std::min(1.0, (qmax - q[k]) / d2));
                } else if (d2 < 0.0) {
                  phi = std::min(phi, par.limiter == LIM_VENKAT ? venkat_phi(qmin - q[k], d2, eps2) : std::min(1.0, (qmin - q[k]) / d2));
                }
              }
              gx *= phi; gy *= phi;
            }
          }
          SXc[static_cast<size_t>(v) * cells + k] = gx;
          SYc[static_cast<size_t>(v) * cells + k] = gy;
        }
      }

    const double* Wr = Wc;
    const double* Wu = Wc + cells;
    const double* Wv = Wc + 2 * cells;
    const double* Wp = Wc + 3 * cells;
    const double* Sr = SXc; const double* Su = SXc + cells; const double* Sv = SXc + 2 * cells; const double* Sp = SXc + 3 * cells;
    const double* Tr = SYc; const double* Tu = SYc + cells; const double* Tv = SYc + 2 * cells; const double* Tp = SYc + 3 * cells;

    // 3a) x faces.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int j = 0; j < ny; ++j) {
      const size_t row = static_cast<size_t>(j + ng) * mx;
      for (int i = 0; i <= nx; ++i) {
        const size_t f = static_cast<size_t>(j) * (nx + 1) + i;
        const size_t L = row + (i - 1 + ng), Rr = row + (i + ng);
        double F[4] = {0, 0, 0, 0};
        if (ct.axl[f] > 0.0) {
          double rL, uL, vL, pL, rR, uR, vR, pR;
          if (regular[L] && regular[Rr]) {
            const double hL = 0.5 * g.dx[i - 1 + ng], hR = 0.5 * g.dx[i + ng];
            rL = Wr[L] + hL * Sr[L]; uL = Wu[L] + hL * Su[L];
            vL = Wv[L] + hL * Sv[L]; pL = Wp[L] + hL * Sp[L];
            rR = Wr[Rr] - hR * Sr[Rr]; uR = Wu[Rr] - hR * Su[Rr];
            vR = Wv[Rr] - hR * Sv[Rr]; pR = Wp[Rr] - hR * Sp[Rr];
          } else {
            const double xf = g.xf[i], yf = ct.axy[f];
            const double dxl = xf - ct.cx[L], dyl = yf - ct.cy[L], dxr = xf - ct.cx[Rr], dyr = yf - ct.cy[Rr];
            rL = Wr[L] + Sr[L] * dxl + Tr[L] * dyl;
            uL = Wu[L] + Su[L] * dxl + Tu[L] * dyl;
            vL = Wv[L] + Sv[L] * dxl + Tv[L] * dyl;
            pL = Wp[L] + Sp[L] * dxl + Tp[L] * dyl;
            rR = Wr[Rr] + Sr[Rr] * dxr + Tr[Rr] * dyr;
            uR = Wu[Rr] + Su[Rr] * dxr + Tu[Rr] * dyr;
            vR = Wv[Rr] + Sv[Rr] * dxr + Tv[Rr] * dyr;
            pR = Wp[Rr] + Sp[Rr] * dxr + Tp[Rr] * dyr;
          }
          admissible_face_state(Wr[L], Wu[L], Wv[L], Wp[L], &rL, &uL, &vL, &pL);
          admissible_face_state(Wr[Rr], Wu[Rr], Wv[Rr], Wp[Rr], &rR, &uR, &vR, &pR);
          face_flux_cut(par, mco, rL, uL, vL, pL, rR, uR, vR, pR, F);
        }
        FXc[0 * nfx + f] = F[0]; FXc[1 * nfx + f] = F[1]; FXc[2 * nfx + f] = F[2]; FXc[3 * nfx + f] = F[3];
      }
    }
    // 3b) y faces.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(cells >= 4096)
#endif
    for (int j = 0; j <= ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const size_t f = static_cast<size_t>(j) * nx + i;
        const size_t L = static_cast<size_t>(j - 1 + ng) * mx + (i + ng), Rr = static_cast<size_t>(j + ng) * mx + (i + ng);
        double F[4] = {0, 0, 0, 0};
        if (ct.ayl[f] > 0.0) {
          double rL, vL, uL, pL, rR, vR, uR, pR;
          if (regular[L] && regular[Rr]) {
            const double hL = 0.5 * g.dy[j - 1 + ng], hR = 0.5 * g.dy[j + ng];
            rL = Wr[L] + hL * Tr[L]; vL = Wv[L] + hL * Tv[L];
            uL = Wu[L] + hL * Tu[L]; pL = Wp[L] + hL * Tp[L];
            rR = Wr[Rr] - hR * Tr[Rr]; vR = Wv[Rr] - hR * Tv[Rr];
            uR = Wu[Rr] - hR * Tu[Rr]; pR = Wp[Rr] - hR * Tp[Rr];
          } else {
            const double yf = g.yf[j], xf = ct.ayx[f];
            const double dxl = xf - ct.cx[L], dyl = yf - ct.cy[L], dxr = xf - ct.cx[Rr], dyr = yf - ct.cy[Rr];
            rL = Wr[L] + Sr[L] * dxl + Tr[L] * dyl;
            vL = Wv[L] + Sv[L] * dxl + Tv[L] * dyl;
            uL = Wu[L] + Su[L] * dxl + Tu[L] * dyl;
            pL = Wp[L] + Sp[L] * dxl + Tp[L] * dyl;
            rR = Wr[Rr] + Sr[Rr] * dxr + Tr[Rr] * dyr;
            vR = Wv[Rr] + Sv[Rr] * dxr + Tv[Rr] * dyr;
            uR = Wu[Rr] + Su[Rr] * dxr + Tu[Rr] * dyr;
            pR = Wp[Rr] + Sp[Rr] * dxr + Tp[Rr] * dyr;
          }
          admissible_face_state(Wr[L], Wv[L], Wu[L], Wp[L], &rL, &vL, &uL, &pL);
          admissible_face_state(Wr[Rr], Wv[Rr], Wu[Rr], Wp[Rr], &rR, &vR, &uR, &pR);
          face_flux_cut(par, mco, rL, vL, uL, pL, rR, vR, uR, pR, F);
        }
        FYc[0 * nfy + f] = F[0]; FYc[1 * nfy + f] = F[2]; FYc[2 * nfy + f] = F[1]; FYc[3 * nfy + f] = F[3];
      }

    // 4) Gather fluxes.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(cells >= 4096)
#endif
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const size_t k = static_cast<size_t>(j + ng) * mx + (i + ng);
        const size_t fe = static_cast<size_t>(j) * (nx + 1) + i, fn = static_cast<size_t>(j) * nx + i;
        if (ct.vol[k] <= 0.0) {
          for (int v = 0; v < NVAR2D; ++v) Rc[static_cast<size_t>(v) * cells + k] = 0.0;
          continue;
        }
        if (ct.flag[k] == 0.0) {  // Full cell: use exactly the body-free solver expression.
          const double idx = 1.0 / g.dx[i + ng], idy = 1.0 / g.dy[j + ng];
          for (int v = 0; v < NVAR2D; ++v) {
            const double* fx = FXc + static_cast<size_t>(v) * nfx;
            const double* fy = FYc + static_cast<size_t>(v) * nfy;
            Rc[static_cast<size_t>(v) * cells + k] = -(fx[fe + 1] - fx[fe]) * idx - (fy[fn + nx] - fy[fn]) * idy;
          }
          continue;
        }
        const double ae = ct.axl[fe + 1], aw = ct.axl[fe], an = ct.ayl[fn + nx], as = ct.ayl[fn];
        const double iv = 1.0 / ct.vol[k];
        // Extrapolate cell pressure linearly to the wall centroid.
        double pw = Wp[k] + Sp[k] * (ct.wcx[k] - ct.cx[k]) + Tp[k] * (ct.wcy[k] - ct.cy[k]);
        if (!(pw > 0.0)) pw = Wp[k];
        for (int v = 0; v < NVAR2D; ++v) {
          const double* fx = FXc + static_cast<size_t>(v) * nfx;
          const double* fy = FYc + static_cast<size_t>(v) * nfy;
          double r = -iv * (ae * fx[fe + 1] - aw * fx[fe] + an * fy[fn + nx] - as * fy[fn]);
          if (v == 1) r -= iv * pw * ct.wax[k];
          if (v == 2) r -= iv * pw * ct.way[k];
          Rc[static_cast<size_t>(v) * cells + k] = r;
        }
      }
  }
}

// ---------------------------------------------------------------------------
// State redistribution (Berger and Giuliani, 2020).
// ---------------------------------------------------------------------------
void cut2d_srd(const Solver2D* s, double* U) {
  const Cut2D& ct = *s->cut;
  if (ct.nsmall == 0) return;
  const Grid2D& g = s->grid;
  const int mx = g.mx, ng = g.ng, nx = g.nx, ny = g.ny;
  const size_t cells = static_cast<size_t>(mx) * g.my;
  const int ns = ct.nsmall;
  std::vector<double> Qh(static_cast<size_t>(ns) * NVAR2D);
  std::vector<double> acc(cells * NVAR2D);
  std::vector<double> gradx(static_cast<size_t>(ns) * NVAR2D), grady(static_cast<size_t>(ns) * NVAR2D), phi(static_cast<size_t>(ns) * NVAR2D);

  for (int c = 0; c < s->ncase; ++c) {
    double* Uc = U + static_cast<size_t>(c) * NVAR2D * cells;
    // 1) Form the neighborhood state Q-hat.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(ns >= 64)
#endif
    for (int q = 0; q < ns; ++q)
      for (int v = 0; v < NVAR2D; ++v) {
        double sum = 0.0;
        for (int m = 0; m < static_cast<int>(ct.scnt[q]); ++m) {
          const size_t r = static_cast<size_t>(ct.smem[25 * static_cast<size_t>(q) + m]);
          sum += ct.vol[r] / ct.nbrcount[r] * Uc[static_cast<size_t>(v) * cells + r];
        }
        Qh[static_cast<size_t>(q) * NVAR2D + v] = sum / ct.svhat[q];
      }
    // Obtain (Qhat,xhat,yhat) for any cell t.
    auto get_hat = [&](size_t t, int v, double* qv, double* xv, double* yv) {
      const int pos = static_cast<int>(ct.smallpos[t]);
      if (pos >= 0) { *qv = Qh[static_cast<size_t>(pos) * NVAR2D + v]; *xv = ct.sxhat[pos]; *yv = ct.syhat[pos]; }
      else { *qv = Uc[static_cast<size_t>(v) * cells + t]; *xv = ct.cx[t]; *yv = ct.cy[t]; }
    };
    // 2) Neighborhood slopes: least-squares Qhat and Barth-Jespersen limiting on 3x3, expanding to 5x3,3x5,5x5.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(ns >= 64)
#endif
    for (int q = 0; q < ns; ++q) {
      if (s->par.limiter == LIM_FIRST_ORDER) {
        for (int v = 0; v < NVAR2D; ++v) {
          gradx[static_cast<size_t>(q) * NVAR2D + v] = 0.0;
          grady[static_cast<size_t>(q) * NVAR2D + v] = 0.0;
        }
        continue;
      }
      const size_t k = static_cast<size_t>(ct.sidx[q]);
      const int i = static_cast<int>(k % mx) - ng, j = static_cast<int>(k / mx) - ng;
      const double xk = ct.sxhat[q], yk = ct.syhat[q];
      const int ext[4][2] = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};
      std::vector<size_t> st;
      bool good = false;
      for (int e = 0; e < 4 && !good; ++e) {
        st.clear();
        double Sxx = 0, Sxy = 0, Syy = 0, maxdx = 0, maxdy = 0;
        for (int dj = -ext[e][1]; dj <= ext[e][1]; ++dj)
          for (int di = -ext[e][0]; di <= ext[e][0]; ++di) {
            if (di == 0 && dj == 0) continue;
            const int ii = i + di, jj = j + dj;
            if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;
            const size_t t = static_cast<size_t>(jj + ng) * mx + (ii + ng);
            if (ct.vol[t] <= 0.0) continue;
            double qv, xv, yv;
            get_hat(t, 0, &qv, &xv, &yv);
            const double dx = xv - xk, dy = yv - yk;
            st.push_back(t);
            Sxx += dx * dx; Sxy += dx * dy; Syy += dy * dy;
            maxdx = std::max(maxdx, std::fabs(dx)); maxdy = std::max(maxdy, std::fabs(dy));
          }
        const double det = Sxx * Syy - Sxy * Sxy;
        if (st.size() >= 2 && det > 1e-8 * Sxx * Syy && maxdx >= 0.5 * g.dx[i + ng] && maxdy >= 0.5 * g.dy[j + ng]) {
          good = true;
          for (int v = 0; v < NVAR2D; ++v) {
            double bx = 0, by = 0;
            const double q0 = Qh[static_cast<size_t>(q) * NVAR2D + v];
            double qmax = q0, qmin = q0;
            for (size_t m = 0; m < st.size(); ++m) {
              double qv, xv, yv;
              get_hat(st[m], v, &qv, &xv, &yv);
              bx += (xv - xk) * (qv - q0);
              by += (yv - yk) * (qv - q0);
              qmax = std::max(qmax, qv); qmin = std::min(qmin, qv);
            }
            double gx = (bx * Syy - by * Sxy) / det, gy = (by * Sxx - bx * Sxy) / det;
            double ph = 1.0;
            // An affine field must be reproduced exactly by SRD.  Roundoff can make the
            // Barth-Jespersen bounds clip such a field when a merged neighbourhood reaches
            // slightly beyond the LS stencil's extrema.  Detect an affine LS fit at machine
            // precision and leave it unlimited; genuinely nonlinear fields still use BJ.
            double fit_error = 0.0, fit_scale = std::fmax(1.0, std::fabs(q0));
            for (size_t m = 0; m < st.size(); ++m) {
              double qv, xv, yv;
              get_hat(st[m], v, &qv, &xv, &yv);
              fit_error = std::fmax(fit_error,
                                    std::fabs((qv - q0) - gx * (xv - xk) - gy * (yv - yk)));
              fit_scale = std::fmax(fit_scale, std::fabs(qv));
            }
            const bool affine = fit_error <= 128.0 * std::numeric_limits<double>::epsilon() * fit_scale;
            if (!affine) {
              for (int m = 0; m < static_cast<int>(ct.scnt[q]); ++m) {
                const size_t r = static_cast<size_t>(ct.smem[25 * static_cast<size_t>(q) + m]);
                const double d2 = gx * (ct.cx[r] - xk) + gy * (ct.cy[r] - yk);
                if (d2 > 0.0) ph = std::min(ph, (qmax - q0) / d2);
                else if (d2 < 0.0) ph = std::min(ph, (qmin - q0) / d2);
              }
            }
            ph = std::max(0.0, std::min(1.0, ph));
            gradx[static_cast<size_t>(q) * NVAR2D + v] = gx * ph;
            grady[static_cast<size_t>(q) * NVAR2D + v] = gy * ph;
          }
        }
      }
      if (!good)
        for (int v = 0; v < NVAR2D; ++v) { gradx[static_cast<size_t>(q) * NVAR2D + v] = 0.0; grady[static_cast<size_t>(q) * NVAR2D + v] = 0.0; }
    }
    // 3) Accumulate each regular cell neighborhood value; add each small-cell reconstruction qhat(x_r) to its members.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(cells >= 4096)
#endif
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const size_t k = static_cast<size_t>(j + ng) * mx + (i + ng);
        for (int v = 0; v < NVAR2D; ++v)
          acc[static_cast<size_t>(v) * cells + k] = (ct.vol[k] > 0.0 && ct.smallpos[k] < 0.0) ? Uc[static_cast<size_t>(v) * cells + k] : 0.0;
      }
    for (int q = 0; q < ns; ++q)
      for (int m = 0; m < static_cast<int>(ct.scnt[q]); ++m) {
        const size_t r = static_cast<size_t>(ct.smem[25 * static_cast<size_t>(q) + m]);
        for (int v = 0; v < NVAR2D; ++v)
          acc[static_cast<size_t>(v) * cells + r] += Qh[static_cast<size_t>(q) * NVAR2D + v] +
              gradx[static_cast<size_t>(q) * NVAR2D + v] * (ct.cx[r] - ct.sxhat[q]) +
              grady[static_cast<size_t>(q) * NVAR2D + v] * (ct.cy[r] - ct.syhat[q]);
      }
    // 4) Final U_r = acc_r/N_r; only small cells or cells with N_r>1 change.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(cells >= 4096)
#endif
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const size_t k = static_cast<size_t>(j + ng) * mx + (i + ng);
        if (ct.vol[k] <= 0.0) continue;
        if (ct.smallpos[k] < 0.0 && ct.nbrcount[k] == 1.0) continue;
        for (int v = 0; v < NVAR2D; ++v)
          Uc[static_cast<size_t>(v) * cells + k] = acc[static_cast<size_t>(v) * cells + k] / ct.nbrcount[k];
      }
  }
}

void cut2d_wall_force(const Solver2D* s, int c, double* fx, double* fy) {
  const Cut2D& ct = *s->cut;
  const Grid2D& g = s->grid;
  const size_t cells = static_cast<size_t>(g.mx) * g.my;
  // Recompute primitives and slopes from current U using workspace arrays.
  Solver2D& sm = const_cast<Solver2D&>(*s);
  solver2d_apply_bc(&sm, sm.U.data());
  solver2d_residual(&sm, sm.U.data(), sm.R.data(), sm.W.data(), sm.SX.data(), sm.SY.data(), sm.FX.data(), sm.FY.data());
  const size_t base = static_cast<size_t>(c) * NVAR2D * cells;
  const double* Wp = s->W.data() + base + 3 * cells;
  const double* Sp = s->SX.data() + base + 3 * cells;
  const double* Tp = s->SY.data() + base + 3 * cells;
  double Fx = 0.0, Fy = 0.0;
  for (int j = 0; j < g.ny; ++j)
    for (int i = 0; i < g.nx; ++i) {
      const size_t k = static_cast<size_t>(j + g.ng) * g.mx + (i + g.ng);
      if (ct.flag[k] != 1.0) continue;
      double pw = Wp[k] + Sp[k] * (ct.wcx[k] - ct.cx[k]) + Tp[k] * (ct.wcy[k] - ct.cy[k]);
      if (!(pw > 0.0)) pw = Wp[k];
      Fx += pw * ct.wax[k];  // Body force = sum p_w*Aw, with Aw pointing into solid.
      Fy += pw * ct.way[k];
    }
  *fx = Fx;
  *fy = Fy;
}

}  // namespace cfd
