// Surface-only geometry and pressure diagnostics for the static Euler cut-cell solver.
#include "euler2d.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace cfd {
namespace {
bool surface_pressures(Solver2D* s, int c, std::vector<double>* pressures) {
  if (!s || !s->cut || c < 0 || c >= s->ncase) return false;
  solver2d_apply_bc(s, s->U.data());
  solver2d_residual(s, s->U.data(), s->R.data(), s->W.data(), s->SX.data(),
                    s->SY.data(), s->FX.data(), s->FY.data());
  const Cut2D& ct = *s->cut;
  const size_t cells = static_cast<size_t>(s->grid.mx) * s->grid.my;
  const size_t offset = (static_cast<size_t>(c) * NVAR2D + 3) * cells;
  pressures->resize(ct.wall_fragments.size());
  for (const WallCell2D& cell : ct.wall_cells) {
    const size_t k = cell.owner_cell;
    // Same sampling point and admissibility fallback as cut2d_residual.
    double p = s->W[offset+k] + s->SX[offset+k]*(ct.wcx[k]-ct.cx[k])
                              + s->SY[offset+k]*(ct.wcy[k]-ct.cy[k]);
    if (!(p > 0.0)) p = s->W[offset+k];
    if (!std::isfinite(p) || !(p > 0.0)) return false;
    for (size_t a = cell.begin; a < cell.begin + cell.count; ++a) (*pressures)[a] = p;
  }
  return true;
}

void add_fragment(const WallFragment2D& f, double p, double xref, double yref,
                  SurfaceIntegral2D* r) {
  const double fx = p*f.nx*f.length, fy = p*f.ny*f.length;
  r->length += f.length;
  r->fx += fx;
  r->fy += fy;
  r->mz += (0.5*(f.x0+f.x1)-xref)*fy - (0.5*(f.y0+f.y1)-yref)*fx;
}

int close_output(FILE* file) {
  const bool failed = std::ferror(file) != 0;
  const int status = std::fclose(file);
  return failed || status != 0 ? 2 : 0;
}
} // namespace

int cut2d_surface_integral(Solver2D* s, int c, int patch_id, double xref, double yref,
                           SurfaceIntegral2D* result) {
  if (!result || patch_id < -1 || !std::isfinite(xref) || !std::isfinite(yref)) return 1;
  std::vector<double> p;
  if (!surface_pressures(s, c, &p)) return 1;
  *result = {0.0, 0.0, 0.0, 0.0};
  for (size_t a = 0; a < p.size(); ++a) {
    const WallFragment2D& f = s->cut->wall_fragments[a];
    if (patch_id == -1 || f.patch_id == patch_id) add_fragment(f, p[a], xref, yref, result);
  }
  return 0;
}

int cut2d_write_surface_integrals(Solver2D* s, int c, const char* path,
                                 double xref, double yref) {
  if (!path || !std::isfinite(xref) || !std::isfinite(yref)) return 1;
  std::vector<double> p;
  if (!surface_pressures(s, c, &p)) return 1;
  std::map<int, SurfaceIntegral2D> patches;
  SurfaceIntegral2D total = {};
  for (size_t a = 0; a < p.size(); ++a) {
    const WallFragment2D& f = s->cut->wall_fragments[a];
    add_fragment(f, p[a], xref, yref, &patches[f.patch_id]);
    add_fragment(f, p[a], xref, yref, &total);
  }
  FILE* file = std::fopen(path, "w");
  if (!file) return 2;
  std::fprintf(file, "# pressure only; forces and moments per unit span; normal fluid-to-solid\n");
  std::fprintf(file, "case,time,body_id,patch_id,xref,yref,length,fx_pressure,fy_pressure,mz_pressure\n");
  patches[-1] = total; // -1 is the aggregate row, physical patches are nonnegative.
  for (const auto& entry : patches) {
    const SurfaceIntegral2D& r = entry.second;
    std::fprintf(file, "%d,%.17e,0,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                 c, s->t[c], entry.first, xref, yref, r.length, r.fx, r.fy, r.mz);
  }
  return close_output(file);
}

int cut2d_write_surface_vtp(Solver2D* s, int c, const char* path) {
  if (!path) return 1;
  std::vector<double> p;
  if (!surface_pressures(s, c, &p)) return 1;
  FILE* file = std::fopen(path, "w");
  if (!file) return 2;
  const auto& fragments = s->cut->wall_fragments;
  const size_t n = fragments.size();
  std::fprintf(file, "<?xml version=\"1.0\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n<PolyData>\n");
  std::fprintf(file, "<!-- Static 2D polygon; body_id=0; normals fluid-to-solid; p_wall uses the solver's per-cell wall pressure. Forces are pressure only, per unit span. -->\n");
  std::fprintf(file, "<FieldData><DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">%.17e</DataArray>"
                     "<DataArray type=\"Int32\" Name=\"case_id\" NumberOfTuples=\"1\" format=\"ascii\">%d</DataArray></FieldData>\n", s->t[c], c);
  std::fprintf(file, "<Piece NumberOfPoints=\"%zu\" NumberOfVerts=\"0\" NumberOfLines=\"%zu\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n", 2*n, n);
  std::fprintf(file, "<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (const auto& f : fragments)
    std::fprintf(file, "%.17e %.17e 0 %.17e %.17e 0\n", f.x0, f.y0, f.x1, f.y1);
  std::fprintf(file, "</DataArray></Points>\n<Lines><DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
  for (size_t a = 0; a < n; ++a) std::fprintf(file, "%zu %zu\n", 2*a, 2*a+1);
  std::fprintf(file, "</DataArray><DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
  for (size_t a = 0; a < n; ++a) std::fprintf(file, "%zu\n", 2*(a+1));
  std::fprintf(file, "</DataArray></Lines>\n<CellData Scalars=\"p_wall\">\n");
  const char* names[] = {"fragment_id", "owner_cell", "source_edge_id", "body_id", "patch_id"};
  for (int field = 0; field < 5; ++field) {
    std::fprintf(file, "<DataArray type=\"Int64\" Name=\"%s\" format=\"ascii\">\n", names[field]);
    for (size_t a = 0; a < n; ++a) {
      const auto& f = fragments[a];
      const size_t value = field == 0 ? a : field == 1 ? f.owner_cell :
                           static_cast<size_t>(field == 2 ? f.source_edge_id : field == 3 ? f.body_id : f.patch_id);
      std::fprintf(file, "%zu\n", value);
    }
    std::fprintf(file, "</DataArray>\n");
  }
  std::fprintf(file, "<DataArray type=\"Float64\" Name=\"normal_fluid_to_solid\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for (const auto& f : fragments) std::fprintf(file, "%.17e %.17e 0\n", f.nx, f.ny);
  std::fprintf(file, "</DataArray><DataArray type=\"Float64\" Name=\"length\" format=\"ascii\">\n");
  for (const auto& f : fragments) std::fprintf(file, "%.17e\n", f.length);
  std::fprintf(file, "</DataArray><DataArray type=\"Float64\" Name=\"p_wall\" format=\"ascii\">\n");
  for (double value : p) std::fprintf(file, "%.17e\n", value);
  std::fprintf(file, "</DataArray><DataArray type=\"Float64\" Name=\"owner_fluid_area\" format=\"ascii\">\n");
  for (const auto& f : fragments) std::fprintf(file, "%.17e\n", s->cut->vol[f.owner_cell]);
  std::fprintf(file, "</DataArray><DataArray type=\"Float64\" Name=\"owner_volume_fraction\" format=\"ascii\">\n");
  for (const auto& f : fragments) {
    const size_t i = f.owner_cell % s->grid.mx, j = f.owner_cell / s->grid.mx;
    std::fprintf(file, "%.17e\n", s->cut->vol[f.owner_cell]/(s->grid.dx[i]*s->grid.dy[j]));
  }
  std::fprintf(file, "</DataArray></CellData>\n</Piece></PolyData></VTKFile>\n");
  return close_output(file);
}
} // namespace cfd
