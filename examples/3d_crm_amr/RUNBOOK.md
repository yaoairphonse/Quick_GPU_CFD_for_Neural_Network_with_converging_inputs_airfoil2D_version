# Procedural Cartesian-AMR CRM runbook

## Purpose and scale

This is the primary 3-D development path.  It discards the NASA volume mesh,
uses only a closed CRM surface, and constructs a pure isotropic Cartesian
octree with cut cells.  It has no prism or wall-normal stretched layer.

The final retained mesh contains 61,482 leaf blocks, 31,478,784 procedural
cells, 21,523,036 agglomerated controls, 1,436,726 cut cells, and 5,173,340
exact wall fragments.  The finest spacing is 0.00390625 MAC.  Each leaf block
contains 8x8x8 Cartesian cells.

Expect several gigabytes of temporary/packed data and a high-memory GPU.  The
complete Level-10 all-wall estimate is 47.6M active cells and does not fit the
current 24-GB solver layout.

## 1. Prepare the closed surface

Use `export_crm_surface.py` on the imported NASA directory or provide an
equivalent closed, consistently oriented NPZ/STL surface:

```bash
python3 solvers/3d/export_crm_surface.py \
  --source output/3d_crm_tetra/mesh \
  --out output/crm_surface
export CRM_SURFACE=$PWD/output/crm_surface/CRM_full_closed.npz
```

The reference surface has 97,492 points and 194,980 triangles.  It must be
closed and manifold after symmetry mirroring/welding.

## 2. Build all three AMR levels

```bash
bash examples/3d_crm_amr/build_meshes.sh build output/3d_crm_amr
```

The script deliberately exposes every stage:

1. surface normalization, orientation, feature/proximity analysis, and a
   reusable `column_plan.npz` containing the exact wall triangles;
2. target-count octree planning with 8-cell blocks and 2:1 balance;
3. triangle/block clipping and finest-voxel fluid/solid/cut classification;
4. exact cut-cell boundary-integral volume, centroid, aperture, and wall
   fragment construction;
5. conservative small-cell agglomeration through the largest usable aperture;
6. sparse block adjacency and regular/cut geometry exception tables;
7. `QCFDAM1` packing of blocks, interfaces, controls, cut faces, wall faces,
   triangle ownership, and source surface;
8. exact control-centroid wall distance through a triangle AABB BVH;
9. all-wall refinement, complete rebuild, targeted upper-wing refinement, and
   another complete rebuild.

The retained final build timings after the parent mesh were approximately
45.42 s interface classification, 257.24 s cut cells, 8.73 s solver tables,
5.09 s packing, and 2.89 s wall distance: 320.30 s total.  Including the
reusable parent build, the from-surface time was about 8 min 52 s.

## 3. Geometry and uniform-flow audits

Before physical flow, run both CPU and CUDA audits:

```bash
build/solvers/3d/amr3d \
  --mesh output/3d_crm_amr/upperwing_l10/packed \
  --out output/3d_crm_amr/audit_cpu --backend cpu \
  --geometry-audit 1 --uniform-audit 1 --iterations 0 --budget 1

build/solvers/3d/amr3d \
  --mesh output/3d_crm_amr/upperwing_l10/packed \
  --out output/3d_crm_amr/audit_cuda --backend cuda \
  --geometry-audit 1 --uniform-audit 1 --iterations 0 --budget 1
```

The retained 21.5M mesh has maximum agglomerated closure 2.28e-10, affine
Green--Gauss error 4.76e-6, uniform-flow residual 2.52e-13, and CPU/CUDA
viscous/SA one-step agreement 4.12e-16 relative Linf.  An audit failure blocks
all aerodynamic interpretation.

## 4. Cold start and reconstruction continuation

For a cold mesh, use first order at moderate CFL, then 25% and 50% mean-flow
reconstruction.  The supplied script encodes this sequence:

```bash
bash examples/3d_crm_amr/run_upperwing.sh \
  build cuda output/3d_crm_amr
```

The procedural AMR solver currently uses a matrix-free scalar-spectral DP-LUR
linearization.  Five sweeps and omega 0.8 are retained.  Do not jump directly
to 100% reconstruction: it was unstable on the base mesh.  A changed operator
causes a legitimate residual jump, so compare trends within one operator stage.

## 5. Coarse-to-fine warm start

The recommended upper-wing workflow starts from a mature 14.0M wall-refined
state and conservatively maps it to the 21.5M nested mesh:

```bash
export WALLREF_STATE=/path/to/wall_refined/state.bin
bash examples/3d_crm_amr/run_upperwing.sh \
  build cuda output/3d_crm_amr
```

`transfer_amr_restart.py` injects source states into ordinary target controls.
For target agglomeration groups it integrates every member's exact fluid volume
and writes the volume-weighted source state.  The retained transfer took 5.86 s,
preserved global conserved integrals to about 1e-15, reproduced a uniform state
to 4.44e-16, and retained positive density and pressure.

The mapping assumes nested octrees, the same root cube, and exactly the same
closed wall.  It is not valid between changed optimization geometries.

## 6. Flow settings and restart discipline

The retained case is Mach 0.85, alpha 2.5 degrees, Re 5M,
`mu-star=1.7e-7`, full-model reference area 7.8185104, moment reference
(4.8074692, 0, 0.6451777), DP-LUR CFL 1, and 50% second-order mean-flow
reconstruction.  SA convection remains first order.

Every continuation writes to a new directory.  Preserve `state.bin`,
`history.csv`, `result.json`, `method.json`, and `surface.csv` together.  A
state without its mesh manifest and method settings is not a reproducible
checkpoint.

## 7. Post-processing

```bash
export CRM_COLUMN_PLAN=$PWD/output/3d_crm_amr/surface_plan/column_plan.npz
bash examples/3d_crm_amr/audit_and_plot.sh \
  build cuda output/3d_crm_amr
```

The slice renderer samples the actual leaf-cell state and reports uncovered
domain/fluid pixels.  The mesh renderer shows real leaf blocks and individual
square cell intersections.  The surface plot paints area-averaged wall-fragment
Cp on the original CRM triangles.

## 8. Convergence and retained result

The AMR executable currently stops by budget/iteration rather than declaring
aerodynamic convergence.  Require:

- all mean-flow and SA residuals below the selected tolerance;
- no positivity failures;
- global and near-field residual trends no longer reversing;
- CL, CD, and CM stable over a long multi-report window;
- pressure/viscous force split stable;
- Cp sections and integrated loads stable across at least two mesh levels.

The retained endpoint is CL/CD/CM = 0.504745/0.026189/-0.102533 versus NASA
0.54239/0.02995/-0.06249.  Wing Cp RMSE is 0.16030.  Mean residual is 1.66e-4,
but SA is 0.00510 and loads still drift.  These numbers show refinement benefit;
they are not a converged validation.

## 9. Next grid

Do not uniformly refine the far field.  Use Cp/residual maps to target the
upper-wing shock, leading/trailing edges, wing-body junction, wing tip, and
near wake.  Reduce per-control GPU memory before a 30--40M run.
