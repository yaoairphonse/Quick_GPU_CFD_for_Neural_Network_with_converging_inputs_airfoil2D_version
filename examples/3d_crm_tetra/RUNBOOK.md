# NASA CRM tetrahedral runbook

## Purpose

This path preserves the ability to consume a large body-fitted tetrahedral
mesh.  It is separate from Cartesian AMR.  The retained NASA DPW4 medium grid
has 10,067,380 tetrahedra and very anisotropic near-wall cells.

## 1. Obtain and verify the public mesh

```bash
export DOWNLOAD_NASA_CRM=1
bash examples/3d_crm_tetra/run.sh build cuda
```

The script downloads the source only when the explicit environment switch is
set.  Expected SHA-256 values are recorded in `source_manifest.json`:

- archive: `0ffdbbfe...d465a2`;
- extracted CGNS: `c031af3f...0341`.

Alternatively set `CRM_CGNS=/absolute/path/to/file.cgns`.

## 2. Import and construct solver tables

The run script performs these stages explicitly:

1. `read_crm_cgns.py` normalizes by the 275.8-inch reference chord and writes
   points, tetrahedra, boundary triangles, zones, and counts.
2. `prepare_tetra ... gauss` builds the `QCFD3D1` owner/neighbour mesh,
   cell-face CSR, face geometry, wall distance, and the separate `QCFDNC1`
   connectivity companion.
3. `build_node_gradient` creates adaptive-dimensional node weights,
   tetrahedral shape gradients, and face-node triples.
4. Every stage checks file sizes, positive geometry, boundary coverage, and
   area-vector closure before the solver starts.

The four-neighbour QR gradient experiment is intentionally not selected.  Its
local condition indicator reached roughly 2.56e22 and rejected the first
update.  Use the node path for current accuracy work.

## 3. Solver start and continuation

The retained development settings are Mach 0.85, alpha 2.5 degrees, Re 5M,
`mu-star=1.7e-7`, half-model reference area 3.9092552, node gradient, Mitchell
viscous face gradient, 50% limited mean-flow reconstruction, first-order SA
convection, block DP-LUR, and CFL 0.03.

Do not switch directly from a first-order checkpoint to an aggressive full-
second-order/CFL state.  A safe sequence is:

1. zero-step reload with the new operator and inspect the residual jump;
2. low-CFL fixed-step probe;
3. 10--25% reconstruction continuation;
4. 50% reconstruction after both mean and SA residuals stabilize;
5. increase CFL only after several jointly improving reports;
6. preserve a metric-best checkpoint and rollback on severe growth.

The provided `run.sh` starts directly at the retained 50% development setting
for reproducibility.  For a new mesh or changed physics, use the staged path.

## 4. Optional hierarchy

```bash
build/solvers/3d/build_agglomeration \
  --mesh output/3d_crm_tetra/mesh/mesh.bin \
  --out output/3d_crm_tetra/mesh/agglomeration_3level.bin \
  --levels 3 --merges-per-level 2
python3 solvers/3d/validate_agglomeration.py \
  --mesh output/3d_crm_tetra/mesh/mesh.bin \
  --hierarchy output/3d_crm_tetra/mesh/agglomeration_3level.bin
```

This provides conservative restriction and prolongated residual smoothing.  It
is not a coarse finite-volume solve and must not be called full FAS multigrid.

## 5. Convergence and known result

Require all six normalized raw residuals below tolerance and stable CL/CD/CM
over a meaningful window.  The retained 600-second legacy endpoint has
CL/CD/CM = 0.581806/0.052443/-0.143849 and SA residual 0.014633.  It is not
converged; pressure and viscous drag remain too large.  The NASA comparison is
not evidence that this transient drag is correct.

## 6. Diagnostics

Use `analyze_crm_mesh_quality.py`, `build_crm_volume_inspector.py`,
`plot_crm_tetra_pressure_proof.py`, and `plot_crm_diagnostics.py`.  The mesh
quality study found a median longest-edge/minimum-altitude ratio near 94.3 and
more than nine million faces above 70 degrees non-orthogonality.
