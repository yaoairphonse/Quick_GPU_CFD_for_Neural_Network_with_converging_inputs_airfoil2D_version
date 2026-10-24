# RAE2822 mixed hybrid-grid runbook

## Purpose and input conventions

This case verifies the 3-D explicit-face solver on a genuinely mixed topology.
The archived 2-D LF mesh contains triangles and quadrilaterals.  A one-cell-
thick spanwise extrusion produces wedges and hexahedra without tetrahedral
subdivision.  The spanwise caps are symmetry boundaries.

The converter records three non-obvious archive conventions:

1. `face_nodes` is zero based and node zero is reserved;
2. positive `face_cells` entries are one based, while zero means boundary;
3. `msh_to_csv[mesh_cell-1]` maps topology cells to the archived polygon order.

External files are not redistributed.  Set paths to `LF1_full_topology.npz`
and `LF1_true_mesh.npz`.

## 1. Build

```bash
cmake -S . -B build -DCFD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j 8 --target sphere3d
```

## 2. Convert and audit the mixed mesh

```bash
export RAE_TOPOLOGY=/path/to/LF1_full_topology.npz
export RAE_TRUE_MESH=/path/to/LF1_true_mesh.npz
python3 tools/hybrid/convert_rae2822_npz.py \
  --topology "$RAE_TOPOLOGY" --true-mesh "$RAE_TRUE_MESH" \
  --out output/3d_rae2822_lf1/mesh --span 0.1
```

The converter checks polygon/face topology, cell closure, total volume, face
orientation, positive distance, and least-squares linear exactness.  LF1 must
contain 5,051 cells: 4,721 wedges and 330 hexahedra.  Compare the generated
report with `reference/mesh_report.json`.

Boundary mapping is wall zones 6/7, farfield zones 8/9, and symmetry caps.  The
current explicit format has no separate pressure-outlet condition, so archived
zone 9 is deliberately mapped to farfield and reported as such.

## 3. Run the retained fixed-CFL sequence

```bash
bash examples/3d_rae2822_hybrid/run.sh build cuda
```

The script runs Mach 0.78, alpha 5 degrees, Re 6.5M, reference area 0.1,
first-order mean and SA convection, five block-DP-LUR sweeps, and fixed CFL
0.1.  It first writes a binary restart, then performs a zero-step CSV reload
for plotting.  Never reuse the input restart directory as output.

## 4. Continue a stopped run

Repeat the same solver command with:

```text
--restart old_stage/state.bin --out new_stage
```

Do not change mesh ordering, Mach, Reynolds number, reference area, or SA
scaling across the restart.  A spatial-order change is allowed as a deliberate
continuation, but it changes the discrete residual and should be staged.

## 5. Acceptance criteria

- no rejected/invalid updates;
- conservation-balance components near floating-point roundoff;
- CPU/CUDA fixed-step states agree near roundoff;
- all six residuals and the load window are stable before validation claims.

The retained final segment reports CL/CD/CM =
0.803917/0.096155/-0.098244 and passed the solver's residual gate.  Its complete
restart lineage and comparison to the intended Fluent reference must still be
consolidated; do not treat a single segment time as total solution time.
