# Full 3-D sphere runbook

## Purpose

This is the self-contained 3-D regression case.  It proves that the solver is
not a two-dimensional extrusion: all three velocity components, closed 3-D
polyhedral volumes, wall forces, positivity, restart, and CPU/CUDA operators
are exercised.

The mesh is a six-block cubed-sphere shell.  Radial layers are exponentially
clustered toward the faceted unit sphere; the outer radius is 8.  The retained
coarse case has 7,680 cells and 46,848 triangular faces.

## 1. Build CPU or CUDA

```bash
cmake -S . -B build_cpu -DCFD_CUDA=OFF -DCFD_OPENMP=ON
cmake --build build_cpu -j 8
ctest --test-dir build_cpu --output-on-failure
```

For CUDA, use `build`, `-DCFD_CUDA=ON`, and architecture 89.

## 2. Generate and audit the mesh

```bash
python3 solvers/3d/make_mesh.py \
  --out output/3d_sphere_cpu/mesh --n 8 --nr 20
```

The generator fails if cell volumes are non-positive, cell area-vector closure
exceeds tolerance, or total cell volume disagrees with the closed boundaries.
Compare `mesh.json` with `reference_mesh.json`.

## 3. Run the smoke case

```bash
bash examples/3d_sphere/run_smoke.sh build_cpu cpu
```

The script uses Mach 1.65, Re 10,000, matrix-free block DP-LUR, CFL 0.1, five
sweeps, 50% limited second-order mean-flow reconstruction, and first-order SA
convection.  Output includes `history.csv`, `result.json`, `state.bin`, and
`surface.csv`.

## 4. Restart without overwriting the input

```bash
build_cpu/solvers/3d/sphere3d \
  --mesh output/3d_sphere_cpu/mesh/mesh.txt \
  --restart output/3d_sphere_cpu/result/state.bin \
  --out output/3d_sphere_cpu/restart \
  --backend cpu --integrator dplur --mach 1.65 --re 10000 \
  --cfl 0.1 --cfl-max 0.1 --dplur-block-coupling 1 \
  --spatial-order 2 --reconstruction-blend 0.5 \
  --iterations 20 --budget 10 --binary-output 1
```

The solver rejects a restart output directory equal to the checkpoint's input
directory.

## 5. CPU/CUDA acceptance checks

Use identical meshes, options, and step counts.  Compare all six conserved
variables after 20 steps.  The retained full-3-D development comparison is at
approximately 1e-15 relative Linf; positivity and conservation identities
passed.  This coarse sphere is a regression, not a grid-independent sphere
drag benchmark.
