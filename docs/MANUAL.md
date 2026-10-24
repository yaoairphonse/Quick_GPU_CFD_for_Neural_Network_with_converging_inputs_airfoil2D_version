# Theory and operation manual

## 1. Scope

Quick CFD is a research code for rapid viscous compressible-flow calculations
and NNLCI data generation.  Development proceeded in four steps:

1. a two-dimensional structured Cartesian finite-volume solver;
2. immersed-boundary and conservative cut-cell airfoil geometry with LF1/LF2;
3. a three-dimensional explicit-face polyhedral consumer for sphere, tetra,
   wedge, and hexahedron meshes;
4. a procedural multi-block Cartesian-AMR cut-cell solver for full aircraft.

The code emphasizes repeatable geometry, conservation, CPU/CUDA agreement,
restartability, and time to a useful qualitative solution.  It is not yet a
certified aerodynamic-drag tool.

## 2. Capability status

| Capability | Status | Important qualification |
|---|---|---|
| 2-D Cartesian cut cells | tested | NACA4412 CL/CM/Cp development comparison |
| 3-D arbitrary explicit faces | tested | tetra, wedge, hexa, sphere polyhedra |
| 3-D multi-block Cartesian AMR | tested | procedural regular faces and 2:1 interfaces |
| Exact wall clipping and small-cell agglomeration | tested | geometry and uniform-flow audits pass |
| CPU and NVIDIA CUDA | tested | fixed-step agreement near roundoff |
| RK3 pseudo-time | tested | steady local pseudo-time, not physical time |
| DP-LUR | tested | explicit mesh has optional five-variable block coupling; AMR uses scalar spectral coupling |
| Limited second-order mean-flow reconstruction | tested development path | 50% blend retained for CRM AMR |
| Second-order SA convection | experimental | first order is the retained setting |
| SA-neg turbulence | implemented | convergence on anisotropic CRM remains limiting |
| Low-Mach preconditioning | implemented in 2-D | density-based; not pressure based |
| Agglomeration residual smoothing | prototype | not FAS multigrid |
| Coarse-to-fine AMR restart mapping | tested | same surface and nested root only |
| Pressure-based SIMPLE/PISO | not implemented | no pressure-correction equation |
| Fourier heat conduction/CHT | not implemented | current mechanical model uses zero conductivity |

## 3. Source architecture

### 3.1 Two-dimensional path

`quick_cfd.py` validates input and orchestrates geometry, LF1/LF2 meshing,
solver execution, and export.  `pipeline/` contains configuration, geometry,
and export logic.  `src/static_amr2d*` contains CPU/CUDA finite-volume,
cut-cell, turbulence, viscous, and reconstruction operators.

### 3.2 Three-dimensional explicit-face path

`solvers/3d/sphere3d` consumes `QCFD3D1` text or binary owner/neighbour meshes.
Every face stores one orientation, centroid, normal, area, distance, boundary
code, and owner/neighbour.  This common representation accepts tetrahedra and
mixed polyhedra without hard-coding a cell type in the residual loop.

### 3.3 Three-dimensional procedural-AMR path

`solvers/3d/amr3d` consumes packed `QCFDAM1` directories.  Regular 8x8x8 block
faces are generated in kernels rather than materialized.  Only block metadata,
2:1 interfaces, physical boundaries, cut-face exceptions, wall fragments, and
agglomerated control geometry are stored.  This is why it is called procedural.

`tools/cartesian_amr/` builds the octree, clips the surface, computes cut
geometry, agglomerates small cells, assembles tables, and packs the mesh.

## 4. Build and runtime environment

Requirements are CMake 3.18+, a C++17 compiler, Python 3, NumPy, SciPy,
Matplotlib, and Shapely.  CUDA builds require the CUDA toolkit and a supported
NVIDIA GPU.  CRM CGNS import currently expects the ParaView-provided VTK CGNS
shared library named in `read_crm_cgns.py`.

CUDA build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCFD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

CPU build:

```bash
cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release \
  -DCFD_CUDA=OFF -DCFD_OPENMP=ON
cmake --build build_cpu -j 8
ctest --test-dir build_cpu --output-on-failure
```

Generated meshes, fields, and checkpoints should be outside the source tree.

## 5. Governing variables and nondimensionalization

The 3-D conserved state is

\[
Q=(\rho,\rho u,\rho v,\rho w,\rho E,\rho\tilde\nu)^T.
\]

Pressure is recovered with the ideal-gas relation

\[
p=(\gamma-1)\left(\rho E-\tfrac12\rho\lVert\mathbf{u}\rVert^2\right).
\]

The standard reference state uses \(\rho_\infty=1\),
\(p_\infty=1/\gamma\), and velocity magnitude \(M_\infty\).  Consequently,

\[
C_p=\frac{p-p_\infty}{\tfrac12 M_\infty^2}.
\]

Lengths are normalized by the case reference chord or sphere radius.  Always
record the reference area and moment point with a state; otherwise integrated
coefficients cannot be reproduced.  For CRM, `mu-star=1.7e-7` corresponds to
Mach 0.85 and Re 5M under the current chord convention.

## 6. Density-based flux formulation

All current flow solvers advance the coupled conservative compressible state.
The inviscid face residual uses an approximate Riemann flux (the production
paths use HLLC-type fluxes; selected 2-D drivers also expose AUSM).  Viscous
fluxes use velocity/SA gradients and molecular plus eddy viscosity.

There is no pressure-correction equation, segregated pressure/velocity loop,
Rhie--Chow interpolation, SIMPLE, SIMPLEC, or PISO.  The 2-D low-Mach option
modifies density-based acoustic scaling and dissipation; it remains a
density-based method.

## 7. SA-neg turbulence model

The turbulence state is the transported working variable \(\tilde\nu\).  In
schematic nondimensional form, the implemented one-equation balance is

\[
\frac{\partial(\rho\tilde\nu)}{\partial t}
+\nabla\cdot(\rho\mathbf{u}\tilde\nu)
=P_{SA}-D_{SA}
+\frac{1}{\sigma}\nabla\cdot[(\mu+\rho\tilde\nu)\nabla\tilde\nu]
+\frac{c_{b2}}{\sigma}\rho\lVert\nabla\tilde\nu\rVert^2.
\]

Production, destruction, damping, eddy viscosity, and negative-working-
variable branches are evaluated by the SA-neg routines.  Wall distance enters
the destruction and modified-vorticity terms.  The no-slip wall sets the
appropriate zero working-variable wall state; the freestream initializes a
small positive value tied to molecular viscosity.

Retained settings use first-order upwind SA convection.  A second-order SA
option exists, but changing it causes a large discrete residual jump and has
not improved CRM convergence.  On the tetrahedral path, the full Mitchell SA
face gradient and its approximate implicit operator can develop a low-frequency
reversal on highly non-orthogonal cells.  A two-point/full-gradient blend was
added as a diagnostic continuation control, not as an accuracy recommendation.
The procedural AMR path uses a corrected two-point viscous face construction.

No k-epsilon, k-omega/SST, transition, DES, or LES model is implemented.

## 8. Two-dimensional Cartesian cut cells

The geometry is an ordered polygon.  Cartesian face apertures and wall
segments are clipped exactly.  Cell fluid area and centroid are recovered from
boundary integrals.  Small cells use conservative state redistribution or
agglomerated controls, depending on the path.  Wall pressure used in the
residual, exported Cp, CL, and CM is reconstructed consistently.

LF1 and LF2 are predefined resolution levels intended for fast surrogate input
generation.  The successful NACA4412 case uses multidimensional limited
least-squares primitive gradients and second-order face reconstruction.  The
viscous/SA discretization should not be described as globally second order.

## 9. Explicit-face unstructured and hybrid 3-D meshes

The residual visits each physical face once.  An interior face contributes
equal and opposite flux to owner and neighbour; a boundary face applies wall,
farfield, or symmetry data.  Cell-face CSR supports gradients and implicit
neighbour actions.

Tetrahedral CRM uses imported CGNS topology.  RAE2822 converts 2-D triangles
and quads into wedges and hexahedra.  Because the solver operates on face
geometry, neither path requires subdivision into tetrahedra.

The sphere generator is a separate six-block cubed-sphere topology.  Its
shared triangular faces are deduplicated, so conservation is tested on a
closed genuinely three-dimensional mesh.

## 10. Multi-block Cartesian AMR

The AMR mesh is genuinely multi-block.  A leaf block contains 8x8x8 uniform
Cartesian cells.  Blocks may be at different octree levels, but face-adjacent
levels differ by at most one.  In the retained CRM mesh the levels span 2--10.

At a same-level boundary, the kernel pairs matching cell faces.  At a 2:1
boundary, one coarse face is divided into the exact fine subfaces.  Each
subface produces one numerical flux that is applied with opposite signs to
the two controls.  The same subfaces are used in the Green--Gauss gradient.
This shared representation is essential: using one geometry for residuals and
another for gradients would lose affine exactness and can create interface
oscillations.

Far from the body, large coarse blocks reduce storage and work.  Near the CRM
surface and shock, Level-9/10 blocks supply small isotropic cells.  An overview
plot normally shows block squares, each of which contains 64 cells in a plane;
individual cell lines require a zoom because a Level-10 edge is only
0.00390625 MAC.

## 11. Three-dimensional cut-cell geometry

For every leaf block intersected by the triangulated wall:

1. surface triangles are clipped to Cartesian cells;
2. a finest-voxel status retains fluid, solid, and cut occupancy across mixed
   levels;
3. wall fragments retain exact polygon points, centroid, oriented area vector,
   and source-triangle ID;
4. Cartesian aperture polygons close the remaining control-volume boundary;
5. the divergence theorem gives fluid volume and first moment;
6. the centroid is first moment divided by volume;
7. area-vector closure and positivity are audited.

Tiny fluid fractions would impose severe explicit/implicit conditioning.  A
small cut cell is conservatively merged through its largest positive open
aperture.  The agglomerated control volume, centroid, member list, apertures,
and wall fragments are retained exactly.  Roundoff slivers are not silently
discarded; their moments are included in a valid neighbouring group.

The background carrier cell is a cube.  A cut cell's actual fluid control
volume is the clipped polyhedron, not the entire cube.

## 12. Wall distance

SA requires distance to the no-slip surface.  `prepare_amr_wall_distance`
builds an axis-aligned bounding-volume hierarchy over the exact source
triangles and performs nearest-triangle queries for all control centroids.
The 7.66M-control case required about 1.73 seconds; the 21.52M case about 2.89
seconds in the retained environment.

## 13. Gradients and second-order reconstruction

The legacy explicit-face gradient is cell based.  The tetrahedral accuracy
path constructs adaptive pseudo-Laplacian cell-to-node weights, exact
tetrahedral shape gradients, and face-node triples.  Mean-flow and SA diffusion
can then use a Mitchell-type face stencil.  This avoids the failed four-face
QR stencil on extreme tetrahedra.

Mean-flow second order reconstructs primitive variables from cell centers to
face centroids.  One Barth--Jespersen limiter per cell and primitive variable
is computed over all faces, preserving a single bounded slope.  Positivity
checks protect reconstructed density and pressure.

`spatial_order=2` therefore means limited second-order mean-flow convection.
It does not imply that turbulence, viscous terms, walls, and every irregular
interface are all globally second order.  CRM AMR currently retains a 0.5
reconstruction blend because full reconstruction was unstable.

### 13.1 Wall pressure, normals, and integrated loads

The 3-D AMR wall normal is not decorative metadata.  Every clipped wall
fragment stores its exact centroid, area, and oriented area vector.  The packed
solver convention is a unit normal from fluid into solid.  Pressure force on
the body is integrated fragment by fragment as

\[
\boldsymbol{F}_p = p_w\,\boldsymbol{n}_{f\rightarrow s}\,A,
\]

and viscous force uses the traction formed from the same normal and fragment
area.  Moments use the fragment centroid.  Fragment loads are subsequently
aggregated onto the original CRM triangles for `surface.csv`; the source-
triangle normal in that file does not replace the fragment-level integration.
The output also audits fragment coverage against every source triangle.

At first order, wall pressure is the owner-control pressure.  At second order,
the pressure used by both the wall flux/load and the surface output is the
limited primitive reconstruction from the control centroid to the exact
fragment centroid, including the configured reconstruction blend and pressure
positivity guard.  It is a numerical reconstruction, not calibration to NASA
data.

A future accuracy experiment may replace this centroid-gradient pressure with
one-sided samples taken along the surface normal into the fluid, followed by a
limited extrapolation back to the wall.  Such a wall-normal probe is not yet
implemented in the production static-AMR 2-D or procedural-AMR 3-D force path.
Pressure, density, and temperature must remain equation-of-state consistent;
they must not be corrected independently.

## 14. Pseudo-time integration

### 14.1 RK3

The explicit path uses a three-stage Runge--Kutta update with local steady
pseudo-time.  Its time coordinate is not physical unsteady time.

### 14.2 DP-LUR

DP-LUR approximately solves a linearized steady update with diagonal spectral
radii and forward/backward neighbour sweeps.  Conceptually,

\[
(D+L+U)\Delta Q=-R(Q),
\]

is approximated by lower and upper relaxation without assembling a global
matrix.  The explicit-face solver optionally applies a matrix-free
five-variable Euler normal-flux Jacobian action to mean-flow neighbour
corrections, with a separate scalar SA update.  The AMR solver currently uses
a less expensive scalar-spectral neighbour action.  More sweeps solve the
approximate system more closely but cannot repair an incompatible spatial
operator; a 5-vs-20-sweep SA experiment demonstrated this limitation.

## 15. Warm start and continuation controls

The safe development sequence is:

1. cold start with first-order convection and modest CFL;
2. verify positivity, conservation, and residual direction;
3. introduce 10--25% mean-flow reconstruction;
4. advance to 50% after both mean and SA residuals improve;
5. increase CFL only after multiple jointly improving reports;
6. cut CFL after persistent mild reversal;
7. rollback after severe growth;
8. preserve a metric-best binary checkpoint independently of the terminal
   state.

The explicit solver implements reconstruction ramps, adaptive CFL growth/cut,
cooldown, persistent-reversal detection, positivity reductions, best-state
storage, and rollback.  The AMR runbook stages separate fixed settings across
restart directories, making each operator change explicit.

## 16. Coarse-to-fine restart mapping

For nested QCFDAM1 meshes with the same root and wall, a mature coarse state
can initialize a refined mesh.  Ordinary target controls inherit the source
control containing their procedural representative.  Target agglomeration
groups receive the exact member-volume-weighted source state.  This preserves
conserved integrals and a uniform state to roundoff.

This is a mesh-refinement transfer, not a geometry-deformation interpolation.
After changing an optimized surface, rebuild and initialize with a separately
validated geometry-aware mapping or a cold start.

## 17. Agglomeration hierarchy versus multigrid

The tetrahedral path can build deterministic coarse groups and perform
volume-conservative restriction plus prolongated residual smoothing.  There is
no coarse finite-volume residual, tau correction, or nonlinear coarse solve.
It is therefore a residual-smoothing prototype, not full FAS multigrid.

## 18. Boundary conditions

- wall: stationary no-slip mechanical viscous wall, SA wall state, zero heat
  conduction in the current model;
- farfield: prescribed compressible freestream/reference state;
- symmetry: zero normal flow with tangential slip, used by the half CRM and
  RAE thin-slab caps;
- pressure outlet: not separately implemented in QCFD3D1; the RAE archive's
  outlet zone is explicitly mapped to farfield and reported.

Rotating the complete geometry and mesh while holding freestream horizontal is
physically equivalent to rotating freestream relative to an unrotated mesh.
Rotating only the wall without rebuilding the volume mesh is invalid.

## 19. Convergence and result provenance

Do not use a visually settled field as the only convergence criterion.  Record:

- all equation residuals, including SA;
- rejected or invalid updates;
- CL, CD, CM and pressure/viscous splits over a long window;
- near-field as well as global residuals;
- mesh, state, method, reference area, moment point, and executable provenance;
- at least two grid levels for an accuracy claim.

The explicit solver's current gate requires all six normalized residuals below
tolerance plus small CL/CD/CM interval changes for several reports.  Drag needs
a stricter long-window drag-count check.  The procedural AMR solver reports
`converged=false` by design and requires offline window assessment.

## 20. Validation hierarchy

1. geometry closure and positive volumes;
2. constant-state/uniform-flow preservation;
3. affine-gradient exactness;
4. CPU/CUDA fixed-step agreement;
5. restart identity;
6. positivity stress test;
7. residual and load-window convergence;
8. grid refinement;
9. comparison with independent CFD or experiment.

Passing a later visual comparison does not excuse failure of an earlier
conservation test.

## 21. Retained results and limitations

- NACA4412 LF1/LF2: fast CL/CM/Cp agreement with XFOIL; drag not validated.
- 3-D sphere: conservation/CPU-CUDA regression; not a drag benchmark.
- RAE2822 LF1: solver gate passed; full lineage and Fluent reference still
  need consolidation.
- CRM tetra: solver and diagnostics work, but SA and drag remain unconverged.
- CRM AMR 21.52M: best current CL/Cp; SA and loads still drift.

The next CRM work is not another arbitrary long run.  First reduce AMR memory,
make the SA face operator and implicit linearization a compatible dissipative
pair, re-converge the 21.52M mesh, and then place a 30--40M targeted refinement
using shock, wall, tip, junction, and wake indicators.

## 22. Reproducible cleanup

Before deleting a large cache, retain at minimum:

- this source package and its checksum manifest;
- external-data URL/checksum manifests;
- one canonical mesh manifest per mesh family;
- one canonical state with result/history/method/surface files;
- geometry and CPU/CUDA audit JSON;
- scripts required to rebuild each omitted large mesh.

Never retain a binary state without the exact mesh ordering and method
settings required to interpret it.

## 23. References

See [REFERENCES.md](REFERENCES.md) for the numerical-method literature,
benchmark sources, data standards, and external software projects.
