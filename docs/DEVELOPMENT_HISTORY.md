# Step-by-step development history

1. **Regular-grid Euler core.** Established conservative 1-D/2-D Euler
   states, Riemann fluxes, limiters, and explicit time integration.
2. **Immersed geometry.** Added signed-distance/immersed-boundary experiments,
   then replaced qualitative wall masking with conservative polygon cut cells.
3. **LF1/LF2 static AMR.** Added two fast Cartesian fidelity levels, exact
   apertures and wall segments, small-cell stabilization, integrated loads, and
   structured exports for NNLCI.
4. **Viscous RANS.** Added compressible mechanical viscous terms, Sutherland
   or constant viscosity, wall distance, and SA-neg on CPU/CUDA.
5. **Low-Mach work.** Added density-based Turkel-style preconditioning and
   matched CPU/CUDA operators. This did not introduce a pressure-based solver.
6. **Second-order 2-D accuracy.** Added agglomerated-control least-squares
   primitive gradients, multidimensional limiting, reconstructed face/wall
   pressure, and the CL/CM/Cp convergence gate. NACA4412 became the retained
   fast accuracy case.
7. **True three-dimensional solver.** Generalized the residual to explicit
   owner/neighbour faces and six conserved variables. Built a six-block
   cubed-sphere case to verify closed 3-D conservation, the third velocity
   component, positivity, restart, and CPU/CUDA agreement.
8. **NASA CRM tetra import.** Imported the public 10.07M-cell CGNS grid,
   preserved ordering and zones, built binary solver/connectivity tables, wall
   distance, restart, surface loads, and diagnostics.
9. **Convergence and accuracy tools.** Added limited reconstruction,
   matrix-free block DP-LUR, adaptive CFL, persistent-reversal detection,
   metric-best checkpoints, rollback, node gradients, Mitchell face gradients,
   SA residual audits, and an agglomeration residual-smoothing prototype.
10. **Mixed hybrid consumer.** Converted RAE2822 triangles/quads into wedges
    and hexahedra without tetrahedral splitting, proving that the explicit-face
    solver consumes mixed 3-D cells.
11. **Direct CRM Cartesian AMR.** Extracted the closed CRM surface, created
    balanced 8-cell octree blocks, exact wall clipping, cut volumes/apertures,
    small-cell agglomeration, procedural regular residuals, conservative 2:1
    subfaces, wall distance, and the compact QCFDAM1 format.
12. **AMR accuracy refinement.** Progressed from 7.66M controls to 14.01M
    all-wall refinement and 21.52M targeted upper-wing Level-10 refinement.
    Geometry, uniform-flow, affine-gradient, and CPU/CUDA audits passed.
13. **Warm fine-grid initialization.** Added exact volume-weighted state
    prolongation between nested AMR meshes. The 14.0M-to-21.5M transfer
    preserved global conserved integrals to roundoff and avoided a full cold
    start.
14. **Honest result packaging.** Added direct leaf-cell slices, true square
    mesh sections, original-triangle surface Cp, canonical runbooks, status
    labels, and non-destructive cache/release manifests.

The remaining numerical milestone is a compatible dissipative SA face
operator/implicit linearization and long-window convergence before a targeted
30--40M CRM grid study.
