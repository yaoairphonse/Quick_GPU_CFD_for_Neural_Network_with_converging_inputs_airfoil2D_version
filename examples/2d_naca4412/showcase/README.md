# NACA 4412 showcase

These figures are rendered from the true Cartesian-AMR member rectangles and
the wall fragments used by the solver.  No scattered interpolation or
`griddata` is used.

- [LF1 field, mesh, and wall Cp](naca4412_lf1.png): 4,718 control volumes.
- [LF2 field, mesh, and wall Cp](naca4412_lf2.png): 9,517 control volumes.

The bundled pictures came from a clean CPU-only, time-budgeted rerun and are
explicitly labelled `not converged`.  They demonstrate the plotting and mesh
paths; the retained converged GPU comparison is summarized separately in
[`../reference/expected_result.json`](../reference/expected_result.json).
