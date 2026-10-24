# Development notes

## Current priorities, updated 2026-10-23

1. Publish a complete English edition of the README, Guide, source comments, CLI descriptions, and generated reports. Preserve the Chinese originals locally under `local_chinese/`.
2. Continue the local low-Mach preconditioning implementation and run a Mach 0.2 NACA4412 case.
3. Implement and run a full 3D viscous sphere case with a one-equation Spalart–Allmaras model and a no-slip wall. Keep the development local until its results have been verified.
4. Update the Guide with methods, reproducible commands, flow visualizations, and the actual validation/convergence status before publishing numerical extensions.

The published pipeline in this English edition remains two-dimensional, with no low-Mach preconditioning switch. Local implementation work must not be described as a released or validated capability.

## Low-Mach verification plan

The earlier priority, recorded on 2026-10-20, was **low-Mach preconditioning before heat transfer**.

- Review the density-based operators and primary references. Establish M=0.1, 0.01, and 0.001 baselines with explicit physical conditions and Reynolds numbers; add the requested M=0.2 NACA4412 case.
- Treat pseudo-time preconditioning and low-Mach flux dissipation consistently, including farfield/outlet boundaries, cut cells, viscosity, SA, and CPU/CUDA paths.
- Separate steady acceleration from physical unsteady evolution. A future dual-time method must preserve the correct physical-time derivative.
- Register pressure/velocity error, conservation, positivity, steady-state, and CPU/GPU tolerances before claiming validation. Retain LF1/LF2, ten-airfoil comparisons, and existing transonic/supersonic capability.
- Measure total time to a useful solution, steps, CUDA event time, advance wall time, and process time. Do not infer speedups from per-step cost or GPU utilization alone.

## Later thermal work

Add fluid Fourier heat fluxes, Pr/Prt, and prescribed-temperature/adiabatic walls. Solid internal temperatures additionally require a solid energy equation and fluid/solid interface coupling.

The current mechanical model has κ=0. Selecting a solid material does not implement heat transfer. Existing temperature output is recovered from total energy and the equation of state.

Entry points: `quick_cfd.py`, `config/default.json`, and `GUIDE.html`. The existing M=1.65 baseline uses SA-neg, Sutherland viscosity, fused kernels, and approximately 4,700/9,500 CVs for LF1/LF2.
