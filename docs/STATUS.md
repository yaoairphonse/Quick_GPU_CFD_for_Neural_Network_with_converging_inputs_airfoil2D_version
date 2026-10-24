# Development status — 2026-10-24

## Retained results

| Case | Size | Result | Status |
|---|---:|---|---|
| NACA4412 LF1 | 4,718 CV | CL 0.66444, Cp RMS 0.01282 vs XFOIL | gate passed |
| NACA4412 LF2 | 9,517 CV | CL 0.66297, Cp RMS 0.01176 vs XFOIL | gate passed |
| 3-D sphere | 7,680 cells | CPU/CUDA about 1e-15 | regression passed |
| RAE2822 hybrid LF1 | 5,051 cells | CL/CD/CM 0.803917/0.096155/-0.098244 | solver gate passed; reference consolidation pending |
| CRM tetra | 10,067,380 cells | CL/CD/CM 0.581806/0.052443/-0.143849 | unconverged |
| CRM AMR base | 7,661,606 controls | CL/CD/CM 0.335620/0.025291/+0.131069 | unconverged |
| CRM AMR wall-refined | 14,011,654 controls | CL/CD/CM 0.357993/0.028121/-0.049050 | unconverged |
| CRM AMR upper-wing L10 | 21,523,036 controls | CL/CD/CM 0.504745/0.026189/-0.102533 | best current, unconverged |

NASA CRM USM3D comparison values are CL/CD/CM
0.54239/0.02995/-0.06249.  Endpoint proximity is not convergence evidence.

## Current blockers

1. SA-dominated low-frequency convergence on CRM.
2. AMR memory per control before a 30--40M mesh.
3. Full second-order viscous/SA verification.
4. Long-window load and pressure/viscous drag convergence.
5. External experimental/Fluent reference packaging.

## Next development phase

The next wall-accuracy experiment is an **optional wall-normal probe
reconstruction**, disabled by default.  The current centroid-gradient wall
state remains the baseline.  The proposed interface is:

```text
--wall-state-reconstruction centroid|normal-probe
--wall-probe-scope pressure|thermodynamic|viscous
--wall-probe-blend 0.0..1.0
--wall-probe-distance-1 1.5
--wall-probe-distance-2 2.5
```

Development will proceed in isolated stages: pressure-only wall flux/load
feedback with SA unchanged; equation-of-state-consistent pressure-density-
temperature reconstruction; and only then an experimental viscous/shear
extension.  Probe data will be frozen during each DP-LUR sweep group and
updated between outer iterations.  The method must not overwrite the first
control's conservative state, and it must pass uniform-flow, conservation,
positivity, CPU/CUDA, restart, and CRM A/B checks before becoming selectable in
a release.  This is a planned reconstruction option, not an implemented wall
function.

In parallel, implement and verify a compatible positive/dissipative SA face
operator and implicit linearization, reduce AMR GPU memory, reconverge the
21.52M mesh, and then perform targeted 30--40M refinement.
