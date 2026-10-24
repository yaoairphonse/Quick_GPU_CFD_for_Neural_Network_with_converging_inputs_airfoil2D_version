# Reproducible example matrix

The retained case families include English runbooks, final development
settings, small reference metadata, and true-mesh showcase figures.

| Example | Self-contained | Purpose | Status |
|---|---|---|---|
| `2d_naca4412` | yes | fast LF1/LF2 low-Mach cut-cell accuracy | reference gate passed; gallery rerun is time limited |
| `2d_rae2822_amr` | geometry bundled | fast transonic polygon-to-AMR path | development state unconverged |
| `3d_sphere` | yes | full-3-D conservation, positivity, restart, CPU/CUDA | verification passed |
| `3d_rae2822_hybrid` | requires archived NPZ inputs | mixed wedge/hexa consumer | LF1 gate passed; LF2 gallery unconverged |
| `3d_crm_tetra` | downloads public NASA archive | imported 10.07M tetra path | solver works; aerodynamic result unconverged |
| `3d_crm_amr` | requires CRM surface and substantial disk/GPU memory | procedural multi-block cut-cell path | audits pass; 21.52M result unconverged |

Run each command from the repository root.  Output belongs under `output/` or
an external cache, never inside the source directories.  A restart must always
write to a new directory so that its input checkpoint is preserved.

## Visual showcase

Every image below is generated from true mesh connectivity or procedural
cell ownership.  None uses `griddata` or another scattered-field interpolation.

| Case | Gallery |
|---|---|
| NACA4412 Cartesian AMR | [LF1](2d_naca4412/showcase/naca4412_lf1.png) · [LF2](2d_naca4412/showcase/naca4412_lf2.png) · [notes](2d_naca4412/showcase/README.md) |
| RAE2822 Cartesian AMR | [LF2 field/mesh/Cp](2d_rae2822_amr/showcase/rae2822_cartesian_amr_lf2.png) · [notes](2d_rae2822_amr/showcase/README.md) |
| 3-D sphere | [surface/field/mesh](3d_sphere/showcase/sphere_m165_exact_mesh.png) · [notes](3d_sphere/showcase/README.md) |
| RAE2822 wedge/hexa | [LF2 field/mesh/Cp](3d_rae2822_hybrid/showcase/rae2822_lf2_hybrid.png) · [notes](3d_rae2822_hybrid/showcase/README.md) |
| CRM tetrahedral | [exact wing section](3d_crm_tetra/showcase/crm_tetra_exact_section.png) · [notes](3d_crm_tetra/showcase/README.md) |
| CRM Cartesian AMR | [symmetry flow](3d_crm_amr/showcase/crm_xz_symmetry_mach_cp.png) · [x=5 full-wing flow](3d_crm_amr/showcase/crm_yz_fullwing_mach_cp.png) · [surface Cp](3d_crm_amr/showcase/crm_surface_cp.png) · [true mesh](3d_crm_amr/showcase/crm_amr_yz_x5_mesh.png) · [notes](3d_crm_amr/showcase/README.md) |
