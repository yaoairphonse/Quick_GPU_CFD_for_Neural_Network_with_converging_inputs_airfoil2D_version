# NASA CRM Cartesian-AMR showcase

All flow slices map QCFDAM1 control-volume averages directly through the
procedural block/cell ownership tables.  Solid pixels come from exact plane
intersections with the CRM surface.  No scattered interpolation is used.

- [Symmetry-near x-z Mach and Cp](crm_xz_symmetry_mach_cp.png)
- [Horizontal x-y wing Mach and Cp](crm_xy_wing_mach_cp.png)
- [Full-span y-z Mach and Cp at x/MAC=5](crm_yz_fullwing_mach_cp.png)
- [Cp painted on the CRM surface](crm_surface_cp.png)
- [True AMR section overview](crm_amr_true_mesh_sections.png)
- [y-z block overview and actual-cell zoom at x/MAC=5](crm_amr_yz_x5_mesh.png)

The mesh has 21,523,036 agglomerated controls and a minimum Cartesian spacing
of 0.00390625 MAC.  The displayed state is a transient development state; it
is not a converged CRM drag result.
