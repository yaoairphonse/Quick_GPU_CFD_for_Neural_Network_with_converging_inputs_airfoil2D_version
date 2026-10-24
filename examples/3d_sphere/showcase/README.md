# Three-dimensional sphere showcase

[The sphere figure](sphere_m165_exact_mesh.png) contains surface Cp, an exact
z=0 Mach cut, an exact z=0 Cp cut, and the true cubed-sphere cell intersections.
Each plane polygon is the analytical hexahedron/plane intersection and receives
its owning control-volume average; no scattered interpolation is used.

The pictured Mach 1.65 case has 7,680 control volumes and reached the recorded
residual gate.  It is a conservation, restart, and CPU/CUDA regression case,
not a sphere-drag validation benchmark.
