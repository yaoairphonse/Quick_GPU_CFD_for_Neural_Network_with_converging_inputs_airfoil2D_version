# RAE2822 Cartesian-AMR showcase

[The LF2 figure](rae2822_cartesian_amr_lf2.png) paints the solver's control-
volume averages directly on their original quadtree member rectangles and
plots the same wall pressure used by the load integration.  No scattered
interpolation is used.

This is a 60-second-class CPU development rerun with 9,399 control volumes.
It is not converged and is included to demonstrate the fast polygon-to-AMR
workflow, not as an aerodynamic validation result.
