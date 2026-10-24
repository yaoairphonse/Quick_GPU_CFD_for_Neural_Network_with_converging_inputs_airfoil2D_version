# Cartesian-AMR preprocessing tools

The production development path uses a closed triangulated wall to create a
pure isotropic Cartesian octree cut-cell mesh. Run the tools in this order:

1. `fast_surface_pipeline.py`: validate/orient the surface and write the
   reusable `column_plan.npz` geometry container;
2. `build_octree_plan.py`: create balanced 8x8x8 leaf blocks;
3. `build_interface_plan.py --surface-layer wall`: clip wall triangles and
   classify fluid/solid/cut procedural cells;
4. `validate_interface_plan.py`: check clipping tables;
5. `build_cut_cells.py --surface-layer wall`: compute volumes, centroids,
   apertures, wall fragments, and conservative small-cell agglomeration;
6. `build_cartesian_solver_tables.py`: assemble sparse block adjacency,
   procedural-control mapping, and geometry exceptions;
7. `pack_cartesian_amr.py`: write a compact `QCFDAM1` directory;
8. `prepare_amr_wall_distance` from `solvers/3d`: add exact BVH wall distance;
9. run `amr3d --geometry-audit 1 --uniform-audit 1` on CPU and CUDA.

`refine_octree_upper_wing.py` creates one additional level on every wall block
or on the selected upper-wing/shock region and applies sparse 2:1 closure.
Repeat stages 3--9 after every refinement.

The original prism-front planning options (`surface-layer outer`,
`run_pipeline.py`, and the NNLCI table exporters) are retained as experimental
research utilities. They are not used by the final pure cut-cell CRM case and
must not be described as a solver-ready prism/Cartesian hybrid mesher.

See `examples/3d_crm_amr/RUNBOOK.md` for exact retained commands.
