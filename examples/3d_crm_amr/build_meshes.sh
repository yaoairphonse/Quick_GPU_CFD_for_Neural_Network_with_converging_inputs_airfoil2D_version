#!/usr/bin/env bash
set -euo pipefail

# This is a large workflow. CRM_SURFACE must be the closed full-model NPZ or STL.
: "${CRM_SURFACE:?Set CRM_SURFACE to CRM_full_closed.npz or CRM_full_closed.stl}"

build_dir="${1:-build}"
case_root="${2:-output/3d_crm_amr}"
plan_dir="${case_root}/surface_plan"
base_dir="${case_root}/base"
wall_dir="${case_root}/wall_refined"
upper_dir="${case_root}/upperwing_l10"

assemble_level() {
  local level_dir="$1"
  local packed_dir="$2"
  mkdir -p "${level_dir}"
  python3 tools/cartesian_amr/build_interface_plan.py \
    --column-plan "${plan_dir}/column_plan.npz" \
    --octree-plan "${level_dir}/octree_blocks.npz" \
    --out "${level_dir}/interface_plan" --surface-layer wall
  python3 tools/cartesian_amr/validate_interface_plan.py \
    --interface "${level_dir}/interface_plan" \
    --column-plan "${plan_dir}/column_plan.npz" \
    --octree-plan "${level_dir}/octree_blocks.npz"
  python3 tools/cartesian_amr/build_cut_cells.py \
    --interface "${level_dir}/interface_plan" \
    --column-plan "${plan_dir}/column_plan.npz" \
    --octree-plan "${level_dir}/octree_blocks.npz" \
    --out "${level_dir}/cut_cells" --surface-layer wall
  python3 tools/cartesian_amr/build_cartesian_solver_tables.py \
    --octree-plan "${level_dir}/octree_blocks.npz" \
    --interface "${level_dir}/interface_plan" \
    --cut-cells "${level_dir}/cut_cells" \
    --out "${level_dir}/solver_tables"
  python3 tools/cartesian_amr/pack_cartesian_amr.py \
    --solver-tables "${level_dir}/solver_tables" \
    --cut-cells "${level_dir}/cut_cells" \
    --interface "${level_dir}/interface_plan" \
    --column-plan "${plan_dir}/column_plan.npz" --out "${packed_dir}"
  "${build_dir}/solvers/3d/prepare_amr_wall_distance" "${packed_dir}" 16
}

python3 tools/cartesian_amr/fast_surface_pipeline.py \
  --surface "${CRM_SURFACE}" --out "${plan_dir}" --no-plot
python3 tools/cartesian_amr/build_octree_plan.py \
  --plan "${plan_dir}/column_plan.npz" --out "${base_dir}" \
  --target-cells 10000000 --block-cells 8 --farfield-padding 10 \
  --maximum-level 8 --refinement-band-cells 3
assemble_level "${base_dir}" "${base_dir}/packed"

python3 tools/cartesian_amr/refine_octree_upper_wing.py \
  --octree "${base_dir}/octree_blocks.npz" \
  --solver-tables "${base_dir}/solver_tables" \
  --packed-amr "${base_dir}/packed" --interface "${base_dir}/interface_plan" \
  --out "${wall_dir}" --scope all-wall
assemble_level "${wall_dir}" "${wall_dir}/packed"

python3 tools/cartesian_amr/refine_octree_upper_wing.py \
  --octree "${wall_dir}/octree_blocks.npz" \
  --solver-tables "${wall_dir}/solver_tables" \
  --packed-amr "${wall_dir}/packed" --interface "${wall_dir}/interface_plan" \
  --out "${upper_dir}" --scope upper-wing \
  --x-min 3.0 --x-max 7.2 --abs-y-min 0.4 --abs-y-max 4.3 --normal-z -0.2
assemble_level "${upper_dir}" "${upper_dir}/packed"

echo "Final packed mesh: ${upper_dir}/packed"
