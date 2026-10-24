#!/usr/bin/env bash
set -euo pipefail

: "${CRM_COLUMN_PLAN:?Set CRM_COLUMN_PLAN to the surface-plan column_plan.npz}"
build_dir="${1:-build}"
backend="${2:-cuda}"
case_root="${3:-output/3d_crm_amr}"
mesh_dir="${case_root}/upperwing_l10/packed"
result_dir="${case_root}/runs/stage3_blend050"
audit_dir="${case_root}/audit_${backend}"
plot_dir="${result_dir}/diagnostics"

"${build_dir}/solvers/3d/amr3d" \
  --mesh "${mesh_dir}" --out "${audit_dir}" --backend "${backend}" \
  --geometry-audit 1 --uniform-audit 1 --iterations 0 --budget 1

python3 solvers/3d/plot_amr_flow_slices.py \
  --mesh "${mesh_dir}" --state "${result_dir}/state.bin" \
  --geometry "${CRM_COLUMN_PLAN}" --out "${plot_dir}"
python3 solvers/3d/plot_amr_mesh_sections.py \
  --mesh "${mesh_dir}" --geometry "${CRM_COLUMN_PLAN}" --out "${plot_dir}"
python3 solvers/3d/plot_amr_surface_cp.py \
  --run "${result_dir}" --geometry "${CRM_COLUMN_PLAN}" \
  --out "${plot_dir}/crm_surface_cp.png"

echo "Audit: ${audit_dir}"
echo "Plots: ${plot_dir}"
