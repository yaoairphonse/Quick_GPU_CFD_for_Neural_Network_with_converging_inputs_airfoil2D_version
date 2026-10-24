#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build}"
backend="${2:-cuda}"
case_root="${3:-output/3d_crm_amr}"
mesh_dir="${case_root}/upperwing_l10/packed"
run_root="${case_root}/runs"
mkdir -p "${run_root}"

common=(
  --mesh "${mesh_dir}" --backend "${backend}" --integrator dplur
  --mach 0.85 --aoa 2.5 --re 5000000 --mu-star 1.7e-7
  --reference-area 7.818510433883157
  --moment-x 4.807469180565627 --moment-y 0 --moment-z 0.6451776649746193
  --dplur-sweeps 5 --dplur-omega 0.8 --check-period 25
)

if [[ -n "${WALLREF_STATE:-}" ]]; then
  python3 solvers/3d/transfer_amr_restart.py \
    --source-mesh "${case_root}/wall_refined/packed" \
    --source-state "${WALLREF_STATE}" --target-mesh "${mesh_dir}" \
    --target-solver-tables "${case_root}/upperwing_l10/solver_tables" \
    --target-interface "${case_root}/upperwing_l10/interface_plan" \
    --target-cut-cells "${case_root}/upperwing_l10/cut_cells" \
    --out "${run_root}/prolongated_state.bin" \
    --report "${run_root}/prolongation_report.json"
  restart=(--restart "${run_root}/prolongated_state.bin")
else
  restart=()
fi

"${build_dir}/solvers/3d/amr3d" "${common[@]}" "${restart[@]}" \
  --out "${run_root}/stage1_order1" --cfl 1 \
  --spatial-order 1 --reconstruction-blend 0 \
  --iterations 100000 --budget 120

"${build_dir}/solvers/3d/amr3d" "${common[@]}" \
  --restart "${run_root}/stage1_order1/state.bin" \
  --out "${run_root}/stage2_blend025" --cfl 1 \
  --spatial-order 2 --reconstruction-blend 0.25 \
  --iterations 100000 --budget 120

"${build_dir}/solvers/3d/amr3d" "${common[@]}" \
  --restart "${run_root}/stage2_blend025/state.bin" \
  --out "${run_root}/stage3_blend050" --cfl 1 \
  --spatial-order 2 --reconstruction-blend 0.5 \
  --iterations 100000 --budget 300

echo "Final staged result: ${run_root}/stage3_blend050/result.json"
