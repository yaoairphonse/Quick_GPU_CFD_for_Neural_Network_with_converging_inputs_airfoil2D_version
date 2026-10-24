#!/usr/bin/env bash
set -euo pipefail

# Required external inputs. The archived files are intentionally not bundled.
: "${RAE_TOPOLOGY:?Set RAE_TOPOLOGY to LF1_full_topology.npz}"
: "${RAE_TRUE_MESH:?Set RAE_TRUE_MESH to LF1_true_mesh.npz}"

build_dir="${1:-build}"
backend="${2:-cuda}"
case_root="output/3d_rae2822_lf1"
mesh_dir="${case_root}/mesh"
result_dir="${case_root}/stage1"
post_dir="${case_root}/post"

python3 tools/hybrid/convert_rae2822_npz.py \
  --topology "${RAE_TOPOLOGY}" --true-mesh "${RAE_TRUE_MESH}" \
  --out "${mesh_dir}" --span 0.1

set +e
"${build_dir}/solvers/3d/sphere3d" \
  --mesh "${mesh_dir}/mesh.txt" --out "${result_dir}" \
  --backend "${backend}" --integrator dplur --gradient legacy \
  --mach 0.78 --aoa 5 --re 6500000 --mu-star 1.2e-7 \
  --reference-area 0.1 --cfl 0.1 --cfl-max 1 \
  --controller-start-iteration 1000000 \
  --dplur-sweeps 5 --dplur-omega 0.8 --dplur-block-coupling 1 \
  --spatial-order 1 --sa-spatial-order 1 \
  --iterations 100000 --budget 60 --check-period 250 \
  --tolerance 0.0001 --binary-output 1 --residual-audit 1
solver_status=$?
set -e
if [[ ${solver_status} -ne 0 && ${solver_status} -ne 3 ]]; then
  exit "${solver_status}"
fi

set +e
"${build_dir}/solvers/3d/sphere3d" \
  --mesh "${mesh_dir}/mesh.txt" --out "${post_dir}" \
  --restart "${result_dir}/state.bin" --backend "${backend}" \
  --integrator dplur --gradient legacy --mach 0.78 --aoa 5 \
  --re 6500000 --mu-star 1.2e-7 --reference-area 0.1 \
  --cfl 0.1 --cfl-max 1 --controller-start-iteration 1000000 \
  --dplur-sweeps 5 --dplur-omega 0.8 --dplur-block-coupling 1 \
  --spatial-order 1 --sa-spatial-order 1 --iterations 0 --budget 1 \
  --field-output csv
solver_status=$?
set -e
if [[ ${solver_status} -ne 0 && ${solver_status} -ne 3 ]]; then
  exit "${solver_status}"
fi

python3 tools/hybrid/plot_rae2822_hybrid_result.py \
  --geometry "${mesh_dir}/geometry.npz" --field "${post_dir}/field.csv" \
  --surface "${post_dir}/surface.csv" \
  --out "${case_root}/rae2822_lf1.png"

echo "Result: ${result_dir}/result.json"
