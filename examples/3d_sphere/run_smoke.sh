#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Arguments: build directory and backend.
build_dir="${1:-build_cpu}"
backend="${2:-cpu}"
case_root="output/3d_sphere_${backend}"
mesh_dir="${case_root}/mesh"
result_dir="${case_root}/result"

python3 solvers/3d/make_mesh.py --out "${mesh_dir}" --n 8 --nr 20
set +e
"${build_dir}/solvers/3d/sphere3d" \
  --mesh "${mesh_dir}/mesh.txt" --out "${result_dir}" \
  --backend "${backend}" --integrator dplur \
  --mach 1.65 --re 10000 --aoa 0 --cfl 0.1 --cfl-max 0.1 \
  --dplur-sweeps 5 --dplur-omega 0.8 --dplur-block-coupling 1 \
  --spatial-order 2 --reconstruction-blend 0.5 \
  --sa-spatial-order 1 --iterations 20 --budget 30 \
  --check-period 20 --binary-output 1
solver_status=$?
set -e
if [[ ${solver_status} -ne 0 && ${solver_status} -ne 3 ]]; then
  exit "${solver_status}"
fi

echo "Result: ${result_dir}/result.json"
