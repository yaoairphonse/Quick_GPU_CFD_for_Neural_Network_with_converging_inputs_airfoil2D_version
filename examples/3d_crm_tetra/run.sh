#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build}"
backend="${2:-cuda}"
case_root="output/3d_crm_tetra"
source_dir="${case_root}/source"
mesh_dir="${case_root}/mesh"
result_dir="${case_root}/node_stage1"
archive="${source_dir}/dpw-wbt0_med-10Mc_5.tgz"
cgns="${source_dir}/dpw-wbt0_med-10Mc_5.cgns"

mkdir -p "${source_dir}" "${mesh_dir}"
if [[ "${DOWNLOAD_NASA_CRM:-0}" == "1" ]]; then
  curl --fail -L -o "${archive}" \
    https://dpw.larc.nasa.gov/DPW4/unstructured_Larc/CellBase/dpw-wbt0_med-10Mc_5.tgz
  echo "0ffdbbfe9862f6a5226e78a42c17e22c72db37d9d5ec38fd6dc60fa4b2d465a2  ${archive}" | sha256sum -c -
  tar -xzf "${archive}" -C "${source_dir}"
fi
: "${CRM_CGNS:=${cgns}}"
[[ -f "${CRM_CGNS}" ]] || { echo "Set CRM_CGNS or DOWNLOAD_NASA_CRM=1"; exit 2; }

python3 solvers/3d/read_crm_cgns.py --source "${CRM_CGNS}" --out "${mesh_dir}"
"${build_dir}/solvers/3d/prepare_tetra" "${mesh_dir}" 16 gauss
"${build_dir}/solvers/3d/build_node_gradient" \
  "${mesh_dir}/connectivity.bin" "${mesh_dir}/mesh.bin" \
  "${mesh_dir}/node_gradient.bin" 16 "${mesh_dir}/node_gradient_audit.json" hybrid

set +e
"${build_dir}/solvers/3d/sphere3d" \
  --mesh "${mesh_dir}/mesh.bin" --connectivity "${mesh_dir}/connectivity.bin" \
  --node-gradient "${mesh_dir}/node_gradient.bin" --gradient node \
  --out "${result_dir}" --backend "${backend}" --integrator dplur \
  --mach 0.85 --aoa 2.5 --re 5000000 --mu-star 1.7e-7 \
  --reference-area 3.9092552169415784 \
  --moment-x 4.807469180565627 --moment-y 0 --moment-z 0.6451776649746193 \
  --cfl 0.03 --cfl-max 0.03 --dplur-sweeps 5 --dplur-omega 0.8 \
  --dplur-block-coupling 1 --spatial-order 2 --reconstruction-blend 0.5 \
  --sa-spatial-order 1 --iterations 50000 --budget 180 \
  --check-period 50 --tolerance 0.001 --binary-output 1 --residual-audit 1
solver_status=$?
set -e
if [[ ${solver_status} -ne 0 && ${solver_status} -ne 3 ]]; then
  exit "${solver_status}"
fi

echo "Result: ${result_dir}/result.json"
