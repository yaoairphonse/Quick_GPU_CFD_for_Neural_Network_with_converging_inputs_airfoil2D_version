#!/usr/bin/env bash
set -euo pipefail

backend="${1:-cpu}"
executable="${2:-build_cpu/quick_airfoil}"
case_root="${3:-output/2d_rae2822_amr}"

set +e
python3 quick_cfd.py run \
  --points examples/2d_rae2822_amr/geometry.csv --point-format polygon \
  --config examples/2d_rae2822_amr/case.json \
  --out "${case_root}" --levels LF2 \
  --backend "${backend}" --executable "${executable}"
status=$?
set -e
if [[ ${status} -ne 0 && ${status} -ne 3 ]]; then
  exit "${status}"
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/quick-cfd-showcase-mpl}" \
python3 tools/plot_cartesian_amr2d_result.py \
  --case "${case_root}/LF2" \
  --geometry examples/2d_rae2822_amr/geometry.csv \
  --out "${case_root}/rae2822_cartesian_amr_lf2.png" \
  --title "RAE2822 Cartesian AMR LF2 | M=0.78 | AOA=5 deg" \
  --bounds -0.35 1.80 -0.50 0.50

echo "Result: ${case_root}/LF2/result.json"
