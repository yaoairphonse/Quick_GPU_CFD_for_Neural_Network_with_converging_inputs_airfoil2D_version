#!/usr/bin/env bash
set -euo pipefail

backend="${1:-cuda}"
executable="${2:-build/quick_airfoil}"
python3 quick_cfd.py run \
  --points examples/2d_naca4412/geometry.csv --point-format polygon \
  --config examples/2d_naca4412/case.json \
  --out output/2d_naca4412 --levels LF1 LF2 \
  --backend "${backend}" --executable "${executable}"

echo "Open output/2d_naca4412/index.html"
