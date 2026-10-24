#!/usr/bin/env python3
"""Compare CRM AMR wall resolution with the NASA 10M tetrahedral mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROBABILITIES = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1]
CELL_DTYPE = np.dtype([
    ("x", "<f8", (3,)), ("volume", "<f8"), ("dwall", "<f8"),
    ("first", "<i4"), ("count", "<i4"),
    ("term_first", "<i4"), ("term_count", "<i4"),
])
FACE_DTYPE = np.dtype({
    "names": ["left", "right", "bc", "x", "normal", "area", "distance"],
    "formats": ["<i4", "<i4", "<i4", ("<f8", (3,)),
                ("<f8", (3,)), "<f8", "<f8"],
    "offsets": [0, 4, 8, 16, 40, 64, 72],
    "itemsize": 80,
})


def summary(values):
    return {
        "probabilities": PROBABILITIES,
        "quantiles": np.quantile(values, PROBABILITIES).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amr", required=True, type=Path)
    parser.add_argument("--solver-tables", required=True, type=Path)
    parser.add_argument("--nasa-native", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    amr_manifest = json.loads((args.amr / "manifest.json").read_text())
    control_count = int(amr_manifest["controls"])
    block_count = int(amr_manifest["blocks"])
    controls = np.memmap(args.amr / "controls.bin", dtype=np.dtype([
        ("x", "<f8", (3,)), ("volume", "<f8"), ("dwall", "<f8")
    ]), mode="r", shape=control_count)
    wall_distance = np.memmap(
        args.amr / "wall_distance.bin", dtype="<f8", mode="r", shape=control_count
    )
    wall_faces = np.memmap(
        args.amr / "wall_faces.bin", dtype=FACE_DTYPE, mode="r",
        shape=int(amr_manifest["wall_faces"]),
    )
    representative = np.load(
        args.solver_tables / "control_representative_procedural_cell.npy",
        mmap_mode="r",
    )
    block_size = np.load(args.solver_tables / "octree_block_size.npy", mmap_mode="r")
    wall_controls = np.unique(wall_faces["left"])
    wall_blocks = np.unique(representative[wall_controls] // 512)
    wall_h = block_size[representative[wall_controls] // 512] / 8
    wall_distance_values = wall_distance[wall_controls]

    face_block = representative[wall_faces["left"]] // 512
    x = wall_faces["x"]
    n = wall_faces["normal"]
    upper_wing = ((x[:, 0] >= 3.0) & (x[:, 0] <= 7.2) &
                  (np.abs(x[:, 1]) >= 0.4) & (np.abs(x[:, 1]) <= 4.3) &
                  (n[:, 2] < -0.2))
    shock_band = upper_wing & (x[:, 0] >= 4.3) & (x[:, 0] <= 6.3)
    upper_wing_blocks = np.unique(face_block[upper_wing])
    shock_blocks = np.unique(face_block[shock_band])

    with (args.nasa_native / "mesh.bin").open("rb") as stream:
        magic = stream.read(8)
        counts = np.fromfile(stream, dtype="<u8", count=4)
    if magic != b"QCFD3D1\0":
        raise RuntimeError("unexpected NASA mesh format")
    nasa_cells, nasa_faces = map(int, counts[:2])
    cells = np.memmap(
        args.nasa_native / "mesh.bin", dtype=CELL_DTYPE, mode="r",
        offset=40, shape=nasa_cells,
    )
    faces = np.memmap(
        args.nasa_native / "mesh.bin", dtype=FACE_DTYPE, mode="r",
        offset=40 + nasa_cells * CELL_DTYPE.itemsize, shape=nasa_faces,
    )
    nasa_wall_cells = np.unique(faces["left"][(faces["right"] < 0) & (faces["bc"] == 1)])
    nasa_wall_distance = cells["dwall"][nasa_wall_cells]
    nasa_effective_h = np.cbrt(cells["volume"][nasa_wall_cells])

    def refinement_estimate(selected_blocks):
        count = len(selected_blocks)
        # A wall-intersected block is approximately half fluid. Replacing one
        # 8^3 parent template by eight children adds about 7*256 active CVs.
        active_increment = 7 * 256 * count
        procedural_increment = 7 * 512 * count
        return {
            "selected_leaf_blocks": count,
            "estimated_controls_before_balance": control_count + active_increment,
            "estimated_procedural_cells_before_balance":
                int(amr_manifest["procedural_cells"]) + procedural_increment,
            "planning_range_with_10_to_30_percent_balance_overhead_controls": [
                int((control_count + active_increment) * 1.10),
                int((control_count + active_increment) * 1.30),
            ],
        }

    report = {
        "cell_counts": {
            "NASA_tetrahedra": nasa_cells,
            "AMR_procedural_cells": int(amr_manifest["procedural_cells"]),
            "AMR_active_controls": control_count,
            "AMR_active_to_NASA_ratio": control_count / nasa_cells,
        },
        "AMR_wall": {
            "unique_wall_controls": len(wall_controls),
            "wall_leaf_blocks": len(wall_blocks),
            "isotropic_spacing_MAC": summary(wall_h),
            "exact_centroid_wall_distance_MAC": summary(wall_distance_values),
            "wall_distance_over_spacing": summary(wall_distance_values / wall_h),
        },
        "NASA_wall": {
            "unique_wall_adjacent_tetrahedra": len(nasa_wall_cells),
            "wall_distance_MAC": summary(nasa_wall_distance),
            "effective_volume_spacing_MAC": summary(nasa_effective_h),
            "wall_distance_over_effective_spacing": summary(
                nasa_wall_distance / nasa_effective_h
            ),
        },
        "one_level_refinement_estimates": {
            "all_wall_blocks": refinement_estimate(wall_blocks),
            "upper_wing_blocks": refinement_estimate(upper_wing_blocks),
            "upper_wing_shock_band_blocks": refinement_estimate(shock_blocks),
        },
        "interpretation": [
            "AMR spacing is isotropic; NASA effective volume spacing hides strongly anisotropic boundary-layer tetrahedra.",
            "Control-count estimates assume wall blocks are half fluid and must be confirmed by rebuilding and 2:1 balancing.",
            "Pressure-focused upper-wing refinement is cheaper than refining every wall block; wall-resolved drag needs anisotropic normal layers.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
