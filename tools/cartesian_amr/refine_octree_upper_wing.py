#!/usr/bin/env python3
"""Refine selected CRM wall-cut blocks by one level with sparse 2:1 closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_octree_plan import refine_blocks


FACE_DTYPE = np.dtype({
    "names": ["left", "right", "bc", "x", "normal", "area", "distance"],
    "formats": ["<i4", "<i4", "<i4", ("<f8", (3,)),
                ("<f8", (3,)), "<f8", "<f8"],
    "offsets": [0, 4, 8, 16, 40, 64, 72],
    "itemsize": 80,
})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--octree", required=True, type=Path)
    parser.add_argument("--solver-tables", required=True, type=Path)
    parser.add_argument("--packed-amr", required=True, type=Path)
    parser.add_argument("--interface", type=Path,
                        help="original interface tables; required for exact all-wall ownership")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scope", choices=("upper-wing", "all-wall"),
                        default="upper-wing")
    parser.add_argument("--x-min", type=float, default=3.0)
    parser.add_argument("--x-max", type=float, default=7.2)
    parser.add_argument("--abs-y-min", type=float, default=0.4)
    parser.add_argument("--abs-y-max", type=float, default=4.3)
    parser.add_argument("--normal-z", type=float, default=-0.2)
    args = parser.parse_args()

    with np.load(args.octree) as source:
        low = np.asarray(source["low"])
        size = np.asarray(source["size"])
        level = np.asarray(source["level"])
        root_low = np.asarray(source["root_low"])
        root_size = float(source["root_size"])
        block_cells = int(source["block_cells"])
    original_count = len(low)
    representative = np.load(
        args.solver_tables / "control_representative_procedural_cell.npy",
        mmap_mode="r",
    )
    offsets = np.load(
        args.solver_tables / "octree_block_neighbor_offsets.npy", mmap_mode="r"
    )
    neighbours = np.load(
        args.solver_tables / "octree_block_neighbors.npy", mmap_mode="r"
    )
    manifest = json.loads((args.packed_amr / "manifest.json").read_text())
    wall = np.memmap(
        args.packed_amr / "wall_faces.bin", dtype=FACE_DTYPE, mode="r",
        shape=int(manifest["wall_faces"]),
    )
    xyz = wall["x"]
    upper = ((xyz[:, 0] >= args.x_min) & (xyz[:, 0] <= args.x_max) &
             (np.abs(xyz[:, 1]) >= args.abs_y_min) &
             (np.abs(xyz[:, 1]) <= args.abs_y_max) &
             (wall["normal"][:, 2] <= args.normal_z))
    if args.interface is not None:
        fragment_cells = np.load(
            args.interface / "interface_fragment_cartesian_cell.npy", mmap_mode="r"
        )
        if len(fragment_cells) != len(wall):
            raise ValueError("interface fragments and packed wall faces disagree")
        face_blocks = np.asarray(fragment_cells) // (block_cells ** 3)
    else:
        if args.scope == "all-wall":
            raise ValueError("--interface is required with --scope all-wall")
        face_blocks = representative[wall["left"]] // (block_cells ** 3)
    requested = np.unique(face_blocks if args.scope == "all-wall" else face_blocks[upper])
    maximum_level = int(level.max()) + 1
    if np.any(level[requested] + 1 > maximum_level):
        raise ValueError("requested refinement exceeds one new octree level")

    marked = np.zeros(original_count, dtype=bool)
    marked[requested] = True
    closure_history = []
    for iteration in range(maximum_level + 1):
        effective = level + marked.astype(level.dtype)
        added = []
        for block in range(original_count):
            start, stop = int(offsets[block]), int(offsets[block + 1])
            adjacent = neighbours[start:stop]
            difference = effective[block] - effective[adjacent]
            if np.any(difference > 1):
                added.extend(adjacent[difference > 1].tolist())
        added = np.unique(added).astype(np.int64) if added else np.empty(0, dtype=np.int64)
        added = added[~marked[added]]
        closure_history.append({"iteration": iteration, "added_coarse_blocks": len(added)})
        if not len(added):
            break
        marked[added] = True
    else:
        raise RuntimeError("2:1 closure did not terminate")

    low, size, level = refine_blocks(low, size, level, marked)
    destination = args.out
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "octree_blocks.npz", low=low, size=size, level=level,
        root_low=root_low, root_size=root_size,
        block_cells=np.int32(block_cells),
    )
    values, counts = np.unique(level, return_counts=True)
    report = {
        "status": "ISOTROPIC_WALL_REFINEMENT_COMPLETE",
        "original_leaf_blocks": original_count,
        "scope": args.scope,
        "requested_wall_blocks": len(requested),
        "two_to_one_closure_blocks": int(np.count_nonzero(marked) - len(requested)),
        "total_refined_parent_blocks": int(np.count_nonzero(marked)),
        "new_leaf_blocks": len(low),
        "procedural_cells": len(low) * block_cells ** 3,
        "minimum_cell_size": float(np.min(size) / block_cells),
        "maximum_level": int(level.max()),
        "level_counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
        "selection": ({
            "x": [args.x_min, args.x_max],
            "abs_y": [args.abs_y_min, args.abs_y_max],
            "fluid_outward_normal_z_lte": args.normal_z,
        } if args.scope == "upper-wing" else {
            "rule": "every original leaf block owning at least one wall fragment",
        }),
        "closure_history": closure_history,
        "mesh_policy": "pure isotropic Cartesian AMR; no wall-normal or prism layers",
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
