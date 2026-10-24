"""Build a fixed-budget Cartesian block-octree plan around a prism front.

This stage is independent of third-party volume meshers.  Each leaf contains a
regular block of cells suitable for GPU kernels.  Refinement follows the outer
boundary-layer surface and stops at an explicit cell budget.  This stage also
enforces 2:1 face balance.  Solid classification, cut-cell geometry, and
conservative prism/octree coupling are subsequent stages; this tool reports them
as unresolved instead of claiming a solver-ready volume.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


CHILD_OFFSETS = np.array(
    [[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)],
    dtype=np.float64
)


def surface_distance_lower_bound(centers, tree, triangle_radius,
                                 neighbors=8):
    k = min(neighbors, len(triangle_radius))
    distance, triangle = tree.query(centers, k=k, workers=-1)
    if k == 1:
        distance = distance[:, None]
        triangle = triangle[:, None]
    return np.min(distance - triangle_radius[triangle], axis=1)


def refine_blocks(low, size, level, selected):
    retained = ~selected
    child_size = size[selected] * 0.5
    child_low = (low[selected, None, :] +
                 CHILD_OFFSETS[None, :, :] * child_size[:, None, None])
    return (
        np.concatenate((low[retained], child_low.reshape(-1, 3))),
        np.concatenate((size[retained], np.repeat(child_size, 8))),
        np.concatenate((level[retained],
                        np.repeat(level[selected] + 1, 8))),
    )


def balance_blocks(low, size, level, root_low, root_size, maximum_level,
                   target_blocks):
    """Enforce a face-neighbor level jump of at most one."""
    if maximum_level > 8:
        raise ValueError(
            "the current exact balance audit supports maximum-level <= 8"
        )
    dimension = 1 << maximum_level
    history = []
    for balance_pass in range(maximum_level):
        owner = np.empty((dimension, dimension, dimension), dtype=np.int32)
        levels = np.empty_like(owner, dtype=np.int8)
        coordinate = np.rint(
            (low - root_low) * dimension / root_size
        ).astype(np.int32)
        span = (1 << (maximum_level - level)).astype(np.int32)
        for index in range(len(low)):
            i, j, k = coordinate[index]
            width = int(span[index])
            owner[i:i + width, j:j + width, k:k + width] = index
            levels[i:i + width, j:j + width, k:k + width] = level[index]
        marked = []
        for axis in range(3):
            left_slice = [slice(None)] * 3
            right_slice = [slice(None)] * 3
            left_slice[axis] = slice(None, -1)
            right_slice[axis] = slice(1, None)
            left_level = levels[tuple(left_slice)]
            right_level = levels[tuple(right_slice)]
            difference = left_level.astype(np.int16) - right_level.astype(np.int16)
            bad = np.abs(difference) > 1
            if np.any(bad):
                left_owner = owner[tuple(left_slice)][bad]
                right_owner = owner[tuple(right_slice)][bad]
                marked.append(np.where(difference[bad] < 0,
                                       left_owner, right_owner))
        if not marked:
            history.append({"pass": balance_pass, "refined_blocks": 0})
            return low, size, level, history, True
        selected_ids = np.unique(np.concatenate(marked))
        selected_ids = selected_ids[level[selected_ids] < maximum_level]
        available = (target_blocks - len(low)) // 7
        complete = len(selected_ids) <= available
        if not complete:
            selected_ids = selected_ids[:max(0, available)]
        history.append({"pass": balance_pass,
                        "requested_blocks": int(len(np.unique(np.concatenate(marked)))),
                        "refined_blocks": int(len(selected_ids)),
                        "budget_limited": not complete})
        if not len(selected_ids):
            return low, size, level, history, False
        selected = np.zeros(len(low), dtype=bool)
        selected[selected_ids] = True
        low, size, level = refine_blocks(low, size, level, selected)
        if not complete:
            return low, size, level, history, False
    return low, size, level, history, False


def plot_plan(path, low, size, level, surface_points):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout="constrained")
    plane_specs = [
        (2, float(np.median(surface_points[:, 2])), 0, 1, "XY block cut"),
        (1, 0.0, 0, 2, "XZ symmetry-plane block cut"),
    ]
    for axis, (normal_axis, location, horizontal, vertical, title) in zip(
            axes[:2], plane_specs):
        selected = ((low[:, normal_axis] <= location) &
                    (low[:, normal_axis] + size >= location))
        chosen = np.flatnonzero(selected)
        segments = []
        colors = []
        for index in chosen:
            x0, y0 = low[index, horizontal], low[index, vertical]
            x1, y1 = x0 + size[index], y0 + size[index]
            segments.extend(([(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)],
                             [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]))
            colors.extend([level[index]] * 4)
        collection = LineCollection(segments, array=np.asarray(colors),
                                    cmap="viridis", linewidths=.35)
        axis.add_collection(collection)
        axis.autoscale()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x / c")
        axis.set_ylabel(("y" if vertical == 1 else "z") + " / c")
        axis.set_title(f"{title} at {location:.3f} c")
        figure.colorbar(collection, ax=axis, label="Octree level")
    values, counts = np.unique(level, return_counts=True)
    axes[2].bar(values, counts, color="#267e9b")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Octree level")
    axes[2].set_ylabel("Leaf blocks")
    axes[2].set_title("Leaf-block distribution")
    figure.suptitle("Fixed-budget off-body Cartesian octree plan")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-cells", type=int, default=5_000_000)
    parser.add_argument("--block-cells", type=int, default=8)
    parser.add_argument("--farfield-padding", type=float, default=10.0)
    parser.add_argument("--maximum-level", type=int, default=9)
    parser.add_argument("--refinement-band-cells", type=float, default=3.0)
    parser.add_argument("--distance-neighbors", type=int, default=8)
    parser.add_argument("--surface-layer", choices=("wall", "outer"),
                        default="outer")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if (args.target_cells < args.block_cells ** 3 or
            args.block_cells < 2 or args.farfield_padding <= 0 or
            args.maximum_level < 1 or args.refinement_band_cells < 0):
        parser.error("invalid octree settings")

    start = time.perf_counter()
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    data = np.load(args.plan)
    points = np.asarray(data["points"], dtype=np.float64)
    triangles = np.asarray(data["triangles"], dtype=np.int64)
    normal = np.asarray(data["normal"], dtype=np.float64)
    column_scale = np.asarray(data["column_scale"], dtype=np.float64)
    offsets = np.asarray(data["offsets"], dtype=np.float64)
    surface_points = points if args.surface_layer == "wall" else (
        points + (offsets[-1] * column_scale)[:, None] * normal
    )
    triangle_points = surface_points[triangles]
    triangle_centers = triangle_points.mean(axis=1)
    triangle_radius = np.linalg.norm(
        triangle_points - triangle_centers[:, None, :], axis=2
    ).max(axis=1)
    tree = cKDTree(triangle_centers)
    stages = {"read_and_index_surface": time.perf_counter() - start}

    desired_low = surface_points.min(axis=0) - args.farfield_padding
    desired_high = surface_points.max(axis=0) + args.farfield_padding
    extent = float(np.max(desired_high - desired_low))
    root_size = float(2 ** np.ceil(np.log2(extent)))
    root_center = 0.5 * (desired_low + desired_high)
    root_low = root_center - 0.5 * root_size
    low = root_low[None, :]
    size = np.array([root_size])
    level = np.array([0], dtype=np.int16)
    block_cell_count = args.block_cells ** 3
    history = []

    checkpoint = time.perf_counter()
    for current_level in range(args.maximum_level):
        candidate_ids = np.flatnonzero(level == current_level)
        if not len(candidate_ids):
            break
        centers = low[candidate_ids] + 0.5 * size[candidate_ids, None]
        distance_lower = surface_distance_lower_bound(
            centers, tree, triangle_radius, args.distance_neighbors
        )
        half_diagonal = 0.5 * np.sqrt(3.0) * size[candidate_ids]
        cell_size = size[candidate_ids] / args.block_cells
        clearance = distance_lower - half_diagonal
        wanted = clearance <= args.refinement_band_cells * cell_size
        wanted_ids = candidate_ids[wanted]
        wanted_clearance = clearance[wanted]
        available = ((args.target_cells // block_cell_count - len(low)) // 7)
        selected_ids = wanted_ids
        budget_limited = False
        if len(selected_ids) > available:
            budget_limited = True
            if available <= 0:
                selected_ids = selected_ids[:0]
            else:
                closest = np.argpartition(wanted_clearance, available - 1)[:available]
                selected_ids = wanted_ids[closest]
        selected = np.zeros(len(low), dtype=bool)
        selected[selected_ids] = True
        history.append({
            "level": current_level,
            "candidate_blocks": int(len(candidate_ids)),
            "surface_requested_blocks": int(len(wanted_ids)),
            "refined_blocks": int(len(selected_ids)),
            "budget_limited": budget_limited,
        })
        if not len(selected_ids):
            break
        low, size, level = refine_blocks(low, size, level, selected)
    stages["octree_refinement"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    target_blocks = args.target_cells // block_cell_count
    low, size, level, balance_history, balance_complete = balance_blocks(
        low, size, level, root_low, root_size, args.maximum_level,
        target_blocks
    )
    stages["two_to_one_balance"] = time.perf_counter() - checkpoint

    centers = low + 0.5 * size[:, None]
    distance_lower = surface_distance_lower_bound(
        centers, tree, triangle_radius, args.distance_neighbors
    )
    half_diagonal = 0.5 * np.sqrt(3.0) * size
    possible_surface_intersection = distance_lower <= half_diagonal
    unresolved = possible_surface_intersection & (level < args.maximum_level)
    cell_size = size / args.block_cells
    np.savez_compressed(
        destination / "octree_blocks.npz", low=low, size=size, level=level,
        distance_lower_bound=distance_lower,
        possible_surface_intersection=possible_surface_intersection,
        root_low=root_low, root_size=root_size,
        block_cells=np.int32(args.block_cells)
    )
    if not args.no_plot:
        checkpoint = time.perf_counter()
        plot_plan(destination / "octree_plan.png", low, size, level,
                  surface_points)
        stages["preview_plot"] = time.perf_counter() - checkpoint

    report = {
        "status": ("OCTREE_PLAN_COMPLETE" if not np.any(unresolved) and balance_complete
                   else "OCTREE_PLAN_BUDGET_LIMITED"),
        "surface_layer": args.surface_layer,
        "scope": (
            "Block-octree topology plan with exact 2:1 face balance; signed "
            "solid classification, cut-cell geometry and conservative "
            "prism-front coupling remain required"
        ),
        "plan": str(Path(args.plan).resolve()),
        "leaf_blocks": len(low),
        "block_cells_per_axis": args.block_cells,
        "represented_cartesian_cells": int(len(low) * block_cell_count),
        "target_cartesian_cells": args.target_cells,
        "minimum_cell_size": float(cell_size.min()),
        "maximum_cell_size": float(cell_size.max()),
        "maximum_level": args.maximum_level,
        "level_counts": {
            str(int(value)): int(count) for value, count in
            zip(*np.unique(level, return_counts=True))
        },
        "possible_surface_intersection_blocks": int(
            np.count_nonzero(possible_surface_intersection)
        ),
        "unresolved_surface_blocks": int(np.count_nonzero(unresolved)),
        "root_low": root_low.tolist(),
        "root_size": root_size,
        "history": history,
        "balance_complete": balance_complete,
        "balance_history": balance_history,
        "stage_seconds": stages,
        "total_seconds": time.perf_counter() - start,
    }
    report_text = json.dumps(report, indent=2) + "\n"
    (destination / "report.json").write_text(report_text)
    (destination / "octree_report.json").write_text(report_text)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
