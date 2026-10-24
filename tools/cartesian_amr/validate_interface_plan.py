"""Independently validate conservative interface-plan binary tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--column-plan", required=True)
    parser.add_argument("--octree-plan", required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    directory = Path(args.interface)
    manifest = json.loads((directory / "manifest.json").read_text())
    arrays = {}
    for name, record in manifest["tables"].items():
        array = np.load(directory / record["file"], mmap_mode="r",
                        allow_pickle=False)
        if list(array.shape) != record["shape"] or array.dtype.str != record["dtype"]:
            raise ValueError(f"table metadata mismatch: {name}")
        arrays[name] = array

    offsets = arrays["interface_fragment_offsets"]
    points = arrays["interface_fragment_points"]
    triangles_of_fragment = arrays["interface_fragment_triangle"]
    cartesian_cells = arrays["interface_fragment_cartesian_cell"]
    centroids = arrays["interface_fragment_centroid"]
    area_vectors = arrays["interface_fragment_area_vector"]
    fragment_count = len(triangles_of_fragment)
    if (offsets[0] != 0 or offsets[-1] != len(points) or
            np.any(offsets[1:] < offsets[:-1]) or
            np.any(np.diff(offsets) < 3)):
        raise ValueError("invalid interface-fragment CSR")

    with np.load(args.column_plan) as plan:
        wall_points = np.asarray(plan["points"], dtype=np.float64)
        surface_triangles = np.asarray(plan["triangles"], dtype=np.int32)
        outer_points = wall_points + (
            float(plan["offsets"][-1]) *
            np.asarray(plan["column_scale"], dtype=np.float64)
        )[:, None] * np.asarray(plan["normal"], dtype=np.float64)
    source = outer_points[surface_triangles]
    source_area_vectors = 0.5 * np.cross(
        source[:, 1] - source[:, 0], source[:, 2] - source[:, 0]
    )
    recovered = np.zeros_like(source_area_vectors)
    np.add.at(recovered, triangles_of_fragment, area_vectors)
    area_error = np.linalg.norm(recovered - source_area_vectors, axis=1)
    source_area = np.linalg.norm(source_area_vectors, axis=1)
    maximum_area_error = float(np.max(area_error / source_area))
    orientation = np.einsum(
        "ij,ij->i", area_vectors, source_area_vectors[triangles_of_fragment]
    )
    if np.any(orientation <= 0) or maximum_area_error > 1e-10:
        raise ValueError("interface orientation or area partition is invalid")

    with np.load(args.octree_plan) as octree:
        block_low = np.asarray(octree["low"], dtype=np.float64)
        block_size = np.asarray(octree["size"], dtype=np.float64)
        block_level = np.asarray(octree["level"], dtype=np.int16)
        block_cells = int(octree["block_cells"])
    cells_per_block = block_cells ** 3
    containment_failures = 0
    tolerance_factor = 2e-12
    chunk = 100_000
    for first in range(0, fragment_count, chunk):
        last = min(fragment_count, first + chunk)
        cells = np.asarray(cartesian_cells[first:last], dtype=np.int64)
        blocks = cells // cells_per_block
        local = cells % cells_per_block
        local_i = local % block_cells
        local_j = (local // block_cells) % block_cells
        local_k = local // (block_cells ** 2)
        cell_size = block_size[blocks] / block_cells
        low = block_low[blocks] + np.column_stack(
            (local_i, local_j, local_k)
        ) * cell_size[:, None]
        high = low + cell_size[:, None]
        tolerance = tolerance_factor * np.maximum(1.0, np.abs(high))
        fragment_lengths = np.diff(offsets[first:last + 1]).astype(np.int64)
        repeated_low = np.repeat(low, fragment_lengths, axis=0)
        repeated_high = np.repeat(high, fragment_lengths, axis=0)
        repeated_tolerance = np.repeat(tolerance, fragment_lengths, axis=0)
        selected_points = np.asarray(
            points[offsets[first]:offsets[last]], dtype=np.float64
        )
        outside = np.any(
            (selected_points < repeated_low - repeated_tolerance) |
            (selected_points > repeated_high + repeated_tolerance), axis=1
        )
        containment_failures += int(np.count_nonzero(outside))
        centroid_outside = np.any(
            (centroids[first:last] < low - tolerance) |
            (centroids[first:last] > high + tolerance), axis=1
        )
        containment_failures += int(np.count_nonzero(centroid_outside))
    if containment_failures:
        raise ValueError(
            f"{containment_failures} fragment coordinates lie outside their cell"
        )

    cut_cells = arrays["cut_cartesian_cells"]
    cut_offsets = arrays["cut_cell_fragment_offsets"]
    cut_fragments = arrays["cut_cell_fragments"]
    if (cut_offsets[0] != 0 or cut_offsets[-1] != len(cut_fragments) or
            len(cut_fragments) != fragment_count or
            np.any(cut_offsets[1:] <= cut_offsets[:-1])):
        raise ValueError("invalid cut-cell fragment CSR")
    for first in range(0, len(cut_cells), chunk):
        last = min(len(cut_cells), first + chunk)
        counts = np.diff(cut_offsets[first:last + 1]).astype(np.int64)
        expected = np.repeat(cut_cells[first:last], counts)
        actual = cartesian_cells[cut_fragments[
            cut_offsets[first]:cut_offsets[last]
        ]]
        if not np.array_equal(expected, actual):
            raise ValueError("cut-cell fragment grouping mismatch")

    status = arrays["cartesian_cell_status"]
    if (np.any(status > 2) or np.any(status[cut_cells] != 2) or
            np.count_nonzero(status == 2) != len(cut_cells)):
        raise ValueError("invalid Cartesian cell classification")
    cut_blocks = cut_cells // cells_per_block
    if np.any(block_level[cut_blocks] != block_level.max()):
        raise ValueError("cut cell found outside the finest octree level")

    cell_status = np.asarray(status).reshape(-1, cells_per_block)
    full_inside_volume = float(np.sum(
        np.count_nonzero(cell_status == 1, axis=1) *
        (block_size / block_cells) ** 3
    ))
    cut_bounding_volume = float(np.sum(
        np.count_nonzero(cell_status == 2, axis=1) *
        (block_size / block_cells) ** 3
    ))
    outer_volume = manifest["outer_surface_signed_volume"]
    volume_bracket_valid = (
        full_inside_volume <= outer_volume <=
        full_inside_volume + cut_bounding_volume
    )
    if not volume_bracket_valid:
        raise ValueError("Cartesian classification does not bracket outer volume")

    report = {
        "status": "INTERFACE_PLAN_VALIDATED",
        "solver_ready": False,
        "fragments": fragment_count,
        "fragment_vertices": len(points),
        "cut_cartesian_cells": len(cut_cells),
        "maximum_triangle_area_vector_relative_error": maximum_area_error,
        "coordinate_containment_failures": containment_failures,
        "full_inside_cell_volume": full_inside_volume,
        "cut_cell_bounding_volume": cut_bounding_volume,
        "outer_surface_volume": outer_volume,
        "volume_bracket_valid": volume_bracket_valid,
        "validation_seconds": time.perf_counter() - start,
    }
    (directory / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
