"""Clip a closed triangle surface into a balanced Cartesian block AMR mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from scipy.ndimage import binary_fill_holes


def compile_clipper(source_directory, destination, supplied):
    if supplied:
        executable = Path(supplied).resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"interface clipper not found: {executable}")
        return executable
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("a C++17 compiler is required to build the interface clipper")
    tool_directory = destination / ".tools"
    tool_directory.mkdir(parents=True, exist_ok=True)
    source = source_directory / "clip_interface.cpp"
    executable = tool_directory / "clip_interface"
    if (not executable.exists() or
            executable.stat().st_mtime_ns < source.stat().st_mtime_ns):
        subprocess.run(
            [compiler, "-O3", "-std=c++17", str(source), "-o", str(executable)],
            check=True
        )
    return executable


def save_array(destination, name, values, description, tables):
    path = destination / f"{name}.npy"
    np.save(path, values, allow_pickle=False)
    tables[name] = {
        "file": path.name,
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "bytes": path.stat().st_size,
        "description": description,
    }


def cartesian_cell_coordinates(cell_ids, block_coordinates, block_levels,
                               block_cells, maximum_level):
    """Return cell origins on the global maximum-level cell lattice."""
    cells_per_block = block_cells ** 3
    blocks = cell_ids // cells_per_block
    local = cell_ids % cells_per_block
    local_i = local % block_cells
    local_j = (local // block_cells) % block_cells
    local_k = local // (block_cells ** 2)
    span = (1 << (maximum_level - block_levels[blocks])).astype(np.int32)
    return (block_coordinates[blocks] * block_cells +
            np.column_stack((local_i, local_j, local_k)) * span[:, None]
            ).astype(np.int32)


def classify_cartesian_cells(cut_cells, fragment_centroids, block_coordinates,
                             block_levels, block_cells, maximum_level,
                             root_low, root_size):
    finest_cell_size = root_size / ((1 << maximum_level) * block_cells)
    cut_coordinates = cartesian_cell_coordinates(
        cut_cells, block_coordinates, block_levels, block_cells, maximum_level
    )
    cells_per_block = block_cells ** 3
    cut_blocks = cut_cells // cells_per_block
    cut_spans = (1 << (maximum_level - block_levels[cut_blocks])).astype(np.int32)
    surface_coordinates = np.floor(
        (fragment_centroids - root_low) / finest_cell_size
    ).astype(np.int32)
    surface_coordinates = np.unique(surface_coordinates, axis=0)
    padding = 2
    grid_low = surface_coordinates.min(axis=0) - padding
    grid_high = surface_coordinates.max(axis=0) + padding + 1
    shape = tuple(map(int, grid_high - grid_low))
    cut_mask = np.zeros(shape, dtype=bool)
    local_surface = surface_coordinates - grid_low
    cut_mask[tuple(local_surface.T)] = True
    surface_linear = np.unique(np.ravel_multi_index(local_surface.T, shape))
    binary_fill_holes(cut_mask, output=cut_mask)
    filled = cut_mask
    if not np.all(filled[tuple(local_surface.T)]):
        raise AssertionError("filled classification lost a cut cell")

    block_count = len(block_coordinates)
    status = np.zeros(block_count * cells_per_block, dtype=np.uint8)
    local_centers = np.asarray([
        (i + 0.5, j + 0.5, k + 0.5)
        for k in range(block_cells)
        for j in range(block_cells)
        for i in range(block_cells)
    ], dtype=np.float64)
    # Convert every leaf-cell center to a maximum-level lattice sample.
    for block in range(block_count):
        span = 1 << (maximum_level - int(block_levels[block]))
        finest = np.floor(
            block_coordinates[block] * block_cells + local_centers * span
        ).astype(np.int32)
        local = finest - grid_low
        valid = np.all((local >= 0) & (local < np.asarray(shape)), axis=1)
        block_status = np.zeros(cells_per_block, dtype=np.uint8)
        selected = local[valid]
        block_status[valid] = filled[tuple(selected.T)].astype(np.uint8)
        first = block * cells_per_block
        status[first:first + cells_per_block] = block_status
    status[cut_cells] = 2

    fine_counts = cut_spans.astype(np.int64) ** 3
    fine_offsets = np.empty(len(cut_cells) + 1, dtype=np.uint64)
    fine_offsets[0] = 0
    np.cumsum(fine_counts, out=fine_offsets[1:])
    fine_owners = np.repeat(np.arange(len(cut_cells), dtype=np.int32), fine_counts)
    fine_local = np.arange(int(fine_offsets[-1]), dtype=np.int64) - np.repeat(
        fine_offsets[:-1].astype(np.int64), fine_counts
    )
    fine_spans = cut_spans[fine_owners].astype(np.int64)
    fine_coordinates = cut_coordinates[fine_owners].astype(np.int64)
    fine_coordinates += np.column_stack((
        fine_local % fine_spans,
        (fine_local // fine_spans) % fine_spans,
        fine_local // (fine_spans ** 2),
    ))
    fine_grid = fine_coordinates - grid_low
    fine_valid = np.all(
        (fine_grid >= 0) & (fine_grid < np.asarray(shape)), axis=1
    )
    fine_status = np.zeros(len(fine_coordinates), dtype=np.uint8)
    selected = fine_grid[fine_valid]
    fine_status[fine_valid] = filled[tuple(selected.T)].astype(np.uint8)
    fine_linear = np.zeros(len(fine_coordinates), dtype=np.int64)
    fine_linear[fine_valid] = np.ravel_multi_index(selected.T, shape)
    locations = np.searchsorted(surface_linear, fine_linear[fine_valid])
    surface_hit = ((locations < len(surface_linear)) &
                   (surface_linear[np.minimum(locations, len(surface_linear) - 1)] ==
                    fine_linear[fine_valid]))
    valid_ids = np.flatnonzero(fine_valid)
    fine_status[valid_ids[surface_hit]] = 2

    voxel_volume = finest_cell_size ** 3
    return status, fine_offsets, fine_status, {
        "classification_grid_low": grid_low.tolist(),
        "classification_grid_shape": list(shape),
        "classification_finest_cell_size": finest_cell_size,
        "digitized_enclosed_volume_including_cut_cells": float(
            np.count_nonzero(filled) * voxel_volume
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column-plan", required=True)
    parser.add_argument("--octree-plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clipper")
    parser.add_argument("--surface-layer", choices=("wall", "outer"),
                        default="outer")
    args = parser.parse_args()
    start = time.perf_counter()
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    source_directory = Path(__file__).resolve().parent
    clipper = compile_clipper(source_directory, destination, args.clipper)

    with np.load(args.column_plan) as plan:
        wall_points = np.asarray(plan["points"], dtype=np.float64)
        triangles = np.asarray(plan["triangles"], dtype=np.int32)
        normals = np.asarray(plan["normal"], dtype=np.float64)
        scales = np.asarray(plan["column_scale"], dtype=np.float64)
        offsets = np.asarray(plan["offsets"], dtype=np.float64)
        surface_points = wall_points if args.surface_layer == "wall" else (
            wall_points + (offsets[-1] * scales)[:, None] * normals
        )
        layer_count = len(offsets) - 1
    with np.load(args.octree_plan) as octree:
        block_low = np.asarray(octree["low"], dtype=np.float64)
        block_size = np.asarray(octree["size"], dtype=np.float64)
        block_level = np.asarray(octree["level"], dtype=np.int16)
        root_low = np.asarray(octree["root_low"], dtype=np.float64)
        root_size = float(octree["root_size"])
        block_cells = int(octree["block_cells"])
    maximum_level = int(block_level.max())
    block_dimension = 1 << maximum_level
    block_coordinates = np.rint(
        (block_low - root_low) * block_dimension / root_size
    ).astype(np.int32)

    with tempfile.TemporaryDirectory(prefix="quick-cfd-interface-") as work_name:
        work = Path(work_name)
        surface_points.astype("<f8").tofile(work / "outer_points.f64")
        triangles.astype("<i4").tofile(work / "triangles.i32")
        block_coordinates.astype("<i4").tofile(work / "block_coordinates.i32")
        block_level.astype("<i2").tofile(work / "block_levels.i16")
        raw_output = work / "output"
        command = [
            str(clipper), str(work / "outer_points.f64"),
            str(work / "triangles.i32"),
            str(work / "block_coordinates.i32"),
            str(work / "block_levels.i16"), str(raw_output),
            str(len(surface_points)), str(len(triangles)), str(len(block_low)),
            *(str(value) for value in root_low), str(root_size),
            str(maximum_level), str(block_cells),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        clipping = json.loads(result.stdout.strip().splitlines()[-1])
        fragment_count = clipping["fragments"]
        vertex_count = clipping["fragment_vertices"]
        fragment_triangles = np.fromfile(
            raw_output / "fragment_triangle.i32", dtype="<i4",
            count=fragment_count
        )
        fragment_cells = np.fromfile(
            raw_output / "fragment_cartesian_cell.i32", dtype="<i4",
            count=fragment_count
        )
        fragment_counts = np.fromfile(
            raw_output / "fragment_count.u8", dtype="u1",
            count=fragment_count
        )
        fragment_points = np.fromfile(
            raw_output / "fragment_points.f64", dtype="<f8",
            count=3 * vertex_count
        ).reshape(-1, 3)
        fragment_centroids = np.fromfile(
            raw_output / "fragment_centroid.f64", dtype="<f8",
            count=3 * fragment_count
        ).reshape(-1, 3)
        fragment_area_vectors = np.fromfile(
            raw_output / "fragment_area_vector.f64", dtype="<f8",
            count=3 * fragment_count
        ).reshape(-1, 3)

    fragment_offsets = np.empty(fragment_count + 1, dtype=np.uint64)
    fragment_offsets[0] = 0
    np.cumsum(fragment_counts, out=fragment_offsets[1:])
    if fragment_offsets[-1] != len(fragment_points):
        raise ValueError("fragment vertex count mismatch")
    cut_cells, inverse = np.unique(fragment_cells, return_inverse=True)
    order = np.argsort(inverse, kind="stable").astype(np.int32)
    cut_counts = np.bincount(inverse, minlength=len(cut_cells)).astype(np.uint64)
    cut_offsets = np.empty(len(cut_cells) + 1, dtype=np.uint64)
    cut_offsets[0] = 0
    np.cumsum(cut_counts, out=cut_offsets[1:])

    original_triangles = surface_points[triangles]
    original_area_vectors = 0.5 * np.cross(
        original_triangles[:, 1] - original_triangles[:, 0],
        original_triangles[:, 2] - original_triangles[:, 0]
    )
    recovered = np.zeros_like(original_area_vectors)
    np.add.at(recovered, fragment_triangles, fragment_area_vectors)
    vector_error = np.linalg.norm(recovered - original_area_vectors, axis=1)
    original_area = np.linalg.norm(original_area_vectors, axis=1)
    maximum_vector_relative_error = float(np.max(vector_error / original_area))
    if maximum_vector_relative_error > 1e-10:
        raise ValueError(
            f"interface area-vector partition error {maximum_vector_relative_error}"
        )

    status, cut_fine_offsets, cut_fine_status, classification = classify_cartesian_cells(
        cut_cells, fragment_centroids, block_coordinates, block_level,
        block_cells, maximum_level, root_low, root_size
    )
    tables = {}
    prism_owner = ((layer_count - 1) * len(triangles) +
                   fragment_triangles).astype(np.int32)
    save_array(destination, "interface_fragment_offsets", fragment_offsets,
               "CSR offsets into interface fragment points.", tables)
    save_array(destination, "interface_fragment_points", fragment_points,
               "Ordered polygon vertices for each matching interface subface.", tables)
    save_array(destination, "interface_fragment_triangle", fragment_triangles,
               "Source outer-prism triangle for each subface.", tables)
    save_array(destination, "interface_fragment_prism_owner", prism_owner,
               "Last-layer prism cell owning each oriented subface.", tables)
    save_array(destination, "interface_fragment_cartesian_cell", fragment_cells,
               "Procedural Cartesian cell adjacent to each subface.", tables)
    save_array(destination, "interface_fragment_centroid", fragment_centroids,
               "Area centroid of each interface subface.", tables)
    save_array(destination, "interface_fragment_area_vector", fragment_area_vectors,
               "Area vector oriented from prism owner toward Cartesian fluid.", tables)
    save_array(destination, "cut_cartesian_cells", cut_cells.astype(np.int32),
               "Sorted unique procedural Cartesian cut-cell identifiers.", tables)
    save_array(destination, "cut_cell_fragment_offsets", cut_offsets,
               "CSR offsets for interface subfaces incident on each cut cell.", tables)
    save_array(destination, "cut_cell_fragments", order,
               "CSR interface-subface identifiers grouped by cut cell.", tables)
    save_array(destination, "cartesian_cell_status", status,
               "0 outside active, 1 fully inside overlap, 2 interface cut cell.", tables)
    save_array(destination, "cut_cell_fine_status_offsets", cut_fine_offsets,
               "CSR offsets for finest-lattice occupancy samples inside each cut cell.", tables)
    save_array(destination, "cut_cell_fine_status", cut_fine_status,
               "0 fluid, 1 solid, 2 surface-intersected finest voxel inside cut cells.", tables)

    signed_outer_volume = float(np.sum(np.einsum(
        "ij,ij->i", original_triangles[:, 0],
        np.cross(original_triangles[:, 1], original_triangles[:, 2])
    )) / 6.0)
    direct = args.surface_layer == "wall"
    report = {
        "mesh_workflow": ("cartesian_amr_cutcell_3d" if direct
                          else "body_fitted_hybrid_3d"),
        "surface_layer": args.surface_layer,
        "status": "INTERFACE_PLAN_VALID",
        "solver_ready": False,
        "scope": (
            "exact interface subfaces and Cartesian overlap classification; "
            "cut-cell closure, volumes, Cartesian apertures, and global faces remain"
        ),
        "connection": ("direct solid-wall cut cells" if direct else
                       "overlap removal plus conformal conservative link"),
        "overset": False,
        "immersed_boundary_forcing": False,
        "cutting_surface_triangles": len(triangles),
        "interface_fragments": fragment_count,
        "interface_fragment_vertices": int(vertex_count),
        "cut_cartesian_cells": len(cut_cells),
        "inside_cartesian_cells_removed": int(np.count_nonzero(status == 1)),
        "outside_cartesian_cells_retained": int(np.count_nonzero(status == 0)),
        "outer_surface_signed_volume": signed_outer_volume,
        "maximum_triangle_area_vector_relative_error": maximum_vector_relative_error,
        "clipping": clipping,
        "classification": classification,
        "tables": tables,
        "total_table_bytes": int(sum(record["bytes"] for record in tables.values())),
        "total_seconds": time.perf_counter() - start,
        "remaining": [
            "close every cut cell with Cartesian aperture polygons",
            "compute exact cut-cell volume and centroid",
            "assemble one global face owner/neighbour table",
            "run finite-volume geometric conservation audits",
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
