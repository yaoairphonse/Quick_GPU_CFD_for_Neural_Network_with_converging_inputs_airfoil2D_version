#!/usr/bin/env python3
"""Pack factorized Cartesian AMR tables into the procedural solver format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np


BLOCK_DTYPE = np.dtype([
    ("low", "<f8", (3,)), ("spacing", "<f8"),
    ("fine_coordinate", "<i4", (3,)), ("fine_span", "<i4"),
    ("first_procedural", "<i4"), ("level", "<i2"),
    ("kind", "u1"), ("padding", "u1"),
])
INTERFACE_DTYPE = np.dtype([
    ("first_block", "<i4"), ("second_block", "<i4"),
    ("first_side", "u1"), ("level_difference", "i1"),
    ("padding", "u1", (2,)),
])
BOUNDARY_DTYPE = np.dtype([
    ("block", "<i4"), ("side", "u1"), ("padding", "u1", (3,)),
])
CONTROL_DTYPE = np.dtype([
    ("x", "<f8", (3,)), ("volume", "<f8"), ("dwall", "<f8"),
])
FACE_DTYPE = np.dtype({
    "names": ["left", "right", "bc", "x", "n", "area", "distance"],
    "formats": ["<i4", "<i4", "<i4", ("<f8", (3,)), ("<f8", (3,)), "<f8", "<f8"],
    "offsets": [0, 4, 8, 16, 40, 64, 72],
    "itemsize": 80,
})


def local_coordinates(ids, block_cells):
    ids = np.asarray(ids, dtype=np.int64)
    return np.column_stack((
        ids % block_cells,
        (ids // block_cells) % block_cells,
        ids // (block_cells ** 2),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-tables", required=True, type=Path)
    parser.add_argument("--cut-cells", required=True, type=Path)
    parser.add_argument("--interface", required=True, type=Path)
    parser.add_argument("--column-plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.solver_tables / "manifest.json").read_text())
    block_cells = int(manifest["block_cells_per_axis"])
    cells_per_block = block_cells ** 3
    block_low = np.load(args.solver_tables / "octree_block_low.npy")
    block_size = np.load(args.solver_tables / "octree_block_size.npy")
    block_level = np.load(args.solver_tables / "octree_block_level.npy")
    block_coordinate = np.load(args.solver_tables / "octree_block_fine_coordinate.npy")
    block_span = np.load(args.solver_tables / "octree_block_fine_span.npy")
    block_kind = np.load(args.solver_tables / "octree_block_kind.npy")
    neighbor_offsets = np.load(args.solver_tables / "octree_block_neighbor_offsets.npy")
    neighbor_blocks = np.load(args.solver_tables / "octree_block_neighbors.npy")
    neighbor_sides = np.load(args.solver_tables / "octree_block_neighbor_sides.npy")
    boundary_mask = np.load(args.solver_tables / "octree_block_boundary_mask.npy")
    procedural_control = np.load(args.solver_tables / "procedural_cell_control.npy")
    control_representative = np.load(
        args.solver_tables / "control_representative_procedural_cell.npy"
    )
    exception_control = np.load(args.solver_tables / "control_geometry_exception.npy")
    exception_volume = np.load(
        args.solver_tables / "control_geometry_exception_volume.npy"
    )
    exception_centroid = np.load(
        args.solver_tables / "control_geometry_exception_centroid.npy"
    )
    negative_control = np.load(args.solver_tables / "cut_face_negative_control.npy")
    positive_control = np.load(args.solver_tables / "cut_face_positive_control.npy")
    wall_control = np.load(args.solver_tables / "wall_fragment_control.npy")
    status = np.load(args.interface / "cartesian_cell_status.npy")
    face_key = np.load(args.cut_cells / "cartesian_aperture_face_key.npy")
    face_area = np.load(args.cut_cells / "cartesian_aperture_area.npy")
    face_centroid = np.load(args.cut_cells / "cartesian_aperture_centroid.npy")
    fragment_centroid = np.load(args.interface / "interface_fragment_centroid.npy")
    fragment_area_vector = np.load(args.interface / "interface_fragment_area_vector.npy")
    fragment_triangle = np.load(args.interface / "interface_fragment_triangle.npy")
    interface_manifest = json.loads((args.interface / "manifest.json").read_text())
    with np.load(args.column_plan) as column:
        surface_points = np.asarray(column["points"], dtype="<f8")
        surface_triangles = np.asarray(column["triangles"], dtype="<i4")

    blocks = np.zeros(len(block_low), dtype=BLOCK_DTYPE)
    blocks["low"] = block_low
    blocks["spacing"] = block_size / block_cells
    blocks["fine_coordinate"] = block_coordinate
    blocks["fine_span"] = block_span
    blocks["first_procedural"] = np.arange(len(blocks), dtype=np.int64) * cells_per_block
    blocks["level"] = block_level
    blocks["kind"] = block_kind

    representative = np.asarray(control_representative, dtype=np.int64)
    representative_blocks = representative // cells_per_block
    local = local_coordinates(representative % cells_per_block, block_cells)
    controls = np.zeros(len(representative), dtype=CONTROL_DTYPE)
    controls["x"] = (block_low[representative_blocks] +
                     (local + 0.5) * blocks["spacing"][representative_blocks, None])
    controls["volume"] = blocks["spacing"][representative_blocks] ** 3
    controls["x"][exception_control] = exception_centroid
    controls["volume"][exception_control] = exception_volume
    if np.any(controls["volume"] <= 0) or np.any(~np.isfinite(controls["x"])):
        raise ValueError("invalid packed control geometry")

    seen = set()
    interface_records = []
    for block in range(len(blocks)):
        for position in range(int(neighbor_offsets[block]), int(neighbor_offsets[block + 1])):
            neighbour = int(neighbor_blocks[position])
            pair = (min(block, neighbour), max(block, neighbour))
            if pair in seen:
                continue
            seen.add(pair)
            side = int(neighbor_sides[position])
            first_side = side if block == pair[0] else side ^ 1
            interface_records.append((pair[0], pair[1], first_side,
                                      int(block_level[pair[1]]) - int(block_level[pair[0]])))
    interfaces = np.zeros(len(interface_records), dtype=INTERFACE_DTYPE)
    for field, column in zip(
            ("first_block", "second_block", "first_side", "level_difference"),
            zip(*interface_records)):
        interfaces[field] = column

    boundary_records = []
    for block, mask in enumerate(boundary_mask):
        for side in range(6):
            if int(mask) & (1 << side):
                boundary_records.append((block, side))
    boundaries = np.zeros(len(boundary_records), dtype=BOUNDARY_DTYPE)
    boundaries["block"] = [record[0] for record in boundary_records]
    boundaries["side"] = [record[1] for record in boundary_records]

    open_cut = ((face_area > 1e-15 * float(np.max(face_area))) &
                (negative_control >= 0) & (positive_control >= 0) &
                (negative_control != positive_control))
    cut_faces = np.zeros(int(np.count_nonzero(open_cut)), dtype=FACE_DTYPE)
    cut_faces["left"] = negative_control[open_cut]
    cut_faces["right"] = positive_control[open_cut]
    cut_faces["bc"] = 0
    cut_faces["x"] = face_centroid[open_cut]
    axes = face_key[open_cut, 0]
    cut_faces["n"][np.arange(len(cut_faces)), axes] = 1.0
    cut_faces["area"] = face_area[open_cut]
    center_delta = controls["x"][cut_faces["right"]] - controls["x"][cut_faces["left"]]
    cut_faces["distance"] = center_delta[np.arange(len(cut_faces)), axes]
    minimum_projected_cut_face_distance = float(np.min(cut_faces["distance"]))
    nonpositive_cut_distance = int(np.count_nonzero(cut_faces["distance"] <= 0))
    if nonpositive_cut_distance:
        # A non-convex agglomerated control can have its volume centroid across
        # a small local aperture.  Match finish_generated_face in amr_solver:
        # retain the conservative face orientation and use the positive
        # Euclidean centroid distance only for spectral/viscous rate scaling.
        selected = cut_faces["distance"] <= 0
        cut_faces["distance"][selected] = np.linalg.norm(
            center_delta[selected], axis=1
        )
    if np.any(cut_faces["distance"] <= 0):
        raise ValueError("a packed cut face has zero center distance")

    wall_faces = np.zeros(len(wall_control), dtype=FACE_DTYPE)
    wall_faces["left"] = wall_control
    wall_faces["right"] = -1
    wall_faces["bc"] = 1
    wall_faces["x"] = fragment_centroid
    wall_faces["area"] = np.linalg.norm(fragment_area_vector, axis=1)
    wall_faces["n"] = -fragment_area_vector / wall_faces["area"][:, None]
    wall_delta = wall_faces["x"] - controls["x"][wall_faces["left"]]
    wall_faces["distance"] = np.einsum("ij,ij->i", wall_delta, wall_faces["n"])
    nonpositive_wall_distance = int(np.count_nonzero(wall_faces["distance"] <= 0))
    if nonpositive_wall_distance:
        # A merged centroid can sit tangentially beyond a strongly curved tiny
        # fragment.  The Euclidean center-to-fragment distance remains a
        # positive, conservative two-point wall length for the first solver.
        selected = wall_faces["distance"] <= 0
        wall_faces["distance"][selected] = np.linalg.norm(wall_delta[selected], axis=1)
    if np.any(wall_faces["distance"] <= 0):
        raise ValueError("a packed wall face has zero center distance")
    if len(fragment_triangle) != len(wall_faces):
        raise ValueError("wall fragment/triangle mapping length mismatch")
    if (np.any(fragment_triangle < 0) or
            np.any(fragment_triangle >= len(surface_triangles))):
        raise ValueError("wall fragment has an invalid source triangle")

    # The leaf-block union has the same minimum as the root cube.
    octree_low = np.min(block_low, axis=0)
    root_size = float(manifest["geometric_conservation"]["root_volume"] ** (1 / 3))
    header = struct.pack(
        "<8sIIII7Q4d", b"QCFDAM1\0", 1, 0x01020304, block_cells, 0,
        len(blocks), len(procedural_control), len(controls), len(interfaces),
        len(boundaries), len(cut_faces), len(wall_faces),
        *octree_low, root_size,
    )
    (args.out / "header.bin").write_bytes(header)
    blocks.tofile(args.out / "blocks.bin")
    interfaces.tofile(args.out / "interfaces.bin")
    boundaries.tofile(args.out / "boundaries.bin")
    controls.tofile(args.out / "controls.bin")
    np.asarray(procedural_control, dtype="<i4").tofile(args.out / "procedural_control.bin")
    np.asarray(status, dtype="u1").tofile(args.out / "procedural_status.bin")
    cut_faces.tofile(args.out / "cut_faces.bin")
    wall_faces.tofile(args.out / "wall_faces.bin")
    np.asarray(fragment_triangle, dtype="<i4").tofile(
        args.out / "wall_triangle.bin"
    )
    surface_points.tofile(args.out / "wall_points.bin")
    surface_triangles.tofile(args.out / "wall_triangles.bin")
    (args.out / "wall_counts.txt").write_text(
        f"{len(surface_points)} {len(surface_triangles)}\n"
    )

    total_bytes = sum(
        path.stat().st_size for path in args.out.iterdir() if path.is_file()
    )
    report = {
        "format": "QCFDAM1 directory",
        "blocks": len(blocks),
        "procedural_cells": len(procedural_control),
        "controls": len(controls),
        "block_interfaces": len(interfaces),
        "farfield_block_sides": len(boundaries),
        "cut_faces": len(cut_faces),
        "wall_faces": len(wall_faces),
        "wall_triangles": len(surface_triangles),
        "wall_fragment_triangle_mapping": "wall_triangle.bin int32",
        "minimum_cut_face_distance": float(np.min(cut_faces["distance"])),
        "minimum_projected_cut_face_distance": minimum_projected_cut_face_distance,
        "cut_faces_using_euclidean_distance": nonpositive_cut_distance,
        "minimum_wall_face_distance": float(np.min(wall_faces["distance"])),
        "wall_fragments_using_euclidean_distance": nonpositive_wall_distance,
        "control_volume_sum": float(np.sum(controls["volume"])),
        "expected_fluid_volume": manifest["geometric_conservation"]["total_fluid_volume"],
        "total_bytes": total_bytes,
        "pack_seconds": time.perf_counter() - start,
        "wall_distance": (
            "required separate wall_distance.bin from prepare_amr_wall_distance"
        ),
        "surface_volume": interface_manifest["outer_surface_signed_volume"],
    }
    if abs(report["control_volume_sum"] - report["expected_fluid_volume"]) > 1e-10:
        raise ValueError("packed control volumes do not conserve global volume")
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
