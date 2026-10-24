#!/usr/bin/env python3
"""Assemble factorized solver tables for a Cartesian AMR cut-cell mesh.

Regular 8x8x8 leaf-block cells and faces remain procedural.  Only AMR block
adjacency, cut-face overrides, wall fragments, and agglomerated/cut control
volume geometry are materialized.  This keeps the complete 3-D topology small
without pretending that the current explicit-face ``sphere3d`` reader already
understands the new format.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np


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


def block_coordinates(low, root_low, root_size, maximum_level):
    dimension = 1 << maximum_level
    return np.rint((low - root_low) * dimension / root_size).astype(np.int32)


def sparse_block_adjacency(coordinates, levels, maximum_level):
    """Build exact balanced-octree neighbours without a dense owner cube."""
    spans = (1 << (maximum_level - levels)).astype(np.int32)
    dimension = 1 << maximum_level
    lookup = {
        (int(c[0]), int(c[1]), int(c[2]), int(span)): block
        for block, (c, span) in enumerate(zip(coordinates, spans))
    }
    offsets = np.empty(len(coordinates) + 1, dtype=np.uint64)
    neighbours = []
    sides = []
    relations = []
    boundary_mask = np.zeros(len(coordinates), dtype=np.uint8)

    for block, (coordinate_array, span_value) in enumerate(zip(coordinates, spans)):
        coordinate = tuple(map(int, coordinate_array))
        span = int(span_value)
        offsets[block] = len(neighbours)
        for side in range(6):
            axis = side // 2
            positive = side % 2
            boundary = ((not positive and coordinate[axis] == 0) or
                        (positive and coordinate[axis] + span == dimension))
            if boundary:
                boundary_mask[block] |= np.uint8(1 << side)
                continue

            same = list(coordinate)
            same[axis] += span if positive else -span
            found = []
            key = (*same, span)
            if key in lookup:
                found = [lookup[key]]
            elif span * 2 <= dimension:
                coarse_span = 2 * span
                coarse = [
                    (value // coarse_span) * coarse_span
                    for value in coordinate
                ]
                coarse[axis] = coordinate[axis] + span if positive else coordinate[axis] - coarse_span
                key = (*coarse, coarse_span)
                if key in lookup:
                    found = [lookup[key]]
            if not found and span % 2 == 0:
                fine_span = span // 2
                transverse = [value for value in range(3) if value != axis]
                face_origin = coordinate[axis] + span if positive else coordinate[axis] - fine_span
                for first_offset in (0, fine_span):
                    for second_offset in (0, fine_span):
                        fine = list(coordinate)
                        fine[axis] = face_origin
                        fine[transverse[0]] += first_offset
                        fine[transverse[1]] += second_offset
                        key = (*fine, fine_span)
                        if key in lookup:
                            found.append(lookup[key])
                if found and len(found) != 4:
                    raise ValueError(f"block {block} side {side} found an incomplete fine quartet")
            if not found:
                raise ValueError(f"block {block} side {side} has no balanced-octree neighbour")
            for neighbour in sorted(found):
                level_difference = int(levels[neighbour]) - int(levels[block])
                if abs(level_difference) > 1:
                    raise ValueError("octree adjacency violates 2:1 balance")
                neighbours.append(neighbour)
                sides.append(side)
                relations.append(level_difference)
    offsets[-1] = len(neighbours)
    neighbours = np.asarray(neighbours, dtype=np.int32)
    sides = np.asarray(sides, dtype=np.uint8)
    relations = np.asarray(relations, dtype=np.int8)

    adjacency = set()
    for block in range(len(coordinates)):
        for position in range(int(offsets[block]), int(offsets[block + 1])):
            adjacency.add((block, int(sides[position]), int(neighbours[position])))
    missing_reciprocals = [
        (block, side, neighbour)
        for block, side, neighbour in adjacency
        if (neighbour, side ^ 1, block) not in adjacency
    ]
    if missing_reciprocals:
        raise ValueError(f"block adjacency is not reciprocal: {missing_reciprocals[:8]}")
    return offsets, neighbours, sides, relations, boundary_mask, spans


def local_coordinates(block_cells):
    local = np.arange(block_cells ** 3, dtype=np.int32)
    return np.column_stack((
        local % block_cells,
        (local // block_cells) % block_cells,
        local // (block_cells ** 2),
    ))


def local_ids(coordinates, block_cells):
    return ((coordinates[:, 2] * block_cells + coordinates[:, 1]) * block_cells +
            coordinates[:, 0]).astype(np.int32)


def face_local_coordinates(axis, positive, block_cells):
    first, second = [value for value in range(3) if value != axis]
    u, v = np.meshgrid(
        np.arange(block_cells, dtype=np.int32),
        np.arange(block_cells, dtype=np.int32), indexing="ij"
    )
    result = np.empty((block_cells ** 2, 3), dtype=np.int32)
    result[:, axis] = block_cells - 1 if positive else 0
    result[:, first] = u.ravel()
    result[:, second] = v.ravel()
    return result


def count_template_pair(first_cells, second_cells, status, control):
    first_status = status[first_cells]
    second_status = status[second_cells]
    regular = (first_status != 2) & (second_status != 2)
    first_active = control[first_cells] >= 0
    second_active = control[second_cells] >= 0
    invalid = regular & (first_active != second_active)
    retained = (regular & first_active & second_active &
                (control[first_cells] != control[second_cells]))
    merged = (regular & first_active & second_active &
              (control[first_cells] == control[second_cells]))
    return int(np.count_nonzero(retained)), int(np.count_nonzero(merged)), int(np.count_nonzero(invalid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--octree-plan", required=True, type=Path)
    parser.add_argument("--interface", required=True, type=Path)
    parser.add_argument("--cut-cells", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)

    with np.load(args.octree_plan) as octree:
        low = np.asarray(octree["low"], dtype=np.float64)
        size = np.asarray(octree["size"], dtype=np.float64)
        level = np.asarray(octree["level"], dtype=np.int16)
        root_low = np.asarray(octree["root_low"], dtype=np.float64)
        root_size = float(octree["root_size"])
        block_cells = int(octree["block_cells"])
    maximum_level = int(np.max(level))
    coordinates = block_coordinates(low, root_low, root_size, maximum_level)
    (neighbor_offsets, neighbor_blocks, neighbor_sides, neighbor_relations,
     boundary_mask, spans) = sparse_block_adjacency(coordinates, level, maximum_level)

    status = np.load(args.interface / "cartesian_cell_status.npy", mmap_mode="r")
    cut_cells = np.load(args.interface / "cut_cartesian_cells.npy", mmap_mode="r")
    fragment_cells = np.load(
        args.interface / "interface_fragment_cartesian_cell.npy", mmap_mode="r"
    )
    fragment_area_vectors = np.load(
        args.interface / "interface_fragment_area_vector.npy", mmap_mode="r"
    )
    interface_manifest = json.loads((args.interface / "manifest.json").read_text())

    cut_volume = np.load(args.cut_cells / "cut_cell_volume.npy", mmap_mode="r")
    cut_centroid = np.load(args.cut_cells / "cut_cell_centroid.npy", mmap_mode="r")
    cut_group = np.load(
        args.cut_cells / "cut_cell_agglomeration_group.npy", mmap_mode="r"
    )
    group_offsets = np.load(
        args.cut_cells / "agglomeration_member_offsets.npy", mmap_mode="r"
    )
    group_members = np.load(
        args.cut_cells / "agglomeration_member_procedural_cell.npy", mmap_mode="r"
    )
    group_volume = np.load(args.cut_cells / "agglomeration_volume.npy", mmap_mode="r")
    group_centroid = np.load(
        args.cut_cells / "agglomeration_centroid.npy", mmap_mode="r"
    )
    face_key = np.load(args.cut_cells / "cartesian_aperture_face_key.npy", mmap_mode="r")
    face_negative = np.load(
        args.cut_cells / "cartesian_aperture_negative_cell.npy", mmap_mode="r"
    )
    face_positive = np.load(
        args.cut_cells / "cartesian_aperture_positive_cell.npy", mmap_mode="r"
    )
    face_area = np.load(args.cut_cells / "cartesian_aperture_area.npy", mmap_mode="r")

    cells_per_block = block_cells ** 3
    procedural_cells = len(low) * cells_per_block
    if len(status) != procedural_cells:
        raise ValueError("status table and octree cell count disagree")
    if len(cut_cells) != len(cut_volume) or np.any(status[cut_cells] != 2):
        raise ValueError("cut-cell tables and status table disagree")

    # Map every active procedural cell to one contiguous control-volume ID.
    representative = np.arange(procedural_cells, dtype=np.int32)
    representative[np.asarray(status) == 1] = -1
    group_size = np.diff(group_offsets).astype(np.int64)
    group_representative = np.asarray(group_members[group_offsets[:-1]], dtype=np.int32)
    member_representative = np.repeat(group_representative, group_size)
    if len(np.unique(group_members)) != len(group_members):
        raise ValueError("an agglomeration member appears in more than one group")
    representative[group_members] = member_representative
    indices = np.arange(procedural_cells, dtype=np.int32)
    control_representative = indices[(representative >= 0) & (representative == indices)]
    representative_to_control = np.full(procedural_cells, -1, dtype=np.int32)
    representative_to_control[control_representative] = np.arange(
        len(control_representative), dtype=np.int32
    )
    procedural_control = np.full(procedural_cells, -1, dtype=np.int32)
    active = representative >= 0
    procedural_control[active] = representative_to_control[representative[active]]
    if np.any(procedural_control[np.asarray(status) != 1] < 0):
        raise ValueError("an active procedural cell has no control volume")

    standalone_cut = np.asarray(cut_group) < 0
    standalone_controls = procedural_control[cut_cells[standalone_cut]]
    grouped_controls = procedural_control[group_representative]
    exception_control = np.concatenate((standalone_controls, grouped_controls))
    exception_kind = np.concatenate((
        np.ones(len(standalone_controls), dtype=np.uint8),
        np.full(len(grouped_controls), 2, dtype=np.uint8),
    ))
    exception_volume = np.concatenate((
        np.asarray(cut_volume[standalone_cut]), np.asarray(group_volume)
    ))
    exception_centroid = np.concatenate((
        np.asarray(cut_centroid[standalone_cut]), np.asarray(group_centroid)
    ))
    order = np.argsort(exception_control)
    exception_control = exception_control[order]
    exception_kind = exception_kind[order]
    exception_volume = exception_volume[order]
    exception_centroid = exception_centroid[order]
    if len(np.unique(exception_control)) != len(exception_control):
        raise ValueError("control-volume geometry exceptions are not unique")
    if np.any(exception_volume <= 0) or np.any(~np.isfinite(exception_centroid)):
        raise ValueError("invalid control-volume geometry exception")

    # Complete wall and cut-face ownership in the new control numbering.
    wall_fragment_control = procedural_control[fragment_cells]
    if np.any(wall_fragment_control < 0):
        raise ValueError("a wall fragment has no active owner")
    wall_area = np.linalg.norm(fragment_area_vectors, axis=1)
    wall_area_sum = float(np.sum(wall_area))

    negative_control = np.where(
        face_negative >= 0, procedural_control[np.maximum(face_negative, 0)], -1
    ).astype(np.int32)
    positive_control = np.where(
        face_positive >= 0, procedural_control[np.maximum(face_positive, 0)], -1
    ).astype(np.int32)
    area_tolerance = 1e-15 * float(np.max(face_area))
    open_face = np.asarray(face_area) > area_tolerance
    invalid_open = open_face & ((negative_control < 0) | (positive_control < 0))
    if np.any(invalid_open):
        bad = np.flatnonzero(invalid_open)[:12]
        raise ValueError(f"open cut-face aperture reaches inactive material: {bad.tolist()}")
    merged_override = open_face & (negative_control == positive_control)
    active_override = open_face & (negative_control != positive_control)

    status_blocks = np.asarray(status).reshape(len(low), cells_per_block)
    block_kind = np.where(
        np.all(status_blocks == 0, axis=1), 0,
        np.where(np.all(status_blocks == 1, axis=1), 1, 2)
    ).astype(np.uint8)

    # Count every global Cartesian face once.  Regular faces incident on cut
    # cells are excluded here and replaced by the aperture table above.
    local = local_coordinates(block_cells)
    internal_pairs = []
    for axis in range(3):
        selected = local[:, axis] + 1 < block_cells
        first_coordinates = local[selected]
        second_coordinates = first_coordinates.copy()
        second_coordinates[:, axis] += 1
        internal_pairs.append((local_ids(first_coordinates, block_cells),
                               local_ids(second_coordinates, block_cells)))
    side_coordinates = [
        face_local_coordinates(side // 2, bool(side % 2), block_cells)
        for side in range(6)
    ]
    side_local_ids = [local_ids(values, block_cells) for values in side_coordinates]

    template_internal_faces = 0
    template_merged_faces = 0
    invalid_fluid_solid_faces = 0
    for block in range(len(low)):
        base = block * cells_per_block
        for first_local, second_local in internal_pairs:
            retained, merged, invalid = count_template_pair(
                base + first_local, base + second_local, status, procedural_control
            )
            template_internal_faces += retained
            template_merged_faces += merged
            invalid_fluid_solid_faces += invalid

    same_level_interfaces = 0
    coarse_fine_interfaces = 0
    same_level_subfaces = 0
    coarse_fine_subfaces = 0
    seen_interfaces = set()
    for block in range(len(low)):
        for position in range(int(neighbor_offsets[block]), int(neighbor_offsets[block + 1])):
            neighbour = int(neighbor_blocks[position])
            side = int(neighbor_sides[position])
            pair = (min(block, neighbour), max(block, neighbour))
            if pair in seen_interfaces:
                continue
            seen_interfaces.add(pair)
            if level[block] == level[neighbour]:
                first_cells = block * cells_per_block + side_local_ids[side]
                second_cells = neighbour * cells_per_block + side_local_ids[side ^ 1]
                same_level_interfaces += 1
                same_level_subfaces += len(first_cells)
            else:
                fine = block if level[block] > level[neighbour] else neighbour
                coarse = neighbour if fine == block else block
                fine_side = side if fine == block else side ^ 1
                fine_coordinates = side_coordinates[fine_side]
                coarse_coordinates = np.empty_like(fine_coordinates)
                axis = fine_side // 2
                coarse_coordinates[:, axis] = block_cells - 1 if fine_side % 2 == 0 else 0
                for transverse in [value for value in range(3) if value != axis]:
                    half = int((coordinates[fine, transverse] -
                                coordinates[coarse, transverse]) // spans[fine])
                    if half not in (0, 1):
                        raise ValueError("fine/coarse transverse offset is not a half block")
                    coarse_coordinates[:, transverse] = (
                        half * block_cells + fine_coordinates[:, transverse]
                    ) // 2
                first_cells = fine * cells_per_block + local_ids(fine_coordinates, block_cells)
                second_cells = coarse * cells_per_block + local_ids(coarse_coordinates, block_cells)
                coarse_fine_interfaces += 1
                coarse_fine_subfaces += len(first_cells)
            retained, merged, invalid = count_template_pair(
                first_cells, second_cells, status, procedural_control
            )
            template_internal_faces += retained
            template_merged_faces += merged
            invalid_fluid_solid_faces += invalid

    farfield_faces = 0
    farfield_area = 0.0
    for block in range(len(low)):
        cell_spacing = size[block] / block_cells
        for side in range(6):
            if not (boundary_mask[block] & (1 << side)):
                continue
            procedural = block * cells_per_block + side_local_ids[side]
            if np.any(np.asarray(status[procedural]) == 2):
                raise ValueError("a cut cell reaches the root farfield boundary")
            active_boundary = procedural_control[procedural] >= 0
            farfield_faces += int(np.count_nonzero(active_boundary))
            farfield_area += float(np.count_nonzero(active_boundary)) * cell_spacing ** 2

    if invalid_fluid_solid_faces:
        raise ValueError(
            f"{invalid_fluid_solid_faces} uncut template faces directly join fluid and solid"
        )
    expected_farfield_area = 6.0 * root_size ** 2
    if abs(farfield_area - expected_farfield_area) > 1e-10 * expected_farfield_area:
        raise ValueError("farfield template area does not close the root cube")

    full_active_volume = float(np.sum(
        np.sum(status_blocks == 0, axis=1) * (size / block_cells) ** 3
    ))
    fluid_volume = full_active_volume + float(np.sum(cut_volume))
    surface_volume = float(interface_manifest["outer_surface_signed_volume"])
    expected_fluid_volume = root_size ** 3 - surface_volume
    volume_balance_error = fluid_volume - expected_fluid_volume
    if abs(volume_balance_error) > 1e-11 * root_size ** 3:
        raise ValueError("global fluid-volume balance failed")

    tables = {}
    save_array(args.out, "octree_block_low", low,
               "Leaf-block minimum physical coordinates.", tables)
    save_array(args.out, "octree_block_size", size,
               "Leaf-block physical edge lengths.", tables)
    save_array(args.out, "octree_block_level", level,
               "Leaf-block refinement levels.", tables)
    save_array(args.out, "octree_block_fine_coordinate", coordinates,
               "Leaf-block origins on the maximum-level block lattice.", tables)
    save_array(args.out, "octree_block_fine_span", spans,
               "Leaf-block spans on the maximum-level block lattice.", tables)
    save_array(args.out, "octree_block_kind", block_kind,
               "0 all-fluid template, 1 all-solid removed, 2 irregular cut block at its leaf level.", tables)
    save_array(args.out, "octree_block_neighbor_offsets", neighbor_offsets,
               "CSR offsets for block-face neighbours.", tables)
    save_array(args.out, "octree_block_neighbors", neighbor_blocks,
               "Face-adjacent leaf blocks; a coarse face has four fine neighbours.", tables)
    save_array(args.out, "octree_block_neighbor_sides", neighbor_sides,
               "Local side 0:-x,1:+x,2:-y,3:+y,4:-z,5:+z.", tables)
    save_array(args.out, "octree_block_neighbor_level_difference", neighbor_relations,
               "Neighbour level minus owner level; restricted to -1,0,+1.", tables)
    save_array(args.out, "octree_block_boundary_mask", boundary_mask,
               "Six low bits identify root farfield block faces.", tables)
    save_array(args.out, "procedural_cell_control", procedural_control,
               "Contiguous control-volume ID per procedural cell, or -1 for solid.", tables)
    save_array(args.out, "control_representative_procedural_cell", control_representative,
               "Representative procedural cell for each contiguous control volume.", tables)
    save_array(args.out, "control_geometry_exception", exception_control,
               "Control IDs whose geometry is not a regular Cartesian cell.", tables)
    save_array(args.out, "control_geometry_exception_kind", exception_kind,
               "1 standalone cut cell, 2 agglomerated cut/full group.", tables)
    save_array(args.out, "control_geometry_exception_volume", exception_volume,
               "Fluid volume for each control geometry exception.", tables)
    save_array(args.out, "control_geometry_exception_centroid", exception_centroid,
               "Fluid first-moment centroid for each control geometry exception.", tables)
    save_array(args.out, "wall_fragment_control", wall_fragment_control.astype(np.int32),
               "Owning control volume for each interface-plan wall fragment.", tables)
    save_array(args.out, "cut_face_negative_control", negative_control,
               "Control on the negative-axis side of each cut-face override.", tables)
    save_array(args.out, "cut_face_positive_control", positive_control,
               "Control on the positive-axis side of each cut-face override.", tables)

    active_controls = len(control_representative)
    override_flux_faces = int(np.count_nonzero(active_override))
    removed_agglomeration_faces = int(np.count_nonzero(merged_override)) + template_merged_faces
    global_faces = (template_internal_faces + override_flux_faces +
                    farfield_faces + len(fragment_cells))
    report = {
        "format": "QCFD_CARTESIAN_AMR_CUTCELL_TABLES",
        "version": 1,
        "mesh_workflow": "cartesian_amr_cutcell_3d",
        "geometry_topology_complete": True,
        "current_sphere3d_reader_compatible": False,
        "reason_not_currently_runnable": (
            "sphere3d presently consumes explicit QCFD3D1 Cell/Face/Term arrays; "
            "a procedural block/cut-face residual and gradient reader is still required"
        ),
        "factorization": (
            "regular 8x8x8 block interiors are implicit; block interfaces, cut-face "
            "overrides, wall fragments, and non-regular control geometry are explicit"
        ),
        "leaf_blocks": len(low),
        "block_cells_per_axis": block_cells,
        "procedural_cells": procedural_cells,
        "active_control_volumes": active_controls,
        "all_fluid_blocks": int(np.count_nonzero(block_kind == 0)),
        "all_solid_blocks": int(np.count_nonzero(block_kind == 1)),
        "irregular_blocks": int(np.count_nonzero(block_kind == 2)),
        "regular_control_volumes": int(active_controls - len(exception_control)),
        "standalone_cut_control_volumes": len(standalone_controls),
        "agglomerated_control_volumes": len(grouped_controls),
        "wall_fragments": len(fragment_cells),
        "wall_area": wall_area_sum,
        "wall_source_area": float(interface_manifest["clipping"]["covered_area"]),
        "wall_area_relative_error": abs(
            wall_area_sum / float(interface_manifest["clipping"]["covered_area"]) - 1.0
        ),
        "cut_face_overrides": len(face_key),
        "open_cut_face_fluxes": override_flux_faces,
        "removed_agglomeration_internal_faces": removed_agglomeration_faces,
        "block_interfaces": {
            "same_level": same_level_interfaces,
            "coarse_fine_quadrants": coarse_fine_interfaces,
            "same_level_subfaces_before_activity_filter": same_level_subfaces,
            "coarse_fine_subfaces_before_activity_filter": coarse_fine_subfaces,
            "maximum_neighbor_level_difference": int(np.max(np.abs(neighbor_relations), initial=0)),
            "reciprocal": True,
        },
        "global_face_counts": {
            "regular_template_internal": template_internal_faces,
            "cut_aperture_internal": override_flux_faces,
            "farfield": farfield_faces,
            "solid_wall_fragments": len(fragment_cells),
            "total": global_faces,
        },
        "geometric_conservation": {
            "root_volume": root_size ** 3,
            "closed_surface_volume": surface_volume,
            "full_fluid_cell_volume": full_active_volume,
            "cut_fluid_cell_volume": float(np.sum(cut_volume)),
            "total_fluid_volume": fluid_volume,
            "root_minus_solid_volume": expected_fluid_volume,
            "volume_balance_error": volume_balance_error,
            "farfield_area": farfield_area,
            "expected_farfield_area": expected_farfield_area,
            "coarse_fine_rule": "each coarse cell face is exactly four fine subfaces",
        },
        "external_geometry_tables": {
            "cut_cell_directory": str(args.cut_cells.resolve()),
            "interface_directory": str(args.interface.resolve()),
            "cut_face_key": "cartesian_aperture_face_key.npy",
            "cut_face_area": "cartesian_aperture_area.npy",
            "cut_face_centroid": "cartesian_aperture_centroid.npy",
            "wall_fragment_area_vector": "interface_fragment_area_vector.npy (negate for outward-from-fluid normal)",
            "wall_fragment_centroid": "interface_fragment_centroid.npy",
        },
        "tables": tables,
        "total_table_bytes": int(sum(value["bytes"] for value in tables.values())),
        "build_seconds": time.perf_counter() - start,
        "remaining": [
            "implement procedural block/cut-face residual iteration in sphere3d",
            "build least-squares or node-based gradient stencils across 2:1 interfaces",
            "add farfield patch metadata and wall-distance tables",
            "run CPU/CUDA uniform-flow and conservation regressions before CRM flow",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
