#!/usr/bin/env python3
"""Extrude a 2-D RAE2822 NPZ mesh into a one-cell-thick QCFD3D1 slab.

The input topology is genuinely mixed: its cells are triangles and quads.  A
single spanwise extrusion therefore produces wedges and hexahedra.  QCFD3D1
stores arbitrary polyhedral cells through owner/neighbour faces, so no
tetrahedral subdivision is needed.

Important input conventions discovered from the archived database:

* ``face_nodes`` is already zero based.  Node zero is a reserved, unused node.
* positive ``face_cells`` values are one based; zero denotes a boundary.
* ``msh_to_csv[mesh_cell - 1]`` maps a topology cell to the ordering used by
  ``cell_centers`` and ``LF*_true_mesh.npz``.

The airfoil becomes a no-slip wall, the upper/lower slab caps are symmetry
planes, and both the far boundary and the archived pressure outlet use the
solver's farfield boundary condition.  That last mapping is recorded in the
report because QCFD3D1 currently has no separate pressure-outlet condition.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


WALL_ZONES = {6, 7}
FARFIELD_ZONES = {8, 9}


def polygon_geometry(vertices: np.ndarray) -> tuple[float, np.ndarray]:
    """Return positive area and the exact area centroid of an ordered polygon."""
    x = vertices[:, 0]
    z = vertices[:, 1]
    x1 = np.roll(x, -1)
    z1 = np.roll(z, -1)
    cross = x * z1 - x1 * z
    signed_area = 0.5 * float(np.sum(cross))
    if not math.isfinite(signed_area) or abs(signed_area) <= 1.0e-18:
        raise ValueError("zero or non-finite polygon area")
    centroid = np.array(
        [np.sum((x + x1) * cross), np.sum((z + z1) * cross)],
        dtype=np.float64,
    ) / (6.0 * signed_area)
    return abs(signed_area), centroid


def point_segment_distances(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Return each point's distance to the closest wall segment."""
    a = segments[:, 0]
    d = segments[:, 1] - a
    denominator = np.einsum("ij,ij->i", d, d)
    if np.any(denominator <= 0.0):
        raise ValueError("wall contains a zero-length segment")
    result = np.full(len(points), np.inf, dtype=np.float64)
    for first in range(0, len(points), 4096):
        p = points[first : first + 4096]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("pwi,wi->pw", ap, d) / denominator, 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * d[None, :, :]
        result[first : first + len(p)] = np.sqrt(
            np.min(np.einsum("pwi,pwi->pw", p[:, None, :] - closest, p[:, None, :] - closest), axis=1)
        )
    return result


def build(topology_path: Path, true_mesh_path: Path, destination: Path, span: float) -> dict:
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("span must be positive")

    with np.load(topology_path) as data:
        nodes = np.asarray(data["nodes"], dtype=np.float64)
        face_nodes = np.asarray(data["face_nodes"], dtype=np.int64)
        input_face_cells = np.asarray(data["face_cells"], dtype=np.int64)
        face_zones = np.asarray(data["face_zones"], dtype=np.int64)
        archived_centers = np.asarray(data["cell_centers"], dtype=np.float64)
        mesh_to_cell = np.asarray(data["msh_to_csv"], dtype=np.int64)
        input_zone_names = str(data["zone_names"].item())
        n_cells = int(data["n_cells"])
    with np.load(true_mesh_path) as data:
        polygon_vertices = np.asarray(data["vertices"], dtype=np.float64)
        polygon_offsets = np.asarray(data["offsets"], dtype=np.int64)

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (N, 2)")
    if face_nodes.shape != input_face_cells.shape or face_nodes.shape[1] != 2:
        raise ValueError("face_nodes and face_cells must have shape (F, 2)")
    if np.min(face_nodes) < 0 or np.max(face_nodes) >= len(nodes):
        raise ValueError("face_nodes is not a valid zero-based index array")
    if np.any(face_nodes == 0):
        raise ValueError("reserved node zero unexpectedly appears in face topology")
    if len(archived_centers) != n_cells or len(mesh_to_cell) != n_cells:
        raise ValueError("cell array sizes disagree")
    if not np.array_equal(np.sort(mesh_to_cell), np.arange(n_cells)):
        raise ValueError("msh_to_csv is not a cell permutation")
    if len(polygon_offsets) != n_cells + 1 or polygon_offsets[-1] != len(polygon_vertices):
        raise ValueError("invalid ragged polygon offsets")
    if np.any(input_face_cells < 0) or np.max(input_face_cells) > n_cells:
        raise ValueError("face_cells is outside the expected one-based range")

    # Convert positive one-based mesh IDs to zero-based output cell IDs.  Zero
    # remains the boundary sentinel until it is changed to -1 below.
    face_cells = np.full_like(input_face_cells, -1)
    positive = input_face_cells > 0
    face_cells[positive] = mesh_to_cell[input_face_cells[positive] - 1]

    areas = np.empty(n_cells, dtype=np.float64)
    centers_2d = np.empty((n_cells, 2), dtype=np.float64)
    polygon_size = np.diff(polygon_offsets)
    for cell in range(n_cells):
        lo, hi = polygon_offsets[cell : cell + 2]
        areas[cell], centers_2d[cell] = polygon_geometry(polygon_vertices[lo:hi])
    center_discrepancy = np.linalg.norm(centers_2d - archived_centers, axis=1)

    # Every polygon vertex must be present in the topology faces incident on
    # the same mapped cell.  This catches both possible off-by-one mistakes.
    incident_nodes: list[set[int]] = [set() for _ in range(n_cells)]
    for face, (left, right) in enumerate(face_cells):
        for cell in (left, right):
            if cell >= 0:
                incident_nodes[int(cell)].update(map(int, face_nodes[face]))
    for cell, node_ids in enumerate(incident_nodes):
        lo, hi = polygon_offsets[cell : cell + 2]
        polygon = polygon_vertices[lo:hi]
        topology_polygon = nodes[np.fromiter(sorted(node_ids), dtype=np.int64)]
        if len(polygon) != len(topology_polygon):
            raise ValueError(f"cell {cell} polygon degree disagrees with its faces")
        separation = np.linalg.norm(polygon[:, None, :] - topology_polygon[None, :, :], axis=2)
        if float(np.max(np.min(separation, axis=1))) > 1.0e-12:
            raise ValueError(f"cell {cell} polygon vertices do not match its faces")

    wall_mask = np.isin(face_zones, list(WALL_ZONES)) & (face_cells[:, 1] < 0)
    wall_segments = nodes[face_nodes[wall_mask]]
    if len(wall_segments) == 0:
        raise ValueError("no airfoil wall segments found")
    wall_distance = point_segment_distances(centers_2d, wall_segments)

    centers = np.column_stack((centers_2d[:, 0], np.zeros(n_cells), centers_2d[:, 1]))
    volumes = areas * span
    cell_faces: list[list[int]] = [[] for _ in range(n_cells)]
    faces: list[dict] = []
    closure = np.zeros((n_cells, 3), dtype=np.float64)

    # Extruded side faces.  The archived edge ordering is consistent with the
    # first cell: (dz, -dx) points from left to right or out of a boundary.
    for input_face, ((node0, node1), (left, right), zone) in enumerate(
        zip(face_nodes, face_cells, face_zones)
    ):
        p0 = nodes[node0]
        p1 = nodes[node1]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(f"face {input_face} has zero length")
        normal_2d = np.array([edge[1], -edge[0]]) / length
        normal = np.array([normal_2d[0], 0.0, normal_2d[1]])
        center = np.array([(p0[0] + p1[0]) * 0.5, 0.0, (p0[1] + p1[1]) * 0.5])
        if left < 0:
            raise ValueError(f"face {input_face} has no left cell")
        if right >= 0:
            delta = centers[right] - centers[left]
            bc = 0
        else:
            delta = center - centers[left]
            if int(zone) in WALL_ZONES:
                bc = 1
            elif int(zone) in FARFIELD_ZONES:
                bc = 2
            else:
                raise ValueError(f"unsupported boundary zone {zone} on face {input_face}")
        distance = float(np.dot(delta, normal))
        if distance <= 0.0:
            raise ValueError(f"face {input_face} has non-positive oriented distance {distance}")
        area_vector = normal * (length * span)
        output_face = len(faces)
        faces.append(
            dict(left=int(left), right=int(right), bc=bc, center=center, normal=normal,
                 area=length * span, distance=distance, input_face=input_face)
        )
        cell_faces[left].append(output_face)
        closure[left] += area_vector
        if right >= 0:
            cell_faces[right].append(output_face)
            closure[right] -= area_vector

    # Add a pair of symmetry caps to every extruded cell.  These make the
    # least-squares stencil full rank while preserving a strictly 2-D flow.
    for cell in range(n_cells):
        for sign in (-1.0, 1.0):
            normal = np.array([0.0, sign, 0.0])
            center = centers[cell].copy()
            center[1] = sign * 0.5 * span
            output_face = len(faces)
            faces.append(
                dict(left=cell, right=-1, bc=3, center=center, normal=normal,
                     area=float(areas[cell]), distance=0.5 * span, input_face=-1)
            )
            cell_faces[cell].append(output_face)
            closure[cell] += normal * areas[cell]

    adjacency: list[int] = []
    terms: list[tuple[int, int, float, float, float]] = []
    cells: list[tuple] = []
    ls_condition = np.empty(n_cells, dtype=np.float64)
    ls_identity_error = np.empty(n_cells, dtype=np.float64)
    for cell, cell_face_ids in enumerate(cell_faces):
        deltas = []
        sources = []
        for face_id in cell_face_ids:
            face = faces[face_id]
            if face["right"] >= 0:
                neighbour = face["right"] if face["left"] == cell else face["left"]
                delta = centers[neighbour] - centers[cell]
                source = neighbour
            else:
                delta = face["center"] - centers[cell]
                source = -face["bc"]
            deltas.append(delta)
            sources.append((source, face_id))
        delta_array = np.asarray(deltas)
        squared_distance = np.einsum("ij,ij->i", delta_array, delta_array)
        if np.any(squared_distance <= 0.0):
            raise ValueError(f"cell {cell} has a zero-length least-squares arm")
        weights = 1.0 / squared_distance
        matrix = np.einsum("i,ij,ik->jk", weights, delta_array, delta_array)
        ls_condition[cell] = np.linalg.cond(matrix)
        coefficients = np.linalg.solve(matrix, (weights[:, None] * delta_array).T).T
        identity = np.einsum("ij,ik->jk", coefficients, delta_array)
        ls_identity_error[cell] = np.max(np.abs(identity - np.eye(3)))
        term_first = len(terms)
        terms.extend(
            (int(source), int(face_id), float(c[0]), float(c[1]), float(c[2]))
            for (source, face_id), c in zip(sources, coefficients)
        )
        face_first = len(adjacency)
        adjacency.extend(cell_face_ids)
        cells.append(
            (*map(float, centers[cell]), float(volumes[cell]), float(wall_distance[cell]),
             face_first, len(cell_face_ids), term_first, len(sources))
        )

    boundary_volume = 0.0
    for face in faces:
        if face["right"] < 0:
            area_vector = face["normal"] * face["area"]
            boundary_volume += float(np.dot(area_vector, face["center"])) / 3.0
    volume_error = abs(float(np.sum(volumes)) - boundary_volume)
    closure_norm = np.linalg.norm(closure, axis=1)

    report = {
        "workflow": "rae2822_npz_mixed_grid_to_qcfd3d1_thin_slab",
        "topology": str(topology_path.resolve()),
        "true_mesh": str(true_mesh_path.resolve()),
        "zone_names": json.loads(input_zone_names),
        "boundary_mapping": {
            "zones_6_7": "bc=1 no-slip wall",
            "zone_8": "bc=2 farfield",
            "zone_9": "bc=2 farfield (QCFD3D1 has no separate pressure outlet)",
            "spanwise_caps": "bc=3 symmetry",
        },
        "span": span,
        "cells": n_cells,
        "triangles": int(np.sum(polygon_size == 3)),
        "quadrilaterals": int(np.sum(polygon_size == 4)),
        "faces": len(faces),
        "input_side_faces": len(face_nodes),
        "interior_side_faces": int(np.sum(face_cells[:, 1] >= 0)),
        "wall_side_faces": int(np.sum(wall_mask)),
        "farfield_side_faces": int(np.sum(np.isin(face_zones, list(FARFIELD_ZONES)) & (face_cells[:, 1] < 0))),
        "symmetry_cap_faces": 2 * n_cells,
        "adjacency_entries": len(adjacency),
        "least_squares_terms": len(terms),
        "minimum_cell_area": float(np.min(areas)),
        "maximum_cell_area": float(np.max(areas)),
        "minimum_volume": float(np.min(volumes)),
        "maximum_volume": float(np.max(volumes)),
        "minimum_wall_distance": float(np.min(wall_distance)),
        "maximum_archived_center_discrepancy": float(np.max(center_discrepancy)),
        "maximum_cell_closure_l2": float(np.max(closure_norm)),
        "volume_balance_error": float(volume_error),
        "minimum_face_distance": float(min(face["distance"] for face in faces)),
        "maximum_least_squares_condition": float(np.max(ls_condition)),
        "maximum_least_squares_identity_error": float(np.max(ls_identity_error)),
        "coordinate_mapping": "input (x,y) -> solver (x,z); extrusion is solver y",
        "mesh_format": "QCFD3D1 text owner/neighbour polyhedra; wedges and hexahedra are not subdivided",
    }

    scale = max(1.0, math.sqrt(float(np.max(areas))))
    if report["maximum_cell_closure_l2"] > 1.0e-10 * scale:
        raise ValueError(f"cell closure audit failed: {report['maximum_cell_closure_l2']}")
    if report["volume_balance_error"] > 1.0e-9 * max(1.0, float(np.sum(volumes))):
        raise ValueError(f"volume balance audit failed: {report['volume_balance_error']}")
    if report["maximum_least_squares_identity_error"] > 1.0e-10:
        raise ValueError("least-squares linear-exactness audit failed")

    destination.mkdir(parents=True, exist_ok=True)
    mesh_path = destination / "mesh.txt"
    with mesh_path.open("w", encoding="utf-8") as stream:
        stream.write(f"{len(cells)} {len(faces)} {len(adjacency)} {len(terms)}\n")
        for cell in cells:
            stream.write(" ".join(map(str, cell)) + "\n")
        for face in faces:
            values = (
                face["left"], face["right"], face["bc"], *face["center"], *face["normal"],
                face["area"], face["distance"],
            )
            stream.write(" ".join(map(str, values)) + "\n")
        stream.write(" ".join(map(str, adjacency)) + "\n")
        for term in terms:
            stream.write(" ".join(map(str, term)) + "\n")
    (destination / "mesh_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        destination / "geometry.npz",
        centers=centers,
        volumes=volumes,
        polygon_vertices=polygon_vertices,
        polygon_offsets=polygon_offsets,
        wall_segments=wall_segments,
        face_nodes=face_nodes,
        face_cells=face_cells,
        face_zones=face_zones,
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--true-mesh", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--span", type=float, default=0.1)
    args = parser.parse_args()
    build(args.topology, args.true_mesh, args.out, args.span)


if __name__ == "__main__":
    main()
