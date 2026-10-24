"""Audit tetrahedral CRM shape, non-orthogonality, and wall-triangle resolution."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


CELL_DTYPE = np.dtype([
    ("x", "<f8", (3,)), ("volume", "<f8"), ("dwall", "<f8"),
    ("first", "<i4"), ("count", "<i4"), ("term_first", "<i4"), ("term_count", "<i4"),
])
FACE_DTYPE = np.dtype({
    "names": ["left", "right", "bc", "x", "n", "area", "distance"],
    "formats": ["<i4", "<i4", "<i4", ("<f8", 3), ("<f8", 3), "<f8", "<f8"],
    "offsets": [0, 4, 8, 16, 40, 64, 72], "itemsize": 80,
})
BOUNDARY_DTYPE = np.dtype([("node", "<i4", (3,)), ("bc", "<i4"), ("patch", "<i4")])


def quantiles(values):
    probabilities = [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1]
    return {str(probability): float(value) for probability, value in zip(probabilities, np.quantile(values, probabilities))}


def plane_segments(triangles, coordinate, location):
    segments = []
    for triangle in triangles:
        intersections = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[first], triangle[second]
            da, db = a[coordinate] - location, b[coordinate] - location
            if da == 0:
                intersections.append(a)
            if da * db < 0:
                intersections.append(a + (-da / (db - da)) * (b - a))
        if len(intersections) >= 2:
            segments.append([[intersections[0][0], intersections[0][2]], [intersections[1][0], intersections[1][2]]])
    return np.asarray(segments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--connectivity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--y", type=float, default=2.1028824141451468)
    parser.add_argument("--chunk", type=int, default=250_000)
    arguments = parser.parse_args(); arguments.out.mkdir(parents=True, exist_ok=True)

    with arguments.mesh.open("rb") as stream:
        magic, cell_count, face_count, _, _ = struct.unpack("<8s4Q", stream.read(40))
    if magic.rstrip(b"\0") != b"QCFD3D1": raise ValueError("expected QCFD3D1 mesh")
    cells = np.memmap(arguments.mesh, dtype=CELL_DTYPE, mode="r", offset=40, shape=(cell_count,))
    faces = np.memmap(arguments.mesh, dtype=FACE_DTYPE, mode="r", offset=40 + cell_count * CELL_DTYPE.itemsize, shape=(face_count,))
    with arguments.connectivity.open("rb") as stream:
        fields = struct.unpack("<8sII10Q", stream.read(96))
    magic, version, endian, point_count, connectivity_cells, boundary_count, incidence_count, points_offset, cell_nodes_offset, _, _, boundary_offset, file_bytes = fields
    if magic.rstrip(b"\0") != b"QCFDNC1" or version != 1 or connectivity_cells != cell_count: raise ValueError("incompatible connectivity")
    points = np.memmap(arguments.connectivity, dtype="<f8", mode="r", offset=points_offset, shape=(point_count, 3))
    cell_nodes = np.memmap(arguments.connectivity, dtype="<i4", mode="r", offset=cell_nodes_offset, shape=(cell_count, 4))
    boundary = np.memmap(arguments.connectivity, dtype=BOUNDARY_DTYPE, mode="r", offset=boundary_offset, shape=(boundary_count,))

    mean_ratio = np.empty(cell_count, dtype=np.float32); aspect = np.empty(cell_count, dtype=np.float32)
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)); face_triples = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    for first in range(0, cell_count, arguments.chunk):
        last = min(first + arguments.chunk, cell_count); vertices = np.asarray(points[cell_nodes[first:last]])
        length2 = np.stack([np.einsum("ij,ij->i", vertices[:, b] - vertices[:, a], vertices[:, b] - vertices[:, a]) for a, b in edge_pairs], axis=1)
        volume = np.asarray(cells["volume"][first:last]); mean_ratio[first:last] = 12 * np.power(3 * volume, 2 / 3) / length2.sum(axis=1)
        face_area = []
        for a, b, c in face_triples:
            face_area.append(0.5 * np.linalg.norm(np.cross(vertices[:, b] - vertices[:, a], vertices[:, c] - vertices[:, a]), axis=1))
        minimum_altitude = 3 * volume / np.max(np.stack(face_area, axis=1), axis=1)
        aspect[first:last] = np.sqrt(length2.max(axis=1)) / minimum_altitude

    internal = np.flatnonzero(faces["right"] >= 0); nonorthogonality = np.empty(len(internal), dtype=np.float32)
    for first in range(0, len(internal), arguments.chunk):
        selected = internal[first:first + arguments.chunk]; face = faces[selected]
        displacement = cells["x"][face["right"]] - cells["x"][face["left"]]
        cosine = np.clip(face["distance"] / np.linalg.norm(displacement, axis=1), 0, 1)
        nonorthogonality[first:first + len(selected)] = np.degrees(np.arccos(cosine))

    wall = boundary[boundary["bc"] == 1]; wall_triangles = np.asarray(points[wall["node"]]); wall_edges = np.stack([
        np.linalg.norm(wall_triangles[:, 1] - wall_triangles[:, 0], axis=1),
        np.linalg.norm(wall_triangles[:, 2] - wall_triangles[:, 1], axis=1),
        np.linalg.norm(wall_triangles[:, 0] - wall_triangles[:, 2], axis=1),
    ], axis=1)
    wall_area = 0.5 * np.linalg.norm(np.cross(wall_triangles[:, 1] - wall_triangles[:, 0], wall_triangles[:, 2] - wall_triangles[:, 0]), axis=1)
    wall_quality = 4 * np.sqrt(3) * wall_area / np.square(wall_edges).sum(axis=1)

    report = {
        "topology": "tetrahedral volume cells with triangular boundary faces", "cells": int(cell_count), "faces": int(face_count),
        "internal_faces": int(len(internal)), "wall_triangles": int(len(wall)),
        "tetra_mean_ratio_1_is_equilateral": quantiles(mean_ratio),
        "longest_edge_over_minimum_altitude": quantiles(aspect),
        "internal_face_nonorthogonality_degrees_0_is_orthogonal": quantiles(nonorthogonality),
        "wall_triangle_quality_1_is_equilateral": quantiles(wall_quality),
        "wall_triangle_sqrt_area_over_MAC": quantiles(np.sqrt(wall_area)),
        "wall_edge_length_over_MAC": quantiles(wall_edges.ravel()),
        "counts": {
            "tetra_mean_ratio_below_0.1": int(np.count_nonzero(mean_ratio < 0.1)),
            "aspect_above_100": int(np.count_nonzero(aspect > 100)),
            "nonorthogonality_above_70_degrees": int(np.count_nonzero(nonorthogonality > 70)),
            "wall_triangle_quality_below_0.2": int(np.count_nonzero(wall_quality < 0.2)),
        },
    }
    (arguments.out / "crm_mesh_quality.json").write_text(json.dumps(report, indent=2) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].hist(mean_ratio, bins=100, range=(0, 1)); axes[0, 0].set(xlabel="tetra mean-ratio quality", ylabel="cells", yscale="log")
    axes[0, 1].hist(aspect, bins=np.logspace(0, 7, 100)); axes[0, 1].set(xlabel="longest edge / minimum altitude", ylabel="cells", xscale="log", yscale="log")
    axes[1, 0].hist(nonorthogonality, bins=90, range=(0, 90)); axes[1, 0].set(xlabel="internal-face non-orthogonality [deg]", ylabel="faces", yscale="log")
    axes[1, 1].hist(np.sqrt(wall_area), bins=np.logspace(-7, 0, 100)); axes[1, 1].set(xlabel="wall triangle sqrt(area) / MAC", ylabel="triangles", xscale="log", yscale="log")
    figure.suptitle("NASA DPW4 CRM 10M tetrahedral mesh quality")
    figure.savefig(arguments.out / "crm_mesh_quality.png", dpi=200); plt.close(figure)

    planform_segments = np.concatenate([wall_triangles[:, [0, 1], :2], wall_triangles[:, [1, 2], :2], wall_triangles[:, [2, 0], :2]])
    section_segments = plane_segments(wall_triangles, 1, arguments.y)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].add_collection(LineCollection(planform_segments, colors="black", linewidths=0.08, rasterized=True)); axes[0].autoscale(); axes[0].set_aspect("equal"); axes[0].set(xlabel="x / MAC", ylabel="y / MAC", title="wall-triangle planform projection")
    if len(section_segments): axes[1].add_collection(LineCollection(section_segments, colors="black", linewidths=0.5)); axes[1].autoscale()
    axes[1].set_aspect("equal"); axes[1].set(xlabel="x / MAC", ylabel="z / MAC", title=f"wall intersection at y={arguments.y:.4f}")
    figure.savefig(arguments.out / "crm_wall_mesh.png", dpi=240); plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
