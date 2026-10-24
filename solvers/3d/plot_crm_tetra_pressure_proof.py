"""Plot pressure on exact cuts through the imported CRM tetrahedral mesh."""
from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quick-cfd-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np


GAMMA = 1.4
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
BOUNDARY_DTYPE = np.dtype([
    ("node", "<i4", (3,)), ("bc", "<i4"), ("patch", "<i4")
])


def read_connectivity(path: Path):
    with path.open("rb") as stream:
        fields = struct.unpack("<8sII10Q", stream.read(96))
    (magic, version, endian, point_count, cell_count, boundary_count,
     _, points_offset, cell_nodes_offset, _, _, boundary_offset,
     file_bytes) = fields
    if (magic.rstrip(b"\0") != b"QCFDNC1" or version != 1 or
            endian != 0x01020304 or file_bytes != path.stat().st_size):
        raise ValueError("incompatible QCFDNC1 connectivity")
    points = np.memmap(path, dtype="<f8", mode="r", offset=points_offset,
                       shape=(point_count, 3))
    tetrahedra = np.memmap(path, dtype="<i4", mode="r",
                           offset=cell_nodes_offset, shape=(cell_count, 4))
    boundary = np.memmap(path, dtype=BOUNDARY_DTYPE, mode="r",
                         offset=boundary_offset, shape=(boundary_count,))
    return points, tetrahedra, boundary


def cut_polygon(vertices: np.ndarray, y: float):
    hits = []
    delta = vertices[:, 1] - y
    for first, second in EDGES:
        a, b = delta[first], delta[second]
        if abs(a) < 1e-13:
            hits.append(vertices[first, (0, 2)])
        if abs(b) < 1e-13:
            hits.append(vertices[second, (0, 2)])
        if a * b < 0:
            fraction = a / (a - b)
            point = vertices[first] + fraction * (vertices[second] - vertices[first])
            hits.append(point[(0, 2),])
    if len(hits) < 3:
        return None
    unique = np.unique(np.round(np.asarray(hits), 13), axis=0)
    if len(unique) < 3:
        return None
    center = unique.mean(axis=0)
    order = np.argsort(np.arctan2(unique[:, 1] - center[1],
                                  unique[:, 0] - center[0]))
    return unique[order]


def wall_segments(points, boundary, y: float, x_limits, z_limits):
    segments = []
    wall = boundary[boundary["bc"] == 1]
    for triangle in points[wall["node"]]:
        if (triangle[:, 1].min() > y or triangle[:, 1].max() < y or
                triangle[:, 0].max() < x_limits[0] or
                triangle[:, 0].min() > x_limits[1] or
                triangle[:, 2].max() < z_limits[0] or
                triangle[:, 2].min() > z_limits[1]):
            continue
        hits = []
        delta = triangle[:, 1] - y
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a, b = delta[first], delta[second]
            if abs(a) < 1e-13:
                hits.append(triangle[first, (0, 2)])
            if abs(b) < 1e-13:
                hits.append(triangle[second, (0, 2)])
            if a * b < 0:
                fraction = a / (a - b)
                point = triangle[first] + fraction * (triangle[second] - triangle[first])
                hits.append(point[(0, 2),])
        if len(hits) >= 2:
            unique = np.unique(np.round(np.asarray(hits), 13), axis=0)
            if len(unique) >= 2:
                segments.append(unique[:2])
    return segments


def collect_cut(points, tetrahedra, state, y, x_limits, z_limits, chunk):
    polygons, cell_ids, cp_values = [], [], []
    pressure_inf = 1.0 / GAMMA
    q_inf = 0.5 * 0.85 ** 2
    for first in range(0, len(tetrahedra), chunk):
        last = min(first + chunk, len(tetrahedra))
        vertices = np.asarray(points[tetrahedra[first:last]])
        selected = ((vertices[:, :, 1].min(axis=1) <= y) &
                    (vertices[:, :, 1].max(axis=1) >= y) &
                    (vertices[:, :, 0].max(axis=1) >= x_limits[0]) &
                    (vertices[:, :, 0].min(axis=1) <= x_limits[1]) &
                    (vertices[:, :, 2].max(axis=1) >= z_limits[0]) &
                    (vertices[:, :, 2].min(axis=1) <= z_limits[1]))
        local_ids = np.flatnonzero(selected)
        if not len(local_ids):
            continue
        conserved = np.asarray(state[first + local_ids])
        rho = conserved[:, 0]
        pressure = (GAMMA - 1.0) * (
            conserved[:, 4] -
            0.5 * np.sum(conserved[:, 1:4] ** 2, axis=1) / rho
        )
        cp = (pressure - pressure_inf) / q_inf
        for local_id, value in zip(local_ids, cp):
            polygon = cut_polygon(vertices[local_id], y)
            if polygon is not None:
                polygons.append(polygon)
                cell_ids.append(first + int(local_id))
                cp_values.append(float(value))
    return polygons, np.asarray(cell_ids), np.asarray(cp_values)


def add_cut(axis, polygons, cp, wall, limits, title, linewidth):
    collection = PolyCollection(
        polygons, array=cp, cmap="turbo", edgecolors=(0.04, 0.07, 0.09, 0.50),
        linewidths=linewidth, rasterized=True,
    )
    collection.set_clim(-1.2, 0.8)
    axis.add_collection(collection)
    axis.add_collection(LineCollection(wall, colors="white", linewidths=2.8))
    axis.add_collection(LineCollection(wall, colors="black", linewidths=1.1))
    axis.set(xlim=limits[0], ylim=limits[1], xlabel="x / MAC", ylabel="z / MAC",
             title=title, aspect="equal")
    return collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectivity", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--y", type=float, default=2.102697720811817)
    parser.add_argument("--chunk", type=int, default=200_000)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    points, tetrahedra, boundary = read_connectivity(args.connectivity)
    if len(tetrahedra) != result["cells"]:
        raise ValueError("result and connectivity cell counts differ")
    state = np.memmap(args.state, dtype="<f8", mode="r",
                      shape=(len(tetrahedra), 6))

    full_limits = ((4.50, 6.15), (0.48, 0.90))
    polygons, cell_ids, cp = collect_cut(
        points, tetrahedra, state, args.y, *full_limits, args.chunk
    )
    wall = wall_segments(points, boundary, args.y, *full_limits)

    zoom_limits = ((4.78, 5.18), (0.625, 0.755))
    zoom_selected = np.array([
        polygon[:, 0].max() >= zoom_limits[0][0] and
        polygon[:, 0].min() <= zoom_limits[0][1] and
        polygon[:, 1].max() >= zoom_limits[1][0] and
        polygon[:, 1].min() <= zoom_limits[1][1]
        for polygon in polygons
    ])
    zoom_polygons = [polygon for polygon, keep in zip(polygons, zoom_selected) if keep]
    zoom_cp = cp[zoom_selected]

    figure, axes = plt.subplots(2, 1, figsize=(16, 10), layout="constrained",
                                gridspec_kw={"height_ratios": [1.0, 1.15]})
    collection = add_cut(
        axes[0], polygons, cp, wall, full_limits,
        f"Complete wing section at y/MAC={args.y:.4f} — {len(polygons):,} original tetrahedra",
        0.075,
    )
    add_cut(
        axes[1], zoom_polygons, zoom_cp, wall, zoom_limits,
        f"Leading-edge zoom — {len(zoom_polygons):,} exact tetrahedron/plane polygons",
        0.28,
    )
    colorbar = figure.colorbar(collection, ax=axes, location="right", shrink=0.88,
                               label="pressure coefficient  $C_p$")
    colorbar.ax.tick_params(labelsize=10)
    figure.suptitle(
        "NASA CRM pressure field on the imported body-fitted tetrahedral mesh\n"
        "This is the mesh used by the previous CRM solves — it is not Cartesian AMR or cut-cell",
        fontsize=17,
    )
    figure.text(
        0.5, 0.005,
        "Each colored polygon is the exact intersection of one 3D tetrahedral control volume with the plane. "
        "Black/white curve: aircraft wall. Latest state is unconverged and is shown only to identify the mesh.",
        ha="center", fontsize=10,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=210)
    plt.close(figure)
    manifest = {
        "figure": args.out.name,
        "mesh_topology": "body-fitted tetrahedra with triangular wall faces",
        "source_cells": len(tetrahedra),
        "plane_y_over_MAC": args.y,
        "shown_intersected_tetrahedra": len(polygons),
        "zoom_intersected_tetrahedra": len(zoom_polygons),
        "unique_cell_ids": len(np.unique(cell_ids)),
        "Cp_range_shown_cells": [float(cp.min()), float(cp.max())],
        "result_converged": bool(result["converged"]),
        "result_CD": result["CD"],
        "field_note": "Cell-centered pressure; no scattered interpolation. Figure is for mesh identification.",
    }
    args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
