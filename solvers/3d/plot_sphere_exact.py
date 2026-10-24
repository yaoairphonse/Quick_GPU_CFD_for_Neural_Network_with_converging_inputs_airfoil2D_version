#!/usr/bin/env python3
"""Plot Sphere3D from exact cubed-sphere cell/face geometry.

The plane figures intersect every hexahedral control volume with z=0 and
paint its cell average on the resulting polygon. No nearest-neighbour,
triangulation, griddata, or other scattered interpolation is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


QUAD_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def unique_points(points, tolerance=1.0e-11):
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - other) <= tolerance for other in unique):
            unique.append(point)
    return unique


def plane_intersection(points, quads, axis=2, station=0.0):
    intersections = []
    for cell_faces in quads:
        candidates = []
        for face in cell_faces:
            vertices = points[face]
            distance = vertices[:, axis] - station
            for first, second in QUAD_EDGES:
                p0, p1 = vertices[first], vertices[second]
                d0, d1 = distance[first], distance[second]
                if d0 * d1 < 0.0:
                    candidates.append(p0 - d0 * (p1 - p0) / (d1 - d0))
                elif abs(d0) <= 1.0e-13:
                    candidates.append(p0)
                elif abs(d1) <= 1.0e-13:
                    candidates.append(p1)
        candidates = unique_points(candidates)
        if len(candidates) < 3:
            intersections.append(None)
            continue
        polygon = np.asarray(candidates)[:, :2]
        center = polygon.mean(axis=0)
        angle = np.arctan2(polygon[:, 1] - center[1],
                           polygon[:, 0] - center[0])
        intersections.append(polygon[np.argsort(angle)])
    return intersections


def wall_section(triangles, axis=2, station=0.0):
    points = []
    for triangle in triangles:
        distance = triangle[:, axis] - station
        for first, second in ((0, 1), (1, 2), (2, 0)):
            p0, p1 = triangle[first], triangle[second]
            d0, d1 = distance[first], distance[second]
            if d0 * d1 < 0.0:
                points.append(p0 - d0 * (p1 - p0) / (d1 - d0))
            elif abs(d0) <= 1.0e-13:
                points.append(p0)
            elif abs(d1) <= 1.0e-13:
                points.append(p1)
    section = np.asarray(unique_points(points))[:, :2]
    center = section.mean(axis=0)
    angle = np.arctan2(section[:, 1] - center[1],
                       section[:, 0] - center[0])
    return section[np.argsort(angle)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--field", required=True, type=Path)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mach", type=float, default=1.65)
    parser.add_argument("--gamma", type=float, default=1.4)
    args = parser.parse_args()

    with np.load(args.geometry) as geometry:
        points = np.asarray(geometry["points"], dtype=np.float64)
        quads = np.asarray(geometry["cell_quads"], dtype=np.int32)
        wall_triangles = np.asarray(geometry["wall_triangles"],
                                    dtype=np.float64)
    field = np.genfromtxt(args.field, delimiter=",", names=True)
    surface = np.genfromtxt(args.surface, delimiter=",", names=True)
    if len(field) != len(quads):
        raise ValueError("field and cubed-sphere cell counts disagree")
    if len(surface) != len(wall_triangles):
        raise ValueError("surface and exact wall-triangle counts disagree")

    all_polygons = plane_intersection(points, quads)
    selected = np.asarray([polygon is not None for polygon in all_polygons])
    polygons = [polygon for polygon in all_polygons if polygon is not None]
    mach = np.asarray(field["Mach"])[selected]
    pressure = np.asarray(field["p_rhoa2"])[selected]
    cp = (pressure - 1.0 / args.gamma) / (0.5 * args.mach ** 2)
    body = wall_section(wall_triangles)

    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    surface_axis = figure.add_subplot(grid[0, 0], projection="3d")
    mach_axis = figure.add_subplot(grid[0, 1])
    cp_axis = figure.add_subplot(grid[1, 0])
    mesh_axis = figure.add_subplot(grid[1, 1])

    surface_cp = np.asarray(surface["Cp"])
    cp_limit = max(0.5, float(np.nanpercentile(np.abs(surface_cp), 99)))
    surface_collection = Poly3DCollection(
        wall_triangles, array=surface_cp, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-cp_limit, vcenter=0.0, vmax=cp_limit),
        edgecolors="none", rasterized=True,
    )
    surface_axis.add_collection3d(surface_collection)
    surface_axis.set(
        xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), zlim=(-1.05, 1.05),
        xlabel="x / R", ylabel="y / R", zlabel="z / R",
        title="Cp on exact inner-wall triangles",
    )
    surface_axis.set_box_aspect((1, 1, 1))
    surface_axis.view_init(22, -125)
    figure.colorbar(surface_collection, ax=surface_axis, shrink=0.65,
                    pad=0.03, label="$C_p$")

    mach_collection = PolyCollection(
        polygons, array=mach, cmap="turbo", norm=Normalize(0.0, 2.0),
        edgecolors="none", rasterized=True,
    )
    mach_axis.add_collection(mach_collection)
    cp_plane_limit = max(0.5, float(np.nanpercentile(np.abs(cp), 99)))
    cp_collection = PolyCollection(
        polygons, array=cp, cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-cp_plane_limit, vcenter=0.0,
                          vmax=cp_plane_limit),
        edgecolors="none", rasterized=True,
    )
    cp_axis.add_collection(cp_collection)
    for axis, collection, title, label in (
        (mach_axis, mach_collection, "Mach on exact z=0 cell polygons", "Mach"),
        (cp_axis, cp_collection, "Cp on exact z=0 cell polygons", "$C_p$"),
    ):
        axis.add_patch(Polygon(body, facecolor="#d9d9d9",
                               edgecolor="black", linewidth=1.0, zorder=5))
        axis.set(xlim=(-2.5, 4.0), ylim=(-2.0, 2.0),
                 xlabel="x / R", ylabel="y / R", title=title)
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(collection, ax=axis, shrink=0.8, label=label)

    mesh_axis.add_collection(PolyCollection(
        polygons, facecolors="none", edgecolors=(0.05, 0.18, 0.25, 0.62),
        linewidths=0.28, rasterized=True,
    ))
    mesh_axis.add_patch(Polygon(body, facecolor="#d9d9d9",
                                edgecolor="black", linewidth=1.0, zorder=5))
    mesh_axis.set(
        xlim=(-2.5, 4.0), ylim=(-2.0, 2.0), xlabel="x / R", ylabel="y / R",
        title="True cubed-sphere control-volume intersections",
    )
    mesh_axis.set_aspect("equal", adjustable="box")

    convergence = "development result"
    result = None
    if args.result and args.result.is_file():
        result = json.loads(args.result.read_text())
        convergence = ("converged" if result.get("converged") else
                       "not converged")
    figure.suptitle(
        f"Full 3D six-block sphere | M={args.mach:g} | {len(field):,} CVs | "
        f"{convergence}\n"
        "Plane colors are piecewise-constant cell averages; no scattered interpolation",
        fontsize=15,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200)
    plt.close(figure)

    metadata = {
        "geometry_name": args.geometry.name,
        "field_name": args.field.name,
        "surface_name": args.surface.name,
        "cells": int(len(field)),
        "plane_cells": int(np.count_nonzero(selected)),
        "wall_triangles": int(len(wall_triangles)),
        "rendering": "exact hexahedron-plane polygons; cell averages; no interpolation",
        "result": result,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
