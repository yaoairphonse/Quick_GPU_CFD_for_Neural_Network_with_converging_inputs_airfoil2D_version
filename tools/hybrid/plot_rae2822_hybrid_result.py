#!/usr/bin/env python3
"""Plot exact mixed-cell RAE2822 polygons and a QCFD3D1 result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--field", required=True, type=Path)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="RAE2822 LF1 hybrid-grid solver check")
    args = parser.parse_args()

    with np.load(args.geometry) as geometry:
        vertices = geometry["polygon_vertices"]
        offsets = geometry["polygon_offsets"]
        face_nodes = geometry["face_nodes"]
        face_cells = geometry["face_cells"]
        face_zones = geometry["face_zones"]
        wall_segments = geometry["wall_segments"]
    field = np.genfromtxt(args.field, delimiter=",", names=True)
    surface = np.genfromtxt(args.surface, delimiter=",", names=True)
    if len(field) != len(offsets) - 1:
        raise ValueError("field and polygon cell counts disagree")

    polygons = [vertices[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    polygon_degrees = np.diff(offsets)
    triangle_count = int(np.count_nonzero(polygon_degrees == 3))
    quad_count = int(np.count_nonzero(polygon_degrees == 4))
    result = None
    converged = False
    if args.result and args.result.is_file():
        result = json.loads(args.result.read_text())
        converged = bool(result.get("converged", False))
    mach = np.asarray(field["Mach"])
    norm = Normalize(vmin=0.0, vmax=1.2)
    cmap = "turbo"

    figure = plt.figure(figsize=(15.5, 8.6), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
    full = figure.add_subplot(grid[0, :])
    zoom = figure.add_subplot(grid[1, 0])
    cp_axis = figure.add_subplot(grid[1, 1])

    full_collection = PolyCollection(polygons, array=mach, cmap=cmap, norm=norm, edgecolors="none")
    full.add_collection(full_collection)
    full.set_xlim(-1.0, 2.0)
    full.set_ylim(-0.75, 0.75)
    full.set_aspect("equal")
    full.set_title("Cell-filled Mach field (exact triangle/quad polygons; no interpolation holes)")
    full.set_xlabel("x / chord")
    full.set_ylabel("z / chord")
    figure.colorbar(full_collection, ax=full, label="Mach", shrink=0.9)

    zoom_collection = PolyCollection(
        polygons, array=mach, cmap=cmap, norm=norm, edgecolors=(0.0, 0.0, 0.0, 0.42), linewidths=0.22
    )
    zoom.add_collection(zoom_collection)
    zoom.set_xlim(-0.08, 1.08)
    zoom.set_ylim(-0.2, 0.2)
    zoom.set_aspect("equal")
    zoom.set_title(
        f"Near-airfoil mixed mesh: {triangle_count:,} triangles + "
        f"{quad_count:,} quads"
    )
    zoom.set_xlabel("x / chord")
    zoom.set_ylabel("z / chord")

    # Paint Cp directly on the physical airfoil segments, while also plotting
    # upper/lower distributions so a malformed surface ordering is obvious.
    segment_points = np.stack(
        (
            np.column_stack((surface["x_R"], surface["z_R"])),
            np.column_stack((surface["x_R"], surface["z_R"])),
        ),
        axis=1,
    )
    # The surface CSV contains face midpoints.  Use the archived exact wall
    # segments for the colored airfoil inset when the counts agree.
    if len(wall_segments) == len(surface):
        segment_points = wall_segments
    cp = np.asarray(surface["Cp"])
    cp_limit = max(0.5, float(np.nanpercentile(np.abs(cp), 98)))
    cp_collection = LineCollection(segment_points, array=cp, cmap="coolwarm", norm=Normalize(-cp_limit, cp_limit), linewidths=5)
    inset = cp_axis.inset_axes([0.48, 0.54, 0.49, 0.39])
    inset.add_collection(cp_collection)
    inset.autoscale()
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("Cp painted on wall", fontsize=9)

    upper = surface["z_R"] >= 0.0
    for mask, label, marker in ((upper, "upper", "o"), (~upper, "lower", "s")):
        order = np.argsort(surface["x_R"][mask])
        cp_axis.plot(surface["x_R"][mask][order], cp[mask][order], marker=marker, markersize=3, linewidth=1, label=label)
    cp_axis.axhline(0.0, color="0.55", linewidth=0.7)
    cp_axis.invert_yaxis()
    cp_axis.set_xlim(-0.02, 1.02)
    cp_axis.set_xlabel("x / chord")
    cp_axis.set_ylabel("pressure coefficient Cp")
    cp_axis.set_title(
        "Wall pressure (" + ("converged state" if converged else
                              "current unconverged state") + ")"
    )
    cp_axis.legend(loc="lower left")

    status = ("converged" if converged else
              "visualization only; SA residual not converged")
    figure.suptitle(args.title + " — " + status, fontsize=15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)
    args.out.with_suffix(".json").write_text(json.dumps({
        "cells": int(len(polygons)),
        "triangles": triangle_count,
        "quadrilaterals": quad_count,
        "rendering": (
            "exact archived triangle/quad polygons painted by the matching "
            "extruded wedge/hexa control-volume average; no interpolation"
        ),
        "result": result,
    }, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
