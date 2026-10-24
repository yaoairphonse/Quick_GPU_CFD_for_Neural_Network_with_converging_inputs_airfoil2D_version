#!/usr/bin/env python3
"""Render hole-free QCFDAM1 flow slices directly from Cartesian leaf cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection
import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union


HEADER_FORMAT = "<8sIIII7Q4d"
BLOCK_DTYPE = np.dtype([
    ("low", "<f8", (3,)), ("spacing", "<f8"),
    ("fine_coordinate", "<i4", (3,)), ("fine_span", "<i4"),
    ("first_procedural", "<i4"), ("level", "<i2"),
    ("kind", "u1"), ("padding", "u1"),
])
CONTROL_DTYPE = np.dtype([
    ("x", "<f8", (3,)), ("volume", "<f8"), ("dwall", "<f8"),
])


def read_header(root: Path):
    values = struct.unpack(HEADER_FORMAT, (root / "header.bin").read_bytes())
    if values[0] != b"QCFDAM1\0" or values[1] != 1:
        raise ValueError("invalid QCFDAM1 header")
    return {
        "block_cells": values[3], "blocks": values[5],
        "procedural_cells": values[6], "controls": values[7],
    }


def primitive_fields(state, gamma, mach_inf, batch=1_000_000):
    count = len(state)
    mach = np.empty(count, dtype=np.float32)
    cp = np.empty(count, dtype=np.float32)
    q_inf = 0.5 * mach_inf ** 2
    p_inf = 1.0 / gamma
    for first in range(0, count, batch):
        last = min(first + batch, count)
        conserved = np.asarray(state[first:last])
        density = conserved[:, 0]
        velocity = conserved[:, 1:4] / density[:, None]
        pressure = (gamma - 1.0) * (
            conserved[:, 4] -
            0.5 * density * np.einsum("ij,ij->i", velocity, velocity)
        )
        sound = np.sqrt(gamma * pressure / density)
        mach[first:last] = np.linalg.norm(velocity, axis=1) / sound
        cp[first:last] = (pressure - p_inf) / q_inf
    return mach, cp


def section_geometry(points, triangles, axis, station, projection):
    vertices = points[triangles]
    selected = ((vertices[:, :, axis].min(axis=1) < station) &
                (vertices[:, :, axis].max(axis=1) > station))
    segments = []
    tolerance = 1e-12
    for triangle in vertices[selected]:
        distance = triangle[:, axis] - station
        intersections = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            d0, d1 = distance[first], distance[second]
            if d0 * d1 < 0:
                fraction = -d0 / (d1 - d0)
                intersections.append(
                    triangle[first] + fraction * (triangle[second] - triangle[first])
                )
            elif abs(d0) <= tolerance and abs(d1) > tolerance:
                intersections.append(triangle[first])
            elif abs(d1) <= tolerance and abs(d0) > tolerance:
                intersections.append(triangle[second])
        if len(intersections) < 2:
            continue
        unique = []
        for point in intersections:
            if not any(np.linalg.norm(point - other) <= 1e-10 for other in unique):
                unique.append(point)
        if len(unique) == 2:
            segments.append(LineString([
                unique[0][list(projection)], unique[1][list(projection)]
            ]))
    if not segments:
        raise ValueError("surface does not intersect requested plane")
    merged = unary_union(segments)
    if merged.geom_type == "LineString":
        outlines = [np.asarray(merged.coords)]
    else:
        outlines = [np.asarray(line.coords) for line in merged.geoms]
    return outlines, len(segments)


def add_surface_intersection(axis, outlines):
    axis.add_collection(LineCollection(
        outlines, colors="black", linewidths=0.7, zorder=5,
    ))


def raster_slice(blocks, block_cells, procedural_control, status,
                 scalar_fields, axis, station, projection, bounds,
                 resolution):
    horizontal, vertical = projection
    nx, ny = resolution
    x_edges = np.linspace(bounds[0][0], bounds[0][1], nx + 1)
    y_edges = np.linspace(bounds[1][0], bounds[1][1], ny + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    fields = [np.full((ny, nx), np.nan, dtype=np.float32)
              for _ in scalar_fields]
    covered = np.zeros((ny, nx), dtype=bool)
    fluid = np.zeros((ny, nx), dtype=bool)
    selected_blocks = np.flatnonzero(
        (blocks["low"][:, axis] <= station) &
        (station < blocks["low"][:, axis] +
                   block_cells * blocks["spacing"])
    )
    for block in selected_blocks:
        low = blocks["low"][block]
        spacing = float(blocks["spacing"][block])
        high = low + block_cells * spacing
        ix0 = int(np.searchsorted(x, low[horizontal], side="left"))
        ix1 = int(np.searchsorted(x, high[horizontal], side="left"))
        iy0 = int(np.searchsorted(y, low[vertical], side="left"))
        iy1 = int(np.searchsorted(y, high[vertical], side="left"))
        ix0, ix1 = max(ix0, 0), min(ix1, nx)
        iy0, iy1 = max(iy0, 0), min(iy1, ny)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        local_horizontal = np.floor(
            (x[ix0:ix1] - low[horizontal]) / spacing
        ).astype(np.int32)
        local_vertical = np.floor(
            (y[iy0:iy1] - low[vertical]) / spacing
        ).astype(np.int32)
        local_axis = int(np.floor((station - low[axis]) / spacing))
        local_axis = min(max(local_axis, 0), block_cells - 1)
        coordinates = [None, None, None]
        coordinates[axis] = np.full(
            (len(local_vertical), len(local_horizontal)), local_axis,
            dtype=np.int32,
        )
        coordinates[horizontal] = np.broadcast_to(
            local_horizontal[None, :],
            (len(local_vertical), len(local_horizontal)),
        )
        coordinates[vertical] = np.broadcast_to(
            local_vertical[:, None],
            (len(local_vertical), len(local_horizontal)),
        )
        local_id = ((coordinates[2] * block_cells + coordinates[1]) *
                    block_cells + coordinates[0])
        procedural = int(blocks["first_procedural"][block]) + local_id
        active = status[procedural] != 1
        control = procedural_control[procedural]
        if np.any(active & (control < 0)):
            raise ValueError("an active raster cell has no solver control")
        for output, values in zip(fields, scalar_fields):
            patch = output[iy0:iy1, ix0:ix1]
            patch[active] = values[control[active]]
        covered[iy0:iy1, ix0:ix1] = True
        fluid_patch = fluid[iy0:iy1, ix0:ix1]
        fluid_patch[active] = True

    xx, yy = np.meshgrid(x, y)
    return x_edges, y_edges, xx, yy, fields, covered, fluid


def render_slice(path, title, x_edges, y_edges, xx, yy, mach, cp,
                 outlines, covered, fluid, labels, cp_limits, mach_limits,
                 metadata, sonic_contour):
    uncovered = ~covered
    missing_fluid = fluid & np.isnan(mach)
    metadata["uncovered_domain_pixels"] = int(np.count_nonzero(uncovered))
    metadata["unfilled_fluid_pixels"] = int(np.count_nonzero(missing_fluid))
    metadata["solid_pixels"] = int(np.count_nonzero(covered & ~fluid))
    if np.any(uncovered):
        raise ValueError(
            f"slice contains {np.count_nonzero(uncovered)} uncovered domain pixels"
        )
    if np.any(missing_fluid):
        raise ValueError(
            f"slice contains {np.count_nonzero(missing_fluid)} unfilled fluid pixels"
        )

    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    cp_norm = TwoSlopeNorm(vmin=cp_limits[0], vcenter=0.0, vmax=cp_limits[1])
    cp_plot = axes[0].pcolormesh(
        x_edges, y_edges, cp, cmap="RdBu_r", norm=cp_norm,
        shading="flat", rasterized=True,
    )
    mach_plot = axes[1].pcolormesh(
        x_edges, y_edges, mach, cmap="turbo", vmin=mach_limits[0],
        vmax=mach_limits[1], shading="flat", rasterized=True,
    )
    finite_mach = mach[np.isfinite(mach)]
    if (sonic_contour != "none" and
            finite_mach.min() <= 1.0 <= finite_mach.max()):
        color = "white" if sonic_contour == "white" else "black"
        axes[1].contour(
            xx, yy, mach, levels=[1.0], colors=color, linewidths=0.8,
            linestyles="dashed",
        )
    for panel in axes:
        panel.set_facecolor("#d9d9d9")
        add_surface_intersection(panel, outlines)
        panel.set_aspect("equal", adjustable="box")
        panel.set_xlim(x_edges[0], x_edges[-1])
        panel.set_ylim(y_edges[0], y_edges[-1])
        panel.set_xlabel(labels[0])
        panel.set_ylabel(labels[1])
    axes[0].set_title(f"pressure coefficient $C_p$ [{cp_limits[0]}, {cp_limits[1]}]")
    sonic_label = ("" if sonic_contour == "none" else
                   f"; {sonic_contour} dashed contour: M=1")
    axes[1].set_title(
        f"Mach [{mach_limits[0]}, {mach_limits[1]}]{sonic_label}"
    )
    figure.colorbar(cp_plot, ax=axes[0], shrink=0.85, label="$C_p$")
    figure.colorbar(mach_plot, ax=axes[1], shrink=0.85, label="Mach")
    figure.suptitle(title + " — unconverged development state")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mach", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--xy-z", type=float, default=0.80)
    parser.add_argument("--yz-x", type=float, default=5.00)
    parser.add_argument("--symmetry-y", type=float, default=1e-6)
    parser.add_argument(
        "--sonic-contour", choices=("none", "black", "white"),
        default="none",
        help="Optional dashed M=1 contour; disabled by default to avoid "
             "confusing sonic islands with geometry holes.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    header = read_header(args.mesh)
    blocks = np.memmap(
        args.mesh / "blocks.bin", dtype=BLOCK_DTYPE, mode="r",
        shape=header["blocks"],
    )
    controls = np.memmap(
        args.mesh / "controls.bin", dtype=CONTROL_DTYPE, mode="r",
        shape=header["controls"],
    )
    procedural_control = np.memmap(
        args.mesh / "procedural_control.bin", dtype="<i4", mode="r",
        shape=header["procedural_cells"],
    )
    status = np.memmap(
        args.mesh / "procedural_status.bin", dtype="u1", mode="r",
        shape=header["procedural_cells"],
    )
    state = np.memmap(
        args.state, dtype="<f8", mode="r", shape=(header["controls"], 6),
    )
    mach, cp = primitive_fields(state, args.gamma, args.mach)
    with np.load(args.geometry) as geometry:
        points = np.asarray(geometry["points"], dtype=np.float64)
        triangles = np.asarray(geometry["triangles"], dtype=np.int32)

    specifications = [
        {
            "name": "crm_xz_symmetry_mach_cp.png", "axis": 1,
            "station": args.symmetry_y, "projection": (0, 2),
            "bounds": ((-1.0, 11.5), (-1.0, 3.0)),
            "resolution": (1800, 720),
            "title": "CRM symmetry-near x–z flow slice at y=0+",
            "labels": ("x / MAC", "z / MAC"),
        },
        {
            "name": "crm_xy_wing_mach_cp.png", "axis": 2,
            "station": args.xy_z, "projection": (0, 1),
            "bounds": ((-1.0, 11.5), (-5.5, 5.5)),
            "resolution": (1600, 1408),
            "title": f"CRM horizontal x–y flow slice at z={args.xy_z:.3f}",
            "labels": ("x / MAC", "y / MAC"),
        },
        {
            "name": "crm_yz_fullwing_mach_cp.png", "axis": 0,
            "station": args.yz_x, "projection": (1, 2),
            "bounds": ((-3.0, 3.0), (0.0, 1.65)),
            "resolution": (1500, 540),
            "title": f"CRM full-span y–z flow slice at x={args.yz_x:.3f}",
            "labels": ("y / MAC", "z / MAC"),
        },
    ]
    report = {
        "mesh_name": args.mesh.name,
        "state_name": args.state.name,
        "provenance": "Large mesh and state are external cache artifacts and are not bundled.",
        "sampling": "direct QCFDAM1 leaf-cell raster; no scattered interpolation",
        "planes": {},
    }
    for specification in specifications:
        outlines, segment_count = section_geometry(
            points, triangles, specification["axis"],
            specification["station"], specification["projection"],
        )
        x_edges, y_edges, xx, yy, fields, covered, fluid = raster_slice(
            blocks, header["block_cells"], procedural_control, status,
            (mach, cp), specification["axis"], specification["station"],
            specification["projection"], specification["bounds"],
            specification["resolution"],
        )
        plane_report = {
            "axis": specification["axis"], "station": specification["station"],
            "bounds": specification["bounds"],
            "resolution": specification["resolution"],
            "surface_intersection_segments": segment_count,
            "surface_intersection_polylines": len(outlines),
        }
        render_slice(
            args.out / specification["name"], specification["title"],
            x_edges, y_edges, xx, yy, fields[0], fields[1], outlines,
            covered, fluid,
            specification["labels"], (-0.8, 0.5), (0.2, 1.15), plane_report,
            args.sonic_contour,
        )
        report["planes"][specification["name"]] = plane_report
    (args.out / "slice_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
