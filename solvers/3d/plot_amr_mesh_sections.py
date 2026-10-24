#!/usr/bin/env python3
"""Plot true Cartesian AMR block and cell intersections for CRM sections."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from plot_amr_flow_slices import (
    BLOCK_DTYPE,
    read_header,
    section_geometry,
)


STATUS_COLORS = {
    0: "#edf8fb",  # regular fluid
    1: "#969696",  # solid/inactive
    2: "#ff7f00",  # cut cell
}


def selected_blocks(blocks, block_cells, axis, station):
    return np.flatnonzero(
        (blocks["low"][:, axis] <= station)
        & (station < blocks["low"][:, axis]
           + block_cells * blocks["spacing"])
    )


def rectangles_for_blocks(blocks, indices, block_cells, projection):
    horizontal, vertical = projection
    polygons = []
    levels = []
    for block_index in indices:
        block = blocks[block_index]
        x0 = float(block["low"][horizontal])
        y0 = float(block["low"][vertical])
        width = block_cells * float(block["spacing"])
        polygons.append(((x0, y0), (x0 + width, y0),
                         (x0 + width, y0 + width), (x0, y0 + width)))
        levels.append(int(block["level"]))
    return polygons, np.asarray(levels)


def rectangles_for_cells(blocks, indices, status, block_cells, axis,
                         station, projection, bounds):
    horizontal, vertical = projection
    polygons = []
    colors = []
    counts = {0: 0, 1: 0, 2: 0}
    spacings = []
    for block_index in indices:
        block = blocks[block_index]
        low = block["low"]
        spacing = float(block["spacing"])
        high_horizontal = low[horizontal] + block_cells * spacing
        high_vertical = low[vertical] + block_cells * spacing
        if (high_horizontal <= bounds[0][0] or low[horizontal] >= bounds[0][1]
                or high_vertical <= bounds[1][0] or low[vertical] >= bounds[1][1]):
            continue
        local_axis = int(np.floor((station - low[axis]) / spacing))
        local_axis = min(max(local_axis, 0), block_cells - 1)
        spacings.append(spacing)
        for j in range(block_cells):
            for i in range(block_cells):
                coordinate = [0, 0, 0]
                coordinate[axis] = local_axis
                coordinate[horizontal] = i
                coordinate[vertical] = j
                local_id = ((coordinate[2] * block_cells + coordinate[1])
                            * block_cells + coordinate[0])
                procedural = int(block["first_procedural"]) + local_id
                cell_status = int(status[procedural])
                x0 = float(low[horizontal] + i * spacing)
                y0 = float(low[vertical] + j * spacing)
                if (x0 + spacing <= bounds[0][0] or x0 >= bounds[0][1]
                        or y0 + spacing <= bounds[1][0] or y0 >= bounds[1][1]):
                    continue
                polygons.append(((x0, y0), (x0 + spacing, y0),
                                 (x0 + spacing, y0 + spacing),
                                 (x0, y0 + spacing)))
                colors.append(STATUS_COLORS[cell_status])
                counts[cell_status] += 1
    return polygons, colors, counts, sorted(set(spacings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    header = read_header(args.mesh)
    blocks = np.memmap(args.mesh / "blocks.bin", dtype=BLOCK_DTYPE, mode="r",
                       shape=header["blocks"])
    status = np.memmap(args.mesh / "procedural_status.bin", dtype="u1",
                       mode="r", shape=header["procedural_cells"])
    with np.load(args.geometry) as geometry:
        points = np.asarray(geometry["points"], dtype=np.float64)
        triangles = np.asarray(geometry["triangles"], dtype=np.int32)

    specifications = [
        {
            "name": "XZ symmetry, y=0+", "axis": 1, "station": 1e-6,
            "projection": (0, 2), "labels": ("x / MAC", "z / MAC"),
            "overview": ((-1.0, 11.5), (-1.0, 3.0)),
            "zoom": ((3.9, 5.5), (0.9, 1.8)),
        },
        {
            "name": "XY wing plane, z=0.800", "axis": 2, "station": 0.8,
            "projection": (0, 1), "labels": ("x / MAC", "y / MAC"),
            "overview": ((-1.0, 11.5), (-5.5, 5.5)),
            "zoom": ((3.7, 5.3), (1.1, 2.5)),
        },
        {
            "name": "YZ wing/fuselage plane, x=5.000", "axis": 0,
            "station": 5.0, "projection": (1, 2),
            "labels": ("y / MAC", "z / MAC"),
            "overview": ((-3.0, 3.0), (0.0, 1.65)),
            "zoom": ((1.40, 2.35), (0.67, 0.82)),
        },
    ]

    level_cmap = ListedColormap(plt.colormaps["viridis"](
        np.linspace(0.05, 0.95, 9)))
    level_norm = BoundaryNorm(np.arange(1.5, 11.5, 1), level_cmap.N)
    figure, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    report = []

    for row, specification in enumerate(specifications):
        indices = selected_blocks(
            blocks, header["block_cells"], specification["axis"],
            specification["station"],
        )
        outlines, segment_count = section_geometry(
            points, triangles, specification["axis"], specification["station"],
            specification["projection"],
        )

        block_polygons, levels = rectangles_for_blocks(
            blocks, indices, header["block_cells"], specification["projection"])
        overview_axis = axes[row, 0]
        block_collection = PolyCollection(
            block_polygons, array=levels, cmap=level_cmap, norm=level_norm,
            edgecolors=(0, 0, 0, 0.24), linewidths=0.08, rasterized=True,
        )
        overview_axis.add_collection(block_collection)
        overview_axis.add_collection(LineCollection(
            outlines, colors="black", linewidths=0.42, zorder=4,
        ))
        overview_axis.set_title(
            f"{specification['name']} — leaf blocks (each block = 8×8 cells)"
        )
        overview_axis.set_xlim(*specification["overview"][0])
        overview_axis.set_ylim(*specification["overview"][1])
        overview_axis.set_aspect("equal", adjustable="box")
        overview_axis.set_xlabel(specification["labels"][0])
        overview_axis.set_ylabel(specification["labels"][1])
        figure.colorbar(block_collection, ax=overview_axis, shrink=0.78,
                        ticks=np.arange(2, 11), label="AMR level")

        cell_polygons, cell_colors, counts, spacings = rectangles_for_cells(
            blocks, indices, status, header["block_cells"],
            specification["axis"], specification["station"],
            specification["projection"], specification["zoom"],
        )
        zoom_axis = axes[row, 1]
        zoom_axis.add_collection(PolyCollection(
            cell_polygons, facecolors=cell_colors,
            edgecolors=(0, 0, 0, 0.78), linewidths=0.16,
            rasterized=True,
        ))
        zoom_axis.add_collection(LineCollection(
            outlines, colors="black", linewidths=1.0, zorder=5,
        ))
        zoom_axis.set_title(
            f"actual Cartesian cells (zoom); cut cells={counts[2]:,}"
        )
        zoom_axis.set_xlim(*specification["zoom"][0])
        zoom_axis.set_ylim(*specification["zoom"][1])
        zoom_axis.set_aspect("equal", adjustable="box")
        zoom_axis.set_xlabel(specification["labels"][0])
        zoom_axis.set_ylabel(specification["labels"][1])
        zoom_axis.legend(handles=[
            Patch(facecolor=STATUS_COLORS[0], edgecolor="black", label="fluid"),
            Patch(facecolor=STATUS_COLORS[1], edgecolor="black", label="solid"),
            Patch(facecolor=STATUS_COLORS[2], edgecolor="black", label="cut cell"),
        ], loc="upper right", fontsize=9)
        report.append((specification["name"], len(indices), len(cell_polygons),
                       counts, spacings, segment_count))

        if specification["axis"] == 0:
            detail, detail_axes = plt.subplots(
                1, 2, figsize=(16, 5.8), constrained_layout=True,
            )
            detail_blocks = PolyCollection(
                block_polygons, array=levels, cmap=level_cmap,
                norm=level_norm, edgecolors=(0, 0, 0, 0.30),
                linewidths=0.12, rasterized=True,
            )
            detail_axes[0].add_collection(detail_blocks)
            detail_axes[0].add_collection(LineCollection(
                outlines, colors="black", linewidths=0.65, zorder=4,
            ))
            detail_axes[0].set_title(
                "YZ x=5.0: true leaf blocks (each block = 8x8x8 cells)"
            )
            detail_axes[0].set_xlim(*specification["overview"][0])
            detail_axes[0].set_ylim(*specification["overview"][1])
            detail.colorbar(
                detail_blocks, ax=detail_axes[0], shrink=0.80,
                ticks=np.arange(2, 11), label="AMR level",
            )

            detail_axes[1].add_collection(PolyCollection(
                cell_polygons, facecolors=cell_colors,
                edgecolors=(0, 0, 0, 0.82), linewidths=0.20,
                rasterized=True,
            ))
            detail_axes[1].add_collection(LineCollection(
                outlines, colors="black", linewidths=1.1, zorder=5,
            ))
            detail_axes[1].set_title(
                f"same plane: actual cells near starboard wing; "
                f"cut cells={counts[2]:,}"
            )
            detail_axes[1].set_xlim(*specification["zoom"][0])
            detail_axes[1].set_ylim(*specification["zoom"][1])
            detail_axes[1].legend(handles=[
                Patch(facecolor=STATUS_COLORS[0], edgecolor="black",
                      label="fluid"),
                Patch(facecolor=STATUS_COLORS[1], edgecolor="black",
                      label="solid"),
                Patch(facecolor=STATUS_COLORS[2], edgecolor="black",
                      label="cut cell"),
            ], loc="upper right", fontsize=9)
            for detail_axis in detail_axes:
                detail_axis.set_aspect("equal", adjustable="box")
                detail_axis.set_xlabel(specification["labels"][0])
                detail_axis.set_ylabel(specification["labels"][1])
            detail.suptitle(
                "CRM procedural AMR mesh at the same YZ section as the flow figure\n"
                "black curve = exact original-surface intersection",
                fontsize=15,
            )
            detail.savefig(args.out / "crm_amr_yz_x5_mesh.png", dpi=240)
            plt.close(detail)

    figure.suptitle(
        "CRM procedural Cartesian AMR: true block and cell cross-sections\n"
        "black curve = original CRM triangle-surface intersection",
        fontsize=16,
    )
    output = args.out / "crm_amr_true_mesh_sections.png"
    figure.savefig(output, dpi=240)
    plt.close(figure)
    print(output)
    for item in report:
        print(item)


if __name__ == "__main__":
    main()
