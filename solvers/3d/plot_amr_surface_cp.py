#!/usr/bin/env python3
"""Paint area-averaged AMR cut-fragment Cp on the exact CRM triangles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def surface_panel(axis, vertices, normals, cp, selector, projection, title, norm):
    chosen = np.flatnonzero(selector)
    polygons = vertices[chosen][:, :, projection]
    collection = PolyCollection(
        polygons, array=cp[chosen], cmap="RdBu_r", norm=norm,
        edgecolors="none", linewidths=0, rasterized=True,
    )
    axis.add_collection(collection)
    flat = polygons.reshape(-1, 2)
    axis.set_xlim(float(flat[:, 0].min()), float(flat[:, 0].max()))
    axis.set_ylim(float(flat[:, 1].min()), float(flat[:, 1].max()))
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"{title} ({len(chosen):,} exact triangles)")
    axis.set_xlabel("x / MAC")
    return collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cp-min", type=float, default=-0.8)
    parser.add_argument("--cp-max", type=float, default=1.2)
    args = parser.parse_args()

    with np.load(args.geometry) as source:
        points = np.asarray(source["points"])
        triangles = np.asarray(source["triangles"])
    table = np.loadtxt(
        args.run / "surface.csv", delimiter=",", skiprows=1,
        usecols=(0, 10),
    )
    triangle_id = table[:, 0].astype(np.int64)
    if not np.array_equal(triangle_id, np.arange(len(triangles))):
        raise ValueError("surface.csv does not contain every source triangle in order")
    cp = table[:, 1]
    vertices = points[triangles]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normals = cross / np.linalg.norm(cross, axis=1)[:, None]
    result = json.loads((args.run / "result.json").read_text())
    norm = TwoSlopeNorm(vmin=args.cp_min, vcenter=0.0, vmax=args.cp_max)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    collection = surface_panel(
        axes[0, 0], vertices, normals, cp, normals[:, 2] > 0.05,
        (0, 1), "upper-facing surface", norm,
    )
    axes[0, 0].set_ylabel("y / MAC")
    surface_panel(
        axes[0, 1], vertices, normals, cp, normals[:, 2] < -0.05,
        (0, 1), "lower-facing surface", norm,
    )
    axes[0, 1].set_ylabel("y / MAC")
    surface_panel(
        axes[1, 0], vertices, normals, cp, normals[:, 1] > 0.05,
        (0, 2), "starboard-facing surface", norm,
    )
    axes[1, 0].set_ylabel("z / MAC")

    history = np.genfromtxt(args.run / "history.csv", delimiter=",", names=True)
    history = np.atleast_1d(history)
    axes[1, 1].plot(history["iteration"], history["CL"], "o-", label="$C_L$")
    axes[1, 1].plot(history["iteration"], history["CD"], "o-", label="$C_D$")
    axes[1, 1].plot(history["iteration"], history["CM_y"], "o-", label="$C_{M_y}$")
    axes[1, 1].set_xlabel("iterations in this restart segment")
    axes[1, 1].set_ylabel("coefficient")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    axes[1, 1].set_title("load evolution (not converged)")

    colorbar = figure.colorbar(collection, ax=axes[:, :], shrink=0.8, pad=0.02)
    colorbar.set_label("pressure coefficient $C_p$")
    figure.suptitle(
        "CRM procedural AMR pressure on original surface; "
        f"CL={result['CL']:.4f}, CD={result['CD']:.4f}, CM={result['CM_y']:.4f}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
