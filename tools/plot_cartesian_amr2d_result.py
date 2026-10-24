#!/usr/bin/env python3
"""Plot a 2-D Cartesian-AMR result from its actual member rectangles.

Every displayed rectangle is an original quadtree member and receives the
piecewise-constant value of its owning (possibly agglomerated) control volume.
The solid is masked by the input polygon. No griddata or scattered
interpolation is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon
import numpy as np


def rectangles(rows):
    return np.asarray([
        ((row[2], row[4]), (row[3], row[4]),
         (row[3], row[5]), (row[2], row[5]))
        for row in rows
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path,
                        help="One LF1/LF2 result directory")
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--bounds", nargs=4, type=float,
                        default=(-0.35, 1.80, -0.50, 0.50),
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    args = parser.parse_args()

    field_path = args.case / "output/field.npz"
    surface_path = args.case / "output/surface.npz"
    members_path = args.case / "internal/flow.agglomeration.npy"
    config_path = args.case / "config.resolved.json"
    result_path = args.case / "result.json"
    for path in (field_path, surface_path, members_path, config_path,
                 result_path, args.geometry):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(field_path) as data:
        field = {key: np.asarray(data[key]) for key in data.files}
    with np.load(surface_path) as data:
        surface = {key: np.asarray(data[key]) for key in data.files}
    members = np.load(members_path)
    polygon = np.loadtxt(
        args.geometry, delimiter=",", skiprows=1, usecols=(0, 1)
    )
    config = json.loads(config_path.read_text())
    result = json.loads(result_path.read_text())

    cell_ids = np.asarray(field["cell_id"], dtype=np.int64)
    if not np.array_equal(cell_ids, np.arange(len(cell_ids))):
        raise ValueError("showcase plot requires an unfiltered full-cell export")
    owners = members[:, 1].astype(np.int64)
    if owners.min() < 0 or owners.max() >= len(cell_ids):
        raise ValueError("member owner is outside the exported field")

    xmin, xmax, ymin, ymax = args.bounds
    visible = ((members[:, 3] >= xmin) & (members[:, 2] <= xmax) &
               (members[:, 5] >= ymin) & (members[:, 4] <= ymax))
    visible_members = members[visible]
    visible_owners = owners[visible]
    cells = rectangles(visible_members)

    freestream = config["freestream"]
    q_inf = (0.5 * freestream["gamma"] * freestream["p_Pa"] *
             freestream["Mach"] ** 2)
    cp = (field["p"] - freestream["p_Pa"]) / q_inf
    mach_values = field["mach"]
    cp_limit = max(0.25, float(np.nanpercentile(
        np.abs(cp[visible_owners]), 99)))
    mach_limits = np.nanpercentile(mach_values[visible_owners], (0.5, 99.5))
    if mach_limits[1] <= mach_limits[0]:
        mach_limits = (0.0, max(1.0, float(mach_limits[1])))

    figure, axes = plt.subplots(2, 2, figsize=(15, 9),
                                constrained_layout=True)
    mach_collection = PolyCollection(
        cells, array=mach_values[visible_owners], cmap="turbo",
        norm=Normalize(*mach_limits), edgecolors="none", rasterized=True,
    )
    cp_collection = PolyCollection(
        cells, array=cp[visible_owners], cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-cp_limit, vcenter=0.0, vmax=cp_limit),
        edgecolors="none", rasterized=True,
    )
    for axis, collection, title, label in (
        (axes[0, 0], mach_collection, "Mach on actual AMR members", "Mach"),
        (axes[0, 1], cp_collection, "Cp on actual AMR members", "$C_p$"),
    ):
        axis.add_collection(collection)
        axis.add_patch(Polygon(
            polygon, facecolor="#d9d9d9", edgecolor="black",
            linewidth=1.0, zorder=5,
        ))
        axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
                 xlabel="x / chord", ylabel="y / chord", title=title)
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(collection, ax=axis, shrink=0.82, label=label)

    mesh_axis = axes[1, 0]
    mesh_axis.add_collection(PolyCollection(
        cells, facecolors="none", edgecolors=(0.05, 0.20, 0.28, 0.70),
        linewidths=0.28, rasterized=True,
    ))
    mesh_axis.add_patch(Polygon(
        polygon, facecolor="#d9d9d9", edgecolor="black",
        linewidth=1.1, zorder=5,
    ))
    mesh_axis.set(xlim=(-0.10, 1.10), ylim=(-0.20, 0.20),
                  xlabel="x / chord", ylabel="y / chord",
                  title="True quadtree member rectangles near the wall")
    mesh_axis.set_aspect("equal", adjustable="box")

    pressure_axis = axes[1, 1]
    patch = surface["patch_id"].astype(np.int64)
    for patch_id, label, color, marker in (
        (1, "upper", "#1766ab", "o"),
        (2, "lower", "#c65b18", "s"),
        (3, "trailing edge", "#555555", "^"),
    ):
        selected = patch == patch_id
        if not np.any(selected):
            continue
        order = np.argsort(surface["x"][selected])
        pressure_axis.plot(
            surface["x"][selected][order], surface["Cp"][selected][order],
            color=color, marker=marker, markersize=2.5, linewidth=1.1,
            label=label,
        )
    pressure_axis.invert_yaxis()
    pressure_axis.set(xlim=(-0.02, 1.02), xlabel="x / chord",
                      ylabel="$C_p$", title="Wall pressure from wall-flux owners")
    pressure_axis.grid(alpha=0.22)
    pressure_axis.legend()

    figure.suptitle(
        f"{args.title} | {result['fluid_control_volumes']:,} control volumes | "
        f"{'converged' if result.get('converged') else 'not converged'}\n"
        "Control-volume values mapped only to their true member rectangles; no interpolation",
        fontsize=15,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=200)
    plt.close(figure)

    metadata = {
        "case_output_name": args.case.name,
        "geometry_name": args.geometry.name,
        "result": result,
        "visible_member_rectangles": int(len(cells)),
        "rendering": (
            "control-volume averages mapped to original AMR member "
            "rectangles; input polygon masks the solid; no interpolation"
        ),
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
