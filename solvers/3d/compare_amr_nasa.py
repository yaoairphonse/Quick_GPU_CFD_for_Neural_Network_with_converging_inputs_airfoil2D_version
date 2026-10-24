#!/usr/bin/env python3
"""Compare procedural-AMR triangle Cp and coefficients with NASA USM3D."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree


def read_reference_pressure(path: Path, chord: float):
    coordinates = []
    cp = []
    patch = -1
    for line in path.read_text().splitlines():
        if "BC#" in line:
            patch = int(re.search(r"BC#\s*(\d+)", line).group(1))
            continue
        fields = line.strip().split(",")
        if len(fields) < 4:
            continue
        try:
            values = [float(value) for value in fields[:4]]
        except ValueError:
            continue
        coordinates.append(np.asarray(values[:3]) / chord)
        cp.append(values[3])
    return np.asarray(coordinates), np.asarray(cp)


def weighted_rmse(error, area, selected):
    return float(np.sqrt(np.sum(area[selected] * error[selected] ** 2) /
                         np.sum(area[selected])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference-directory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chord", type=float, default=275.8)
    args = parser.parse_args()

    result = json.loads((args.run / "result.json").read_text())
    with (args.reference_directory / "NASA_wbt0_M85_CFD_orig.csv").open(
            newline="") as stream:
        rows = list(csv.DictReader(stream))
    reference = next(
        {key: float(value) for key, value in row.items() if key and value}
        for row in rows if row.get("alpha") and abs(float(row["alpha"]) - 2.5) < 1e-12
    )
    reference_xyz, reference_cp = read_reference_pressure(
        args.reference_directory / "NASA_wbt0_M85_a2.5_pr_orig.csv", args.chord
    )

    with np.load(args.geometry) as geometry:
        points = np.asarray(geometry["points"])
        triangles = np.asarray(geometry["triangles"])
    vertices = points[triangles]
    centers = vertices.mean(axis=1)
    query = centers.copy()
    query[:, 1] = np.abs(query[:, 1])
    distance, nearest = cKDTree(reference_xyz).query(query)
    nasa_cp = reference_cp[nearest]
    local = np.loadtxt(
        args.run / "surface.csv", delimiter=",", skiprows=1, usecols=(0, 7, 10)
    )
    if not np.array_equal(local[:, 0].astype(np.int64), np.arange(len(triangles))):
        raise ValueError("surface triangle ordering mismatch")
    area = local[:, 1]
    local_cp = local[:, 2]
    error = local_cp - nasa_cp
    close = distance <= 0.005
    close_002 = distance <= 0.02
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    wing = ((np.abs(centers[:, 1]) >= 0.4) & (centers[:, 0] >= 3.0) &
            (centers[:, 0] <= 7.2) & (np.abs(normals[:, 2]) >= 0.2) & close)
    wing_upper = wing & (normals[:, 2] > 0)
    wing_lower = wing & (normals[:, 2] < 0)

    coefficients = {}
    for local_key, nasa_key in (("CL", "CL"), ("CD", "CD"), ("CM_y", "CM")):
        local_value = float(result[local_key])
        nasa_value = reference[nasa_key]
        coefficients[local_key] = {
            "local": local_value,
            "NASA_USM3D": nasa_value,
            "relative_error_percent": 100 * (local_value - nasa_value) / abs(nasa_value),
        }
    report = {
        "reference": "NASA USM3D computed CFD, original WBT0",
        "coefficients": coefficients,
        "surface_sampling": "nearest NASA half-model node to mirrored source-triangle centroid",
        "matched_area_fraction_within_0p005_MAC": float(area[close].sum() / area.sum()),
        "matched_area_fraction_within_0p02_MAC": float(area[close_002].sum() / area.sum()),
        "median_match_distance_MAC": float(np.median(distance)),
        "maximum_match_distance_MAC": float(distance.max()),
        "whole_surface_area_weighted_Cp_RMSE": weighted_rmse(error, area, close),
        "whole_surface_nearest_Cp_RMSE_without_distance_filter": weighted_rmse(
            error, area, np.ones(len(area), dtype=bool)
        ),
        "whole_surface_Cp_RMSE_within_0p02_MAC": weighted_rmse(
            error, area, close_002
        ),
        "wing_area_weighted_Cp_RMSE": weighted_rmse(error, area, wing),
        "wing_upper_Cp_RMSE": weighted_rmse(error, area, wing_upper),
        "wing_lower_Cp_RMSE": weighted_rmse(error, area, wing_lower),
        "matched_triangles": int(np.count_nonzero(close)),
        "wing_triangles": int(np.count_nonzero(wing)),
        "converged": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
