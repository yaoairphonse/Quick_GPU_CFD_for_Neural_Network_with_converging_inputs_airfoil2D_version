"""Build a fast, compact boundary-layer column plan from a closed triangle surface.

This is the first stage of the automatic hybrid mesher.  It validates and orients
the surface, detects sharp features, constructs outward wall-normal columns,
limits their height using curvature and opposing-surface proximity, and audits
every proposed triangular prism.  The compact column representation can be
expanded without repeating geometry analysis.  It is not yet an off-body volume
mesher; the outer prism surface must later be connected to a tetrahedral or
Cartesian/octree farfield.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import struct
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_binary_stl(path: Path):
    """Load a binary STL and weld exactly coincident float32 vertices."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(80)
        count = struct.unpack("<I", stream.read(4))[0]
    if size != 84 + 50 * count:
        raise ValueError("only binary STL input is supported")
    record_type = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
         ("attribute", "<u2")]
    )
    records = np.memmap(path, dtype=record_type, mode="r", offset=84,
                        shape=(count,))
    raw_points = np.asarray(records["vertices"]).reshape(-1, 3)
    points, inverse = np.unique(raw_points, axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    patches = np.asarray(records["attribute"], dtype=np.int32)
    return points.astype(np.float64), triangles.astype(np.int64), patches


def load_surface(path: Path):
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        points = np.asarray(data["points"], dtype=np.float64)
        triangles = np.asarray(data["triangles"], dtype=np.int64)
        patches = np.asarray(
            data["patch"] if "patch" in data.files
            else np.zeros(len(triangles)), dtype=np.int32
        )
        return points, triangles, patches
    if path.suffix.lower() == ".stl":
        return load_binary_stl(path)
    raise ValueError("surface must be a connected NPZ file or binary STL")


def surface_topology(points, triangles, patches, feature_angle_degrees,
                     patch_features):
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (M,3)")
    if len(patches) != len(triangles):
        raise ValueError("patch count does not match triangle count")
    if triangles.min(initial=0) < 0 or triangles.max(initial=-1) >= len(points):
        raise ValueError("triangle vertex index is out of range")

    vertices = points[triangles]
    area_vectors = 0.5 * np.cross(vertices[:, 1] - vertices[:, 0],
                                  vertices[:, 2] - vertices[:, 0])
    areas = np.linalg.norm(area_vectors, axis=1)
    scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    degenerate = areas <= max(scale * scale * 1e-16, 1e-30)
    if np.any(degenerate):
        raise ValueError(f"surface has {np.count_nonzero(degenerate)} degenerate triangles")

    signed_volume = np.einsum(
        "ij,ij->i", vertices[:, 0],
        np.cross(vertices[:, 1], vertices[:, 2])
    ).sum() / 6.0
    if signed_volume < 0:
        triangles = triangles[:, ::-1].copy()
        vertices = points[triangles]
        area_vectors = -area_vectors
        signed_volume = -signed_volume
    if signed_volume <= scale ** 3 * 1e-15:
        raise ValueError("surface has zero or numerically negligible enclosed volume")

    directed = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                               triangles[:, [2, 0]]))
    face_ids = np.tile(np.arange(len(triangles), dtype=np.int64), 3)
    direction = directed[:, 0] < directed[:, 1]
    edges = np.sort(directed, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_faces = face_ids[order]
    sorted_direction = direction[order]
    unique_edges, starts, counts = np.unique(
        sorted_edges, axis=0, return_index=True, return_counts=True
    )
    open_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    if open_edges or nonmanifold_edges:
        raise ValueError(
            f"surface is not closed manifold: {open_edges} open and "
            f"{nonmanifold_edges} nonmanifold edges"
        )
    if not np.all(counts == 2):
        raise ValueError("surface edge incidence is not exactly two")
    first = starts
    second = starts + 1
    orientation_errors = int(np.count_nonzero(
        sorted_direction[first] == sorted_direction[second]
    ))
    if orientation_errors:
        raise ValueError(
            f"surface has {orientation_errors} inconsistently oriented edge pairs"
        )

    normals = area_vectors / areas[:, None]
    edge_face_a = sorted_faces[first]
    edge_face_b = sorted_faces[second]
    cos_dihedral = np.einsum(
        "ij,ij->i", normals[edge_face_a], normals[edge_face_b]
    ).clip(-1.0, 1.0)
    dihedral = np.arccos(cos_dihedral)
    feature = dihedral >= np.deg2rad(feature_angle_degrees)
    if patch_features:
        feature |= patches[edge_face_a] != patches[edge_face_b]

    return {
        "triangles": triangles,
        "vertices": vertices,
        "area_vectors": area_vectors,
        "areas": areas,
        "normals": normals,
        "signed_volume": float(signed_volume),
        "edges": unique_edges,
        "edge_face_a": edge_face_a,
        "edge_face_b": edge_face_b,
        "dihedral": dihedral,
        "feature": feature,
        "orientation_errors": orientation_errors,
    }


def vertex_geometry(points, topology, proximity_neighbors,
                    curvature_fraction, proximity_fraction):
    triangles = topology["triangles"]
    edges = topology["edges"]
    area_vectors = topology["area_vectors"]

    # Corner-angle weighting prevents a large neighboring triangle from forcing
    # the extrusion direction behind a much smaller incident triangle.  That
    # failure mode inverted hundreds of first-layer CRM prisms with an
    # area-weighted normal even at a two-microchord height.
    face_normal = topology["normals"]
    face_vertices = points[triangles]
    vertex_normal = np.zeros_like(points)
    for corner in range(3):
        edge_a = face_vertices[:, (corner + 1) % 3] - face_vertices[:, corner]
        edge_b = face_vertices[:, (corner + 2) % 3] - face_vertices[:, corner]
        edge_a /= np.linalg.norm(edge_a, axis=1)[:, None]
        edge_b /= np.linalg.norm(edge_b, axis=1)[:, None]
        corner_angle = np.arccos(
            np.einsum("ij,ij->i", edge_a, edge_b).clip(-1.0, 1.0)
        )
        np.add.at(vertex_normal, triangles[:, corner],
                  face_normal * corner_angle[:, None])
    normal_length = np.linalg.norm(vertex_normal, axis=1)
    weak_normals = normal_length <= 1e-14 * normal_length.max()
    if np.any(weak_normals):
        raise ValueError(
            f"{np.count_nonzero(weak_normals)} vertices have an undefined averaged normal"
        )
    vertex_normal /= normal_length[:, None]

    edge_vector = points[edges[:, 1]] - points[edges[:, 0]]
    edge_length = np.linalg.norm(edge_vector, axis=1)
    local_edge = np.full(len(points), np.inf)
    np.minimum.at(local_edge, edges[:, 0], edge_length)
    np.minimum.at(local_edge, edges[:, 1], edge_length)

    curvature = topology["dihedral"] / np.maximum(edge_length, 1e-30)
    maximum_curvature = np.zeros(len(points))
    np.maximum.at(maximum_curvature, edges[:, 0], curvature)
    np.maximum.at(maximum_curvature, edges[:, 1], curvature)
    curvature_cap = curvature_fraction / np.maximum(maximum_curvature, 1e-14)

    # Opposing normals identify nearby surfaces such as thin trailing edges or
    # narrow gaps.  This is deliberately a conservative vertex-level screen;
    # the future volume stage will add exact swept-triangle collision tests.
    tree = cKDTree(points)
    k = min(max(2, proximity_neighbors), len(points))
    distances, neighbors = tree.query(points, k=k, workers=-1)
    if k == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    normal_dot = np.einsum(
        "ijk,ik->ij", vertex_normal[neighbors], vertex_normal
    )
    candidate = (neighbors != np.arange(len(points))[:, None]) & (normal_dot < -0.2)
    opposing_distance = np.min(np.where(candidate, distances, np.inf), axis=1)
    proximity_cap = proximity_fraction * opposing_distance

    return {
        "normal": vertex_normal,
        "local_edge": local_edge,
        "maximum_curvature": maximum_curvature,
        "curvature_cap": curvature_cap,
        "opposing_distance": opposing_distance,
        "proximity_cap": proximity_cap,
    }


def cumulative_offsets(first_height, growth, layers):
    heights = first_height * growth ** np.arange(layers, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(heights)))


def smooth_column_scale(scale, edges, maximum_ratio, iterations):
    for _ in range(iterations):
        previous = scale.copy()
        np.minimum.at(scale, edges[:, 0], maximum_ratio * previous[edges[:, 1]])
        np.minimum.at(scale, edges[:, 1], maximum_ratio * previous[edges[:, 0]])
    return scale


def prism_determinants(bottom, top):
    a, b, c = bottom[:, 0], bottom[:, 1], bottom[:, 2]
    aa, bb, cc = top[:, 0], top[:, 1], top[:, 2]
    d0 = np.einsum("ij,ij->i", np.cross(b - a, c - a), aa - a)
    d1 = np.einsum("ij,ij->i", np.cross(c - b, aa - b), bb - b)
    d2 = np.einsum("ij,ij->i", np.cross(aa - c, bb - c), cc - c)
    return np.column_stack((d0, d1, d2))


def audit_columns(points, topology, geometry, offsets, column_scale,
                  repair_passes, edges, scale_ratio, smoothing_iterations):
    triangles = topology["triangles"]
    normal = geometry["normal"]
    repaired_triangles = 0
    passes_used = 0

    def layer_points(offset):
        return points + (offset * column_scale)[:, None] * normal

    for repair_pass in range(repair_passes + 1):
        invalid_vertices = np.zeros(len(points), dtype=bool)
        invalid_total = 0
        for layer in range(len(offsets) - 1):
            bottom = layer_points(offsets[layer])[triangles]
            top = layer_points(offsets[layer + 1])[triangles]
            determinants = prism_determinants(bottom, top)
            bad = np.any(determinants <= 1e-30, axis=1)
            invalid_total += int(np.count_nonzero(bad))
            if np.any(bad):
                invalid_vertices[triangles[bad].ravel()] = True
        if invalid_total == 0:
            break
        if repair_pass == repair_passes:
            raise ValueError(
                f"{invalid_total} proposed prisms remain inverted after "
                f"{repair_passes} repair passes"
            )
        repaired_triangles += invalid_total
        column_scale[invalid_vertices] *= 0.5
        smooth_column_scale(column_scale, edges, scale_ratio,
                            smoothing_iterations)
        passes_used += 1

    minimum_determinant = np.inf
    maximum_determinant = 0.0
    total_prism_volume = 0.0
    minimum_alignment = 1.0
    for layer in range(len(offsets) - 1):
        bottom_points = layer_points(offsets[layer])
        top_points = layer_points(offsets[layer + 1])
        bottom = bottom_points[triangles]
        top = top_points[triangles]
        determinants = prism_determinants(bottom, top)
        minimum_determinant = min(minimum_determinant, float(determinants.min()))
        maximum_determinant = max(maximum_determinant, float(determinants.max()))
        total_prism_volume += float(determinants.sum() / 6.0)
        displacement = top.mean(axis=1) - bottom.mean(axis=1)
        displacement /= np.maximum(np.linalg.norm(displacement, axis=1)[:, None],
                                   1e-300)
        alignment = np.einsum("ij,ij->i", displacement, topology["normals"])
        minimum_alignment = min(minimum_alignment, float(alignment.min()))

    return {
        "repair_passes_used": passes_used,
        "prisms_repaired_across_passes": repaired_triangles,
        "minimum_subtet_determinant": minimum_determinant,
        "maximum_subtet_determinant": maximum_determinant,
        "total_prism_volume": total_prism_volume,
        "minimum_face_normal_alignment": minimum_alignment,
    }


def write_binary_ply(path, points, triangles):
    points32 = np.asarray(points, dtype="<f4")
    triangles32 = np.asarray(triangles, dtype="<i4")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points32)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(triangles32)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    face_type = np.dtype([("count", "u1"), ("vertices", "<i4", (3,))])
    faces = np.empty(len(triangles32), dtype=face_type)
    faces["count"] = 3
    faces["vertices"] = triangles32
    with path.open("wb") as stream:
        stream.write(header)
        points32.tofile(stream)
        faces.tofile(stream)


def write_preview_plot(path, points, triangles, column_scale, requested_total,
                       elapsed_seconds):
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = points[triangles].mean(axis=1)
    face_scale = column_scale[triangles].mean(axis=1)
    stride = max(1, len(triangles) // 60000)
    sample = np.arange(0, len(triangles), stride)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2),
                               layout="constrained")
    for axis, horizontal, vertical, title in (
            (axes[0], 0, 1, "Top view: column-height scale"),
            (axes[1], 0, 2, "Side view: column-height scale")):
        image = axis.scatter(centers[sample, horizontal],
                             centers[sample, vertical],
                             c=face_scale[sample], s=1.0, cmap="viridis",
                             vmin=0, vmax=1, rasterized=True)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x / c")
        axis.set_ylabel(("y" if vertical == 1 else "z") + " / c")
        axis.set_title(title)
    figure.colorbar(image, ax=axes[:2], shrink=.82, pad=.025,
                    label="Local thickness / requested thickness")
    axes[2].hist(column_scale, bins=np.linspace(0, 1, 51),
                 color="#267e9b", edgecolor="none")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Column-height scale")
    axes[2].set_ylabel("Vertices")
    axes[2].set_title("Collision/curvature limiting")
    axes[2].text(.03, .97,
                 f"Requested total: {requested_total:.4g} c\n"
                 f"Pipeline: {elapsed_seconds:.2f} s",
                 transform=axes[2].transAxes, va="top")
    figure.suptitle("Automatic CRM boundary-layer column plan")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def collision_check(executable, surface_path, pair_path):
    result = subprocess.run(
        [str(executable), str(surface_path), str(pair_path)],
        text=True, capture_output=True
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stdout + result.stderr)
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    with pair_path.open("rb") as stream:
        count = int(np.fromfile(stream, dtype="<u8", count=1)[0])
        pairs = np.fromfile(stream, dtype="<i4").reshape(-1, 2)
    if count != len(pairs) or count != summary["collisions"]:
        raise RuntimeError("collision checker output count mismatch")
    return summary, pairs


def repair_global_collisions(destination, executable, points, triangles,
                             normal, requested_total, column_scale, edges,
                             scale_ratio, smoothing_iterations, passes):
    wall_path = destination / "_wall_collision_check.ply"
    pair_path = destination / "_collision_pairs.bin"
    write_binary_ply(wall_path, points, triangles)
    wall_summary, wall_pairs = collision_check(executable, wall_path, pair_path)
    wall_path.unlink()
    pair_path.unlink()
    if len(wall_pairs):
        raise ValueError(
            f"input wall surface has {len(wall_pairs)} non-adjacent "
            "triangle intersections"
        )

    history = []
    preview_path = destination / "_outer_collision_check.ply"
    for repair_pass in range(passes + 1):
        outer = points + (requested_total * column_scale)[:, None] * normal
        write_binary_ply(preview_path, outer, triangles)
        summary, pairs = collision_check(executable, preview_path, pair_path)
        history.append(summary)
        if not len(pairs):
            preview_path.unlink()
            pair_path.unlink()
            return {"wall": wall_summary, "outer_passes": history,
                    "repair_passes_used": repair_pass}
        if repair_pass == passes:
            raise ValueError(
                f"outer layer has {len(pairs)} triangle intersections after "
                f"{passes} global repair passes"
            )
        affected = np.unique(triangles[pairs.ravel()].ravel())
        column_scale[affected] *= 0.5
        smooth_column_scale(column_scale, edges, scale_ratio,
                            smoothing_iterations)
    raise AssertionError("unreachable collision repair state")


def materialize_shell(destination, points, triangles, normal, offsets,
                      column_scale):
    layer_points = np.lib.format.open_memmap(
        destination / "layer_points.npy", mode="w+", dtype="<f8",
        shape=(len(offsets), len(points), 3)
    )
    for layer, offset in enumerate(offsets):
        layer_points[layer] = points + (offset * column_scale)[:, None] * normal
    layer_points.flush()

    prisms = np.lib.format.open_memmap(
        destination / "prisms.npy", mode="w+", dtype="<i4",
        shape=((len(offsets) - 1) * len(triangles), 6)
    )
    npoint = len(points)
    ntriangle = len(triangles)
    for layer in range(len(offsets) - 1):
        block = prisms[layer * ntriangle:(layer + 1) * ntriangle]
        block[:, :3] = triangles + layer * npoint
        block[:, 3:] = triangles + (layer + 1) * npoint
    prisms.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--first-height", type=float, default=2e-6)
    parser.add_argument("--growth", type=float, default=1.25)
    parser.add_argument("--feature-angle", type=float, default=35.0)
    parser.add_argument("--patch-features", action="store_true")
    parser.add_argument("--curvature-fraction", type=float, default=0.30)
    parser.add_argument("--proximity-fraction", type=float, default=0.35)
    parser.add_argument("--proximity-neighbors", type=int, default=32)
    parser.add_argument("--scale-ratio", type=float, default=1.30)
    parser.add_argument("--smoothing-iterations", type=int, default=8)
    parser.add_argument("--repair-passes", type=int, default=8)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--collision-checker")
    parser.add_argument("--collision-repair-passes", type=int, default=8)
    args = parser.parse_args()
    if (args.layers < 1 or args.first_height <= 0 or args.growth < 1 or
            not 0 < args.curvature_fraction <= 1 or
            not 0 < args.proximity_fraction <= 1 or args.scale_ratio < 1):
        parser.error("invalid layer or limiter settings")

    start = time.perf_counter()
    stage_times = {}
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    points, triangles, patches = load_surface(Path(args.surface))
    stage_times["read_surface"] = time.perf_counter() - start

    checkpoint = time.perf_counter()
    topology = surface_topology(
        points, triangles, patches, args.feature_angle, args.patch_features
    )
    triangles = topology["triangles"]
    stage_times["topology_and_features"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    geometry = vertex_geometry(
        points, topology, args.proximity_neighbors,
        args.curvature_fraction, args.proximity_fraction
    )
    stage_times["normals_curvature_and_proximity"] = time.perf_counter() - checkpoint

    offsets = cumulative_offsets(args.first_height, args.growth, args.layers)
    requested_total = float(offsets[-1])
    cap = np.minimum(geometry["curvature_cap"], geometry["proximity_cap"])
    column_scale = np.minimum(1.0, cap / requested_total)
    column_scale[~np.isfinite(column_scale)] = 1.0
    smooth_column_scale(column_scale, topology["edges"], args.scale_ratio,
                        args.smoothing_iterations)

    checkpoint = time.perf_counter()
    prism_audit = audit_columns(
        points, topology, geometry, offsets, column_scale,
        args.repair_passes, topology["edges"], args.scale_ratio,
        args.smoothing_iterations
    )
    stage_times["prism_audit_and_repair"] = time.perf_counter() - checkpoint

    collision_audit = None
    if args.collision_checker:
        checkpoint = time.perf_counter()
        collision_audit = repair_global_collisions(
            destination, Path(args.collision_checker), points, triangles,
            geometry["normal"], requested_total, column_scale,
            topology["edges"], args.scale_ratio, args.smoothing_iterations,
            args.collision_repair_passes
        )
        # Global repair only shortens columns.  Re-audit all final prisms so the
        # report describes the arrays that are actually written.
        prism_audit = audit_columns(
            points, topology, geometry, offsets, column_scale,
            args.repair_passes, topology["edges"], args.scale_ratio,
            args.smoothing_iterations
        )
        stage_times["global_collision_audit"] = time.perf_counter() - checkpoint

    outer_points = points + (requested_total * column_scale)[:, None] * geometry["normal"]
    np.savez_compressed(
        destination / "column_plan.npz",
        points=points.astype(np.float64), triangles=triangles.astype(np.int32),
        patch=patches.astype(np.int32), normal=geometry["normal"].astype(np.float64),
        column_scale=column_scale.astype(np.float64), offsets=offsets
    )
    write_binary_ply(destination / "outer_layer_preview.ply", outer_points,
                     triangles)
    if args.materialize:
        checkpoint = time.perf_counter()
        materialize_shell(destination, points, triangles, geometry["normal"],
                          offsets, column_scale)
        stage_times["materialize_shell"] = time.perf_counter() - checkpoint

    if not args.no_plot:
        checkpoint = time.perf_counter()
        write_preview_plot(destination / "column_plan.png", points, triangles,
                           column_scale, requested_total,
                           time.perf_counter() - start)
        stage_times["preview_plot"] = time.perf_counter() - checkpoint

    feature_vertices = np.zeros(len(points), dtype=bool)
    feature_vertices[topology["edges"][topology["feature"]].ravel()] = True
    finite_opposing = geometry["opposing_distance"][
        np.isfinite(geometry["opposing_distance"])
    ]
    report = {
        "status": ("BOUNDARY_LAYER_PLAN_VALID" if collision_audit
                   else "BOUNDARY_LAYER_PLAN_LOCAL_AUDIT_ONLY"),
        "scope": (
            "Validated wall-normal prism columns; wall and final outer "
            "surfaces receive an exact non-adjacent triangle collision audit "
            "when a checker is supplied. Continuous swept-front collision, "
            "off-body cells and conservative coupling remain required"
        ),
        "input": str(Path(args.surface).resolve()),
        "vertices": len(points),
        "surface_triangles": len(triangles),
        "surface_edges": len(topology["edges"]),
        "patches": int(len(np.unique(patches))),
        "signed_enclosed_volume": topology["signed_volume"],
        "feature_angle_degrees": args.feature_angle,
        "feature_edges": int(np.count_nonzero(topology["feature"])),
        "feature_vertices": int(np.count_nonzero(feature_vertices)),
        "surface_edge_length_quantiles": dict(zip(
            ["min", "p01", "p50", "p99", "max"],
            map(float, np.quantile(geometry["local_edge"], [0, .01, .5, .99, 1]))
        )),
        "requested_layers": args.layers,
        "requested_first_height": args.first_height,
        "growth_ratio": args.growth,
        "requested_total_thickness": requested_total,
        "column_scale_quantiles": dict(zip(
            ["min", "p01", "p50", "p99", "max"],
            map(float, np.quantile(column_scale, [0, .01, .5, .99, 1]))
        )),
        "columns_limited_percent": float(100 * np.mean(column_scale < .999999)),
        "opposing_surface_distance_min": (
            float(finite_opposing.min()) if len(finite_opposing) else None
        ),
        "expanded_boundary_layer_points": int((args.layers + 1) * len(points)),
        "expanded_prism_cells": int(args.layers * len(triangles)),
        "materialized": bool(args.materialize),
        "prism_audit": prism_audit,
        "global_collision_audit": collision_audit,
        "stage_seconds": stage_times,
        "total_seconds": time.perf_counter() - start,
    }
    report_text = json.dumps(report, indent=2) + "\n"
    (destination / "report.json").write_text(report_text)
    (destination / "boundary_layer_report.json").write_text(report_text)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
