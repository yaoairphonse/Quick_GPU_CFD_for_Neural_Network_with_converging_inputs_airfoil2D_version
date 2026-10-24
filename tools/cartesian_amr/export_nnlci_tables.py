"""Export reusable memory-mapped connectivity and point-location tables.

The current hybrid mesh is not solver-ready because prism/octree clipping and
coupling are unfinished.  This exporter deliberately writes two separate
submesh tables and records that limitation in the manifest.  Stable, explicit
prism connectivity is materialized; regular Cartesian cells remain factorized
inside octree blocks to avoid hundreds of megabytes of redundant indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


WALL_NEIGHBOR = -1
UNRESOLVED_INTERFACE_NEIGHBOR = -2
PRISM_EDGE_VERTICES = np.array(((0, 1), (1, 2), (2, 0)), dtype=np.int64)


class TableWriter:
    def __init__(self, destination):
        self.destination = destination
        self.tables = {}

    def create(self, name, dtype, shape, description):
        path = self.destination / f"{name}.npy"
        array = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape)
        )
        self.tables[name] = {
            "file": path.name,
            "dtype": np.dtype(dtype).str,
            "shape": [int(value) for value in shape],
            "description": description,
        }
        return array

    def finish(self):
        for name, record in self.tables.items():
            record["bytes"] = (self.destination / record["file"]).stat().st_size


def surface_adjacency(triangles):
    triangle_count = len(triangles)
    edge_vertices = triangles[:, PRISM_EDGE_VERTICES].reshape(-1, 2)
    edge_keys = np.sort(edge_vertices, axis=1)
    triangle_ids = np.repeat(np.arange(triangle_count, dtype=np.int32), 3)
    local_edges = np.tile(np.arange(3, dtype=np.int8), triangle_count)
    order = np.lexsort((edge_keys[:, 1], edge_keys[:, 0]))
    edge_keys = edge_keys[order]
    triangle_ids = triangle_ids[order]
    local_edges = local_edges[order]
    if len(edge_keys) % 2 or not np.array_equal(edge_keys[0::2], edge_keys[1::2]):
        raise ValueError("surface is not a closed two-manifold")
    if len(edge_keys) > 2 and np.any(
            np.all(edge_keys[1:-1] == edge_keys[:-2], axis=1) &
            np.all(edge_keys[1:-1] == edge_keys[2:], axis=1)):
        raise ValueError("surface contains a non-manifold edge")

    first_triangle = triangle_ids[0::2]
    second_triangle = triangle_ids[1::2]
    first_local = local_edges[0::2]
    second_local = local_edges[1::2]
    neighbors = np.empty((triangle_count, 3), dtype=np.int32)
    triangle_edges = np.empty((triangle_count, 3), dtype=np.int32)
    edge_ids = np.arange(len(first_triangle), dtype=np.int32)
    neighbors[first_triangle, first_local] = second_triangle
    neighbors[second_triangle, second_local] = first_triangle
    triangle_edges[first_triangle, first_local] = edge_ids
    triangle_edges[second_triangle, second_local] = edge_ids
    edge_triangles = np.column_stack((first_triangle, second_triangle))
    return edge_keys[0::2].astype(np.int32), edge_triangles, triangle_edges, neighbors


def prism_geometry(bottom, top):
    a, b, c = bottom[:, 0], bottom[:, 1], bottom[:, 2]
    aa, bb, cc = top[:, 0], top[:, 1], top[:, 2]
    determinants = np.column_stack((
        np.einsum("ij,ij->i", np.cross(b - a, c - a), aa - a),
        np.einsum("ij,ij->i", np.cross(c - b, aa - b), bb - b),
        np.einsum("ij,ij->i", np.cross(aa - c, bb - c), cc - c),
    ))
    if np.any(determinants <= 0):
        raise ValueError("an exported prism has a non-positive subtetrahedron")
    tetra_centroids = np.stack((
        (a + b + c + aa) * 0.25,
        (b + c + aa + bb) * 0.25,
        (c + aa + bb + cc) * 0.25,
    ), axis=1)
    determinant_sum = determinants.sum(axis=1)
    centroids = np.einsum(
        "ij,ijk->ik", determinants, tetra_centroids
    ) / determinant_sum[:, None]
    return determinant_sum / 6.0, centroids


def build_column_bvh(bounds, leaf_size=8):
    indices = np.arange(len(bounds), dtype=np.int32)
    centers = 0.5 * (bounds[:, :3] + bounds[:, 3:])
    node_bounds = []
    node_meta = []

    def build(first, last):
        node_id = len(node_bounds)
        selected = indices[first:last]
        box = np.concatenate((bounds[selected, :3].min(axis=0),
                              bounds[selected, 3:].max(axis=0)))
        node_bounds.append(box)
        node_meta.append([-1, -1, first, last - first])
        if last - first <= leaf_size:
            return node_id
        spread = np.ptp(centers[selected], axis=0)
        axis = int(np.argmax(spread))
        middle = (first + last) // 2
        partition = np.argpartition(
            centers[selected, axis], middle - first
        )
        indices[first:last] = selected[partition]
        left = build(first, middle)
        right = build(middle, last)
        node_meta[node_id] = [left, right, first, 0]
        return node_id

    root = build(0, len(bounds))
    return (np.asarray(node_bounds, dtype=np.float64),
            np.asarray(node_meta, dtype=np.int32), indices, root)


def write_array(writer, name, values, description):
    output = writer.create(name, values.dtype, values.shape, description)
    output[...] = values
    output.flush()
    return output


def export_prisms(plan, writer):
    points = np.asarray(plan["points"], dtype=np.float64)
    triangles = np.asarray(plan["triangles"], dtype=np.int32)
    patches = np.asarray(plan["patch"], dtype=np.int32)
    normals = np.asarray(plan["normal"], dtype=np.float64)
    scales = np.asarray(plan["column_scale"], dtype=np.float64)
    offsets = np.asarray(plan["offsets"], dtype=np.float64)
    vertex_count = len(points)
    triangle_count = len(triangles)
    layer_count = len(offsets) - 1
    node_count = (layer_count + 1) * vertex_count
    cell_count = layer_count * triangle_count
    if node_count >= np.iinfo(np.int32).max or cell_count >= np.iinfo(np.int32).max:
        raise ValueError("32-bit prism identifiers would overflow")

    edges, edge_triangles, triangle_edges, triangle_neighbors = \
        surface_adjacency(triangles)
    write_array(writer, "surface_triangles", triangles,
                "Wall-surface triangle nodes; zero-based.")
    write_array(writer, "surface_triangle_patch", patches,
                "Original boundary patch identifier per surface triangle.")
    write_array(writer, "surface_edges", edges,
                "Unique closed-surface edge endpoint nodes.")
    write_array(writer, "surface_edge_triangles", edge_triangles,
                "The two surface triangles incident on each edge.")
    write_array(writer, "surface_triangle_edges", triangle_edges,
                "Three unique edge identifiers per surface triangle.")
    write_array(writer, "surface_triangle_neighbors", triangle_neighbors,
                "Triangle across each local edge (0-1, 1-2, 2-0).")
    write_array(writer, "prism_layer_offsets", offsets,
                "Requested cumulative wall-normal offsets before local scaling.")
    write_array(writer, "prism_column_scale", scales,
                "Per-surface-vertex multiplier applied to every layer offset.")
    write_array(writer, "prism_column_normal", normals,
                "Angle-weighted extrusion direction per surface vertex.")

    nodes = writer.create(
        "prism_nodes", "<f8", (node_count, 3),
        "Expanded layer-node coordinates in layer-major order."
    )
    for layer, offset in enumerate(offsets):
        first = layer * vertex_count
        nodes[first:first + vertex_count] = (
            points + (offset * scales)[:, None] * normals
        )
    nodes.flush()

    cell_nodes = writer.create(
        "prism_cell_nodes", "<i4", (cell_count, 6),
        "Six node identifiers per prism: bottom triangle then top triangle."
    )
    cell_neighbors = writer.create(
        "prism_cell_neighbors", "<i4", (cell_count, 5),
        "Neighbors across bottom, top, and local edges 0-1, 1-2, 2-0."
    )
    cell_centroids = writer.create(
        "prism_cell_centroids", "<f8", (cell_count, 3),
        "Exact volume-weighted centroids from the audited three-tet split."
    )
    cell_volumes = writer.create(
        "prism_cell_volumes", "<f8", (cell_count,),
        "Positive cell volumes from the audited three-tet split."
    )
    column_bounds = np.empty((triangle_count, 6), dtype=np.float64)
    for layer in range(layer_count):
        cell_first = layer * triangle_count
        cell_last = cell_first + triangle_count
        bottom_ids = triangles + layer * vertex_count
        top_ids = triangles + (layer + 1) * vertex_count
        cell_nodes[cell_first:cell_last, :3] = bottom_ids
        cell_nodes[cell_first:cell_last, 3:] = top_ids
        cell_neighbors[cell_first:cell_last, 0] = (
            np.arange(triangle_count, dtype=np.int32) +
            (layer - 1) * triangle_count
            if layer else WALL_NEIGHBOR
        )
        cell_neighbors[cell_first:cell_last, 1] = (
            np.arange(triangle_count, dtype=np.int32) +
            (layer + 1) * triangle_count
            if layer + 1 < layer_count else UNRESOLVED_INTERFACE_NEIGHBOR
        )
        cell_neighbors[cell_first:cell_last, 2:] = (
            triangle_neighbors + layer * triangle_count
        )
        bottom = np.asarray(nodes[bottom_ids], dtype=np.float64)
        top = np.asarray(nodes[top_ids], dtype=np.float64)
        volumes, centroids = prism_geometry(bottom, top)
        cell_volumes[cell_first:cell_last] = volumes
        cell_centroids[cell_first:cell_last] = centroids
        if layer == 0:
            column_bounds[:, :3] = np.minimum(
                bottom.min(axis=1), top.min(axis=1)
            )
            column_bounds[:, 3:] = np.maximum(
                bottom.max(axis=1), top.max(axis=1)
            )
        else:
            column_bounds[:, :3] = np.minimum(
                column_bounds[:, :3], top.min(axis=1)
            )
            column_bounds[:, 3:] = np.maximum(
                column_bounds[:, 3:], top.max(axis=1)
            )
    for array in (cell_nodes, cell_neighbors, cell_centroids, cell_volumes):
        array.flush()

    flat_vertices = triangles.ravel()
    flat_triangles = np.repeat(
        np.arange(triangle_count, dtype=np.int32), 3
    )
    order = np.argsort(flat_vertices, kind="stable")
    sorted_vertices = flat_vertices[order]
    sorted_triangles = flat_triangles[order]
    incidence_count = np.bincount(
        sorted_vertices, minlength=vertex_count
    ).astype(np.uint64)
    base_offsets = np.empty(vertex_count + 1, dtype=np.uint64)
    base_offsets[0] = 0
    np.cumsum(incidence_count, out=base_offsets[1:])
    rank = (np.arange(len(sorted_vertices), dtype=np.uint64) -
            np.repeat(base_offsets[:-1], incidence_count.astype(np.int64)))
    node_counts = np.tile(incidence_count, layer_count + 1)
    if layer_count > 1:
        node_counts[vertex_count:layer_count * vertex_count] *= 2
    node_cell_offsets = writer.create(
        "prism_node_cell_offsets", "<u8", (node_count + 1,),
        "CSR offsets for incident prism cells of each node."
    )
    node_cell_offsets[0] = 0
    np.cumsum(node_counts, out=node_cell_offsets[1:])
    incidence_total = int(node_cell_offsets[-1])
    if incidence_total != 6 * cell_count:
        raise AssertionError("node-to-cell incidence count mismatch")
    node_cells = writer.create(
        "prism_node_cells", "<i4", (incidence_total,),
        "CSR values containing incident prism cell identifiers."
    )
    node_cells[:] = -1
    for layer in range(layer_count + 1):
        node_ids = layer * vertex_count + sorted_vertices
        target = np.asarray(node_cell_offsets[node_ids], dtype=np.uint64) + rank
        if layer > 0:
            node_cells[target] = ((layer - 1) * triangle_count +
                                  sorted_triangles)
        if layer < layer_count:
            if layer > 0:
                target = target + incidence_count[sorted_vertices]
            node_cells[target] = (layer * triangle_count + sorted_triangles)
    if np.any(node_cells < 0):
        raise AssertionError("node-to-cell CSR contains an unwritten entry")
    node_cell_offsets.flush()
    node_cells.flush()

    directed_sources = np.concatenate((edges[:, 0], edges[:, 1]))
    directed_targets = np.concatenate((edges[:, 1], edges[:, 0]))
    order = np.argsort(directed_sources, kind="stable")
    directed_sources = directed_sources[order]
    directed_targets = directed_targets[order]
    surface_neighbor_count = np.bincount(
        directed_sources, minlength=vertex_count
    ).astype(np.uint64)
    surface_neighbor_offsets = np.empty(vertex_count + 1, dtype=np.uint64)
    surface_neighbor_offsets[0] = 0
    np.cumsum(surface_neighbor_count, out=surface_neighbor_offsets[1:])
    surface_rank = (
        np.arange(len(directed_sources), dtype=np.uint64) -
        np.repeat(surface_neighbor_offsets[:-1],
                  surface_neighbor_count.astype(np.int64))
    )
    node_neighbor_count = np.tile(surface_neighbor_count, layer_count + 1)
    node_neighbor_count[:layer_count * vertex_count] += 1
    node_neighbor_count[vertex_count:] += 1
    node_neighbor_offsets = writer.create(
        "prism_node_neighbor_offsets", "<u8", (node_count + 1,),
        "CSR offsets for nodes joined by prism edges."
    )
    node_neighbor_offsets[0] = 0
    np.cumsum(node_neighbor_count, out=node_neighbor_offsets[1:])
    node_neighbors = writer.create(
        "prism_node_neighbors", "<i4", (int(node_neighbor_offsets[-1]),),
        "CSR values containing edge-adjacent prism nodes."
    )
    node_neighbors[:] = -1
    all_vertices = np.arange(vertex_count, dtype=np.int64)
    for layer in range(layer_count + 1):
        expanded_sources = layer * vertex_count + directed_sources
        target = (np.asarray(node_neighbor_offsets[expanded_sources],
                             dtype=np.uint64) + surface_rank)
        node_neighbors[target] = layer * vertex_count + directed_targets
        vertical_target = (
            np.asarray(node_neighbor_offsets[
                layer * vertex_count + all_vertices
            ], dtype=np.uint64) + surface_neighbor_count
        )
        if layer > 0:
            node_neighbors[vertical_target] = (
                (layer - 1) * vertex_count + all_vertices
            )
            vertical_target += 1
        if layer < layer_count:
            node_neighbors[vertical_target] = (
                (layer + 1) * vertex_count + all_vertices
            )
    if np.any(node_neighbors < 0):
        raise AssertionError("node-neighbor CSR contains an unwritten entry")
    node_neighbor_offsets.flush()
    node_neighbors.flush()

    bvh_bounds, bvh_meta, bvh_order, bvh_root = build_column_bvh(column_bounds)
    write_array(writer, "prism_column_bounds", column_bounds,
                "AABB of each complete wall-normal prism column.")
    write_array(writer, "prism_bvh_bounds", bvh_bounds,
                "AABB for each prism-column BVH node.")
    write_array(writer, "prism_bvh_meta", bvh_meta,
                "BVH rows: left child, right child, first permutation index, count.")
    write_array(writer, "prism_bvh_order", bvh_order,
                "Column identifiers referenced by BVH leaf ranges.")
    return {
        "vertices_per_surface_layer": vertex_count,
        "surface_triangles": triangle_count,
        "layers": layer_count,
        "nodes": node_count,
        "cells": cell_count,
        "node_neighbor_incidences": int(node_neighbor_offsets[-1]),
        "cell_type": "triangular_prism",
        "cell_id": "layer * surface_triangles + surface_triangle",
        "node_id": "layer * vertices_per_surface_layer + surface_vertex",
        "neighbor_slots": ["bottom", "top", "edge_0_1", "edge_1_2", "edge_2_0"],
        "negative_neighbors": {
            str(WALL_NEIGHBOR): "solid_wall",
            str(UNRESOLVED_INTERFACE_NEIGHBOR): "unresolved_prism_octree_interface",
        },
        "point_locator": {
            "kind": "aabb_bvh_then_three_tetrahedra_per_prism",
            "root": bvh_root,
            "tetrahedra_local_nodes": [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
        },
        "unique_face_topology": {
            "triangular_faces": "surface triangle at each layer interface",
            "quadrilateral_faces": "surface edge at each prism layer",
            "materialized": False,
            "reason": "the final global owner/neighbour table must be built after prism/octree clipping",
        },
    }


def export_octree(octree, writer):
    low = np.asarray(octree["low"], dtype=np.float64)
    size = np.asarray(octree["size"], dtype=np.float64)
    level = np.asarray(octree["level"], dtype=np.int16)
    root_low = np.asarray(octree["root_low"], dtype=np.float64)
    root_size = float(octree["root_size"])
    block_cells = int(octree["block_cells"])
    surface_candidate = np.asarray(
        octree["possible_surface_intersection"], dtype=np.uint8
    )
    surface_distance = np.asarray(
        octree["distance_lower_bound"], dtype=np.float64
    )
    maximum_level = int(level.max())
    dimension = 1 << maximum_level
    coordinates = np.rint(
        (low - root_low) * dimension / root_size
    ).astype(np.int32)
    spans = (1 << (maximum_level - level)).astype(np.int32)
    block_count = len(low)

    write_array(writer, "octree_block_low", low,
                "Minimum coordinate of each leaf block.")
    write_array(writer, "octree_block_size", size,
                "Physical edge length of each cubic leaf block.")
    write_array(writer, "octree_block_level", level,
                "Refinement level of each leaf block.")
    write_array(writer, "octree_block_fine_coordinate", coordinates,
                "Integer block origins on the maximum-level lattice.")
    write_array(writer, "octree_block_fine_span", spans,
                "Block widths on the maximum-level lattice.")
    write_array(writer, "octree_block_surface_candidate", surface_candidate,
                "One when the block may intersect the outer prism front.")
    write_array(writer, "octree_block_surface_distance_lower_bound", surface_distance,
                "Conservative lower bound on distance to the outer prism front.")

    owner = writer.create(
        "octree_finest_owner", "<i4", (dimension, dimension, dimension),
        "O(1) leaf-block locator on the maximum-level lattice."
    )
    owner[:] = -1
    for block, (coordinate, span) in enumerate(zip(coordinates, spans)):
        i, j, k = map(int, coordinate)
        width = int(span)
        region = owner[i:i + width, j:j + width, k:k + width]
        if np.any(region >= 0):
            raise ValueError("octree leaf blocks overlap")
        region[:] = block
    if np.any(owner < 0):
        raise ValueError("octree leaf blocks do not cover the root")
    owner.flush()

    neighbor_offsets = np.empty(block_count + 1, dtype=np.uint64)
    neighbor_ids = []
    neighbor_sides = []
    boundary_mask = np.zeros(block_count, dtype=np.uint8)
    for block, (coordinate, span) in enumerate(zip(coordinates, spans)):
        i, j, k = map(int, coordinate)
        width = int(span)
        neighbor_offsets[block] = len(neighbor_ids)
        faces = (
            None if i == 0 else owner[i - 1, j:j + width, k:k + width],
            None if i + width == dimension else owner[i + width, j:j + width, k:k + width],
            None if j == 0 else owner[i:i + width, j - 1, k:k + width],
            None if j + width == dimension else owner[i:i + width, j + width, k:k + width],
            None if k == 0 else owner[i:i + width, j:j + width, k - 1],
            None if k + width == dimension else owner[i:i + width, j:j + width, k + width],
        )
        for side, face in enumerate(faces):
            if face is None:
                boundary_mask[block] |= np.uint8(1 << side)
                continue
            adjacent = np.unique(face)
            adjacent = adjacent[adjacent != block]
            neighbor_ids.extend(map(int, adjacent))
            neighbor_sides.extend([side] * len(adjacent))
    neighbor_offsets[-1] = len(neighbor_ids)
    write_array(writer, "octree_block_neighbor_offsets", neighbor_offsets,
                "CSR offsets for face-adjacent leaf blocks.")
    write_array(writer, "octree_block_neighbors",
                np.asarray(neighbor_ids, dtype=np.int32),
                "CSR leaf-block neighbor identifiers; 2:1 faces may have four.")
    write_array(writer, "octree_block_neighbor_sides",
                np.asarray(neighbor_sides, dtype=np.uint8),
                "Local face for each CSR neighbor: -x,+x,-y,+y,-z,+z.")
    write_array(writer, "octree_block_boundary_mask", boundary_mask,
                "Six low bits mark leaf-block faces on the farfield root boundary.")
    return {
        "leaf_blocks": block_count,
        "block_cells_per_axis": block_cells,
        "procedural_cells": block_count * block_cells ** 3,
        "maximum_level": maximum_level,
        "root_low": root_low.tolist(),
        "root_size": root_size,
        "point_locator": {
            "kind": "dense_maximum_level_owner_then_structured_local_index",
            "lattice_dimension": dimension,
        },
        "cell_connectivity": {
            "materialized": False,
            "reason": "regular 8x8x8 block connectivity is exact and cheaper to reconstruct",
            "local_cell_id": "(k * block_cells + j) * block_cells + i",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column-plan", required=True)
    parser.add_argument("--octree-plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    writer = TableWriter(destination)
    with np.load(args.column_plan) as plan:
        prism = export_prisms(plan, writer)
    with np.load(args.octree_plan) as octree:
        octree_record = export_octree(octree, writer)
    writer.finish()
    manifest = {
        "format": "QCFD_NNLCI_MESH_TABLES",
        "version": 1,
        "mesh_workflow": "body_fitted_hybrid_3d",
        "endianness": "little",
        "index_base": 0,
        "solver_ready": False,
        "reason_not_solver_ready": (
            "solid classification, Cartesian cut cells, and the conservative "
            "prism/octree interface have not been assembled"
        ),
        "stable_identifier_policy": (
            "identifiers are deterministic for unchanged surface ordering, "
            "layer settings, and octree settings"
        ),
        "prism_submesh": prism,
        "octree_submesh": octree_record,
        "tables": writer.tables,
        "total_binary_bytes": int(sum(
            table["bytes"] for table in writer.tables.values()
        )),
        "export_seconds": time.perf_counter() - start,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
