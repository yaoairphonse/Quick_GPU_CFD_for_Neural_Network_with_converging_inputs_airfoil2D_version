"""Validate stored NNLCI mesh connectivity and point-location tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from nnlci_table_reader import MeshTables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    start = time.perf_counter()
    mesh = MeshTables(args.mesh)
    manifest = mesh.manifest
    rng = np.random.default_rng(20261023)

    for name in manifest["tables"]:
        mesh.table(name)

    neighbors = mesh.table("prism_cell_neighbors")
    cell_count = len(neighbors)
    chunk_size = 250_000
    reciprocal_failures = 0
    for first in range(0, cell_count, chunk_size):
        last = min(cell_count, first + chunk_size)
        rows = neighbors[first:last]
        sources = np.repeat(
            np.arange(first, last, dtype=np.int32), rows.shape[1]
        )
        adjacent = rows.ravel()
        selected = adjacent >= 0
        adjacent = adjacent[selected]
        sources = sources[selected]
        reciprocal_failures += int(np.count_nonzero(~np.any(
            neighbors[adjacent] == sources[:, None], axis=1
        )))
    if reciprocal_failures:
        raise ValueError(f"{reciprocal_failures} prism neighbors are not reciprocal")

    node_offsets = mesh.table("prism_node_cell_offsets")
    node_cells = mesh.table("prism_node_cells")
    if (node_offsets[0] != 0 or node_offsets[-1] != len(node_cells) or
            np.any(node_offsets[1:] < node_offsets[:-1]) or
            np.any(node_cells < 0) or np.any(node_cells >= cell_count)):
        raise ValueError("invalid node-to-cell CSR")
    if len(node_cells) != 6 * cell_count:
        raise ValueError("node-to-cell incidence total is not six per prism")
    node_neighbor_offsets = mesh.table("prism_node_neighbor_offsets")
    node_neighbors = mesh.table("prism_node_neighbors")
    node_count = manifest["prism_submesh"]["nodes"]
    if (node_neighbor_offsets[0] != 0 or
            node_neighbor_offsets[-1] != len(node_neighbors) or
            np.any(node_neighbor_offsets[1:] < node_neighbor_offsets[:-1]) or
            np.any(node_neighbors < 0) or np.any(node_neighbors >= node_count)):
        raise ValueError("invalid node-neighbor CSR")
    edge_sample_count = min(args.samples * 16, len(node_neighbors))
    sampled_edge_entries = rng.choice(
        len(node_neighbors), edge_sample_count, replace=False
    )
    source_nodes = np.searchsorted(
        node_neighbor_offsets, sampled_edge_entries, side="right"
    ) - 1
    node_neighbor_reciprocal_failures = 0
    for source, entry in zip(source_nodes, sampled_edge_entries):
        target = int(node_neighbors[entry])
        if int(source) not in mesh.prism_node_neighbors(target):
            node_neighbor_reciprocal_failures += 1
    if node_neighbor_reciprocal_failures:
        raise ValueError(
            f"{node_neighbor_reciprocal_failures} sampled node neighbors are not reciprocal"
        )
    volumes = mesh.table("prism_cell_volumes")
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("prism volumes are not finite and positive")

    sample_count = min(args.samples, cell_count)
    sampled_cells = rng.choice(cell_count, sample_count, replace=False)
    centroids = mesh.table("prism_cell_centroids")
    point_location_failures = 0
    for cell in sampled_cells:
        location = mesh.locate_prism(centroids[cell])
        if location is None or location["cell"] != cell:
            point_location_failures += 1
    if point_location_failures:
        raise ValueError(
            f"{point_location_failures} sampled prism centroids were not located"
        )

    owner = mesh.table("octree_finest_owner")
    block_count = manifest["octree_submesh"]["leaf_blocks"]
    if np.any(owner < 0) or np.any(owner >= block_count):
        raise ValueError("invalid octree owner locator")
    block_offsets = mesh.table("octree_block_neighbor_offsets")
    block_neighbors = mesh.table("octree_block_neighbors")
    if (block_offsets[0] != 0 or block_offsets[-1] != len(block_neighbors) or
            np.any(block_offsets[1:] < block_offsets[:-1])):
        raise ValueError("invalid octree neighbor CSR")
    neighbor_sets = [set(map(int, block_neighbors[block_offsets[i]:
                                                    block_offsets[i + 1]]))
                     for i in range(block_count)]
    block_reciprocal_failures = sum(
        neighbor not in range(block_count) or block not in neighbor_sets[neighbor]
        for block, values in enumerate(neighbor_sets) for neighbor in values
    )
    if block_reciprocal_failures:
        raise ValueError(
            f"{block_reciprocal_failures} octree neighbors are not reciprocal"
        )

    report = {
        "status": "NNLCI_TABLES_VALID",
        "solver_ready": manifest["solver_ready"],
        "prism_cells": cell_count,
        "prism_nodes": manifest["prism_submesh"]["nodes"],
        "node_cell_incidences": len(node_cells),
        "node_neighbor_incidences": len(node_neighbors),
        "sampled_node_neighbor_entries": edge_sample_count,
        "node_neighbor_reciprocal_failures": node_neighbor_reciprocal_failures,
        "prism_neighbor_reciprocal_failures": reciprocal_failures,
        "sampled_prism_point_locations": sample_count,
        "prism_point_location_failures": point_location_failures,
        "octree_blocks": block_count,
        "octree_neighbor_reciprocal_failures": block_reciprocal_failures,
        "binary_bytes": manifest["total_binary_bytes"],
        "validation_seconds": time.perf_counter() - start,
    }
    destination = Path(args.mesh) / "validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
