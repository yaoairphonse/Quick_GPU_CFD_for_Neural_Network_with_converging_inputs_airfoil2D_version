"""Validate every table in a QCFDNC1 connectivity companion file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


HEADER = struct.Struct("<8sII10Q")
BOUNDARY_DTYPE = np.dtype([
    ("node", "<i4", (3,)),
    ("bc", "<i4"),
    ("patch", "<i4"),
])


def validate(path: Path) -> dict:
    with path.open("rb") as stream:
        values = HEADER.unpack(stream.read(HEADER.size))
    (magic, version, endian, point_count, cell_count, boundary_count,
     incidence_count, points_offset, cell_nodes_offset, node_offsets_offset,
     node_cells_offset, boundary_offset, file_bytes) = values
    if magic != b"QCFDNC1\0" or version != 1 or endian != 0x01020304:
        raise ValueError("unsupported connectivity header")

    expected_offsets = [
        HEADER.size,
        points_offset + point_count * 3 * 8,
        cell_nodes_offset + cell_count * 4 * 4,
        node_offsets_offset + (point_count + 1) * 8,
        node_cells_offset + incidence_count * 4,
        boundary_offset + boundary_count * BOUNDARY_DTYPE.itemsize,
    ]
    actual_offsets = [points_offset, cell_nodes_offset, node_offsets_offset,
                      node_cells_offset, boundary_offset, file_bytes]
    if actual_offsets != expected_offsets or path.stat().st_size != file_bytes:
        raise ValueError("non-contiguous offsets or incorrect file length")
    if incidence_count != 4 * cell_count:
        raise ValueError("a tetrahedral file must have four incidences per cell")

    points = np.memmap(path, dtype="<f8", mode="r", offset=points_offset,
                       shape=(point_count, 3))
    cell_nodes = np.memmap(path, dtype="<i4", mode="r",
                           offset=cell_nodes_offset, shape=(cell_count, 4))
    node_offsets = np.memmap(path, dtype="<u8", mode="r",
                             offset=node_offsets_offset,
                             shape=(point_count + 1,))
    node_cells = np.memmap(path, dtype="<i4", mode="r",
                           offset=node_cells_offset, shape=(incidence_count,))
    boundary = np.memmap(path, dtype=BOUNDARY_DTYPE, mode="r",
                         offset=boundary_offset, shape=(boundary_count,))

    if not np.isfinite(points).all():
        raise ValueError("non-finite point coordinates")
    if cell_nodes.min() < 0 or cell_nodes.max() >= point_count:
        raise ValueError("cell-to-node index outside the point table")
    if node_offsets[0] != 0 or node_offsets[-1] != incidence_count:
        raise ValueError("invalid node-to-cell CSR endpoints")
    if np.any(node_offsets[1:] < node_offsets[:-1]):
        raise ValueError("node-to-cell CSR offsets are not monotone")
    if node_cells.min() < 0 or node_cells.max() >= cell_count:
        raise ValueError("node-to-cell index outside the cell table")
    if boundary["node"].min() < 0 or boundary["node"].max() >= point_count:
        raise ValueError("boundary triangle index outside the point table")
    if not np.isin(boundary["bc"], [1, 2, 3]).all():
        raise ValueError("unknown native boundary-condition code")
    if np.any(boundary["patch"] < 1):
        raise ValueError("boundary patch identifiers must be positive")

    cell_occurrences = np.zeros(cell_count, dtype=np.uint8)
    chunk = 4_000_000
    for first in range(0, incidence_count, chunk):
        last = min(first + chunk, incidence_count)
        cell_occurrences += np.bincount(
            node_cells[first:last], minlength=cell_count).astype(np.uint8)
    if not np.all(cell_occurrences == 4):
        raise ValueError("each tetrahedron must occur in four node rows")

    # Exact reverse-table validation in bounded-memory node chunks.
    node_chunk = 100_000
    for first_node in range(0, point_count, node_chunk):
        last_node = min(first_node + node_chunk, point_count)
        first_incidence = int(node_offsets[first_node])
        last_incidence = int(node_offsets[last_node])
        degree = np.diff(node_offsets[first_node:last_node + 1]).astype(np.int64)
        expanded_nodes = np.repeat(
            np.arange(first_node, last_node, dtype=np.int32), degree)
        incident_cells = node_cells[first_incidence:last_incidence]
        if not np.any(cell_nodes[incident_cells] == expanded_nodes[:, None],
                      axis=1).all():
            raise ValueError("node-to-cell CSR is not the reverse cell table")

    for first in range(0, cell_count, chunk):
        block = np.sort(cell_nodes[first:first + chunk], axis=1)
        if np.any(block[:, 1:] == block[:, :-1]):
            raise ValueError("a tetrahedron contains a repeated point")

    degree = np.diff(node_offsets)
    return {
        "format": "QCFDNC1",
        "version": version,
        "file_bytes": file_bytes,
        "points": point_count,
        "cells": cell_count,
        "cell_node_incidences": incidence_count,
        "boundary_triangles": boundary_count,
        "boundary_zones": int(np.unique(boundary["patch"]).size),
        "node_degree_min": int(degree.min()),
        "node_degree_max": int(degree.max()),
        "node_degree_mean": float(degree.mean()),
        "full_reverse_table_check": True,
        "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("connectivity", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    result = validate(arguments.connectivity)
    text = json.dumps(result, indent=2) + "\n"
    if arguments.report:
        arguments.report.write_text(text)
    print(text, end="")
