"""Memory-mapped reader and point locator for QCFD NNLCI mesh tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PRISM_TETRAHEDRA = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))


def tetrahedron_weights(point, vertices, tolerance=2e-10):
    matrix = np.column_stack((vertices[1] - vertices[0],
                              vertices[2] - vertices[0],
                              vertices[3] - vertices[0]))
    try:
        coordinates = np.linalg.solve(matrix, point - vertices[0])
    except np.linalg.LinAlgError:
        return None
    weights = np.array((1.0 - coordinates.sum(), *coordinates))
    if np.all(weights >= -tolerance) and np.all(weights <= 1.0 + tolerance):
        return weights
    return None


class MeshTables:
    def __init__(self, directory, interface_directory=None):
        self.directory = Path(directory)
        self.manifest = json.loads(
            (self.directory / "manifest.json").read_text()
        )
        if self.manifest.get("format") != "QCFD_NNLCI_MESH_TABLES":
            raise ValueError("unsupported mesh-table format")
        self._arrays = {}
        default_interface = self.directory.parent / "interface_plan"
        self.interface_directory = (
            Path(interface_directory) if interface_directory else default_interface
        )
        interface_manifest = self.interface_directory / "manifest.json"
        self.interface_manifest = (
            json.loads(interface_manifest.read_text())
            if interface_manifest.is_file() else None
        )
        self._interface_arrays = {}

    def table(self, name):
        if name not in self._arrays:
            record = self.manifest["tables"][name]
            array = np.load(
                self.directory / record["file"], mmap_mode="r",
                allow_pickle=False
            )
            if list(array.shape) != record["shape"] or array.dtype.str != record["dtype"]:
                raise ValueError(f"table metadata mismatch: {name}")
            self._arrays[name] = array
        return self._arrays[name]

    def interface_table(self, name):
        if self.interface_manifest is None:
            raise RuntimeError("no interface plan is available")
        if name not in self._interface_arrays:
            record = self.interface_manifest["tables"][name]
            self._interface_arrays[name] = np.load(
                self.interface_directory / record["file"], mmap_mode="r",
                allow_pickle=False
            )
        return self._interface_arrays[name]

    def prism_cell_neighbors(self, cell):
        return np.asarray(self.table("prism_cell_neighbors")[cell])

    def prism_node_cells(self, node):
        offsets = self.table("prism_node_cell_offsets")
        cells = self.table("prism_node_cells")
        return np.asarray(cells[offsets[node]:offsets[node + 1]])

    def prism_node_neighbors(self, node):
        offsets = self.table("prism_node_neighbor_offsets")
        neighbors = self.table("prism_node_neighbors")
        return np.asarray(neighbors[offsets[node]:offsets[node + 1]])

    def candidate_prism_columns(self, point):
        point = np.asarray(point, dtype=np.float64)
        bounds = self.table("prism_bvh_bounds")
        meta = self.table("prism_bvh_meta")
        order = self.table("prism_bvh_order")
        root = self.manifest["prism_submesh"]["point_locator"]["root"]
        candidates = []
        stack = [root]
        while stack:
            node = stack.pop()
            box = bounds[node]
            if np.any(point < box[:3]) or np.any(point > box[3:]):
                continue
            left, right, first, count = map(int, meta[node])
            if count:
                candidates.extend(map(int, order[first:first + count]))
            else:
                stack.append(right)
                stack.append(left)
        return candidates

    def locate_prism(self, point):
        point = np.asarray(point, dtype=np.float64)
        nodes = self.table("prism_nodes")
        cell_nodes = self.table("prism_cell_nodes")
        triangle_count = self.manifest["prism_submesh"]["surface_triangles"]
        layer_count = self.manifest["prism_submesh"]["layers"]
        for column in self.candidate_prism_columns(point):
            for layer in range(layer_count):
                cell = layer * triangle_count + column
                node_ids = np.asarray(cell_nodes[cell], dtype=np.int64)
                vertices = np.asarray(nodes[node_ids], dtype=np.float64)
                if (np.any(point < vertices.min(axis=0)) or
                        np.any(point > vertices.max(axis=0))):
                    continue
                for tetrahedron, local_nodes in enumerate(PRISM_TETRAHEDRA):
                    ids = node_ids[list(local_nodes)]
                    weights = tetrahedron_weights(
                        point, np.asarray(nodes[ids], dtype=np.float64)
                    )
                    if weights is not None:
                        return {
                            "cell": cell,
                            "column": column,
                            "layer": layer,
                            "tetrahedron": tetrahedron,
                            "interpolation_node_ids": ids.tolist(),
                            "interpolation_weights": weights.tolist(),
                        }
        return None

    def locate_octree_cell(self, point):
        point = np.asarray(point, dtype=np.float64)
        record = self.manifest["octree_submesh"]
        root_low = np.asarray(record["root_low"], dtype=np.float64)
        root_size = float(record["root_size"])
        dimension = record["point_locator"]["lattice_dimension"]
        relative = (point - root_low) / root_size
        if np.any(relative < 0) or np.any(relative >= 1):
            return None
        fine_index = np.floor(relative * dimension).astype(np.int64)
        block = int(self.table("octree_finest_owner")[tuple(fine_index)])
        block_low = np.asarray(self.table("octree_block_low")[block])
        block_size = float(self.table("octree_block_size")[block])
        block_cells = int(record["block_cells_per_axis"])
        local = np.floor(
            (point - block_low) * block_cells / block_size
        ).astype(np.int64)
        local = np.clip(local, 0, block_cells - 1)
        local_cell = int(
            (local[2] * block_cells + local[1]) * block_cells + local[0]
        )
        cell_low = block_low + local * (block_size / block_cells)
        coordinate = (point - cell_low) / (block_size / block_cells)
        x, y, z = coordinate
        weights = [
            (1 - x) * (1 - y) * (1 - z), x * (1 - y) * (1 - z),
            (1 - x) * y * (1 - z), x * y * (1 - z),
            (1 - x) * (1 - y) * z, x * (1 - y) * z,
            (1 - x) * y * z, x * y * z,
        ]
        corners = [
            [int(local[0] + i), int(local[1] + j), int(local[2] + k)]
            for k in (0, 1) for j in (0, 1) for i in (0, 1)
        ]
        return {
            "block": block,
            "local_ijk": local.tolist(),
            "local_cell": local_cell,
            "procedural_cell": block * block_cells ** 3 + local_cell,
            "cell_low": cell_low.tolist(),
            "cell_size": block_size / block_cells,
            "trilinear_coordinate": coordinate.tolist(),
            "interpolation_local_node_ijk": corners,
            "interpolation_weights": weights,
        }

    def locate(self, point):
        prism = self.locate_prism(point)
        octree = self.locate_octree_cell(point)
        cartesian_status = None
        active_region = "outside_domain"
        if octree is not None:
            if self.interface_manifest is None:
                active_region = "unclassified_cartesian_candidate"
            else:
                status = int(self.interface_table("cartesian_cell_status")[
                    octree["procedural_cell"]
                ])
                cartesian_status = {
                    "code": status,
                    "meaning": {
                        0: "outside_active",
                        1: "removed_overlap_or_solid",
                        2: "unresolved_cut_cell",
                    }[status],
                }
                active_region = {
                    0: "regular_cartesian",
                    1: "removed",
                    2: "unresolved_cut_cell",
                }[status]
        if prism is not None:
            active_region = "body_fitted_prism"
        return {
            "point": list(map(float, point)),
            "active_region": active_region,
            "prism": prism,
            "octree": octree,
            "cartesian_status": cartesian_status,
            "solver_ready": self.manifest["solver_ready"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--interface")
    parser.add_argument("--point", nargs=3, type=float, required=True)
    args = parser.parse_args()
    mesh = MeshTables(args.mesh, args.interface)
    print(json.dumps(mesh.locate(args.point), indent=2))


if __name__ == "__main__":
    main()
