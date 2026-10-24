"""Plot CRM volume-pressure slices and wall pressure from a binary checkpoint."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree


CELL_DTYPE = np.dtype([
    ("x", "<f8", (3,)), ("volume", "<f8"), ("dwall", "<f8"),
    ("first", "<i4"), ("count", "<i4"),
    ("term_first", "<i4"), ("term_count", "<i4"),
])
FACE_DTYPE = np.dtype({
    "names": ["left", "right", "bc", "x", "n", "area", "distance"],
    "formats": ["<i4", "<i4", "<i4", ("<f8", 3), ("<f8", 3), "<f8", "<f8"],
    "offsets": [0, 4, 8, 16, 40, 64, 72], "itemsize": 80,
})
BOUNDARY_DTYPE = np.dtype([
    ("node", "<i4", (3,)), ("bc", "<i4"), ("patch", "<i4"),
])


def load_cells(path: Path):
    with path.open("rb") as stream:
        magic, cells, faces, indices, terms = struct.unpack("<8s4Q", stream.read(40))
    if magic.rstrip(b"\0") != b"QCFD3D1" or cells < 1:
        raise ValueError("expected a QCFD3D1 binary mesh")
    expected_minimum = 40 + cells * CELL_DTYPE.itemsize
    if path.stat().st_size < expected_minimum:
        raise ValueError("truncated mesh cell table")
    return np.memmap(path, dtype=CELL_DTYPE, mode="r", offset=40, shape=(cells,)), faces


def load_faces(path: Path, cell_count: int, face_count: int):
    return np.memmap(
        path, dtype=FACE_DTYPE, mode="r",
        offset=40 + cell_count * CELL_DTYPE.itemsize, shape=(face_count,),
    )


def interpolate_plane(horizontal, vertical, fields, horizontal_points=900):
    """Interpolate an unstructured cut to a regular, locally supported display grid."""
    horizontal = np.asarray(horizontal); vertical = np.asarray(vertical)
    points = np.column_stack((horizontal, vertical))
    horizontal_grid = np.linspace(float(horizontal.min()), float(horizontal.max()), horizontal_points)
    aspect = (vertical.max() - vertical.min()) / max(horizontal.max() - horizontal.min(), 1e-12)
    vertical_points = int(np.clip(round(horizontal_points * aspect), 220, 650))
    vertical_grid = np.linspace(float(vertical.min()), float(vertical.max()), vertical_points)
    grid_x, grid_y = np.meshgrid(horizontal_grid, vertical_grid)
    query = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    triangulation = Delaunay(points)
    tree = cKDTree(points)
    neighbor_count = min(12, len(points))
    local_spacing = tree.query(points, k=neighbor_count, workers=-1)[0]
    if neighbor_count > 1:
        local_spacing = local_spacing[:, -1]
    nearest_distance, nearest_index = tree.query(query, k=1, workers=-1)
    grid_diagonal = np.hypot(horizontal_grid[1] - horizontal_grid[0], vertical_grid[1] - vertical_grid[0])
    unsupported = nearest_distance > np.maximum(2.5 * local_spacing[nearest_index], 2.0 * grid_diagonal)

    grids = []
    for values in fields:
        grid = np.asarray(LinearNDInterpolator(triangulation, np.asarray(values), fill_value=np.nan)(query)).reshape(grid_x.shape)
        grid.ravel()[unsupported] = np.nan
        grids.append(grid)
    return horizontal_grid, vertical_grid, grids


def interpolate_boundary_mesh(triangulation, fields, x_bounds, z_bounds, nx=900, nz=320):
    """Rasterize the actual symmetry triangles without bridging the solid."""
    grid_x = np.linspace(*x_bounds, nx)
    grid_z = np.linspace(*z_bounds, nz)
    xx, zz = np.meshgrid(grid_x, grid_z)
    grids = []
    for field in fields:
        interpolator = mtri.LinearTriInterpolator(triangulation, field)
        grids.append(np.ma.filled(interpolator(xx, zz), np.nan))
    return grid_x, grid_z, grids


def wall_seam_segments(boundary, points):
    """Return the physical wall/symmetry intersection as exact mesh edges."""
    wall_triangles = boundary["node"][boundary["bc"] == 1]
    edges = np.sort(np.concatenate((
        wall_triangles[:, [0, 1]], wall_triangles[:, [1, 2]], wall_triangles[:, [2, 0]],
    )), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    seam = unique_edges[counts == 1]
    if np.max(np.abs(points[seam, 1])) > 1e-8:
        raise ValueError("wall has open edges away from the symmetry plane")
    return np.asarray(points[seam][:, :, [0, 2]])


TETRA_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def tetra_plane_mesh(tables, axis, station, horizontal, vertical, bounds, chunk_size=500_000):
    """Intersect tetrahedra with a plane and share crossing-edge vertices."""
    points = tables["points"]
    cell_nodes = tables["cell_nodes"]
    selected = []
    for first in range(0, len(cell_nodes), chunk_size):
        last = min(first + chunk_size, len(cell_nodes))
        coordinates = np.asarray(points[cell_nodes[first:last]])
        plane_values = coordinates[:, :, axis]
        crosses = (plane_values.min(axis=1) < station) & (plane_values.max(axis=1) > station)
        near_horizontal = (
            (coordinates[:, :, horizontal].max(axis=1) >= bounds[0][0])
            & (coordinates[:, :, horizontal].min(axis=1) <= bounds[0][1])
        )
        near_vertical = (
            (coordinates[:, :, vertical].max(axis=1) >= bounds[1][0])
            & (coordinates[:, :, vertical].min(axis=1) <= bounds[1][1])
        )
        selected.append(np.flatnonzero(crosses & near_horizontal & near_vertical) + first)
    cut_cells = np.concatenate(selected)
    tetrahedra = np.asarray(cell_nodes[cut_cells])
    distances = points[tetrahedra, axis] - station
    local_vertices = np.full((len(tetrahedra), 6), -1, dtype=np.int32)
    edge_parts = []
    positions = []
    for edge_number, (a, b) in enumerate(TETRA_EDGES):
        crossing = np.flatnonzero((distances[:, a] < 0) != (distances[:, b] < 0))
        pairs = np.sort(tetrahedra[crossing][:, [a, b]], axis=1)
        positions.append((edge_number, crossing, len(pairs)))
        edge_parts.append(pairs)
    all_edges = np.concatenate(edge_parts)
    unique_edges, inverse = np.unique(all_edges, axis=0, return_inverse=True)
    cursor = 0
    for edge_number, crossing, count in positions:
        local_vertices[crossing, edge_number] = inverse[cursor:cursor + count]
        cursor += count
    count = (local_vertices >= 0).sum(axis=1)
    if not np.all((count == 3) | (count == 4)):
        raise ValueError("degenerate tetra-plane crossing; choose a plane away from exact mesh nodes")
    triangles = np.sort(local_vertices[count == 3], axis=1)[:, 3:]
    quads = np.sort(local_vertices[count == 4], axis=1)[:, 2:]
    endpoints = np.asarray(points[unique_edges])
    fraction = (station - endpoints[:, 0, axis]) / (endpoints[:, 1, axis] - endpoints[:, 0, axis])
    cut_points = endpoints[:, 0] + fraction[:, None] * (endpoints[:, 1] - endpoints[:, 0])
    if len(quads):
        coordinates = cut_points[quads][:, :, [horizontal, vertical]]
        centers = coordinates.mean(axis=1)
        angles = np.arctan2(
            coordinates[:, :, 1] - centers[:, None, 1],
            coordinates[:, :, 0] - centers[:, None, 0],
        )
        ordered = np.take_along_axis(quads, np.argsort(angles, axis=1), axis=1)
        triangles = np.concatenate((
            triangles, ordered[:, [0, 1, 2]], ordered[:, [0, 2, 3]],
        ))
    mesh = mtri.Triangulation(cut_points[:, horizontal], cut_points[:, vertical], triangles)
    return mesh, cut_points, unique_edges, fraction, len(cut_cells)


def wall_plane_segments(tables, axis, station, horizontal, vertical):
    """Intersect the actual wall triangles with an interior cutting plane."""
    points = tables["points"]
    faces = np.asarray(tables["boundary"]["node"][tables["boundary"]["bc"] == 1])
    distance = points[faces, axis] - station
    crossing_faces = (distance.min(axis=1) < 0) & (distance.max(axis=1) > 0)
    faces = faces[crossing_faces]
    distance = distance[crossing_faces]
    crossing_points = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cross = (distance[:, a] < 0) != (distance[:, b] < 0)
        start, end = points[faces[cross, a]], points[faces[cross, b]]
        weight = (station - start[:, axis]) / (end[:, axis] - start[:, axis])
        crossing_points.append((cross, start + weight[:, None] * (end - start)))
    segments = np.full((len(faces), 2, 2), np.nan)
    cursor = np.zeros(len(faces), dtype=np.int8)
    for mask, coordinates in crossing_points:
        rows = np.flatnonzero(mask)
        segments[rows, cursor[rows]] = coordinates[:, [horizontal, vertical]]
        cursor[rows] += 1
    if not np.all(cursor == 2):
        raise ValueError("degenerate wall-plane crossing")
    return segments


def select_slice(cells, axis: int, location: float, bounds, maximum: int, maximum_half_width: float):
    xyz = cells["x"]
    mask = np.ones(len(cells), dtype=bool)
    for coordinate, (lower, upper) in enumerate(bounds):
        if coordinate != axis:
            mask &= (xyz[:, coordinate] >= lower) & (xyz[:, coordinate] <= upper)
    mask &= np.abs(xyz[:, axis] - location) <= maximum_half_width
    candidates = np.flatnonzero(mask)
    if not len(candidates):
        raise ValueError("slice plot bounds contain no cells")
    distance = np.abs(xyz[candidates, axis] - location)
    count = min(maximum, len(candidates))
    chosen = np.argpartition(distance, count - 1)[:count] if count < len(candidates) else np.arange(count)
    indices = candidates[chosen]
    order = np.argsort(cells["volume"][indices])[::-1]
    indices = indices[order]
    return indices, float(distance[chosen].max())


def primitive_cp(state, indices, gamma: float, mach: float):
    conserved = np.asarray(state[indices])
    density = conserved[:, 0]
    pressure = (gamma - 1) * (
        conserved[:, 4] - np.einsum("ij,ij->i", conserved[:, 1:4], conserved[:, 1:4]) / (2 * density)
    )
    return (pressure - 1 / gamma) / (0.5 * mach * mach)


def primitive_fields(state, indices, gamma: float, mach: float):
    conserved = np.asarray(state[indices]); density = conserved[:, 0]
    velocity = conserved[:, 1:4] / density[:, None]
    pressure = (gamma - 1) * (conserved[:, 4] - 0.5 * density * np.einsum("ij,ij->i", velocity, velocity))
    local_mach = np.linalg.norm(velocity, axis=1) / np.sqrt(gamma * pressure / density)
    cp = (pressure - 1 / gamma) / (0.5 * mach * mach)
    return cp, local_mach, velocity


def load_node_gradient(connectivity_path: Path, gradient_path: Path, cell_count: int, face_count: int):
    with connectivity_path.open("rb") as stream:
        fields = struct.unpack("<8sII10Q", stream.read(96))
    magic, version, _, point_count, connectivity_cells, boundary_count, incidence_count, points_offset, cell_nodes_offset, node_offsets_offset, node_cells_offset, boundary_offset, _ = fields
    if magic.rstrip(b"\0") != b"QCFDNC1" or version != 1 or connectivity_cells != cell_count:
        raise ValueError("incompatible connectivity companion")
    with gradient_path.open("rb") as stream:
        gradient_fields = struct.unpack("<8sII8Q", stream.read(80))
    gradient_magic, gradient_version, _, gradient_points, gradient_cells, gradient_faces, gradient_incidences, weights_offset, gradients_offset, face_nodes_offset, gradient_bytes = gradient_fields
    if gradient_magic.rstrip(b"\0") != b"QCFDNG1" or gradient_version != 2 or gradient_points != point_count or gradient_cells != cell_count or gradient_faces != face_count or gradient_incidences != incidence_count or gradient_path.stat().st_size != gradient_bytes:
        raise ValueError("incompatible node-gradient companion")
    return {
        "point_count": point_count,
        "points": np.memmap(connectivity_path, dtype="<f8", mode="r", offset=points_offset, shape=(point_count, 3)),
        "cell_nodes": np.memmap(connectivity_path, dtype="<i4", mode="r", offset=cell_nodes_offset, shape=(cell_count, 4)),
        "node_offsets": np.memmap(connectivity_path, dtype="<u8", mode="r", offset=node_offsets_offset, shape=(point_count + 1,)),
        "node_cells": np.memmap(connectivity_path, dtype="<i4", mode="r", offset=node_cells_offset, shape=(incidence_count,)),
        "weights": np.memmap(gradient_path, dtype="<f8", mode="r", offset=weights_offset, shape=(incidence_count,)),
        "shape_gradients": np.memmap(gradient_path, dtype="<f8", mode="r", offset=gradients_offset, shape=(cell_count, 4, 3)),
        "face_nodes": np.memmap(gradient_path, dtype="<i4", mode="r", offset=face_nodes_offset, shape=(face_count, 3)),
        "boundary": np.memmap(connectivity_path, dtype=BOUNDARY_DTYPE, mode="r", offset=boundary_offset, shape=(boundary_count,)),
    }


def interpolate_node_velocity(state, tables, node_chunk=100_000):
    node_velocity = np.empty((tables["point_count"], 3), dtype=np.float64)
    offsets = tables["node_offsets"]
    for first_node in range(0, tables["point_count"], node_chunk):
        last_node = min(first_node + node_chunk, tables["point_count"])
        first_incidence, last_incidence = int(offsets[first_node]), int(offsets[last_node])
        incident_cells = tables["node_cells"][first_incidence:last_incidence]
        local_offsets = np.asarray(offsets[first_node:last_node], dtype=np.int64) - first_incidence
        weights = tables["weights"][first_incidence:last_incidence]
        density = state[incident_cells, 0]
        for component in range(3):
            contributions = weights * state[incident_cells, component + 1] / density
            node_velocity[first_node:last_node, component] = np.add.reduceat(contributions, local_offsets)
    return node_velocity


def interpolate_node_primitives(state, tables, node_indices, gamma: float, mach: float):
    """Interpolate cell Cp and Mach to selected mesh nodes with stored weights."""
    result_cp = np.empty(len(node_indices)); result_mach = np.empty(len(node_indices))
    offsets = tables["node_offsets"]
    for output_index, node in enumerate(node_indices):
        first, last = int(offsets[node]), int(offsets[node + 1])
        incident_cells = tables["node_cells"][first:last]
        weights = tables["weights"][first:last]
        cp, local_mach, _ = primitive_fields(state, incident_cells, gamma, mach)
        result_cp[output_index] = weights @ cp
        result_mach[output_index] = weights @ local_mach
    return result_cp, result_mach


def color_norm(*arrays):
    values = np.concatenate([np.asarray(value)[np.isfinite(value)] for value in arrays])
    lower, upper = np.quantile(values, [0.01, 0.99])
    lower = min(float(lower), -1e-6); upper = max(float(upper), 1e-6)
    return TwoSlopeNorm(vmin=lower, vcenter=0, vmax=upper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--connectivity", type=Path)
    parser.add_argument("--node-gradient", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mach", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--aoa", type=float, default=2.5)
    parser.add_argument("--reference-area", type=float, default=3.9092552169415784)
    parser.add_argument("--y", type=float)
    parser.add_argument("--z", type=float)
    parser.add_argument("--vortex-x", type=float)
    parser.add_argument("--maximum-points", type=int, default=250_000)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    cells, face_count = load_cells(arguments.mesh)
    expected_state_bytes = len(cells) * 6 * 8
    if arguments.state.stat().st_size != expected_state_bytes:
        raise ValueError("checkpoint size does not match the mesh")
    state = np.memmap(arguments.state, dtype="<f8", mode="r", shape=(len(cells), 6))
    if (arguments.connectivity is None) != (arguments.node_gradient is None):
        raise ValueError("--connectivity and --node-gradient must be supplied together")
    tables = None
    if arguments.connectivity and arguments.node_gradient:
        tables = load_node_gradient(arguments.connectivity, arguments.node_gradient, len(cells), face_count)
    surface = np.genfromtxt(arguments.surface, delimiter=",", names=True)
    sx, sy, sz, surface_cp = (surface[name] for name in ("x_R", "y_R", "z_R", "Cp"))
    y_station = arguments.y if arguments.y is not None else float(sy.min() + 0.50 * (sy.max() - sy.min()))
    # A constant-z plane cannot cut the entire swept, dihedral wing. This
    # station crosses the inner/main wing; --z can inspect farther outboard.
    z_station = arguments.z if arguments.z is not None else 0.65

    # A near-field window broad enough to show the fuselage shock and wake,
    # without shrinking the view to the wall bounding box or plotting 100 MAC
    # of intentionally coarse farfield mesh.
    x_bounds = (float(sx.min() - 1.0), float(sx.max() + 2.0))
    y_bounds = (0.0, float(sy.max() + 1.0))
    z_bounds = (float(sz.min() - 0.85), float(sz.max() + 1.50))
    if tables is not None:
        xy_mesh, xy_xyz, xy_edges, xy_fraction, xy_cells = tetra_plane_mesh(
            tables, 2, z_station, 0, 1, (x_bounds, y_bounds),
        )
        xy_nodes = np.unique(xy_edges)
        node_cp, node_mach = interpolate_node_primitives(
            state, tables, xy_nodes, arguments.gamma, arguments.mach,
        )
        edge_nodes = np.searchsorted(xy_nodes, xy_edges)
        xy_cp = (1 - xy_fraction) * node_cp[edge_nodes[:, 0]] + xy_fraction * node_cp[edge_nodes[:, 1]]
        xy_mach = (1 - xy_fraction) * node_mach[edge_nodes[:, 0]] + xy_fraction * node_mach[edge_nodes[:, 1]]
        xy_wall_segments = wall_plane_segments(tables, 2, z_station, 0, 1)
        z_half_width = 0.0
        xy_indices = None
    else:
        xy_indices, z_half_width = select_slice(cells, 2, z_station, (x_bounds, y_bounds, (0, 0)), arguments.maximum_points, 0.01 * (sz.max() - sz.min()))
        xy_cp, xy_mach, _ = primitive_fields(state, xy_indices, arguments.gamma, arguments.mach)
        xy_xyz = np.asarray(cells["x"][xy_indices])
        xy_cells = len(xy_indices)
        xy_mesh = None
        xy_wall_segments = None

    if tables is not None:
        points = tables["points"]
        symmetry_faces = np.asarray(tables["boundary"]["node"][tables["boundary"]["bc"] == 3])
        symmetry_vertices = points[symmetry_faces]
        face_in_bounds = (
            (symmetry_vertices[:, :, 0].max(axis=1) >= x_bounds[0])
            & (symmetry_vertices[:, :, 0].min(axis=1) <= x_bounds[1])
            & (symmetry_vertices[:, :, 2].max(axis=1) >= z_bounds[0])
            & (symmetry_vertices[:, :, 2].min(axis=1) <= z_bounds[1])
        )
        symmetry_faces = symmetry_faces[face_in_bounds]
        symmetry_nodes, local_face_nodes = np.unique(symmetry_faces, return_inverse=True)
        symmetry_xyz = np.asarray(points[symmetry_nodes])
        symmetry_cp, symmetry_mach = interpolate_node_primitives(
            state, tables, symmetry_nodes, arguments.gamma, arguments.mach,
        )
        symmetry_mesh = mtri.Triangulation(
            symmetry_xyz[:, 0], symmetry_xyz[:, 2], local_face_nodes.reshape(-1, 3),
        )
        seam_segments = wall_seam_segments(tables["boundary"], points)
        symmetry_half_width = 0.0
    else:
        symmetry_indices, symmetry_half_width = select_slice(cells, 1, 0.0, (x_bounds, (0, 0), z_bounds), arguments.maximum_points, 0.005 * (sy.max() - sy.min()))
        symmetry_cp, symmetry_mach, _ = primitive_fields(state, symmetry_indices, arguments.gamma, arguments.mach)
        symmetry_xyz = np.asarray(cells["x"][symmetry_indices])
        symmetry_nodes = None
        symmetry_mesh = None
        seam_segments = None
    symmetry_wall = sy <= max(symmetry_half_width, 1e-8)
    xy_wall = np.abs(sz - z_station) <= z_half_width
    if symmetry_mesh is not None:
        symmetry_grid_x, symmetry_grid_z, symmetry_grids = interpolate_boundary_mesh(
            symmetry_mesh, (symmetry_cp, symmetry_mach), x_bounds, z_bounds,
        )
    else:
        symmetry_grid_x, symmetry_grid_z, symmetry_grids = interpolate_plane(
            symmetry_xyz[:, 0], symmetry_xyz[:, 2], (symmetry_cp, symmetry_mach),
        )
    symmetry_cp_grid, symmetry_mach_grid = symmetry_grids
    if xy_mesh is not None:
        xy_grid_x, xy_grid_y, xy_grids = interpolate_boundary_mesh(
            xy_mesh, (xy_cp, xy_mach), x_bounds, y_bounds, nx=1000, nz=500,
        )
    else:
        xy_grid_x, xy_grid_y, xy_grids = interpolate_plane(xy_xyz[:, 0], xy_xyz[:, 1], (xy_cp, xy_mach))
    xy_cp_grid, xy_mach_grid = xy_grids
    norm = color_norm(symmetry_cp, xy_cp, surface_cp)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    pressure_plot = axes[0].pcolormesh(symmetry_grid_x, symmetry_grid_z, symmetry_cp_grid, cmap="RdBu_r", norm=color_norm(symmetry_cp), shading="auto", rasterized=True)
    mach_plot = axes[1].pcolormesh(symmetry_grid_x, symmetry_grid_z, symmetry_mach_grid, cmap="turbo", vmin=0.20, vmax=1.10, shading="auto", rasterized=True)
    if np.nanmin(symmetry_mach_grid) <= 1 <= np.nanmax(symmetry_mach_grid):
        axes[1].contour(symmetry_grid_x, symmetry_grid_z, symmetry_mach_grid, levels=[1.0], colors="black", linewidths=0.8)
    for axis in axes:
        if seam_segments is not None:
            axis.add_collection(LineCollection(seam_segments, colors="black", linewidths=0.5))
        else:
            axis.scatter(sx[symmetry_wall], sz[symmetry_wall], s=0.5, c="black", linewidths=0, rasterized=True)
        axis.set(xlabel="x / MAC", ylabel="z / MAC", xlim=x_bounds, ylim=z_bounds); axis.set_aspect("equal", adjustable="box")
    axes[0].set_title("pressure coefficient $C_p$"); axes[1].set_title("Mach; black contour: M=1")
    figure.colorbar(pressure_plot, ax=axes[0], label="$C_p$"); figure.colorbar(mach_plot, ax=axes[1], label="Mach")
    symmetry_label = "exact symmetry-boundary nodes" if symmetry_nodes is not None else f"cell slab half-width={symmetry_half_width:.3g}"
    figure.suptitle(f"CRM symmetry-plane near field at y=0; {symmetry_label}")
    figure.savefig(arguments.out / "crm_xz_symmetry_mach_pressure.png", dpi=220); plt.close(figure)

    shock_x_bounds = (1.15, 2.15); shock_z_bounds = (1.05, 1.65)
    shock_nodes = (
        (symmetry_xyz[:, 0] >= shock_x_bounds[0]) & (symmetry_xyz[:, 0] <= shock_x_bounds[1])
        & (symmetry_xyz[:, 2] >= shock_z_bounds[0]) & (symmetry_xyz[:, 2] <= shock_z_bounds[1])
    )
    if symmetry_mesh is not None:
        shock_grid_x, shock_grid_z, shock_grids = interpolate_boundary_mesh(
            symmetry_mesh, (symmetry_cp, symmetry_mach), shock_x_bounds, shock_z_bounds,
            nx=1200, nz=720,
        )
        shock_cp_grid, shock_mach_grid = shock_grids
    else:
        shock_grid_x, shock_grid_z, shock_cp_grid, shock_mach_grid = (
            symmetry_grid_x, symmetry_grid_z, symmetry_cp_grid, symmetry_mach_grid,
        )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    pressure_plot = axes[0].pcolormesh(
        shock_grid_x, shock_grid_z, shock_cp_grid, cmap="RdBu_r",
        norm=color_norm(symmetry_cp[shock_nodes]), shading="auto", rasterized=True,
    )
    mach_plot = axes[1].pcolormesh(
        shock_grid_x, shock_grid_z, shock_mach_grid, cmap="turbo",
        vmin=0.75, vmax=1.03, shading="auto", rasterized=True,
    )
    axes[1].contour(shock_grid_x, shock_grid_z, shock_mach_grid, levels=[0.95, 1.0], colors=["white", "black"], linewidths=[0.7, 1.2])
    for axis in axes:
        if seam_segments is not None:
            axis.add_collection(LineCollection(seam_segments, colors="black", linewidths=0.85))
        else:
            axis.scatter(sx[symmetry_wall], sz[symmetry_wall], s=1.0, c="black", linewidths=0, rasterized=True)
        axis.set(xlabel="x / MAC", ylabel="z / MAC", xlim=shock_x_bounds, ylim=shock_z_bounds)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_title("pressure coefficient $C_p$")
    axes[1].set_title("Mach; white M=0.95, black M=1")
    figure.colorbar(pressure_plot, ax=axes[0], label="$C_p$"); figure.colorbar(mach_plot, ax=axes[1], label="Mach")
    figure.suptitle("CRM fuselage dorsal transonic-pocket / shock zoom on y=0")
    figure.savefig(arguments.out / "crm_xz_fuselage_shock_zoom.png", dpi=240); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    pressure_plot = axes[0].pcolormesh(xy_grid_x, xy_grid_y, xy_cp_grid, cmap="RdBu_r", norm=color_norm(xy_cp), shading="auto", rasterized=True)
    mach_plot = axes[1].pcolormesh(xy_grid_x, xy_grid_y, xy_mach_grid, cmap="turbo", vmin=0.20, vmax=1.15, shading="auto", rasterized=True)
    if np.nanmin(xy_mach_grid) <= 1 <= np.nanmax(xy_mach_grid):
        axes[1].contour(xy_grid_x, xy_grid_y, xy_mach_grid, levels=[1.0], colors="black", linewidths=0.8)
    for axis in axes:
        if xy_wall_segments is not None:
            axis.add_collection(LineCollection(xy_wall_segments, colors="black", linewidths=0.55))
        else:
            axis.scatter(sx[xy_wall], sy[xy_wall], s=0.4, c="black", linewidths=0, rasterized=True)
        axis.set(xlabel="x / MAC", ylabel="y / MAC", xlim=x_bounds, ylim=y_bounds); axis.set_aspect("equal", adjustable="box")
    axes[0].set_title("pressure coefficient $C_p$"); axes[1].set_title("Mach; black contour: M=1")
    figure.colorbar(pressure_plot, ax=axes[0], label="$C_p$"); figure.colorbar(mach_plot, ax=axes[1], label="Mach")
    xy_label = "exact tetrahedron/plane intersections" if xy_mesh is not None else f"sampled half-width={z_half_width:.3g}"
    figure.suptitle(f"CRM horizontal flow slice at z={z_station:.4f}; {xy_label}")
    figure.savefig(arguments.out / "crm_xy_mach_pressure.png", dpi=220); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for axis, grid_y, grid_cp, wall_x, wall_y, title, ylabel, limits in (
        (axes[0], symmetry_grid_z, symmetry_cp_grid, sx[symmetry_wall], sz[symmetry_wall], "x–z symmetry plane, y=0", "z / MAC", z_bounds),
        (axes[1], xy_grid_y, xy_cp_grid, sx[xy_wall], sy[xy_wall], f"x–y plane, z={z_station:.4f}", "y / MAC", y_bounds),
    ):
        grid_x = symmetry_grid_x if axis is axes[0] else xy_grid_x
        pressure_plot = axis.pcolormesh(grid_x, grid_y, grid_cp, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
        if axis is axes[0] and seam_segments is not None:
            axis.add_collection(LineCollection(seam_segments, colors="black", linewidths=0.5))
        elif axis is axes[1] and xy_wall_segments is not None:
            axis.add_collection(LineCollection(xy_wall_segments, colors="black", linewidths=0.55))
        else:
            axis.scatter(wall_x, wall_y, s=0.4, c="black", linewidths=0, rasterized=True)
        axis.set(xlabel="x / MAC", ylabel=ylabel, title=title, xlim=x_bounds, ylim=limits); axis.set_aspect("equal", adjustable="box")
    figure.colorbar(pressure_plot, ax=axes, label="$C_p$")
    figure.suptitle("CRM near-field pressure; geometry-aware symmetry and tetrahedron cuts")
    figure.savefig(arguments.out / "crm_pressure_slices.png", dpi=220); plt.close(figure)

    vortex_manifest = None
    if arguments.connectivity and arguments.node_gradient:
        tip_selection = sy > sy.min() + 0.75 * (sy.max() - sy.min())
        # Stay just aft of the tip: the wake mesh coarsens rapidly farther downstream.
        vortex_x = arguments.vortex_x if arguments.vortex_x is not None else float(sx[tip_selection].max() + 0.10)
        vortex_y_bounds = (float(sy.max() - 0.90), float(sy.max() + 0.35))
        vortex_z_bounds = (float(sz[tip_selection].min() - 0.25), float(sz[tip_selection].max() + 0.30))
        vortex_indices, vortex_half_width = select_slice(cells, 0, vortex_x, ((0, 0), vortex_y_bounds, vortex_z_bounds), arguments.maximum_points, 0.03)
        _, _, vortex_velocity = primitive_fields(state, vortex_indices, arguments.gamma, arguments.mach)
        node_velocity = interpolate_node_velocity(state, tables)
        nodal = node_velocity[tables["cell_nodes"][vortex_indices]]; shape = tables["shape_gradients"][vortex_indices]
        velocity_gradient = np.einsum("miv,mij->mvj", nodal, shape)
        streamwise_vorticity = velocity_gradient[:, 2, 1] - velocity_gradient[:, 1, 2]
        cross_velocity = vortex_velocity[:, 1:3].copy(); cross_velocity[:, 1] -= arguments.mach * np.sin(np.deg2rad(arguments.aoa)); cross_speed = np.linalg.norm(cross_velocity, axis=1)
        vortex_xyz = cells["x"][vortex_indices]; vorticity_limit = np.quantile(np.abs(streamwise_vorticity[np.isfinite(streamwise_vorticity)]), 0.995)
        vortex_grid_y, vortex_grid_z, vortex_grids = interpolate_plane(
            vortex_xyz[:, 1], vortex_xyz[:, 2], (streamwise_vorticity, cross_speed), horizontal_points=650,
        )
        vorticity_grid, cross_speed_grid = vortex_grids
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
        vortex_plot = axes[0].pcolormesh(vortex_grid_y, vortex_grid_z, vorticity_grid, cmap="RdBu_r", vmin=-vorticity_limit, vmax=vorticity_limit, shading="auto", rasterized=True)
        speed_limits = np.quantile(cross_speed[np.isfinite(cross_speed)], [0.01, 0.995]); speed_plot = axes[1].pcolormesh(vortex_grid_y, vortex_grid_z, cross_speed_grid, cmap="magma", vmin=speed_limits[0], vmax=speed_limits[1], shading="auto", rasterized=True)
        arrow_count = min(250, len(vortex_indices)); arrow_indices = np.linspace(0, len(vortex_indices) - 1, arrow_count, dtype=int)
        axes[1].quiver(
            vortex_xyz[arrow_indices, 1], vortex_xyz[arrow_indices, 2],
            cross_velocity[arrow_indices, 0], cross_velocity[arrow_indices, 1],
            color="black", alpha=0.65, angles="xy", scale_units="xy", scale=1.5, width=0.002,
        )
        for axis in axes: axis.set(xlabel="y / MAC", ylabel="z / MAC"); axis.set_aspect("equal", adjustable="box")
        axes[0].set_title("streamwise vorticity $\\omega_x$"); axes[1].set_title("crossflow speed and direction")
        figure.colorbar(vortex_plot, ax=axes[0], label="$\\omega_x$ [a/MAC]"); figure.colorbar(speed_plot, ax=axes[1], label="crossflow speed / a∞")
        figure.suptitle(f"CRM wake plane at x={vortex_x:.4f}; sampled half-width={vortex_half_width:.3g}")
        figure.savefig(arguments.out / "crm_yz_tip_vortex.png", dpi=220); plt.close(figure)
        vortex_manifest = {"x": vortex_x, "slab_half_width": vortex_half_width, "points": len(vortex_indices), "vorticity_range": [float(streamwise_vorticity.min()), float(streamwise_vorticity.max())]}

    upper = (surface["nz"] > 0.15) & (sy > sy.min() + 0.02 * (sy.max() - sy.min()))
    lower = (surface["nz"] < -0.15) & (sy > sy.min() + 0.02 * (sy.max() - sy.min()))
    surface_center_error = None
    if tables is not None:
        faces = load_faces(arguments.mesh, len(cells), face_count)
        wall_face_ids = np.flatnonzero((faces["right"] < 0) & (faces["bc"] == 1))
        if len(wall_face_ids) != len(surface):
            raise ValueError("surface CSV does not match wall-face ordering")
        wall_triangles = np.asarray(tables["points"][tables["face_nodes"][wall_face_ids]])
        surface_center_error = float(np.max(np.linalg.norm(wall_triangles.mean(axis=1) - np.column_stack((sx, sy, sz)), axis=1)))
        if surface_center_error > 1e-9:
            raise ValueError("reconstructed wall triangles do not match surface CSV")
        mirrored_triangles = wall_triangles.copy(); mirrored_triangles[:, :, 1] *= -1
        full_triangles = np.concatenate((wall_triangles, mirrored_triangles))
        full_cp = np.tile(surface_cp, 2)
        surface_norm = color_norm(surface_cp)
        surface_colors = plt.get_cmap("RdBu_r")(surface_norm(full_cp))
        figure, axes = plt.subplots(
            1, 2, figsize=(16, 7), constrained_layout=True,
            subplot_kw={"projection": "3d", "computed_zorder": False},
        )
        for axis, elevation, title in ((axes[0], 52, "upper surface"), (axes[1], -38, "lower surface")):
            collection = Poly3DCollection(
                full_triangles, facecolors=surface_colors, edgecolors="none", linewidths=0,
                antialiased=False, rasterized=True,
            )
            axis.add_collection3d(collection)
            axis.set(
                xlim=(float(sx.min()), float(sx.max())), ylim=(-float(sy.max()), float(sy.max())),
                zlim=(float(sz.min()), float(sz.max())), title=title,
            )
            axis.set_box_aspect(
                (float(np.ptp(sx)), float(2 * sy.max()), float(2.5 * np.ptp(sz))), zoom=1.25,
            )
            axis.set_proj_type("ortho"); axis.view_init(elev=elevation, azim=-90); axis.set_axis_off()
        scalar_map = plt.cm.ScalarMappable(norm=surface_norm, cmap="RdBu_r"); scalar_map.set_array(surface_cp)
        figure.colorbar(scalar_map, ax=axes, shrink=0.78, pad=0.015, label="$C_p$")
        figure.suptitle("CRM surface pressure $C_p$ — exact wall triangles; full model mirrored from half-domain")
        figure.savefig(arguments.out / "crm_surface_pressure.png", dpi=220)
        plt.close(figure)

    section_width = 0.005 * (sy.max() - sy.min())
    section = np.abs(sy - y_station) <= section_width
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    axis.scatter(sx[section & upper], surface_cp[section & upper], s=9, label="$n_z>0.15$", alpha=0.8)
    axis.scatter(sx[section & lower], surface_cp[section & lower], s=9, label="$n_z<-0.15$", alpha=0.8)
    axis.invert_yaxis(); axis.grid(alpha=0.25); axis.legend()
    axis.set(xlabel="x / MAC", ylabel="$C_p$", title=f"surface-pressure section near y={y_station:.4f}")
    figure.savefig(arguments.out / "crm_surface_cp_section.png", dpi=220)
    plt.close(figure)

    np.savez_compressed(arguments.out / "crm_pressure_slice_data.npz",
        symmetry_xyz=symmetry_xyz, symmetry_cp=symmetry_cp, symmetry_mach=symmetry_mach,
        xy_xyz=xy_xyz, xy_cp=xy_cp, xy_mach=xy_mach,
        y_station=y_station, z_station=z_station,
        symmetry_slab_half_width=symmetry_half_width, z_slab_half_width=z_half_width)
    manifest = {
        "mesh": str(arguments.mesh), "state": str(arguments.state), "surface": str(arguments.surface),
        "cells": len(cells), "faces": face_count, "wall_faces": len(surface),
        "mach": arguments.mach, "gamma": arguments.gamma,
        "xy": {
            "z": z_station,
            "sampling": "tetrahedron-plane intersections" if xy_mesh is not None else "cell-center slab",
            "slab_half_width": z_half_width, "points": len(xy_xyz), "intersected_cells": xy_cells,
        },
        "xz_symmetry": {
            "y": 0.0, "sampling": "symmetry boundary triangles" if symmetry_nodes is not None else "cell-center slab",
            "slab_half_width": symmetry_half_width, "points": len(symmetry_xyz),
        },
        "nearfield_plot_bounds": {"x": x_bounds, "y": y_bounds, "z": z_bounds},
        "Cp_surface_range": [float(np.nanmin(surface_cp)), float(np.nanmax(surface_cp))],
        "note": "The xz plane uses symmetry boundary triangles, xy uses exact tetrahedron-plane intersections, and yz remains a masked cell-center-slab interpolation. Surface Cp is painted on actual wall triangles from owner-cell samples. Coefficients are from an unconverged checkpoint.",
    }
    if symmetry_mesh is not None:
        manifest["xz_symmetry"]["triangles"] = len(symmetry_mesh.triangles)
        manifest["xz_symmetry"]["wall_seam_edges"] = len(seam_segments)
    if surface_center_error is not None:
        manifest["surface_geometry"] = {
            "computed_half_triangles": len(surface), "displayed_full_triangles": 2 * len(surface),
            "face_center_match_linf": surface_center_error,
        }
    if vortex_manifest is not None: manifest["yz_tip_vortex"] = vortex_manifest
    angle = np.deg2rad(arguments.aoa); drag_direction = np.array([np.cos(angle), 0, np.sin(angle)]); lift_direction = np.array([-np.sin(angle), 0, np.cos(angle)])
    pressure_force = np.array([(surface["p"] * surface[name] * surface["area_R2"]).sum() for name in ("nx", "ny", "nz")])
    total_force = np.array([surface[name].sum() for name in ("force_x", "force_y", "force_z")]); viscous_force = total_force - pressure_force
    coefficient_scale = 0.5 * arguments.mach**2 * arguments.reference_area
    manifest["load_split"] = {name: {"CD": float(force @ drag_direction / coefficient_scale), "CL": float(force @ lift_direction / coefficient_scale)} for name, force in (("pressure", pressure_force), ("viscous", viscous_force), ("total", total_force))}
    if arguments.history:
        history = np.genfromtxt(arguments.history, delimiter=",", names=True)
        reference = {"CL": 0.54239, "CD": 0.02995, "CM": -0.06249}
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        axes[0].semilogy(history["iteration"], history["res_mean_max"], label="mean-flow max")
        axes[0].semilogy(history["iteration"], history["res_SA"], label="SA")
        axes[0].axhline(1e-3, color="black", linestyle="--", linewidth=1, label="current tolerance")
        axes[0].set(xlabel="continuation iteration", ylabel="volume-weighted raw residual", title="residual history"); axes[0].grid(alpha=0.25); axes[0].legend()
        for name in ("CL", "CD", "CM"):
            error = 100 * (history[name] - reference[name]) / abs(reference[name])
            axes[1].plot(history["iteration"], error, label=name)
        axes[1].axhline(0, color="black", linewidth=1); axes[1].set(xlabel="continuation iteration", ylabel="difference from NASA USM3D [%]", title="transient load drift—not validation"); axes[1].grid(alpha=0.25); axes[1].legend()
        figure.savefig(arguments.out / "crm_convergence_history.png", dpi=220); plt.close(figure)
        if len(history) >= 2:
            latest, previous = history[-1], history[-2]
            manifest["last_check_interval"] = {
                "iterations": int(latest["iteration"] - previous["iteration"]),
                "CL_delta": float(latest["CL"] - previous["CL"]),
                "CD_delta": float(latest["CD"] - previous["CD"]),
                "CM_delta": float(latest["CM"] - previous["CM"]),
                "CL_relative_change": float(abs(latest["CL"] - previous["CL"]) / max(0.1, abs(latest["CL"]))),
                "CD_relative_change": float(abs(latest["CD"] - previous["CD"]) / max(0.01, abs(latest["CD"]))),
                "CM_relative_change": float(abs(latest["CM"] - previous["CM"]) / max(0.01, abs(latest["CM"]))),
            }
    (arguments.out / "diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
