"""Close Cartesian cut cells around a clipped closed triangle surface.

Cartesian aperture areas are recovered from the oriented intersection curves
on each grid face.  The same aperture record is shared by both adjacent cells.
Cut-cell volume and centroid then follow from boundary integrals.  This stage
does not assemble regular octree faces or a complete solver mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


FACE_AXES = ((1, 2), (2, 0), (0, 1))
FACE_COORDINATE_BITS = 20
FACE_COORDINATE_MASK = (1 << FACE_COORDINATE_BITS) - 1


def encode_face(axis, plane, first, second):
    return ((np.asarray(axis, dtype=np.uint64) << np.uint64(60)) |
            (np.asarray(plane, dtype=np.uint64) << np.uint64(40)) |
            (np.asarray(first, dtype=np.uint64) << np.uint64(20)) |
            np.asarray(second, dtype=np.uint64))


def decode_faces(keys):
    keys = np.asarray(keys, dtype=np.uint64)
    return np.column_stack((
        (keys >> np.uint64(60)) & np.uint64(3),
        (keys >> np.uint64(40)) & np.uint64(FACE_COORDINATE_MASK),
        (keys >> np.uint64(20)) & np.uint64(FACE_COORDINATE_MASK),
        keys & np.uint64(FACE_COORDINATE_MASK),
    )).astype(np.int32)


def cell_coordinates(cell_ids, block_coordinates, block_levels,
                     maximum_level, block_cells):
    """Return finest-lattice cell origins and spans."""
    cells_per_block = block_cells ** 3
    blocks = np.asarray(cell_ids, dtype=np.int64) // cells_per_block
    local = np.asarray(cell_ids, dtype=np.int64) % cells_per_block
    local_coordinates = np.column_stack((
        local % block_cells,
        (local // block_cells) % block_cells,
        local // (block_cells ** 2),
    ))
    span = (1 << (maximum_level - block_levels[blocks])).astype(np.int32)
    origins = (block_coordinates[blocks] * block_cells +
               local_coordinates * span[:, None]).astype(np.int32)
    return origins, span


def procedural_cells_at_fine_coordinates(coordinates, owner, block_coordinates,
                                         block_levels, maximum_level,
                                         block_cells):
    coordinates = np.asarray(coordinates, dtype=np.int64)
    dimension = owner.shape[0] * block_cells
    result = np.full(len(coordinates), -1, dtype=np.int32)
    valid = np.all((coordinates >= 0) & (coordinates < dimension), axis=1)
    selected = coordinates[valid]
    blocks = np.asarray(owner[tuple((selected // block_cells).T)], dtype=np.int64)
    span = (1 << (maximum_level - block_levels[blocks])).astype(np.int64)
    origins = block_coordinates[blocks].astype(np.int64) * block_cells
    local = (selected - origins) // span[:, None]
    local_id = ((local[:, 2] * block_cells + local[:, 1]) * block_cells +
                local[:, 0])
    result[valid] = blocks * (block_cells ** 3) + local_id
    return result


def create_cut_faces(cut_cells, coordinates, spans):
    """Split every cut-cell boundary into maximum-level atomic subfaces."""
    count = len(cut_cells)
    counts = 6 * spans.astype(np.int64) ** 2
    offsets = np.empty(count + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    owners = np.repeat(np.arange(count, dtype=np.int32), counts)
    local = np.arange(int(offsets[-1]), dtype=np.int64) - np.repeat(
        offsets[:-1].astype(np.int64), counts
    )
    owner_spans = spans[owners].astype(np.int64)
    face_size = owner_spans ** 2
    sides = (local // face_size).astype(np.uint8)
    subface = local % face_size
    first_offset = subface % owner_spans
    second_offset = subface // owner_spans
    owner_coordinates = coordinates[owners]
    keys = np.empty(len(owners), dtype=np.uint64)
    for axis, (first_axis, second_axis) in enumerate(FACE_AXES):
        for positive in (0, 1):
            side = 2 * axis + positive
            selected = sides == side
            keys[selected] = encode_face(
                axis,
                owner_coordinates[selected, axis] + positive * owner_spans[selected],
                owner_coordinates[selected, first_axis] + first_offset[selected],
                owner_coordinates[selected, second_axis] + second_offset[selected],
            )
    unique, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.astype(np.int32)
    negative_cut = np.full(len(unique), -1, dtype=np.int32)
    positive_cut = np.full(len(unique), -1, dtype=np.int32)
    negative_side = (sides & 1) == 0
    positive_cut[inverse[negative_side]] = owners[negative_side]
    negative_cut[inverse[~negative_side]] = owners[~negative_side]
    return unique, offsets, inverse, negative_cut, positive_cut


def extract_face_segments(fragment_offsets, fragment_points, fragment_cells,
                          fragment_centroids, fragment_area_vectors,
                          block_coordinates, block_levels, maximum_level,
                          block_cells, root_low, cell_size, face_keys):
    counts = np.diff(fragment_offsets).astype(np.int64)
    fragment_of_edge = np.repeat(
        np.arange(len(fragment_cells), dtype=np.int32), counts
    )
    point_ids = np.arange(len(fragment_points), dtype=np.int64)
    next_ids = point_ids + 1
    next_ids[fragment_offsets[1:] - 1] = fragment_offsets[:-1]
    first_points = np.asarray(fragment_points, dtype=np.float64)
    second_points = np.asarray(fragment_points[next_ids], dtype=np.float64)
    fragment_coordinates, fragment_spans = cell_coordinates(
        fragment_cells, block_coordinates, block_levels, maximum_level,
        block_cells
    )
    edge_coordinates = fragment_coordinates[fragment_of_edge]
    edge_spans = fragment_spans[fragment_of_edge]
    fragment_voxels = np.floor(
        (np.asarray(fragment_centroids) - root_low) / cell_size
    ).astype(np.int32)
    edge_voxels = fragment_voxels[fragment_of_edge]
    tolerance = 2e-10 * cell_size
    key_parts = []
    endpoint_parts = []
    normal_parts = []
    cell_parts = []
    fragment_parts = []
    for axis, (first_axis, second_axis) in enumerate(FACE_AXES):
        first_plane = np.rint(
            (first_points[:, axis] - root_low[axis]) / cell_size
        ).astype(np.int32)
        second_plane = np.rint(
            (second_points[:, axis] - root_low[axis]) / cell_size
        ).astype(np.int32)
        on_plane = ((np.abs(first_points[:, axis] -
                            (root_low[axis] + first_plane * cell_size)) <= tolerance) &
                    (np.abs(second_points[:, axis] -
                            (root_low[axis] + second_plane * cell_size)) <= tolerance) &
                    (first_plane == second_plane))
        for side in (0, 1):
            plane_index = edge_coordinates[:, axis] + side * edge_spans
            selected = on_plane & (first_plane == plane_index)
            selected &= np.linalg.norm(
                second_points - first_points, axis=1
            ) > 1e-13 * cell_size
            if not np.any(selected):
                continue
            coordinates = edge_voxels[selected]
            planes = first_plane[selected]
            keys = encode_face(
                axis, planes,
                coordinates[:, first_axis], coordinates[:, second_axis]
            )
            locations = np.searchsorted(face_keys, keys)
            if (np.any(locations == len(face_keys)) or
                    np.any(face_keys[locations] != keys)):
                raise ValueError("an interface edge references an unknown cut-cell face")
            origin_first = root_low[first_axis] + coordinates[:, first_axis] * cell_size
            origin_second = root_low[second_axis] + coordinates[:, second_axis] * cell_size
            p = first_points[selected]
            q = second_points[selected]
            endpoints = np.column_stack((
                (p[:, first_axis] - origin_first) / cell_size,
                (p[:, second_axis] - origin_second) / cell_size,
                (q[:, first_axis] - origin_first) / cell_size,
                (q[:, second_axis] - origin_second) / cell_size,
            ))
            normals = fragment_area_vectors[fragment_of_edge[selected]][
                :, (first_axis, second_axis)
            ]
            key_parts.append(locations.astype(np.int32))
            endpoint_parts.append(endpoints)
            normal_parts.append(normals)
            cell_parts.append(fragment_cells[fragment_of_edge[selected]])
            fragment_parts.append(fragment_of_edge[selected])
    face_ids = np.concatenate(key_parts)
    endpoints = np.concatenate(endpoint_parts)
    projected_normals = np.concatenate(normal_parts)
    segment_cells = np.concatenate(cell_parts)
    segment_fragments = np.concatenate(fragment_parts)
    order = np.argsort(face_ids, kind="stable")
    face_ids = face_ids[order]
    endpoints = endpoints[order]
    projected_normals = projected_normals[order]
    segment_cells = segment_cells[order]
    segment_fragments = segment_fragments[order]
    segment_counts = np.bincount(face_ids, minlength=len(face_keys)).astype(np.uint64)
    segment_offsets = np.empty(len(face_keys) + 1, dtype=np.uint64)
    segment_offsets[0] = 0
    np.cumsum(segment_counts, out=segment_offsets[1:])
    return (segment_offsets, endpoints, projected_normals, segment_cells,
            segment_fragments)


def canonical_segment_key(segment, scale=10**6):
    values = np.rint(np.clip(segment, 0.0, 1.0) * scale).astype(np.int64)
    first = (values[0], values[1])
    second = (values[2], values[3])
    return (*first, *second) if first <= second else (*second, *first)


def oriented_unique_segments(endpoints, normals, cells):
    groups = {}
    tangent = 0
    for index, segment in enumerate(endpoints):
        key = canonical_segment_key(segment)
        groups.setdefault(key, []).append(index)
    unique = []
    for indices in groups.values():
        # Two copies from the same Cartesian side are an internal surface-mesh
        # edge lying on the grid face. Copies contributed from opposite sides
        # represent the true surface/grid intersection and are retained once.
        contributing_cells = np.unique(cells[indices])
        if len(indices) % 2 == 0 and len(contributing_cells) == 1:
            continue
        index = indices[0]
        segment = endpoints[index]
        normal = np.mean(normals[indices], axis=0)
        p = np.asarray(segment[:2], dtype=np.float64)
        q = np.asarray(segment[2:], dtype=np.float64)
        tangent_vector = q - p
        left = np.array((-tangent_vector[1], tangent_vector[0]))
        score = float(np.dot(normal, left))
        if abs(score) <= 1e-16 * max(np.linalg.norm(normal), 1e-300):
            tangent += 1
            continue
        if score < 0:
            p, q = q, p
        unique.append((p, q))
    return unique, tangent


def distance_to_segment_squared(point, start, end):
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator == 0:
        return float(np.dot(point - start, point - start))
    fraction = np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0)
    nearest = start + fraction * delta
    return float(np.dot(point - nearest, point - nearest))


def left_of_nearest_segment(point, segments):
    best = None
    best_distance = np.inf
    for start, end in segments:
        distance = distance_to_segment_squared(point, start, end)
        if distance < best_distance:
            best_distance = distance
            best = (start, end)
    start, end = best
    delta = end - start
    return delta[0] * (point[1] - start[1]) - delta[1] * (point[0] - start[0]) > 0


def perimeter_parameter(point, tolerance=2e-9):
    u, v = point
    if abs(v) <= tolerance:
        return float(np.clip(u, 0.0, 1.0))
    if abs(u - 1.0) <= tolerance:
        return 1.0 + float(np.clip(v, 0.0, 1.0))
    if abs(v - 1.0) <= tolerance:
        return 2.0 + 1.0 - float(np.clip(u, 0.0, 1.0))
    if abs(u) <= tolerance:
        return 3.0 + 1.0 - float(np.clip(v, 0.0, 1.0))
    return None


def perimeter_point(parameter):
    parameter = float(parameter % 4.0)
    if parameter <= 1.0:
        return np.array((parameter, 0.0))
    if parameter <= 2.0:
        return np.array((1.0, parameter - 1.0))
    if parameter <= 3.0:
        return np.array((3.0 - parameter, 1.0))
    return np.array((0.0, 4.0 - parameter))


def perimeter_segments(segments):
    # Each oriented intersection segment keeps fluid on its left. Therefore a
    # segment ending on the counterclockwise square perimeter starts a fluid
    # perimeter interval, while a segment starting there ends that interval.
    # This directed topology selects the closure without a fragile nearest-
    # triangle point classification.
    event_balance = {}
    for start, end in segments:
        for point, delta in ((start, -1), (end, 1)):
            parameter = perimeter_parameter(point)
            if parameter is not None:
                key = int(round((parameter % 4.0) * 1e9))
                event_balance[key] = event_balance.get(key, 0) + delta
    events = sorted(
        (key / 1e9, balance) for key, balance in event_balance.items()
        if balance
    )
    if not events:
        return []
    if len(events) % 2 or any(abs(balance) != 1 for _, balance in events):
        raise ValueError("an aperture curve has inconsistent directed crossings")
    for index, (_, balance) in enumerate(events):
        if events[(index + 1) % len(events)][1] != -balance:
            raise ValueError("aperture perimeter crossings do not alternate")

    result = []
    for index, (start, balance) in enumerate(events):
        if balance < 0:
            continue
        end = events[(index + 1) % len(events)][0]
        if end <= start:
            end += 4.0
        boundaries = [start]
        boundaries.extend(
            float(corner) for corner in range(int(np.floor(start)) + 1,
                                               int(np.ceil(end)))
            if start < corner < end
        )
        boundaries.append(end)
        for first, second in zip(boundaries[:-1], boundaries[1:]):
            if second > first + 1e-13:
                result.append((perimeter_point(first), perimeter_point(second)))
    return result


def aperture_from_segments(segments):
    boundary = list(segments) + perimeter_segments(segments)
    twice_area = 0.0
    first_u = 0.0
    first_v = 0.0
    for start, end in boundary:
        cross_value = start[0] * end[1] - start[1] * end[0]
        twice_area += cross_value
        first_u += (start[0] + end[0]) * cross_value
        first_v += (start[1] + end[1]) * cross_value
    # Two valid closures of open intersection curves differ by the complete
    # counterclockwise square boundary. Normalize the oriented result to the
    # physical [0,1] aperture while preserving its first moments.
    if twice_area < -4e-9:
        twice_area += 2.0
        first_u += 3.0
        first_v += 3.0
    elif twice_area > 2.0 + 4e-9:
        twice_area -= 2.0
        first_u -= 3.0
        first_v -= 3.0
    area = 0.5 * twice_area
    if area < -2e-9 or area > 1.0 + 2e-9:
        raise ValueError(f"invalid Cartesian aperture fraction {area}")
    area = float(np.clip(area, 0.0, 1.0))
    if area <= 1e-14:
        return 0.0, np.array((0.5, 0.5))
    centroid = np.array((first_u, first_v)) / (3.0 * twice_area)
    return area, centroid


def closest_point_on_triangle(point, a, b, c):
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = point - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = point - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denominator = 1.0 / (va + vb + vc)
    return a + (vb * denominator) * ab + (vc * denominator) * ac


def point_outside_local_surface(point, fragment_ids, fragment_triangles,
                                surface_points, surface_triangles):
    triangle_ids = np.unique(fragment_triangles[fragment_ids])
    best_distance = np.inf
    best_sign = None
    for triangle in triangle_ids:
        a, b, c = surface_points[surface_triangles[triangle]]
        nearest = closest_point_on_triangle(point, a, b, c)
        delta = point - nearest
        distance = float(np.dot(delta, delta))
        if distance < best_distance:
            normal = np.cross(b - a, c - a)
            best_distance = distance
            best_sign = float(np.dot(delta, normal))
    if best_sign is None or abs(best_sign) < 1e-24:
        raise ValueError("cannot classify a Cartesian aperture face")
    return best_sign > 0


def fragment_axis_second_moments(offsets, points, area_vectors):
    result = np.zeros_like(area_vectors)
    counts = np.diff(offsets).astype(np.int64)
    for count in np.unique(counts):
        fragment_ids = np.flatnonzero(counts == count)
        for start in range(0, len(fragment_ids), 200_000):
            selected = fragment_ids[start:start + 200_000]
            point_ids = offsets[selected, None].astype(np.int64) + np.arange(count)
            polygon = np.asarray(points[point_ids])
            a = polygon[:, 0]
            integrals = np.zeros((len(selected), 3), dtype=np.float64)
            for index in range(1, int(count) - 1):
                b, c = polygon[:, index], polygon[:, index + 1]
                triangle_area = 0.5 * np.linalg.norm(
                    np.cross(b - a, c - a), axis=1
                )
                integrals += triangle_area[:, None] * (
                    a * a + b * b + c * c + a * b + b * c + c * a
                ) / 6.0
            area = np.linalg.norm(area_vectors[selected], axis=1)
            result[selected] = area_vectors[selected] / area[:, None] * integrals
    return result


def save_array(destination, name, values, description, tables):
    path = destination / f"{name}.npy"
    np.save(path, values, allow_pickle=False)
    tables[name] = {
        "file": path.name, "dtype": values.dtype.str,
        "shape": list(values.shape), "bytes": path.stat().st_size,
        "description": description,
    }


def build_finest_owner(block_coordinates, block_levels, maximum_level):
    """Map each finest block lattice point to its balanced-octree leaf."""
    dimension = 1 << maximum_level
    owner = np.full((dimension, dimension, dimension), -1, dtype=np.int32)
    for block, (coordinate, level) in enumerate(zip(block_coordinates,
                                                     block_levels)):
        span = 1 << (maximum_level - int(level))
        i, j, k = map(int, coordinate)
        region = owner[i:i + span, j:j + span, k:k + span]
        if np.any(region >= 0):
            raise ValueError("octree leaf blocks overlap")
        region[...] = block
    if np.any(owner < 0):
        raise ValueError("octree leaf blocks do not cover the root")
    return owner


def procedural_cell_geometry(cell_ids, block_low, block_size, block_cells):
    """Return center and uncut volume for procedural block-local cell IDs."""
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    cells_per_block = block_cells ** 3
    blocks = cell_ids // cells_per_block
    local = cell_ids % cells_per_block
    local_coordinates = np.column_stack((
        local % block_cells,
        (local // block_cells) % block_cells,
        local // (block_cells ** 2),
    ))
    spacing = block_size[blocks] / block_cells
    centers = block_low[blocks] + (local_coordinates + 0.5) * spacing[:, None]
    return centers, spacing ** 3, spacing


def build_agglomeration_plan(cut_cells, cut_face_offsets, cut_cell_faces,
                             negative_cells,
                             positive_cells, face_area, status, cut_volume,
                             cut_first_moment, cut_closure, cut_surface_measure,
                             block_low, block_size, block_cells,
                             threshold_fraction=0.5):
    """Agglomerate sub-threshold cut cells through their widest open face.

    This is the three-dimensional counterpart of the production 2-D rule:
    connected control volumes are merged until their combined volume reaches
    half the largest uncut member volume.  Only cut-cell faces are required,
    because absorbing one complete Cartesian neighbour immediately satisfies
    the threshold.  The returned groups are geometric plans; the next stage
    still has to remove their internal faces from the global AMR face table.
    """
    cut_cells = np.asarray(cut_cells, dtype=np.int64)
    faces = np.asarray(cut_cell_faces, dtype=np.int64)
    incidence_counts = np.diff(cut_face_offsets).astype(np.int64)
    owner_cut_indices = np.repeat(
        np.arange(len(cut_cells), dtype=np.int32), incidence_counts
    )
    owners = cut_cells[owner_cut_indices]
    negative = np.asarray(negative_cells[faces], dtype=np.int64)
    positive = np.asarray(positive_cells[faces], dtype=np.int64)
    owner_on_negative = negative == owners
    owner_on_positive = positive == owners
    if np.any(~(owner_on_negative | owner_on_positive)):
        raise ValueError("a cut-cell face does not contain its procedural owner")
    neighbours = np.where(owner_on_negative, positive, negative)
    areas = np.asarray(face_area[faces], dtype=np.float64)
    active_neighbour = neighbours >= 0
    active_neighbour &= status[np.maximum(neighbours, 0)] != 1
    open_face = areas > 1e-15 * max(float(np.max(areas)), 1e-300)
    valid = active_neighbour & open_face
    has_open = np.zeros(len(cut_cells), dtype=bool)
    has_open[owner_cut_indices[valid]] = True
    _, cut_uncut_volume, _ = procedural_cell_geometry(
        cut_cells, block_low, block_size, block_cells
    )
    fallback_cells = ((cut_volume < 1e-6 * cut_uncut_volume) & ~has_open)
    fallback = active_neighbour & fallback_cells[owner_cut_indices]
    valid |= fallback
    edge_owner = owners[valid]
    edge_neighbour = neighbours[valid]
    edge_area = areas[valid]

    nodes = np.unique(np.concatenate((cut_cells, edge_neighbour)))
    edge_first = np.searchsorted(nodes, edge_owner)
    edge_second = np.searchsorted(nodes, edge_neighbour)
    if np.any(nodes[edge_first] != edge_owner) or np.any(nodes[edge_second] != edge_neighbour):
        raise ValueError("agglomeration edge mapping failed")

    centers, uncut_volume, spacing = procedural_cell_geometry(
        nodes, block_low, block_size, block_cells
    )
    node_volume = uncut_volume.copy()
    node_moment = node_volume[:, None] * centers
    node_closure = np.zeros((len(nodes), 3), dtype=np.float64)
    node_surface_measure = 6.0 * spacing ** 2
    cut_locations = np.searchsorted(nodes, cut_cells)
    if np.any(nodes[cut_locations] != cut_cells):
        raise ValueError("cut-cell agglomeration mapping failed")
    node_volume[cut_locations] = cut_volume
    node_moment[cut_locations] = cut_first_moment
    node_closure[cut_locations] = cut_closure
    node_surface_measure[cut_locations] = cut_surface_measure

    parent = np.arange(len(nodes), dtype=np.int64)
    group_volume = node_volume.copy()
    group_target = threshold_fraction * uncut_volume
    group_moment = node_moment.copy()
    group_closure = node_closure.copy()
    group_surface_measure = node_surface_measure.copy()

    def find_one(index):
        index = int(index)
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def roots_all():
        roots = np.arange(len(parent), dtype=np.int64)
        while True:
            updated = parent[roots]
            if np.array_equal(updated, roots):
                return roots
            roots = updated

    merge_iterations = 0
    merge_count = 0
    for merge_iterations in range(1, 65):
        roots = roots_all()
        parent[:] = roots
        root_mask = roots == np.arange(len(nodes))
        deficient = root_mask & (group_volume < group_target * (1.0 - 1e-13))
        if not np.any(deficient):
            break
        edge_root_first = roots[edge_first]
        edge_root_second = roots[edge_second]
        cross = edge_root_first != edge_root_second
        candidate_group = np.concatenate((edge_root_first[cross], edge_root_second[cross]))
        candidate_other = np.concatenate((edge_root_second[cross], edge_root_first[cross]))
        candidate_area = np.concatenate((edge_area[cross], edge_area[cross]))
        candidate_valid = deficient[candidate_group]
        candidate_group = candidate_group[candidate_valid]
        candidate_other = candidate_other[candidate_valid]
        candidate_area = candidate_area[candidate_valid]
        if len(candidate_group) == 0:
            missing = np.flatnonzero(deficient)[:12]
            raise ValueError(
                f"small cut-cell groups have no open neighbour: {nodes[missing].tolist()}"
            )
        order = np.lexsort((-candidate_area, candidate_group))
        sorted_group = candidate_group[order]
        first = np.r_[True, sorted_group[1:] != sorted_group[:-1]]
        chosen = order[first]
        # The most deficient groups merge first.  Re-resolve roots after each
        # union so simultaneous choices cannot create cycles.
        ratios = group_volume[candidate_group[chosen]] / group_target[candidate_group[chosen]]
        chosen = chosen[np.argsort(ratios, kind="stable")]
        progress = 0
        for choice in chosen:
            root = find_one(candidate_group[choice])
            if group_volume[root] >= group_target[root] * (1.0 - 1e-13):
                continue
            other = find_one(candidate_other[choice])
            if root == other:
                continue
            parent[root] = other
            group_volume[other] += group_volume[root]
            group_target[other] = max(group_target[other], group_target[root])
            group_moment[other] += group_moment[root]
            group_closure[other] += group_closure[root]
            group_surface_measure[other] += group_surface_measure[root]
            progress += 1
        merge_count += progress
        if progress == 0:
            missing = np.flatnonzero(deficient)[:12]
            raise ValueError(
                f"small cut-cell agglomeration stalled: {nodes[missing].tolist()}"
            )
    else:
        raise ValueError("small cut-cell agglomeration exceeded 64 passes")

    roots = roots_all()
    parent[:] = roots
    counts = np.bincount(roots, minlength=len(nodes))
    nontrivial_roots = np.flatnonzero(counts > 1)
    minimum_member = np.full(len(nodes), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(minimum_member, roots, nodes)
    nontrivial_roots = nontrivial_roots[np.argsort(minimum_member[nontrivial_roots])]
    root_to_group = np.full(len(nodes), -1, dtype=np.int32)
    root_to_group[nontrivial_roots] = np.arange(len(nontrivial_roots), dtype=np.int32)
    node_group = root_to_group[roots]
    member_mask = node_group >= 0
    member_order = np.lexsort((nodes[member_mask], node_group[member_mask]))
    member_cells = nodes[member_mask][member_order]
    member_groups = node_group[member_mask][member_order]
    member_counts = np.bincount(member_groups, minlength=len(nontrivial_roots))
    member_offsets = np.empty(len(nontrivial_roots) + 1, dtype=np.uint64)
    member_offsets[0] = 0
    np.cumsum(member_counts, out=member_offsets[1:])
    cut_group = node_group[cut_locations]
    if np.any((cut_volume < threshold_fraction * uncut_volume[cut_locations]) &
              (cut_group < 0)):
        raise ValueError("a sub-threshold cut cell remained unagglomerated")

    volumes = group_volume[nontrivial_roots]
    targets = group_target[nontrivial_roots]
    centroids = group_moment[nontrivial_roots] / volumes[:, None]
    closure_relative = np.linalg.norm(group_closure[nontrivial_roots], axis=1) / np.maximum(
        group_surface_measure[nontrivial_roots], 1e-300
    )
    return {
        "nodes": nodes,
        "cut_group": cut_group,
        "member_offsets": member_offsets,
        "member_cells": member_cells,
        "volume": volumes,
        "target": targets,
        "centroid": centroids,
        "closure_relative": closure_relative,
        "merge_iterations": merge_iterations,
        "merge_count": merge_count,
        "edge_count": len(edge_area),
        "roundoff_slivers_with_zero_open_area": int(np.count_nonzero(fallback_cells)),
        "roundoff_fallback_edges": int(np.count_nonzero(fallback)),
        "small_cut_count": int(np.count_nonzero(
            cut_volume < threshold_fraction * uncut_volume[cut_locations]
        )),
        "original_active_cells": int(np.count_nonzero(status != 1)),
        "agglomerated_active_cells": int(
            np.count_nonzero(status != 1) - len(member_cells) + len(nontrivial_roots)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--column-plan", required=True)
    parser.add_argument("--octree-plan", required=True)
    parser.add_argument("--nnlci-tables")
    parser.add_argument("--out", required=True)
    parser.add_argument("--surface-layer", choices=("wall", "outer"),
                        default="outer")
    args = parser.parse_args()
    start = time.perf_counter()
    interface = Path(args.interface)
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)

    fragment_offsets = np.load(interface / "interface_fragment_offsets.npy", mmap_mode="r")
    fragment_points = np.load(interface / "interface_fragment_points.npy", mmap_mode="r")
    fragment_cells = np.load(interface / "interface_fragment_cartesian_cell.npy", mmap_mode="r")
    fragment_triangles = np.load(interface / "interface_fragment_triangle.npy", mmap_mode="r")
    fragment_centroids = np.load(interface / "interface_fragment_centroid.npy", mmap_mode="r")
    fragment_area_vectors = np.load(interface / "interface_fragment_area_vector.npy", mmap_mode="r")
    cut_cells = np.load(interface / "cut_cartesian_cells.npy", mmap_mode="r")
    cut_fragment_offsets = np.load(interface / "cut_cell_fragment_offsets.npy", mmap_mode="r")
    cut_fragments = np.load(interface / "cut_cell_fragments.npy", mmap_mode="r")
    status = np.load(interface / "cartesian_cell_status.npy", mmap_mode="r")
    cut_fine_status_offsets = np.load(
        interface / "cut_cell_fine_status_offsets.npy", mmap_mode="r"
    )
    cut_fine_status = np.load(
        interface / "cut_cell_fine_status.npy", mmap_mode="r"
    )

    with np.load(args.column_plan) as plan:
        wall_points = np.asarray(plan["points"], dtype=np.float64)
        surface_triangles = np.asarray(plan["triangles"], dtype=np.int32)
        surface_points = wall_points if args.surface_layer == "wall" else (
            wall_points + (
                float(plan["offsets"][-1]) *
                np.asarray(plan["column_scale"], dtype=np.float64)
            )[:, None] * np.asarray(plan["normal"], dtype=np.float64)
        )
    with np.load(args.octree_plan) as octree:
        block_low = np.asarray(octree["low"], dtype=np.float64)
        block_size = np.asarray(octree["size"], dtype=np.float64)
        block_level = np.asarray(octree["level"], dtype=np.int16)
        root_low = np.asarray(octree["root_low"], dtype=np.float64)
        root_size = float(octree["root_size"])
        block_cells = int(octree["block_cells"])
    maximum_level = int(block_level.max())
    block_dimension = 1 << maximum_level
    cell_size = root_size / (block_dimension * block_cells)
    block_coordinates = np.rint(
        (block_low - root_low) * block_dimension / root_size
    ).astype(np.int32)
    owner = (np.load(Path(args.nnlci_tables) / "octree_finest_owner.npy",
                     mmap_mode="r") if args.nnlci_tables else
             build_finest_owner(block_coordinates, block_level, maximum_level))

    cut_coordinates, cut_spans = cell_coordinates(
        cut_cells, block_coordinates, block_level, maximum_level, block_cells
    )
    (face_keys, cut_face_offsets, cut_cell_faces,
     negative_cut, positive_cut) = create_cut_faces(
        cut_cells, cut_coordinates, cut_spans
    )
    face_records = decode_faces(face_keys)
    negative_coordinates = np.empty((len(face_keys), 3), dtype=np.int32)
    positive_coordinates = np.empty_like(negative_coordinates)
    for axis, (first_axis, second_axis) in enumerate(FACE_AXES):
        selected = face_records[:, 0] == axis
        plane = face_records[selected, 1]
        first = face_records[selected, 2]
        second = face_records[selected, 3]
        negative_coordinates[selected, axis] = plane - 1
        positive_coordinates[selected, axis] = plane
        negative_coordinates[selected, first_axis] = first
        positive_coordinates[selected, first_axis] = first
        negative_coordinates[selected, second_axis] = second
        positive_coordinates[selected, second_axis] = second
    negative_cells = procedural_cells_at_fine_coordinates(
        negative_coordinates, owner, block_coordinates, block_level,
        maximum_level, block_cells
    )
    positive_cells = procedural_cells_at_fine_coordinates(
        positive_coordinates, owner, block_coordinates, block_level,
        maximum_level, block_cells
    )
    def atomic_status(coordinates, cells):
        result = np.where(cells >= 0, status[np.maximum(cells, 0)], 0).astype(np.uint8)
        selected = (cells >= 0) & (result == 2)
        if not np.any(selected):
            return result
        selected_cells = cells[selected]
        cut_indices = np.searchsorted(cut_cells, selected_cells)
        if (np.any(cut_indices == len(cut_cells)) or
                np.any(cut_cells[cut_indices] != selected_cells)):
            raise ValueError("surface-intersected procedural cell is absent from cut table")
        spans = cut_spans[cut_indices].astype(np.int64)
        local = coordinates[selected].astype(np.int64) - cut_coordinates[cut_indices]
        if np.any(local < 0) or np.any(local >= spans[:, None]):
            raise ValueError("atomic coordinate lies outside its procedural cut cell")
        local_ids = ((local[:, 2] * spans + local[:, 1]) * spans + local[:, 0])
        result[selected] = cut_fine_status[
            cut_fine_status_offsets[cut_indices].astype(np.int64) + local_ids
        ]
        return result

    negative_status = atomic_status(negative_coordinates, negative_cells)
    positive_status = atomic_status(positive_coordinates, positive_cells)

    (segment_offsets, segment_endpoints, segment_normals, segment_cells,
     segment_fragments) = extract_face_segments(
        fragment_offsets, fragment_points, fragment_cells,
        fragment_centroids, fragment_area_vectors, block_coordinates,
        block_level, maximum_level, block_cells, root_low, cell_size, face_keys
    )
    face_area_fraction = np.zeros(len(face_keys), dtype=np.float64)
    face_centroid_uv = np.full((len(face_keys), 2), 0.5, dtype=np.float64)
    tangent_segments = 0
    ambiguous_full_faces = 0
    negative_cut_only = (negative_status == 2) & (positive_status != 2)
    positive_cut_only = (positive_status == 2) & (negative_status != 2)
    neither_cut = (negative_status != 2) & (positive_status != 2)
    inconsistent = neither_cut & (negative_status != positive_status)
    if np.any(inconsistent):
        bad = np.flatnonzero(inconsistent)[:12]
        raise ValueError(
            f"atomic faces cross material without surface fragments: {bad.tolist()}"
        )
    face_area_fraction[negative_cut_only] = (
        positive_status[negative_cut_only] == 0
    )
    face_area_fraction[positive_cut_only] = (
        negative_status[positive_cut_only] == 0
    )
    face_area_fraction[neither_cut] = (negative_status[neither_cut] == 0)
    both_cut = np.flatnonzero((negative_status == 2) & (positive_status == 2))
    for face in both_cut:
        negative = int(negative_status[face])
        positive = int(positive_status[face])
        first = int(segment_offsets[face])
        last = int(segment_offsets[face + 1])
        if negative == 2 and positive == 2:
            fragment_groups = []
            for cut_index in (negative_cut[face], positive_cut[face]):
                if cut_index >= 0:
                    fragment_groups.append(cut_fragments[
                        cut_fragment_offsets[cut_index]:
                        cut_fragment_offsets[cut_index + 1]
                    ])
            local_fragments = np.unique(np.concatenate(fragment_groups))
            axis, plane, first_coordinate, second_coordinate = face_records[face]
            first_axis, second_axis = FACE_AXES[axis]

            def outside_at(uv):
                point = np.empty(3)
                point[axis] = root_low[axis] + plane * cell_size
                point[first_axis] = (
                    root_low[first_axis] + (first_coordinate + uv[0]) * cell_size
                )
                point[second_axis] = (
                    root_low[second_axis] + (second_coordinate + uv[1]) * cell_size
                )
                return point_outside_local_surface(
                    point, local_fragments, fragment_triangles,
                    surface_points, surface_triangles
                )

            segments, tangent = oriented_unique_segments(
                segment_endpoints[first:last], segment_normals[first:last],
                segment_cells[first:last]
            )
            tangent_segments += tangent
            if segments:
                try:
                    area, centroid = aperture_from_segments(segments)
                except ValueError as error:
                    raise ValueError(
                        f"face {face}, key {face_records[face].tolist()}, "
                        f"raw segments {last - first}, unique segments {segments}"
                    ) from error
                face_area_fraction[face] = area
                face_centroid_uv[face] = centroid
            else:
                outside = outside_at(np.array((0.5, 0.5)))
                face_area_fraction[face] = float(outside)
                ambiguous_full_faces += 1
        else:
            raise ValueError("invalid atomic cut-face classification")
    partial_faces = int(np.count_nonzero(
        (face_area_fraction > 1e-12) & (face_area_fraction < 1 - 1e-12)
    ))
    full_faces = int(np.count_nonzero(face_area_fraction >= 1 - 1e-12))
    empty_faces = int(np.count_nonzero(face_area_fraction <= 1e-12))

    face_area = face_area_fraction * cell_size ** 2
    face_centroid = np.empty((len(face_keys), 3), dtype=np.float64)
    for axis, (first_axis, second_axis) in enumerate(FACE_AXES):
        selected = face_records[:, 0] == axis
        face_centroid[selected, axis] = (
            root_low[axis] + face_records[selected, 1] * cell_size
        )
        face_centroid[selected, first_axis] = root_low[first_axis] + (
            face_records[selected, 2] + face_centroid_uv[selected, 0]
        ) * cell_size
        face_centroid[selected, second_axis] = root_low[second_axis] + (
            face_records[selected, 3] + face_centroid_uv[selected, 1]
        ) * cell_size

    # Retain a restartable diagnostic checkpoint before the more expensive
    # cut-cell moment integration. It is intentionally not a validated output.
    np.save(destination / "_diagnostic_face_records.npy", face_records)
    np.save(destination / "_diagnostic_face_area_fraction.npy", face_area_fraction)
    np.save(destination / "_diagnostic_face_centroid.npy", face_centroid)
    np.save(destination / "_diagnostic_cut_cell_faces.npy", cut_cell_faces)
    np.save(destination / "_diagnostic_cut_cell_face_offsets.npy", cut_face_offsets)
    np.save(destination / "_diagnostic_negative_status.npy", negative_status)
    np.save(destination / "_diagnostic_positive_status.npy", positive_status)

    fragment_second = fragment_axis_second_moments(
        fragment_offsets, fragment_points, fragment_area_vectors
    )
    cut_count = len(cut_cells)
    cut_centroid = np.zeros((len(cut_cells), 3), dtype=np.float64)
    cut_first_moment = np.zeros((len(cut_cells), 3), dtype=np.float64)
    cut_closure = np.zeros((len(cut_cells), 3), dtype=np.float64)
    cut_surface_measure = np.zeros(len(cut_cells), dtype=np.float64)
    volume_flux = np.zeros(cut_count, dtype=np.float64)

    # Reduce the Cartesian boundary integrals through the cut-cell CSR table.
    # A coarse cut cell owns several finest-lattice apertures on each side, so
    # the old fixed-six-face loop was both slow and no longer sufficient.
    face_incidence_counts = np.diff(cut_face_offsets).astype(np.int64)
    incidence_cut = np.repeat(
        np.arange(cut_count, dtype=np.int32), face_incidence_counts
    )
    incidence_face = np.asarray(cut_cell_faces, dtype=np.int32)
    incidence_axis = np.asarray(face_records[incidence_face, 0], dtype=np.int8)
    incidence_plane = face_records[incidence_face, 1]
    owner_plane = cut_coordinates[incidence_cut, incidence_axis]
    negative_side = incidence_plane == owner_plane
    positive_side = incidence_plane == (
        owner_plane + cut_spans[incidence_cut]
    )
    if np.any(negative_side == positive_side):
        bad = np.flatnonzero(negative_side == positive_side)[:12]
        raise ValueError(
            "atomic apertures are not on their cut-cell boundaries: "
            f"{bad.tolist()}"
        )
    signed_area = np.where(negative_side, -1.0, 1.0) * face_area[incidence_face]
    cut_surface_measure += np.bincount(
        incidence_cut, weights=face_area[incidence_face], minlength=cut_count
    )
    for axis in range(3):
        selected = incidence_axis == axis
        owners = incidence_cut[selected]
        weights = signed_area[selected]
        coordinate = face_centroid[incidence_face[selected], axis]
        cut_closure[:, axis] += np.bincount(
            owners, weights=weights, minlength=cut_count
        )
        volume_flux += np.bincount(
            owners, weights=weights * coordinate, minlength=cut_count
        )
        cut_first_moment[:, axis] += np.bincount(
            owners, weights=0.5 * weights * coordinate ** 2,
            minlength=cut_count
        )

    # The clipped surface fragments complete each closed fluid boundary.  The
    # fragment table is already grouped by cut cell, so the same conservative
    # surface and volume moments can be reduced without a Python cell loop.
    fragment_incidence_counts = np.diff(cut_fragment_offsets).astype(np.int64)
    fragment_cut = np.repeat(
        np.arange(cut_count, dtype=np.int32), fragment_incidence_counts
    )
    ordered_fragment = np.asarray(cut_fragments, dtype=np.int32)
    ordered_vector = fragment_area_vectors[ordered_fragment]
    ordered_centroid = fragment_centroids[ordered_fragment]
    ordered_second = fragment_second[ordered_fragment]
    for axis in range(3):
        cut_closure[:, axis] -= np.bincount(
            fragment_cut, weights=ordered_vector[:, axis], minlength=cut_count
        )
        cut_first_moment[:, axis] -= 0.5 * np.bincount(
            fragment_cut, weights=ordered_second[:, axis], minlength=cut_count
        )
    volume_flux -= np.bincount(
        fragment_cut,
        weights=np.einsum("ij,ij->i", ordered_vector, ordered_centroid),
        minlength=cut_count
    )
    cut_surface_measure += np.bincount(
        fragment_cut, weights=np.linalg.norm(ordered_vector, axis=1),
        minlength=cut_count
    )

    cut_volume = volume_flux / 3.0
    positive_volume = cut_volume > 0
    cut_centroid[positive_volume] = (
        cut_first_moment[positive_volume] / cut_volume[positive_volume, None]
    )
    closure_relative = np.linalg.norm(cut_closure, axis=1) / np.maximum(
        cut_surface_measure, 1e-300
    )

    _, full_volume, _ = procedural_cell_geometry(
        cut_cells, block_low, block_size, block_cells
    )
    volume_fraction = cut_volume / full_volume
    roundoff_tolerance = 1e-6
    invalid_volume = (~np.isfinite(cut_volume) |
                      (volume_fraction < -roundoff_tolerance) |
                      (volume_fraction > 1 + roundoff_tolerance))
    stable_geometry = ((volume_fraction >= 1e-3) &
                       (volume_fraction <= 1 + roundoff_tolerance))
    maximum_closure = float(np.max(closure_relative))
    maximum_stable_closure = float(np.max(
        closure_relative[stable_geometry], initial=0.0
    ))
    if np.any(invalid_volume) or maximum_stable_closure > 2e-8:
        worst = np.argsort(closure_relative)[-12:][::-1]
        raise ValueError(
            f"cut-cell audit failed: invalid volumes={np.count_nonzero(invalid_volume)}, "
            f"closure counts >1e-8/>1e-5/>1e-2="
            f"{np.count_nonzero(closure_relative > 1e-8)}/"
            f"{np.count_nonzero(closure_relative > 1e-5)}/"
            f"{np.count_nonzero(closure_relative > 1e-2)}, "
            f"maximum closure={maximum_closure}, worst cut indices={worst.tolist()}, "
            f"worst procedural cells={np.asarray(cut_cells[worst]).tolist()}, "
            f"worst volume fractions={(cut_volume[worst] / full_volume[worst]).tolist()}"
        )

    agglomeration = build_agglomeration_plan(
        cut_cells, cut_face_offsets, cut_cell_faces, negative_cells,
        positive_cells, face_area,
        status, cut_volume, cut_first_moment, cut_closure,
        cut_surface_measure, block_low, block_size, block_cells,
        threshold_fraction=0.5
    )
    if (len(agglomeration["volume"]) and
            (np.any(agglomeration["volume"] <= 0) or
             np.any(agglomeration["volume"] <
                    agglomeration["target"] * (1 - 1e-12)))):
        raise ValueError("agglomerated control-volume threshold audit failed")
    maximum_group_closure = float(np.max(
        agglomeration["closure_relative"], initial=0.0
    ))
    if maximum_group_closure > 2e-8:
        raise ValueError(
            f"agglomerated area-vector closure audit failed: {maximum_group_closure}"
        )

    tables = {}
    save_array(destination, "cartesian_aperture_face_key", face_records,
               "Rows are axis, plane index, and two in-plane cell indices.", tables)
    save_array(destination, "cartesian_aperture_negative_cell", negative_cells,
               "Procedural Cartesian cell on the negative-axis side.", tables)
    save_array(destination, "cartesian_aperture_positive_cell", positive_cells,
               "Procedural Cartesian cell on the positive-axis side.", tables)
    save_array(destination, "cartesian_aperture_area", face_area,
               "Fluid aperture area shared by both adjacent Cartesian cells.", tables)
    save_array(destination, "cartesian_aperture_centroid", face_centroid,
               "Area centroid of each Cartesian aperture.", tables)
    save_array(destination, "cut_cell_faces", cut_cell_faces,
               "Atomic Cartesian aperture-face identifiers grouped per cut cell.", tables)
    save_array(destination, "cut_cell_face_offsets", cut_face_offsets,
               "CSR offsets into cut_cell_faces.", tables)
    save_array(destination, "cut_cell_volume", cut_volume,
               "Fluid volume from the closed boundary integral; sub-roundoff slivers are retained only as agglomeration members.", tables)
    save_array(destination, "cut_cell_centroid", cut_centroid,
               "First-moment centroid for positive-volume cut cells; agglomerated centroids are authoritative for slivers.", tables)
    save_array(destination, "cut_cell_first_moment", cut_first_moment,
               "Unnormalized volume first moment, including numerical slivers.", tables)
    save_array(destination, "cut_cell_closure_relative", closure_relative,
               "Relative area-vector closure error per cut cell.", tables)
    save_array(destination, "cut_cell_agglomeration_group",
               agglomeration["cut_group"],
               "Agglomeration group per cut cell, or -1 for a standalone cut cell.", tables)
    save_array(destination, "agglomeration_member_offsets",
               agglomeration["member_offsets"],
               "CSR offsets into agglomeration_member_procedural_cell.", tables)
    save_array(destination, "agglomeration_member_procedural_cell",
               agglomeration["member_cells"],
               "Procedural cut/full Cartesian cells in each conservative group.", tables)
    save_array(destination, "agglomeration_volume", agglomeration["volume"],
               "Combined fluid volume of each agglomerated control volume.", tables)
    save_array(destination, "agglomeration_target_volume",
               agglomeration["target"],
               "Minimum half-parent-cell volume required for each group.", tables)
    save_array(destination, "agglomeration_centroid",
               agglomeration["centroid"],
               "Volume first-moment centroid of each agglomerated group.", tables)
    save_array(destination, "agglomeration_closure_relative",
               agglomeration["closure_relative"],
               "Relative summed area-vector closure error of each group.", tables)
    direct = args.surface_layer == "wall"
    report = {
        "mesh_workflow": ("cartesian_amr_cutcell_3d" if direct
                          else "body_fitted_hybrid_3d"),
        "surface_layer": args.surface_layer,
        "status": "CUT_CELL_GEOMETRY_AGGLOMERATED_VALID",
        "solver_ready": False,
        "cut_cells": len(cut_cells),
        "unique_cartesian_faces": len(face_keys),
        "partial_aperture_faces": partial_faces,
        "full_aperture_faces": full_faces,
        "empty_aperture_faces": empty_faces,
        "cut_cut_faces_without_intersection_segments": ambiguous_full_faces,
        "ignored_tangent_segments": tangent_segments,
        "minimum_volume_fraction": float(np.min(cut_volume / full_volume)),
        "maximum_volume_fraction": float(np.max(cut_volume / full_volume)),
        "maximum_relative_area_vector_closure": maximum_closure,
        "stable_cut_cell_volume_fraction_floor": 1e-3,
        "maximum_stable_cut_cell_relative_area_vector_closure": maximum_stable_closure,
        "roundoff_volume_fraction_tolerance": roundoff_tolerance,
        "roundoff_nonpositive_cut_cells": int(np.count_nonzero(cut_volume <= 0)),
        "roundoff_overfull_cut_cells": int(np.count_nonzero(cut_volume > full_volume)),
        "total_cut_fluid_volume": float(np.sum(cut_volume)),
        "agglomeration": {
            "threshold_fraction_of_parent_cell": 0.5,
            "small_cut_cells": agglomeration["small_cut_count"],
            "groups": len(agglomeration["volume"]),
            "members": len(agglomeration["member_cells"]),
            "merges": agglomeration["merge_count"],
            "passes": agglomeration["merge_iterations"],
            "candidate_open_faces": agglomeration["edge_count"],
            "roundoff_slivers_with_zero_open_area": agglomeration[
                "roundoff_slivers_with_zero_open_area"
            ],
            "roundoff_fallback_edges": agglomeration["roundoff_fallback_edges"],
            "minimum_group_to_target_ratio": float(np.min(
                agglomeration["volume"] / agglomeration["target"], initial=np.inf
            )),
            "maximum_group_relative_area_vector_closure": maximum_group_closure,
            "original_active_cartesian_cells": agglomeration["original_active_cells"],
            "agglomerated_active_control_volumes": agglomeration["agglomerated_active_cells"],
        },
        "tables": tables,
        "total_table_bytes": int(sum(value["bytes"] for value in tables.values())),
        "total_seconds": time.perf_counter() - start,
        "remaining": [
            "remove agglomeration-internal faces while assembling the global table",
            "assemble regular and 2:1 octree faces into one global table",
            "audit global volume and face cancellation",
            "build GPU gradient and solver tables",
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
