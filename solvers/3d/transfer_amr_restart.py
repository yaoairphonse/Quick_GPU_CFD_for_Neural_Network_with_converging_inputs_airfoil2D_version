#!/usr/bin/env python3
"""Conservatively prolong a QCFDAM1 restart onto a nested refined mesh.

The source state is piecewise constant on source control volumes.  Ordinary
target controls inherit the state of the source control containing their
procedural representative.  For a target agglomeration group, every member's
exact fluid volume is integrated separately and the group receives the
volume-weighted source state.  Because the Cartesian octrees are nested and
share the same exact cut surface, this preserves the global conserved
integrals to floating-point summation error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np


HEADER_FORMAT = "<8sIIII7Q4d"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)
BLOCK_DTYPE = np.dtype([
    ("low", "<f8", (3,)), ("spacing", "<f8"),
    ("fine_coordinate", "<i4", (3,)), ("fine_span", "<i4"),
    ("first_procedural", "<i4"), ("level", "<i2"),
    ("kind", "u1"), ("padding", "u1"),
])
CONTROL_DTYPE = np.dtype([
    ("x", "<f8", (3,)), ("volume", "<f8"), ("dwall", "<f8"),
])


def read_header(root: Path):
    raw = (root / "header.bin").read_bytes()
    if len(raw) != HEADER_BYTES:
        raise ValueError(f"invalid QCFDAM1 header size in {root}")
    values = struct.unpack(HEADER_FORMAT, raw)
    if values[0] != b"QCFDAM1\0" or values[1] != 1 or values[2] != 0x01020304:
        raise ValueError(f"invalid QCFDAM1 header in {root}")
    return {
        "block_cells": values[3],
        "blocks": values[5],
        "procedural_cells": values[6],
        "controls": values[7],
        "root_low": np.asarray(values[12:15], dtype=np.float64),
        "root_size": values[15],
    }


def exact_memmap(path: Path, dtype, count, shape=None):
    dtype = np.dtype(dtype)
    expected = int(count) * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(
            f"invalid byte count for {path}: {path.stat().st_size}, expected {expected}"
        )
    return np.memmap(path, dtype=dtype, mode="r", shape=shape or (int(count),))


def procedural_centers(ids, blocks, block_cells):
    ids = np.asarray(ids, dtype=np.int64)
    cells_per_block = block_cells ** 3
    block = ids // cells_per_block
    local = ids % cells_per_block
    coordinate = np.column_stack((
        local % block_cells,
        (local // block_cells) % block_cells,
        local // (block_cells ** 2),
    ))
    return blocks["low"][block] + (
        coordinate + 0.5
    ) * blocks["spacing"][block, None]


def build_source_owner(blocks):
    dimension = int(np.max(
        blocks["fine_coordinate"].astype(np.int64) +
        blocks["fine_span"].astype(np.int64)[:, None]
    ))
    owner = np.full((dimension, dimension, dimension), -1, dtype=np.int32)
    for block, (coordinate, span_value) in enumerate(zip(
            blocks["fine_coordinate"], blocks["fine_span"])):
        i, j, k = map(int, coordinate)
        span = int(span_value)
        region = owner[i:i + span, j:j + span, k:k + span]
        if np.any(region >= 0):
            raise ValueError("source octree leaf blocks overlap")
        region[...] = block
    if np.any(owner < 0):
        raise ValueError("source octree leaf blocks do not cover the root")
    return owner


def locate_source_controls(points, source_header, source_blocks,
                           source_owner, source_procedural_control):
    dimension = source_owner.shape[0]
    lattice = np.floor(
        (points - source_header["root_low"]) *
        (dimension / source_header["root_size"])
    ).astype(np.int64)
    lattice = np.clip(lattice, 0, dimension - 1)
    block = source_owner[lattice[:, 0], lattice[:, 1], lattice[:, 2]]
    local = np.floor(
        (points - source_blocks["low"][block]) /
        source_blocks["spacing"][block, None]
    ).astype(np.int64)
    local = np.clip(local, 0, source_header["block_cells"] - 1)
    b = source_header["block_cells"]
    procedural = source_blocks["first_procedural"][block].astype(np.int64) + (
        (local[:, 2] * b + local[:, 1]) * b + local[:, 0]
    )
    control = np.asarray(source_procedural_control[procedural], dtype=np.int32)
    if np.any(control < 0):
        bad = np.flatnonzero(control < 0)[:12]
        raise ValueError(
            "active target representatives mapped into source solid cells: "
            f"{bad.tolist()}"
        )
    return control


def state_integral(state, volume, chunk=1_000_000):
    result = np.zeros(6, dtype=np.float64)
    for first in range(0, len(volume), chunk):
        last = min(first + chunk, len(volume))
        result += np.einsum(
            "i,ij->j", np.asarray(volume[first:last]),
            np.asarray(state[first:last]), optimize=True
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mesh", required=True, type=Path)
    parser.add_argument("--source-state", required=True, type=Path)
    parser.add_argument("--target-mesh", required=True, type=Path)
    parser.add_argument("--target-solver-tables", required=True, type=Path)
    parser.add_argument("--target-interface", required=True, type=Path)
    parser.add_argument("--target-cut-cells", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=1_000_000)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.batch < 1:
        raise ValueError("batch must be positive")

    source_header = read_header(args.source_mesh)
    target_header = read_header(args.target_mesh)
    if (source_header["block_cells"] != target_header["block_cells"] or
            not np.array_equal(source_header["root_low"], target_header["root_low"]) or
            source_header["root_size"] != target_header["root_size"]):
        raise ValueError("source and target AMR roots are not identical")

    source_blocks = exact_memmap(
        args.source_mesh / "blocks.bin", BLOCK_DTYPE, source_header["blocks"]
    )
    target_blocks = exact_memmap(
        args.target_mesh / "blocks.bin", BLOCK_DTYPE, target_header["blocks"]
    )
    source_controls = exact_memmap(
        args.source_mesh / "controls.bin", CONTROL_DTYPE, source_header["controls"]
    )
    target_controls = exact_memmap(
        args.target_mesh / "controls.bin", CONTROL_DTYPE, target_header["controls"]
    )
    source_procedural_control = exact_memmap(
        args.source_mesh / "procedural_control.bin", "<i4",
        source_header["procedural_cells"]
    )
    target_procedural_control = exact_memmap(
        args.target_mesh / "procedural_control.bin", "<i4",
        target_header["procedural_cells"]
    )
    target_status = exact_memmap(
        args.target_mesh / "procedural_status.bin", "u1",
        target_header["procedural_cells"]
    )
    source_state = exact_memmap(
        args.source_state, "<f8", 6 * source_header["controls"],
        (source_header["controls"], 6)
    )
    target_representative = np.load(
        args.target_solver_tables / "control_representative_procedural_cell.npy",
        mmap_mode="r"
    )
    if len(target_representative) != target_header["controls"]:
        raise ValueError("target representative/control count mismatch")

    source_owner = build_source_owner(source_blocks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    target_state = np.memmap(
        args.out, dtype="<f8", mode="w+", shape=(target_header["controls"], 6)
    )
    target_source_control = np.empty(target_header["controls"], dtype=np.int32)
    for first in range(0, target_header["controls"], args.batch):
        last = min(first + args.batch, target_header["controls"])
        points = procedural_centers(
            target_representative[first:last], target_blocks,
            target_header["block_cells"]
        )
        mapped = locate_source_controls(
            points, source_header, source_blocks, source_owner,
            source_procedural_control
        )
        target_source_control[first:last] = mapped
        target_state[first:last] = source_state[mapped]

    # Replace injection by an exact member-volume average for every target
    # agglomeration group that can span more than one source control.
    group_offsets = np.load(
        args.target_cut_cells / "agglomeration_member_offsets.npy", mmap_mode="r"
    )
    group_members = np.load(
        args.target_cut_cells / "agglomeration_member_procedural_cell.npy",
        mmap_mode="r"
    )
    group_count = len(group_offsets) - 1
    group_sizes = np.diff(group_offsets).astype(np.int64)
    group_ids = np.repeat(np.arange(group_count, dtype=np.int32), group_sizes)
    group_controls_all = np.asarray(
        target_procedural_control[group_members], dtype=np.int32
    )
    first_members = np.asarray(group_members[group_offsets[:-1]], dtype=np.int64)
    group_control = np.asarray(
        target_procedural_control[first_members], dtype=np.int32
    )
    if (np.any(group_control < 0) or
            np.any(group_controls_all != group_control[group_ids])):
        raise ValueError("target agglomeration members do not share one control")

    member_blocks = np.asarray(group_members, dtype=np.int64) // (
        target_header["block_cells"] ** 3
    )
    member_volume = target_blocks["spacing"][member_blocks] ** 3
    member_status = target_status[group_members]
    if np.any(member_status == 1):
        raise ValueError("a target agglomeration group contains a solid cell")
    cut_selected = member_status == 2
    target_cut_ids = np.load(
        args.target_interface / "cut_cartesian_cells.npy", mmap_mode="r"
    )
    target_cut_volume = np.load(
        args.target_cut_cells / "cut_cell_volume.npy", mmap_mode="r"
    )
    cut_location = np.searchsorted(target_cut_ids, group_members[cut_selected])
    if (np.any(cut_location == len(target_cut_ids)) or
            np.any(target_cut_ids[cut_location] != group_members[cut_selected])):
        raise ValueError("a grouped target cut cell is absent from the cut-volume table")
    member_volume[cut_selected] = target_cut_volume[cut_location]

    member_points = procedural_centers(
        group_members, target_blocks, target_header["block_cells"]
    )
    member_source_control = locate_source_controls(
        member_points, source_header, source_blocks, source_owner,
        source_procedural_control
    )
    group_volume = np.bincount(
        group_ids, weights=member_volume, minlength=group_count
    )
    packed_group_volume = target_controls["volume"][group_control]
    group_volume_relative_error = np.max(
        np.abs(group_volume - packed_group_volume) /
        np.maximum(np.abs(packed_group_volume), 1e-300)
    )
    if group_volume_relative_error > 2e-12:
        raise ValueError(
            f"target agglomeration volume mismatch: {group_volume_relative_error}"
        )
    for variable in range(6):
        integrated = np.bincount(
            group_ids,
            weights=(member_volume *
                     np.asarray(source_state[member_source_control, variable])),
            minlength=group_count
        )
        target_state[group_control, variable] = integrated / group_volume

    target_state.flush()
    del source_owner

    source_integral = state_integral(
        source_state, source_controls["volume"], args.batch
    )
    target_integral = state_integral(
        target_state, target_controls["volume"], args.batch
    )
    integral_difference = target_integral - source_integral
    relative_integral_error = np.abs(integral_difference) / np.maximum(
        np.abs(source_integral), 1e-300
    )
    scaled_integral_error = float(
        np.max(np.abs(integral_difference)) /
        np.max(np.abs(source_integral))
    )

    minimum_density = np.inf
    minimum_pressure = np.inf
    minimum_sa = np.inf
    finite = True
    gamma = 1.4
    for first in range(0, target_header["controls"], args.batch):
        last = min(first + args.batch, target_header["controls"])
        state = np.asarray(target_state[first:last])
        density = state[:, 0]
        pressure = (gamma - 1.0) * (
            state[:, 4] -
            0.5 * np.sum(state[:, 1:4] ** 2, axis=1) / density
        )
        sa = state[:, 5] / density
        finite &= bool(np.all(np.isfinite(state)))
        minimum_density = min(minimum_density, float(np.min(density)))
        minimum_pressure = min(minimum_pressure, float(np.min(pressure)))
        minimum_sa = min(minimum_sa, float(np.min(sa)))
    if not finite or minimum_density <= 0 or minimum_pressure <= 0:
        raise ValueError("prolongated restart is non-finite or non-positive")

    uniform = np.asarray((1.0, 0.2, -0.3, 0.4, 2.5, 1e-6))
    uniform_group = group_volume[:, None] * uniform[None, :] / (
        packed_group_volume[:, None]
    )
    uniform_relative_error = float(np.max(
        np.abs(uniform_group - uniform) /
        np.maximum(np.abs(uniform), 1e-300)
    ))
    report = {
        "format": "QCFDAM1 nested restart transfer",
        "method": (
            "piecewise-constant source injection with exact target-member "
            "volume averaging for agglomerated controls"
        ),
        "source_mesh": str(args.source_mesh.resolve()),
        "source_state": str(args.source_state.resolve()),
        "source_controls": int(source_header["controls"]),
        "target_mesh": str(args.target_mesh.resolve()),
        "target_controls": int(target_header["controls"]),
        "target_agglomeration_groups": int(group_count),
        "target_agglomeration_members": int(len(group_members)),
        "maximum_group_volume_relative_error": float(group_volume_relative_error),
        "uniform_state_relative_error": uniform_relative_error,
        "source_conserved_integral": source_integral.tolist(),
        "target_conserved_integral": target_integral.tolist(),
        "conserved_integral_difference": integral_difference.tolist(),
        "conserved_integral_relative_error": relative_integral_error.tolist(),
        "maximum_conserved_integral_relative_error": float(np.max(relative_integral_error)),
        "maximum_conserved_integral_error_scaled_by_largest_integral": scaled_integral_error,
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
        "minimum_SA_kinematic": minimum_sa,
        "finite": finite,
        "output_bytes": args.out.stat().st_size,
        "seconds": time.perf_counter() - start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
