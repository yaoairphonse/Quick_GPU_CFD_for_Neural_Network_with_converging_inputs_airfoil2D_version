"""Validate every parent map in a QCFDMG1 aggregation hierarchy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


HEADER = struct.Struct("<8s4I2Q")
LEVEL_HEADER = struct.Struct("<2Q")


def validate(path: Path) -> dict:
    with path.open("rb") as stream:
        values = HEADER.unpack(stream.read(HEADER.size))
    (magic, version, endian, level_count, merges_per_level,
     fine_cell_count, file_bytes) = values
    if magic != b"QCFDMG1\0" or version != 1 or endian != 0x01020304:
        raise ValueError("unsupported agglomeration header")
    if level_count < 2 or merges_per_level < 1:
        raise ValueError("invalid hierarchy dimensions")
    if path.stat().st_size != file_bytes:
        raise ValueError("incorrect hierarchy file length")

    offset = HEADER.size
    expected_children = fine_cell_count
    counts = [fine_cell_count]
    statistics = []
    for level in range(level_count - 1):
        with path.open("rb") as stream:
            stream.seek(offset)
            child_count, parent_count = LEVEL_HEADER.unpack(
                stream.read(LEVEL_HEADER.size))
        offset += LEVEL_HEADER.size
        if child_count != expected_children or parent_count < 1:
            raise ValueError("non-contiguous hierarchy counts")
        parents = np.memmap(path, dtype="<i4", mode="r", offset=offset,
                            shape=(child_count,))
        if parents.min() != 0 or parents.max() != parent_count - 1:
            raise ValueError("parent identifiers are not dense and in range")
        children = np.bincount(parents, minlength=parent_count)
        if np.any(children == 0):
            raise ValueError("empty aggregate")
        statistics.append({
            "child_level": level,
            "parent_level": level + 1,
            "children": int(child_count),
            "parents": int(parent_count),
            "coarsening_ratio": float(child_count / parent_count),
            "children_per_parent_min": int(children.min()),
            "children_per_parent_max": int(children.max()),
            "children_per_parent_mean": float(children.mean()),
        })
        offset += child_count * 4
        expected_children = parent_count
        counts.append(parent_count)
    if offset != file_bytes:
        raise ValueError("trailing or missing hierarchy bytes")
    return {
        "format": "QCFDMG1",
        "version": version,
        "file_bytes": file_bytes,
        "levels": level_count,
        "merges_per_level": merges_per_level,
        "cell_counts": counts,
        "maps": statistics,
        "dense_parent_maps": True,
        "no_empty_aggregates": True,
        "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hierarchy", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    result = validate(arguments.hierarchy)
    text = json.dumps(result, indent=2) + "\n"
    if arguments.report:
        arguments.report.write_text(text)
    print(text, end="")
