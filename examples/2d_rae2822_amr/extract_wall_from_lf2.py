#!/usr/bin/env python3
"""Extract the ordered RAE2822 wall polygon from an archived LF2 topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.topology) as data:
        nodes = np.asarray(data["nodes"], dtype=np.float64)
        face_nodes = np.asarray(data["face_nodes"], dtype=np.int64)
        face_cells = np.asarray(data["face_cells"], dtype=np.int64)
        zones = np.asarray(data["face_zones"], dtype=np.int64)

    mask = np.isin(zones, (6, 7)) & np.any(face_cells == 0, axis=1)
    edges = face_nodes[mask]
    edge_zones = zones[mask]
    zone_by_edge = {
        tuple(sorted(map(int, edge))): int(zone)
        for edge, zone in zip(edges, edge_zones)
    }
    adjacency = {}
    for first, second in edges:
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    if not adjacency or any(len(neighbours) != 2
                            for neighbours in adjacency.values()):
        raise ValueError("wall edges do not form one closed degree-two cycle")

    start = min(adjacency, key=lambda node: (nodes[node, 0], nodes[node, 1]))
    ordered = [start]
    previous = None
    current = start
    while True:
        options = [node for node in adjacency[current] if node != previous]
        following = options[0]
        if following == start:
            break
        if following in ordered:
            raise ValueError("wall traversal closed on the wrong node")
        ordered.append(following)
        previous, current = current, following
    if len(ordered) != len(adjacency):
        raise ValueError("wall contains more than one cycle")

    def area(node_ids):
        xy = nodes[node_ids]
        return 0.5 * float(np.sum(
            xy[:, 0] * np.roll(xy[:, 1], -1) -
            xy[:, 1] * np.roll(xy[:, 0], -1)
        ))

    if area(ordered) < 0.0:
        ordered = [ordered[0], *reversed(ordered[1:])]
    xy = nodes[ordered]
    patches = []
    for index, first in enumerate(ordered):
        second = ordered[(index + 1) % len(ordered)]
        zone = zone_by_edge[tuple(sorted((first, second)))]
        patches.append(1 if zone == 7 else 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        stream.write("x,y,patch\n")
        for point, patch in zip(xy, patches):
            stream.write(f"{point[0]:.17g},{point[1]:.17g},{patch}\n")
    manifest = {
        "source": str(args.topology.resolve()),
        "source_sha256": sha256(args.topology),
        "output_sha256": sha256(args.out),
        "vertices": len(ordered),
        "signed_area": area(ordered),
        "source_wall_zones": {"6": "lower", "7": "upper"},
        "coordinate_transform": "none",
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
