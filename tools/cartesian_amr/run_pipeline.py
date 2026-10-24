"""Run the current in-house CRM hybrid-mesh planning stages with one command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(command):
    print("Running:", " ".join(map(str, command)), flush=True)
    result = subprocess.run(command, text=True)
    if result.returncode:
        raise RuntimeError(f"stage failed with exit code {result.returncode}")


def prepare_collision_checker(source_directory, destination, supplied_path):
    if supplied_path:
        checker = Path(supplied_path).resolve()
        if not checker.is_file():
            raise FileNotFoundError(f"collision checker not found: {checker}")
        return checker

    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError(
            "a C++17 compiler is required to build the collision checker"
        )
    source = source_directory / "surface_self_intersection.cpp"
    tool_directory = destination / ".tools"
    tool_directory.mkdir(parents=True, exist_ok=True)
    checker = tool_directory / "check_surface"
    if (not checker.exists() or
            checker.stat().st_mtime_ns < source.stat().st_mtime_ns):
        run([compiler, "-O3", "-std=c++17", str(source), "-o", str(checker)])
    return checker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--collision-checker",
        help="optional prebuilt checker; otherwise it is built automatically"
    )
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--first-height", type=float, default=2e-6)
    parser.add_argument("--growth", type=float, default=1.25)
    parser.add_argument("--target-cartesian-cells", type=int,
                        default=10_000_000)
    parser.add_argument("--farfield-padding", type=float, default=10.0)
    parser.add_argument("--maximum-level", type=int, default=8)
    parser.add_argument("--materialize-prisms", action="store_true")
    parser.add_argument(
        "--export-nnlci-tables", action="store_true",
        help="write and validate reusable memory-mapped connectivity tables"
    )
    parser.add_argument(
        "--build-interface", action="store_true",
        help="classify overlap and clip matching prism/Cartesian subfaces"
    )
    args = parser.parse_args()

    start = time.perf_counter()
    source_directory = Path(__file__).resolve().parent
    destination = Path(args.out).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    collision_checker = prepare_collision_checker(
        source_directory, destination, args.collision_checker
    )
    surface_command = [
        sys.executable, str(source_directory / "fast_surface_pipeline.py"),
        "--surface", str(Path(args.surface).resolve()),
        "--out", str(destination),
        "--layers", str(args.layers),
        "--first-height", str(args.first_height),
        "--growth", str(args.growth),
        "--collision-checker", str(collision_checker),
    ]
    if args.materialize_prisms:
        surface_command.append("--materialize")
    run(surface_command)
    run([
        sys.executable, str(source_directory / "build_octree_plan.py"),
        "--plan", str(destination / "column_plan.npz"),
        "--out", str(destination),
        "--target-cells", str(args.target_cartesian_cells),
        "--block-cells", "8",
        "--farfield-padding", str(args.farfield_padding),
        "--maximum-level", str(args.maximum_level),
        "--refinement-band-cells", "3",
    ])
    interface = None
    if args.build_interface:
        interface_directory = destination / "interface_plan"
        run([
            sys.executable, str(source_directory / "build_interface_plan.py"),
            "--column-plan", str(destination / "column_plan.npz"),
            "--octree-plan", str(destination / "octree_blocks.npz"),
            "--out", str(interface_directory),
        ])
        run([
            sys.executable, str(source_directory / "validate_interface_plan.py"),
            "--interface", str(interface_directory),
            "--column-plan", str(destination / "column_plan.npz"),
            "--octree-plan", str(destination / "octree_blocks.npz"),
        ])
        interface = {
            "manifest": json.loads(
                (interface_directory / "manifest.json").read_text()
            ),
            "validation": json.loads(
                (interface_directory / "validation.json").read_text()
            ),
        }
    nnlci = None
    if args.export_nnlci_tables:
        table_directory = destination / "nnlci_tables"
        run([
            sys.executable, str(source_directory / "export_nnlci_tables.py"),
            "--column-plan", str(destination / "column_plan.npz"),
            "--octree-plan", str(destination / "octree_blocks.npz"),
            "--out", str(table_directory),
        ])
        run([
            sys.executable, str(source_directory / "validate_nnlci_tables.py"),
            "--mesh", str(table_directory),
        ])
        nnlci = {
            "manifest": json.loads(
                (table_directory / "manifest.json").read_text()
            ),
            "validation": json.loads(
                (table_directory / "validation.json").read_text()
            ),
        }
    boundary = json.loads(
        (destination / "boundary_layer_report.json").read_text()
    )
    octree = json.loads((destination / "octree_report.json").read_text())
    combined = {
        "mesh_workflow": "body_fitted_hybrid_3d",
        "status": (
            "PLANNING_STAGES_COMPLETE"
            if boundary["status"] == "BOUNDARY_LAYER_PLAN_VALID" and
            octree["status"] == "OCTREE_PLAN_COMPLETE"
            else "PLANNING_STAGE_INCOMPLETE"
        ),
        "solver_ready": False,
        "boundary_layer": boundary,
        "octree": octree,
        "interface_plan": interface,
        "nnlci_tables": nnlci,
        "planning_cell_upper_bound": (
            boundary["expanded_prism_cells"] +
            octree["represented_cartesian_cells"]
        ),
        "total_pipeline_seconds": time.perf_counter() - start,
        "remaining": (
            [
                "close every cut cell with Cartesian aperture polygons",
                "compute exact cut-cell volume and centroid",
                "global face owner/neighbour assembly",
                "finite-volume geometric conservation audit",
                "GPU solver-table assembly",
            ] if interface else [
                "signed solid classification",
                "exact Cartesian cut-cell geometry",
                "conservative prism-front coupling",
                "global face owner/neighbour assembly",
                "GPU solver-table assembly",
            ]
        ),
    }
    (destination / "pipeline_report.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
