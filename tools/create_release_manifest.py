#!/usr/bin/env python3
"""Write deterministic SHA-256 and JSON manifests for the source package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"RELEASE_MANIFEST.sha256", "RELEASE_CONTENTS.json"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    records = []
    for path in sorted(candidate for candidate in ROOT.rglob("*") if candidate.is_file()):
        relative = path.relative_to(ROOT)
        if relative.name in EXCLUDED or "__pycache__" in relative.parts:
            continue
        records.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        })
    (ROOT / "RELEASE_MANIFEST.sha256").write_text(
        "".join(f"{record['sha256']}  {record['path']}\n" for record in records),
        encoding="utf-8",
    )
    (ROOT / "RELEASE_CONTENTS.json").write_text(
        json.dumps({
            "format": "quick-cfd-source-release-v1",
            "files": len(records),
            "bytes": sum(record["bytes"] for record in records),
            "records": records,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifested {len(records)} files")


if __name__ == "__main__":
    main()
