#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, ensure_dir, write_json


def validate(path: Path) -> dict:
    result = {
        "path": str(path),
        "exists": path.exists(),
        "status": "FAIL",
        "checks": [],
        "warnings": [],
        "errors": [],
    }
    if not path.exists():
        result["errors"].append("file does not exist")
        return result
    result["size_bytes"] = path.stat().st_size
    if result["size_bytes"] <= 0:
        result["errors"].append("file is empty")
        return result

    try:
        import trimesh
    except Exception as exc:
        result["status"] = "PARTIAL"
        result["warnings"].append(f"trimesh unavailable; only file existence checked: {exc!r}")
        return result

    try:
        loaded = trimesh.load(path, force="scene")
        geometries = list(getattr(loaded, "geometry", {}).values()) if hasattr(loaded, "geometry") else [loaded]
        vertex_count = 0
        face_count = 0
        texture_visual_count = 0
        material_count = 0
        bounds = []
        for geom in geometries:
            vertices = getattr(geom, "vertices", None)
            faces = getattr(geom, "faces", None)
            if vertices is not None:
                vertex_count += len(vertices)
            if faces is not None:
                face_count += len(faces)
            if hasattr(geom, "bounds") and geom.bounds is not None:
                bounds.append(geom.bounds.tolist())
            visual = getattr(geom, "visual", None)
            if getattr(visual, "kind", None) == "texture":
                texture_visual_count += 1
                if getattr(visual, "material", None) is not None:
                    material_count += 1
        result["vertex_count"] = vertex_count
        result["face_count"] = face_count
        result["texture_visual_count"] = texture_visual_count
        result["material_count"] = material_count
        result["bounds"] = bounds
        if vertex_count <= 0:
            result["errors"].append("no vertices")
        if face_count <= 0:
            result["errors"].append("no faces")
        result["status"] = "PASS" if not result["errors"] else "FAIL"
        return result
    except Exception as exc:
        result["errors"].append(repr(exc))
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate that a GLB exists and contains loadable geometry.")
    parser.add_argument("glb")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.glb)
    if not path.is_absolute():
        path = ROOT / path
    report = validate(path)

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        ensure_dir(out)
        write_json(out / "glb_validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
