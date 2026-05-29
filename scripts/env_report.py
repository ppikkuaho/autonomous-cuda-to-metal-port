#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT, collect_environment, ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Mac/PyTorch/repo environment details for a harness run.")
    parser.add_argument("--out", default="artifacts/runs/m0_env", help="Output artifact directory.")
    args = parser.parse_args()

    out = ensure_dir(ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out))
    env = collect_environment()
    env["run_id"] = out.name

    write_json(out / "manifest.json", env)
    (out / "env.txt").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")

    commits = env["repos"]
    (out / "git_commits.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in commits.items()) + "\n"
    )

    torch = env["software"]["torch"]
    print(f"wrote {out}")
    print(f"python_arch={env['software']['python_arch']}")
    print(f"torch_available={torch.get('available')}")
    print(f"mps_available={torch.get('mps_available')}")
    print(f"upstream_pixal3d={commits.get('upstream_pixal3d_commit')}")
    print(f"upstream_trellis_mac={commits.get('upstream_trellis_mac_commit')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

