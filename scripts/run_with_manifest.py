#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from _common import ROOT, collect_environment, ensure_dir, write_json


def parse_env(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"env must be KEY=VALUE, got {value!r}")
        key, val = value.split("=", 1)
        parsed[key] = val
    return parsed


def parse_secret_env_files(values: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    parsed = {}
    manifest = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"secret env must be KEY=PATH, got {value!r}")
        key, path_str = value.split("=", 1)
        path = Path(path_str).expanduser()
        token = None
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            stripped = line.strip()
            if key == "HF_TOKEN" and not stripped.startswith("hf_"):
                continue
            if stripped:
                token = stripped
                break
        if token is None:
            raise ValueError(f"secret env file is empty: {path}")
        parsed[key] = token
        manifest[key] = str(path)
    return parsed, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command and record logs plus a manifest.")
    parser.add_argument("--out", required=True, help="Output artifact directory.")
    parser.add_argument("--name", default=None, help="Optional run name.")
    parser.add_argument("--cwd", default=None, help="Working directory for command.")
    parser.add_argument("--env", action="append", default=[], help="Extra env var KEY=VALUE. Repeatable.")
    parser.add_argument("--secret-env-file", action="append", default=[], help="Secret env var KEY=PATH. The file's last non-empty line is used and the value is not written to the manifest.")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("missing command after --")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    ensure_dir(out)

    cwd = Path(args.cwd) if args.cwd else ROOT
    if not cwd.is_absolute():
        cwd = ROOT / cwd

    env = os.environ.copy()
    env.update(parse_env(args.env))
    secret_env, secret_env_manifest = parse_secret_env_files(args.secret_env_file)
    env.update(secret_env)

    manifest = collect_environment()
    manifest.update(
        {
            "run_id": args.name or out.name,
            "command": cmd,
            "cwd": str(cwd),
            "started_unix": time.time(),
            "extra_env": parse_env(args.env),
            "secret_env_files": secret_env_manifest,
        }
    )
    write_json(out / "manifest.json", manifest)

    stdout_path = out / "stdout.log"
    stderr_path = out / "stderr.log"
    started = time.time()
    def pump(src, dst_file, mirror) -> None:
        for line in iter(src.readline, ""):
            dst_file.write(line)
            dst_file.flush()
            mirror.write(line)
            mirror.flush()

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        stdout_thread = threading.Thread(target=pump, args=(proc.stdout, stdout, sys.stdout), daemon=True)
        stderr_thread = threading.Thread(target=pump, args=(proc.stderr, stderr, sys.stderr), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        return_code = proc.wait()
        stdout_thread.join()
        stderr_thread.join()

    manifest["ended_unix"] = time.time()
    manifest["duration_s"] = round(time.time() - started, 6)
    manifest["return_code"] = return_code
    write_json(out / "manifest.json", manifest)
    (out / "timings.jsonl").write_text(
        f'{{"name":"command","duration_s":{manifest["duration_s"]},"return_code":{return_code}}}\n'
    )
    print(f"return_code={return_code}")
    print(f"wrote={out}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
