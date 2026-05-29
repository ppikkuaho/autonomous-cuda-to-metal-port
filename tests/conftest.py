from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
WORK_PIXAL3D = ROOT / "work" / "Pixal3D"
UPSTREAM_PIXAL3D = ROOT / "upstream" / "Pixal3D"
UPSTREAM_TRELLIS_MAC = ROOT / "upstream" / "trellis-mac"

sys.path.insert(0, str(SCRIPTS))


def env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def pytest_configure(config):
    config.addinivalue_line("markers", "milestone(name): milestone covered by the test")


@pytest.fixture
def repo_root() -> Path:
    return ROOT

