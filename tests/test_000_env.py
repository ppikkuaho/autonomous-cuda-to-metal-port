from __future__ import annotations

import platform

import pytest


@pytest.mark.milestone("M0")
def test_python_is_arm64():
    assert platform.machine() == "arm64"


@pytest.mark.milestone("M0")
def test_torch_mps_available():
    torch = pytest.importorskip("torch")
    assert torch.backends.mps.is_built()
    assert torch.backends.mps.is_available()


@pytest.mark.milestone("M0")
def test_expected_repo_layout(repo_root):
    # This repository ships the Mac-port changes as a patch plus the
    # verification harness; the Pixal3D model tree itself is obtained by
    # cloning upstream and applying patches/pixal3d-mac-port.patch.
    assert (repo_root / "patches" / "pixal3d-mac-port.patch").is_file()
    assert (repo_root / "patches" / "apply.sh").is_file()
    assert (repo_root / "scripts" / "env_report.py").is_file()
    assert (repo_root / "scripts" / "static_cuda_scan.py").is_file()

