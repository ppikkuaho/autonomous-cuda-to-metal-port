from __future__ import annotations

import os
import sys

import pytest

from conftest import WORK_PIXAL3D, env_enabled


pytestmark = [pytest.mark.milestone("M3"), pytest.mark.requires_pixal3d]


def require_enabled():
    if not env_enabled("RUN_PIXAL3D_IMPORT_TESTS"):
        pytest.skip("M3 import tests are gated; set RUN_PIXAL3D_IMPORT_TESTS=1 when import-only porting starts")


def test_import_pixal3d_pipeline():
    require_enabled()
    sys.path.insert(0, str(WORK_PIXAL3D))
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline  # noqa: F401


def test_import_inference_module_without_cuda_extensions():
    require_enabled()
    sys.path.insert(0, str(WORK_PIXAL3D))
    import inference  # noqa: F401


def test_inference_import_uses_mac_safe_defaults(monkeypatch):
    require_enabled()
    for name in ["inference", "o_voxel"]:
        sys.modules.pop(name, None)
    monkeypatch.delenv("ATTN_BACKEND", raising=False)
    monkeypatch.delenv("SPARSE_ATTN_BACKEND", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    sys.path.insert(0, str(WORK_PIXAL3D))
    import inference  # noqa: F401

    assert os.environ.get("ATTN_BACKEND") == "sdpa"
    assert os.environ.get("SPARSE_ATTN_BACKEND") == "sdpa"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert "o_voxel" not in sys.modules


def test_no_cuda_available_required():
    require_enabled()
    torch = pytest.importorskip("torch")
    assert torch.backends.mps.is_available()
    assert os.environ.get("ATTN_BACKEND", "sdpa") in {"sdpa", "naive", "flash_attn"}
