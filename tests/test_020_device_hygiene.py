from __future__ import annotations

import pytest

from conftest import env_enabled


pytestmark = pytest.mark.milestone("M4")


def test_cuda_runtime_guard_example(monkeypatch):
    torch = pytest.importorskip("torch")

    def forbid_cuda(*args, **kwargs):
        raise AssertionError("Unexpected CUDA call in Mac port path")

    monkeypatch.setattr(torch.Tensor, "cuda", forbid_cuda, raising=False)
    x = torch.zeros(1)
    with pytest.raises(AssertionError):
        x.cuda()


def test_strict_device_scan_is_gated():
    if not env_enabled("STRICT_DEVICE_SCAN"):
        pytest.skip("Strict scan is enabled after device hygiene port work starts")

