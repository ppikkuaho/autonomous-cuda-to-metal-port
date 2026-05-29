from __future__ import annotations

import pytest

from projection_compat import grid_sample_2d


pytestmark = pytest.mark.milestone("M5")


def make_feature_map(torch, device: str):
    h = 8
    w = 8
    ys = torch.linspace(0, 1, h, device=device)
    xs = torch.linspace(0, 1, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    fmap = torch.stack([xx, yy, xx + yy], dim=0).unsqueeze(0)
    return fmap.to(torch.float32)


def sample(torch, device: str, padding_mode: str = "zeros", compat: bool = False):
    fmap = make_feature_map(torch, device)
    # Four deterministic points in normalized grid_sample coordinates.
    grid = torch.tensor(
        [[[-1.25, -0.75], [0.0, 0.0], [0.5, -0.25], [1.2, 0.75]]],
        device=device,
        dtype=torch.float32,
    )
    grid = grid.view(1, 4, 1, 2)
    sampler = grid_sample_2d if compat else torch.nn.functional.grid_sample
    out = sampler(
        fmap,
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode=padding_mode,
    )
    return out.squeeze(-1).detach().cpu()


def test_cpu_mps_grid_sample_projection_match():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    cpu = sample(torch, "cpu", padding_mode="zeros")
    mps = sample(torch, "mps", padding_mode="zeros")
    torch.testing.assert_close(mps, cpu, atol=1e-4, rtol=1e-4)


def test_projection_ops_are_finite_on_mps():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    out = sample(torch, "mps", padding_mode="zeros")
    assert torch.isfinite(out).all()
    assert out.abs().sum().item() > 0


def test_mps_grid_sample_border_padding_native_status():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    try:
        out = sample(torch, "mps", padding_mode="border")
    except RuntimeError as exc:
        pytest.xfail(f"Native MPS border padding unsupported in this torch build: {exc}")
    assert torch.isfinite(out).all()


def test_border_padding_compat_matches_cpu_native_on_cpu():
    torch = pytest.importorskip("torch")
    native = sample(torch, "cpu", padding_mode="border", compat=False)
    compat = sample(torch, "cpu", padding_mode="border", compat=True)
    torch.testing.assert_close(compat, native, atol=1e-6, rtol=1e-6)


def test_border_padding_compat_matches_cpu_native_on_mps():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native = sample(torch, "cpu", padding_mode="border", compat=False)
    compat = sample(torch, "mps", padding_mode="border", compat=True)
    torch.testing.assert_close(compat, native, atol=1e-4, rtol=1e-4)
