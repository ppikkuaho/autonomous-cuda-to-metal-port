from __future__ import annotations

import sys

import pytest

from conftest import WORK_PIXAL3D, env_enabled


pytestmark = [pytest.mark.milestone("M6"), pytest.mark.requires_backend]


def require_enabled():
    if not env_enabled("RUN_SPARSE_CONV_TESTS"):
        pytest.skip("Sparse conv tests start after backend selection")


def sparse_modules():
    require_enabled()
    sys.path.insert(0, str(WORK_PIXAL3D))
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.modules.sparse import config as sparse_config
    from pixal3d.modules.sparse.conv import SparseConv3d
    from pixal3d.modules.sparse.conv import conv as conv_module

    sparse_config.set_conv_backend("none")
    conv_module._backends.clear()
    return SparseTensor, SparseConv3d


def test_conv_none_dispatch_does_not_import_cuda_backends():
    require_enabled()
    for name in ["flex_gemm", "spconv", "torchsparse"]:
        sys.modules.pop(name, None)

    torch = pytest.importorskip("torch")
    SparseTensor, SparseConv3d = sparse_modules()
    x = SparseTensor(
        feats=torch.ones(1, 1),
        coords=torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
        shape=torch.Size([1, 1]),
    )
    conv = SparseConv3d(1, 1, kernel_size=3, bias=False)
    out = conv(x)

    assert out.feats.shape == (1, 1)
    assert "flex_gemm" not in sys.modules
    assert "spconv" not in sys.modules
    assert "torchsparse" not in sys.modules


def available_devices():
    torch = pytest.importorskip("torch")
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


@pytest.mark.parametrize("device", available_devices())
def test_sparse_varlen_reductions_match_manual_reference(device):
    torch = pytest.importorskip("torch")
    SparseTensor, _SparseConv3d = sparse_modules()

    coords = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 2, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    feats = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [-1.0, 2.0, 0.5],
            [3.0, -2.0, 1.5],
        ],
        dtype=torch.float32,
        device=device,
    )
    x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([2, 3]))

    expected_mean = torch.stack([feats[:3].mean(), feats[3:].mean()]).reshape(2, 1)
    expected_sum = torch.stack([feats[:3].sum(), feats[3:].sum()])
    expected_std = torch.stack([feats[:3].std(unbiased=False), feats[3:].std(unbiased=False)]).reshape(2, 1)

    torch.testing.assert_close(x.mean(dim=1, keepdim=True).cpu(), expected_mean.cpu())
    torch.testing.assert_close(x.sum(dim=1).cpu(), expected_sum.cpu())
    torch.testing.assert_close(x.std(dim=1, keepdim=True).cpu(), expected_std.cpu(), atol=1e-5, rtol=1e-5)


def slow_subm_reference(feats, coords, weight, bias=None, dilation=(1, 1, 1)):
    Co, Kd, Kh, Kw, _Ci = weight.shape
    out = feats.new_zeros((feats.shape[0], Co))
    coord_to_idx = {}
    coords_cpu = coords.cpu()
    for idx, coord in enumerate(coords_cpu.tolist()):
        key = tuple(coord)
        if key in coord_to_idx:
            raise ValueError(f"duplicate coordinate in reference: {key}")
        coord_to_idx[key] = idx

    dz, dy, dx = dilation
    for tgt_idx, coord in enumerate(coords_cpu.tolist()):
        batch, z, y, x_coord = coord
        for kz in range(Kd):
            for ky in range(Kh):
                for kx in range(Kw):
                    src_key = (
                        batch,
                        z + (kz - Kd // 2) * dz,
                        y + (ky - Kh // 2) * dy,
                        x_coord + (kx - Kw // 2) * dx,
                    )
                    src_idx = coord_to_idx.get(src_key)
                    if src_idx is not None:
                        out[tgt_idx] += feats[src_idx] @ weight[:, kz, ky, kx, :].T
    if bias is not None:
        out = out + bias
    return out


@pytest.mark.parametrize("device", available_devices())
def test_conv_none_matches_slow_reference_for_multibatch(device):
    torch = pytest.importorskip("torch")
    SparseTensor, SparseConv3d = sparse_modules()

    coords = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 2, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    feats = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, -1.0],
            [0.5, 4.0],
            [-2.0, 1.0],
            [1.5, 0.25],
        ],
        dtype=torch.float32,
        device=device,
    )
    x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([2, 2]))
    conv = SparseConv3d(2, 3, kernel_size=3, bias=True).to(device)
    with torch.no_grad():
        values = torch.arange(conv.weight.numel(), dtype=torch.float32, device=device)
        conv.weight.copy_((values.reshape_as(conv.weight) % 7 - 3) / 5)
        conv.bias.copy_(torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32, device=device))

    out = conv(x)
    expected = slow_subm_reference(feats, coords, conv.weight, conv.bias)

    assert out.coords.device == coords.device
    assert torch.equal(out.coords, coords)
    assert out.feats.shape == (coords.shape[0], 3)
    torch.testing.assert_close(out.feats.cpu(), expected.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", available_devices())
def test_conv_none_axis_orientation_sentinel(device):
    torch = pytest.importorskip("torch")
    SparseTensor, SparseConv3d = sparse_modules()

    coords = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 2, 1, 1],
            [0, 1, 2, 1],
            [0, 1, 1, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    feats = torch.tensor([[0.0], [1.0], [1.0], [1.0]], dtype=torch.float32, device=device)
    x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 1]))
    conv = SparseConv3d(1, 1, kernel_size=3, bias=False).to(device)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 2, 1, 1, 0] = 10.0
        conv.weight[0, 1, 2, 1, 0] = 20.0
        conv.weight[0, 1, 1, 2, 0] = 30.0

    out = conv(x)

    torch.testing.assert_close(out.feats[0].cpu(), torch.tensor([60.0]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", available_devices())
def test_conv_none_respects_dilation(device):
    torch = pytest.importorskip("torch")
    SparseTensor, SparseConv3d = sparse_modules()

    coords = torch.tensor(
        [
            [0, 2, 2, 2],
            [0, 2, 2, 4],
            [0, 2, 4, 2],
            [0, 4, 2, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    feats = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2) / 3
    x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 2]))
    conv = SparseConv3d(2, 2, kernel_size=3, dilation=2, bias=False).to(device)
    with torch.no_grad():
        conv.weight.fill_(0)
        conv.weight[:, 1, 1, 1, :] = torch.eye(2, dtype=torch.float32, device=device)
        conv.weight[:, 1, 1, 2, :] = torch.tensor(
            [[0.5, -0.25], [0.75, 0.125]], dtype=torch.float32, device=device
        )

    out = conv(x)
    expected = slow_subm_reference(feats, coords, conv.weight, None, dilation=(2, 2, 2))

    torch.testing.assert_close(out.feats.cpu(), expected.cpu(), atol=1e-5, rtol=1e-5)


def test_conv_none_cache_does_not_reuse_stale_coordinates():
    torch = pytest.importorskip("torch")
    SparseTensor, SparseConv3d = sparse_modules()

    coords = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 2]], dtype=torch.long)
    feats = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 1]))
    conv = SparseConv3d(1, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 1, 1, 2, 0] = 10.0

    out_a = conv(x)
    moved = x.replace(feats, torch.tensor([[0, 1, 1, 1], [0, 1, 2, 1]], dtype=torch.long))
    out_b = conv(moved)

    torch.testing.assert_close(out_a.feats[0], torch.tensor([10.0]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_b.feats[0], torch.tensor([0.0]), atol=1e-5, rtol=1e-5)


def test_conv_none_rejects_duplicate_coordinates():
    torch = pytest.importorskip("torch")
    SparseTensor, SparseConv3d = sparse_modules()

    coords = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.long)
    feats = torch.ones(2, 1)
    x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 1]))
    conv = SparseConv3d(1, 1, kernel_size=3, bias=False)

    with pytest.raises(ValueError, match="Duplicate sparse coordinate"):
        conv(x)


def test_conv_none_empty_tensor_with_explicit_shape():
    torch = pytest.importorskip("torch")
    SparseTensor, SparseConv3d = sparse_modules()

    coords = torch.empty((0, 4), dtype=torch.long)
    feats = torch.empty((0, 2), dtype=torch.float32)
    x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([1, 2]))
    conv = SparseConv3d(2, 4, kernel_size=3, bias=True)

    out = conv(x)

    assert out.feats.shape == (0, 4)
    assert out.coords.shape == (0, 4)


@pytest.mark.parametrize("device", available_devices())
def test_conv_none_runs_pixal3d_sparse_model_block(device):
    torch = pytest.importorskip("torch")
    SparseTensor, _SparseConv3d = sparse_modules()
    from pixal3d.models.sc_vaes.sparse_unet_vae import SparseConvNeXtBlock3d

    coords = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 2, 1],
            [0, 2, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 2],
        ],
        dtype=torch.long,
        device=device,
    )
    feats = torch.linspace(-1.0, 1.0, steps=coords.shape[0] * 8, device=device).reshape(coords.shape[0], 8)
    x = SparseTensor(feats=feats, coords=coords, shape=torch.Size([2, 8]))
    block = SparseConvNeXtBlock3d(channels=8, mlp_ratio=2.0, use_checkpoint=False).to(device)

    out = block(x)

    assert torch.equal(out.coords, coords)
    assert out.feats.shape == feats.shape
    assert torch.isfinite(out.feats).all()
