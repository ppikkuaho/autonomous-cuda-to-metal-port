from __future__ import annotations

import sys
import types

import pytest

from conftest import WORK_PIXAL3D, env_enabled


pytestmark = [pytest.mark.milestone("M8"), pytest.mark.requires_pixal3d]


def require_enabled():
    if not env_enabled("RUN_IMAGE_COND_TESTS"):
        pytest.skip("Image conditioning tests start after import-only and device hygiene milestones")


def projection_modules():
    require_enabled()
    sys.path.insert(0, str(WORK_PIXAL3D))
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid, sample_features

    return ProjGrid, sample_features


def image_conditioning_module():
    require_enabled()
    sys.path.insert(0, str(WORK_PIXAL3D))
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj

    return image_conditioned_proj


def available_devices():
    torch = pytest.importorskip("torch")
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


@pytest.mark.parametrize("device", available_devices())
def test_sample_features_border_compat_matches_cpu_native(device):
    torch = pytest.importorskip("torch")
    torch_f = pytest.importorskip("torch.nn.functional")
    _ProjGrid, sample_features = projection_modules()

    fmap_cpu = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    queries_cpu = torch.tensor([[[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]]], dtype=torch.float32)
    expected = torch_f.grid_sample(
        fmap_cpu,
        queries_cpu.view(1, 3, 1, 2),
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    ).squeeze(-1)

    got = sample_features(fmap_cpu.to(device), queries_cpu.to(device)).cpu()

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", available_devices())
def test_proj_grid_synthetic_features_are_finite(device):
    torch = pytest.importorskip("torch")
    ProjGrid, _sample_features = projection_modules()

    grid = ProjGrid(grid_resolution=3, image_resolution=64).to(device)
    y, x = torch.meshgrid(
        torch.linspace(0, 1, 8, device=device),
        torch.linspace(0, 1, 8, device=device),
        indexing="ij",
    )
    features = torch.stack([x, y, x + y], dim=-1).unsqueeze(0)

    projected = grid(
        features,
        camera_angle_x=torch.tensor([0.8], dtype=torch.float32, device=device),
        distance=torch.tensor([2.0], dtype=torch.float32, device=device),
        mesh_scale=torch.tensor([1.0], dtype=torch.float32, device=device),
    )

    assert projected.shape == (1, 27, 3)
    assert torch.isfinite(projected).all()
    assert projected.std() > 0


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("target_size", [8, (8, 10), [8, 10]])
def test_interpolated_naf_fallback_preserves_device_dtype_and_shape(device, target_size):
    torch = pytest.importorskip("torch")
    module = image_conditioning_module()

    fallback = module.InterpolatedNAFUpsampler().to(device)
    guide = torch.rand(1, 3, 16, 16, device=device, dtype=torch.float32)
    lr_features = torch.rand(1, 4, 2, 2, device=device, dtype=torch.float32)
    output = fallback(guide, lr_features, target_size)

    expected_size = (target_size, target_size) if isinstance(target_size, int) else tuple(target_size)
    assert output.shape == (1, 4, *expected_size)
    assert output.device == lr_features.device
    assert output.dtype == lr_features.dtype
    assert torch.isfinite(output).all()


class _FakeDinoEmbeddings(pytest.importorskip("torch").nn.Module):
    def __init__(self, patch_size: int, hidden_size: int, num_register_tokens: int):
        super().__init__()
        torch = pytest.importorskip("torch")
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_register_tokens = num_register_tokens
        self.patch_embeddings = torch.nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False
        )

    def forward(self, image, bool_masked_pos=None):
        torch = pytest.importorskip("torch")
        batch, _channels, height, width = image.shape
        patch_count = (height // self.patch_size) * (width // self.patch_size)
        token_count = 1 + self.num_register_tokens + patch_count
        values = torch.arange(
            token_count * self.hidden_size,
            device=image.device,
            dtype=image.dtype,
        ).reshape(1, token_count, self.hidden_size)
        return values.expand(batch, -1, -1)


class _FakeDinoModel(pytest.importorskip("torch").nn.Module):
    def __init__(self, patch_size: int = 4, hidden_size: int = 6, num_register_tokens: int = 2):
        super().__init__()
        torch = pytest.importorskip("torch")
        self.config = types.SimpleNamespace(
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_register_tokens=num_register_tokens,
        )
        self.embeddings = _FakeDinoEmbeddings(patch_size, hidden_size, num_register_tokens)
        self.model = torch.nn.Module()
        self.model.layer = torch.nn.ModuleList()
        self.norm = torch.nn.LayerNorm(hidden_size)

    def rope_embeddings(self, image):
        return None


@pytest.mark.parametrize("device", available_devices())
def test_dinov3_extractor_interpolated_naf_keeps_projection_contract(monkeypatch, device):
    torch = pytest.importorskip("torch")
    import torch.hub

    module = image_conditioning_module()
    monkeypatch.setenv("PIXAL3D_NAF_BACKEND", "interpolate")
    monkeypatch.setattr(
        module.DINOv3ViTModel,
        "from_pretrained",
        staticmethod(lambda _model_name: _FakeDinoModel()),
    )
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("torch.hub.load should not run")),
    )

    extractor = module.DinoV3ProjFeatureExtractor(
        "fake-dino",
        image_size=8,
        grid_resolution=2,
        use_naf_upsample=True,
        naf_target_size=8,
    ).to(device)
    image = torch.rand(1, 3, 8, 8, device=device)

    global_features, proj_features = extractor(
        image,
        camera_angle_x=torch.tensor([0.8], device=device),
        distance=torch.tensor([2.0], device=device),
        mesh_scale=torch.tensor([1.0], device=device),
    )

    assert extractor.proj_channels == extractor.embed_dim * 2
    assert isinstance(extractor.naf_model, module.InterpolatedNAFUpsampler)
    assert global_features.shape == (1, 3, extractor.embed_dim)
    assert proj_features.shape == (1, 8, extractor.embed_dim * 2)
    assert global_features.device == image.device
    assert proj_features.device == image.device
    assert torch.isfinite(global_features).all()
    assert torch.isfinite(proj_features).all()


@pytest.mark.parametrize("device", available_devices())
def test_dinov3_extractor_without_naf_keeps_original_projection_width(monkeypatch, device):
    torch = pytest.importorskip("torch")

    module = image_conditioning_module()
    monkeypatch.setenv("PIXAL3D_NAF_BACKEND", "interpolate")
    monkeypatch.setattr(
        module.DINOv3ViTModel,
        "from_pretrained",
        staticmethod(lambda _model_name: _FakeDinoModel()),
    )

    extractor = module.DinoV3ProjFeatureExtractor(
        "fake-dino",
        image_size=8,
        grid_resolution=2,
        use_naf_upsample=False,
    ).to(device)
    image = torch.rand(1, 3, 8, 8, device=device)

    _global_features, proj_features = extractor(
        image,
        camera_angle_x=torch.tensor([0.8], device=device),
        distance=torch.tensor([2.0], device=device),
        mesh_scale=torch.tensor([1.0], device=device),
    )

    assert extractor.proj_channels == extractor.embed_dim
    assert extractor.naf_model is None
    assert proj_features.shape == (1, 8, extractor.embed_dim)
