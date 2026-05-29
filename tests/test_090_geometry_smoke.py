from __future__ import annotations

import inspect
import sys

import pytest

from conftest import WORK_PIXAL3D, env_enabled


pytestmark = [pytest.mark.milestone("M10"), pytest.mark.requires_pixal3d]


def require_enabled():
    if not env_enabled("RUN_GEOMETRY_SMOKE_TESTS"):
        pytest.skip("Geometry smoke test starts after pipeline load")


def test_pipeline_run_exposes_geometry_only_flag():
    require_enabled()
    sys.path.insert(0, str(WORK_PIXAL3D))
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline

    signature = inspect.signature(Pixal3DImageTo3DPipeline.run)

    assert "texture" in signature.parameters
    assert signature.parameters["texture"].default is True


def test_geometry_only_export_helper_roundtrips_glb(tmp_path):
    require_enabled()
    torch = pytest.importorskip("torch")
    trimesh = pytest.importorskip("trimesh")
    sys.path.insert(0, str(WORK_PIXAL3D))
    from inference import export_geometry_glb

    class Mesh:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)

    out = tmp_path / "geometry_only.glb"
    export_geometry_glb(Mesh(), str(out))
    loaded = trimesh.load(out, force="mesh")

    assert out.exists()
    assert len(loaded.vertices) == 4
    assert len(loaded.faces) == 2


def test_textured_export_fallback_writes_texture_visual_glb(tmp_path):
    require_enabled()
    torch = pytest.importorskip("torch")
    trimesh = pytest.importorskip("trimesh")
    sys.path.insert(0, str(WORK_PIXAL3D))
    from inference import export_textured_glb_fallback, sample_vertex_base_colors

    class Mesh:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)
        origin = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        voxel_size = 1.0
        coords = vertices.to(torch.long)
        attrs = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.5, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.5, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
                [1.0, 1.0, 0.0, 0.0, 0.5, 1.0],
            ],
            dtype=torch.float32,
        )
        layout = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}

    class Pipeline:
        pbr_attr_layout = Mesh.layout

    colors, match_fraction = sample_vertex_base_colors(Mesh(), Pipeline())
    assert match_fraction == 1.0
    assert colors.shape == (4, 3)

    class MeshWithMissingVoxel(Mesh):
        coords = Mesh.coords[:3]
        attrs = Mesh.attrs[:3]

    _colors, partial_match_fraction = sample_vertex_base_colors(MeshWithMissingVoxel(), Pipeline())
    assert 0.0 < partial_match_fraction < 1.0

    out = tmp_path / "textured_fallback.glb"
    export_textured_glb_fallback(Mesh(), Pipeline(), str(out), texture_size=64)
    loaded = trimesh.load(out, force="scene")
    geom = next(iter(loaded.geometry.values()))

    assert out.exists()
    assert len(geom.vertices) == 4
    assert len(geom.faces) == 2
    assert geom.visual.kind == "texture"
