from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from conftest import ROOT, WORK_PIXAL3D, env_enabled


pytestmark = [pytest.mark.milestone("M7"), pytest.mark.requires_backend]


def require_enabled():
    if not env_enabled("RUN_MESH_TESTS"):
        pytest.skip("Mesh extraction tests start after backend selection")


def mesh_modules():
    require_enabled()
    sys.path.insert(0, str(WORK_PIXAL3D))
    from pixal3d.utils.mesh_extract import flexible_dual_grid_to_mesh
    from pixal3d.representations import Mesh

    return flexible_dual_grid_to_mesh, Mesh


def artifact_dir() -> Path | None:
    raw = os.environ.get("MESH_TEST_OUTPUT_DIR")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def single_quad_dual_grid(torch, device):
    coords = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=torch.long,
        device=device,
    )
    dual_vertices = torch.full((4, 3), 0.5, dtype=torch.float32, device=device)
    intersected = torch.zeros((4, 3), dtype=torch.bool, device=device)
    intersected[0, 0] = True
    split_weight = torch.ones((4, 1), dtype=torch.float32, device=device)
    return coords, dual_vertices, intersected, split_weight


def available_devices():
    torch = pytest.importorskip("torch")
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


@pytest.mark.parametrize("device", available_devices())
def test_flexible_dual_grid_to_mesh_single_quad(device):
    torch = pytest.importorskip("torch")
    flexible_dual_grid_to_mesh, Mesh = mesh_modules()
    coords, dual_vertices, intersected, split_weight = single_quad_dual_grid(torch, device)

    vertices, faces = flexible_dual_grid_to_mesh(
        coords,
        dual_vertices,
        intersected,
        split_weight,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=2,
    )
    mesh = Mesh(vertices, faces)

    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)
    assert torch.isfinite(mesh.vertices).all()
    assert mesh.faces.min() >= 0
    assert mesh.faces.max() < mesh.vertices.shape[0]
    expected_bounds = torch.tensor([[-0.25, -0.25, -0.25], [-0.25, 0.25, 0.25]], dtype=torch.float32)
    torch.testing.assert_close(
        torch.stack([mesh.vertices.min(dim=0).values.cpu(), mesh.vertices.max(dim=0).values.cpu()]),
        expected_bounds,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("device", available_devices())
def test_flexible_dual_grid_to_mesh_empty_flags(device):
    torch = pytest.importorskip("torch")
    flexible_dual_grid_to_mesh, _Mesh = mesh_modules()
    coords, dual_vertices, intersected, split_weight = single_quad_dual_grid(torch, device)
    intersected.zero_()

    vertices, faces = flexible_dual_grid_to_mesh(
        coords,
        dual_vertices,
        intersected,
        split_weight,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=2,
    )

    assert vertices.shape == (0, 3)
    assert faces.shape == (0, 3)


def test_fdg_vae_import_uses_local_mesh_fallback_without_o_voxel():
    require_enabled()
    sys.modules.pop("o_voxel", None)
    sys.modules.pop("o_voxel.convert", None)
    sys.path.insert(0, str(WORK_PIXAL3D))

    from pixal3d.models.sc_vaes import fdg_vae

    assert fdg_vae.flexible_dual_grid_to_mesh.__module__ == "pixal3d.utils.mesh_extract"
    assert "o_voxel._C" not in sys.modules


def test_mesh_cpu_cleanup_fallbacks_without_cumesh():
    torch = pytest.importorskip("torch")
    _flexible_dual_grid_to_mesh, Mesh = mesh_modules()

    mesh = Mesh(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long),
    )
    mesh.fill_holes()
    mesh.simplify(target=1)
    mesh.remove_faces(torch.tensor([True, False]))

    assert mesh.faces.shape == (1, 3)
    assert torch.equal(mesh.faces[0].cpu(), torch.tensor([0, 1, 3], dtype=torch.int32))


def test_synthetic_mesh_exports_and_reloads_glb(tmp_path):
    torch = pytest.importorskip("torch")
    trimesh = pytest.importorskip("trimesh")
    flexible_dual_grid_to_mesh, _Mesh = mesh_modules()
    coords, dual_vertices, intersected, split_weight = single_quad_dual_grid(torch, torch.device("cpu"))

    vertices, faces = flexible_dual_grid_to_mesh(
        coords,
        dual_vertices,
        intersected,
        split_weight,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=2,
    )
    mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(), process=False)
    out_dir = artifact_dir() or tmp_path
    out_file = out_dir / "synthetic_dual_grid_quad.glb"
    mesh.export(out_file)

    loaded = trimesh.load(out_file, force="mesh")

    assert out_file.exists()
    assert len(loaded.vertices) == 4
    assert len(loaded.faces) == 2
