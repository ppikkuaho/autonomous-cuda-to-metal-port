from __future__ import annotations

import json
import subprocess
import sys

import pytest

from conftest import ROOT, WORK_PIXAL3D, env_enabled


pytestmark = [pytest.mark.milestone("M9"), pytest.mark.requires_pixal3d]


def require_enabled():
    if not env_enabled("RUN_PIPELINE_LOAD_TESTS"):
        pytest.skip("Pipeline load test starts after image conditioning checks")


def test_pipeline_load_script_help():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pipeline_load_smoke.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert proc.returncode == 0
    assert "--allow-external-gate" in proc.stdout


def test_pipeline_load_script_records_external_gate(monkeypatch, tmp_path):
    require_enabled()
    sys.path.insert(0, str(WORK_PIXAL3D))
    import scripts.pipeline_load_smoke as smoke

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path):
            raise RuntimeError("GatedRepoError: 401 Client Error. Cannot access gated repo")

    monkeypatch.setitem(sys.modules, "pixal3d.pipelines", type("P", (), {"Pixal3DImageTo3DPipeline": FakePipeline}))

    def fake_resolve_device(device=None):
        return "mps"

    monkeypatch.setitem(sys.modules, "pixal3d.utils.device", type("D", (), {"resolve_device": fake_resolve_device}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline_load_smoke.py",
            "--model",
            "TencentARC/Pixal3D",
            "--out",
            str(tmp_path),
            "--allow-external-gate",
        ],
    )

    assert smoke.main() == 0
    report = json.loads((tmp_path / "pipeline_load_report.json").read_text())
    assert report["status"] == "PARTIAL"
    assert report["error_category"] == "external_hf_access_gate"


def test_birefnet_is_lazy_and_device_aware():
    require_enabled()
    sys.path.insert(0, str(WORK_PIXAL3D))
    from pixal3d.pipelines.rembg.BiRefNet import BiRefNet

    rembg = BiRefNet("briaai/RMBG-2.0")
    rembg.to("cpu")

    assert rembg.model is None
    assert str(rembg.device) == "cpu"
