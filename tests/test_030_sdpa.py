from __future__ import annotations

import pytest


pytestmark = pytest.mark.milestone("M5")


def run_sdpa(torch, device: str, q_cpu, k_cpu, v_cpu):
    q = q_cpu.to(device)
    k = k_cpu.to(device)
    v = v_cpu.to(device)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v).detach().cpu()


def test_cpu_mps_sdpa_match_small_tensor():
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    torch.manual_seed(1234)
    q = torch.randn(2, 4, 8, 16, device="cpu", dtype=torch.float32)
    k = torch.randn(2, 4, 8, 16, device="cpu", dtype=torch.float32)
    v = torch.randn(2, 4, 8, 16, device="cpu", dtype=torch.float32)
    cpu = run_sdpa(torch, "cpu", q, k, v)
    mps = run_sdpa(torch, "mps", q, k, v)
    torch.testing.assert_close(mps, cpu, atol=1e-4, rtol=1e-4)
