from __future__ import annotations

import pytest

from conftest import env_enabled


pytestmark = [pytest.mark.milestone("M11"), pytest.mark.requires_pixal3d]


def test_glb_export_gated():
    if not env_enabled("RUN_GLB_EXPORT_TESTS"):
        pytest.skip("GLB export test starts after geometry smoke")

