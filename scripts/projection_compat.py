from __future__ import annotations

import torch
import torch.nn.functional as F


def _border_clamp_grid_for_zero_padding(
    grid: torch.Tensor,
    height: int,
    width: int,
    align_corners: bool,
) -> torch.Tensor:
    """Map border-padding semantics to a clamped grid usable with zero padding.

    MPS currently rejects `grid_sample(..., padding_mode="border")` on this
    machine. For bilinear 2D sampling, border padding is equivalent to clamping
    sampling coordinates to the image border before interpolation. With
    `align_corners=False`, the normalized coordinates of border pixel centers are
    not exactly -1 and 1, so the clamp range must be adjusted per dimension.
    """
    clamped = grid.clone()
    if align_corners:
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
    else:
        x_min, x_max = -1.0 + 1.0 / width, 1.0 - 1.0 / width
        y_min, y_max = -1.0 + 1.0 / height, 1.0 - 1.0 / height
    clamped[..., 0] = clamped[..., 0].clamp(x_min, x_max)
    clamped[..., 1] = clamped[..., 1].clamp(y_min, y_max)
    return clamped


def grid_sample_2d(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """Device-compatible 2D grid_sample wrapper for Pixal3D projection tests.

    Supports the Pixal3D projection use case: 4D image feature maps and 2D
    sampling grids. `padding_mode="border"` is emulated by clamping the grid and
    then calling `grid_sample` with zero padding, which avoids the native MPS
    unsupported border-padding path.
    """
    if input.ndim != 4:
        raise ValueError(f"expected 4D input [N,C,H,W], got shape {tuple(input.shape)}")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(f"expected 4D grid [N,H_out,W_out,2], got shape {tuple(grid.shape)}")
    if mode not in {"bilinear", "nearest"}:
        raise ValueError(f"unsupported mode for compatibility wrapper: {mode}")

    if padding_mode == "border":
        _, _, height, width = input.shape
        grid = _border_clamp_grid_for_zero_padding(grid, height, width, align_corners)
        padding_mode = "zeros"

    return F.grid_sample(
        input,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

