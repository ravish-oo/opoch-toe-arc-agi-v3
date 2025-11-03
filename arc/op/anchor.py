# arc/op/anchor.py
# WO-01: Anchor grid to origin
# Implements 01_engineering_spec.md §2

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class AnchorRc:
    """
    Anchor offset receipt.

    Contract (02_determinism_addendum.md §1.2):
    Anchor offsets are signed integers (dr, dc) encoded as ZigZag LEB128 varints
    when serialized to bytes.
    """
    dr: int
    dc: int


def anchor_to_origin(G: np.ndarray) -> tuple[np.ndarray, AnchorRc]:
    """
    Translate grid so bbox of nonzero support touches (0,0).

    Algorithm:
    1. Find all nonzero positions
    2. If all zeros: return (G, (0, 0))
    3. Otherwise: find min row and min col of nonzero positions
    4. Shift grid so min row → 0, min col → 0 (NON-SHRINKING)

    Contract (01_engineering_spec.md line 46):
    "Anchor: translate so bbox of nonzero support touches (0,0)"

    Contract (F2 Fix - WO01_SWEEP_BUG_REPORT.md):
    "Make anchoring non-shrinking. Π must only shift the bbox to (0,0) and keep
    the original H×W. Never crop on anchor. With a non-shrinking anchor, applying
    Π again returns the same array and pose, so Π²=Π holds."

    Args:
        G: input grid (H×W)

    Returns:
        (anchored_grid, AnchorRc(dr, dc))
        - anchored_grid: translated grid (SAME SHAPE as input)
        - dr, dc: original top-left position of bbox (shift applied)
    """
    # Find nonzero positions
    nz = np.argwhere(G != 0)

    # All zeros case: no shift needed
    if nz.size == 0:
        return G.copy(), AnchorRc(dr=0, dc=0)

    # Find bbox of nonzero support
    rmin = int(nz[:, 0].min())
    cmin = int(nz[:, 1].min())

    # Already at origin: no shift needed (IDEMPOTENCE GUARD)
    if rmin == 0 and cmin == 0:
        return G.copy(), AnchorRc(dr=0, dc=0)

    # NON-SHRINKING SHIFT: Keep original shape, shift content to (0,0)
    # Before (BUGGY): out = G[rmin:, cmin:].copy()  # ❌ crops, changes shape
    # After (CORRECT): shift without changing H×W
    H, W = G.shape

    # WO-11G: Bounds check before slicing
    if rmin < 0 or cmin < 0 or rmin >= H or cmin >= W:
        raise RuntimeError(
            f"anchor_oob: bbox_min=({rmin},{cmin}) vs grid=({H},{W})"
        )

    out = np.zeros_like(G)  # Same shape as input
    out[0 : H - rmin, 0 : W - cmin] = G[rmin:H, cmin:W]  # Shift content

    return out, AnchorRc(dr=rmin, dc=cmin)


def unanchor_from_origin(G: np.ndarray, anchor: AnchorRc) -> np.ndarray:
    """
    Inverse of anchor_to_origin: shift grid back to original position.

    Used in unpresentation (U⁻¹).

    Contract (F2 Fix - WO01_SWEEP_BUG_REPORT.md):
    With non-shrinking anchor, G is already the correct size. We shift content
    back without changing the grid shape.

    Args:
        G: anchored grid (same shape as original due to non-shrinking anchor)
        anchor: AnchorRc with (dr, dc) shift

    Returns:
        Grid shifted back by (dr, dc) (same shape as input)
    """
    dr, dc = anchor.dr, anchor.dc

    # No shift: return copy
    if dr == 0 and dc == 0:
        return G.copy()

    # NON-SHRINKING UNANCHOR: Shift content back within same grid size
    # Before (OLD): expanded canvas (H+dr, W+dc) - incompatible with non-shrinking anchor
    # After (CORRECT): shift within same H×W
    H, W = G.shape

    # WO-11G: Bounds check before slicing
    if dr < 0 or dc < 0 or dr >= H or dc >= W:
        raise RuntimeError(
            f"unanchor_oob: anchor=({dr},{dc}) vs grid=({H},{W})"
        )

    out = np.zeros_like(G)  # Same shape as input

    # Shift content from [0:H-dr, 0:W-dc] back to [dr:H, dc:W]
    out[dr:H, dc:W] = G[0 : H - dr, 0 : W - dc]

    return out
