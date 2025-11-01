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
    4. Shift grid so min row → 0, min col → 0

    Contract (01_engineering_spec.md line 46):
    "Anchor: translate so bbox of nonzero support touches (0,0)"

    Args:
        G: input grid (H×W)

    Returns:
        (anchored_grid, AnchorRc(dr, dc))
        - anchored_grid: translated grid
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

    # Already at origin: no shift needed
    if rmin == 0 and cmin == 0:
        return G.copy(), AnchorRc(dr=0, dc=0)

    # Crop grid to remove empty rows/cols before bbox
    # This reduces the grid size by (rmin, cmin)
    out = G[rmin:, cmin:].copy()

    return out, AnchorRc(dr=rmin, dc=cmin)


def unanchor_from_origin(G: np.ndarray, anchor: AnchorRc) -> np.ndarray:
    """
    Inverse of anchor_to_origin: shift grid back to original position.

    Used in unpresentation (U⁻¹).

    Args:
        G: anchored grid
        anchor: AnchorRc with (dr, dc) shift

    Returns:
        Grid shifted back by (dr, dc)
    """
    dr, dc = anchor.dr, anchor.dc

    # No shift: return copy
    if dr == 0 and dc == 0:
        return G.copy()

    # Expand canvas and shift back
    H, W = G.shape
    out = np.zeros((H + dr, W + dc), dtype=G.dtype)
    out[dr : dr + H, dc : dc + W] = G

    return out
