#!/usr/bin/env python3
# arc/op/border_scalar.py
# WO-10B: Border-Scalar Engine

"""
Contract (WO-10B):
Reconstruct small outputs with constant border/interior fills derived from
largest 4-CC border color and minimal interior 4-CC color.

Frozen algorithm (Spec-B "a"):
1. Frame bands: outermost rows/cols as border (row=0 | row=H*-1 | col=0 | col=W*-1)
2. Border color: max area 4-CC on border band; ties→smallest color (A1, C1)
3. Interior color: min area >0 4-CC on interior; ties→smallest color (A2)
4. Verify trainings: check Yᵢ border=border_color, interior=interior_color
5. Apply: paint border with border_color, interior with interior_color

Engineering = Math:
- CC areas exact integers
- Tie-breaks frozen (smallest color)
- Fail-closed (verification failure → ok=False)
- No thresholds, no learning
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import numpy as np

from arc.op.components import cc4_by_color
from arc.op.hash import hash_bytes
from arc.op.admit import empty_domains, _set_bit, _normalize_scope


@dataclass
class BorderScalarFitRc:
    """
    Border-Scalar fit receipt.

    Contract (WO-10B):
    Records border_color, interior_color, rule, verification status.
    """
    engine: Literal["border_scalar"] = "border_scalar"
    border_color: int = 0
    interior_color: int = 0
    rule: str = ""
    band_source: str = "outermost_rows_cols"
    fit_verified_on: List[str] = field(default_factory=list)
    counts: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""


def _extract_border_region(G: np.ndarray) -> np.ndarray:
    """
    Extract border region (outermost rows/cols).

    Contract:
    Border = {(r,c) : r=0 or r=H-1 or c=0 or c=W-1}

    Returns:
        Boolean mask (H x W) marking border pixels
    """
    H, W = G.shape
    border_mask = np.zeros((H, W), dtype=bool)

    if H > 0 and W > 0:
        border_mask[0, :] = True   # Top row
        border_mask[H-1, :] = True # Bottom row
        border_mask[:, 0] = True   # Left col
        border_mask[:, W-1] = True # Right col

    return border_mask


def _extract_interior_region(G: np.ndarray) -> np.ndarray:
    """
    Extract interior region (exclude border).

    Returns:
        Boolean mask (H x W) marking interior pixels
    """
    border_mask = _extract_border_region(G)
    return ~border_mask


def _compute_cc_areas_on_mask(G: np.ndarray, mask: np.ndarray) -> Dict[int, List[int]]:
    """
    Compute 4-CC areas per color restricted to mask region.

    Contract (D2):
    Use 4-connectivity only.

    Returns:
        {color: [area1, area2, ...]} for each color present in masked region
    """
    H, W = G.shape

    # Mask the grid
    masked_grid = np.where(mask, G, -1)  # Use -1 for masked-out regions

    # Extract CCs for each color
    cc_areas_per_color = {}

    unique_colors = np.unique(masked_grid)
    for color in unique_colors:
        if color == -1:
            continue  # Skip masked regions

        color_mask = (masked_grid == color)

        # Get CCs for this color using cc4_by_color
        # Need to create a grid with only this color
        single_color_grid = np.where(color_mask, color, 0).astype(np.uint8)
        cc_masks, cc_rc = cc4_by_color(single_color_grid)

        # Compute areas for each CC
        areas = []
        for i, cc_mask in enumerate(cc_masks):
            if np.any(cc_mask > 0):
                # cc_mask may be cropped to bbox - we need to count pixels in original mask region
                # Get the invariant for this component to find its bbox
                inv = cc_rc.invariants[i]
                r0 = inv['anchor_r']
                c0 = inv['anchor_c']
                h_cc, w_cc = cc_mask.shape

                # Count pixels that are both in cc_mask AND in original mask region
                area = 0
                for r_local in range(h_cc):
                    for c_local in range(w_cc):
                        if cc_mask[r_local, c_local] > 0:
                            r_global = r0 + r_local
                            c_global = c0 + c_local
                            # Check if this pixel is in the original mask region
                            if 0 <= r_global < H and 0 <= c_global < W and mask[r_global, c_global]:
                                area += 1

                if area > 0:
                    areas.append(int(area))

        if areas:
            cc_areas_per_color[int(color)] = sorted(areas, reverse=True)

    return cc_areas_per_color


def fit_border_scalar(
    train_pairs: List[Tuple[str, np.ndarray, np.ndarray]]  # (train_id, X_t, Y_t) in Π frame
) -> Tuple[bool, BorderScalarFitRc]:
    """
    Fit border-scalar engine.

    Contract (WO-10B):
    1. Extract border/interior regions from first training input
    2. Find border_color: max CC area on border, tie→min color
    3. Find interior_color: min CC area >0 on interior, tie→min color
    4. Verify ALL trainings: Y border=border_color, interior=interior_color
    5. Return ok=True if all verify, else ok=False

    Args:
        train_pairs: [(train_id, X_t, Y_t), ...] where X_t, Y_t in Π frame

    Returns:
        (ok, BorderScalarFitRc)
    """
    if not train_pairs:
        return False, BorderScalarFitRc()

    # Use first training input to determine border/interior colors
    train_id_0, X0, Y0 = train_pairs[0]
    H, W = X0.shape

    # Extract regions
    border_mask = _extract_border_region(X0)
    interior_mask = _extract_interior_region(X0)

    # Compute CC areas on border
    border_cc_areas = _compute_cc_areas_on_mask(X0, border_mask)

    # Find border_color: max area, tie→min color (A1, C1)
    border_color = 0
    max_border_area = 0

    for color, areas in sorted(border_cc_areas.items()):  # sorted by color for tie-break
        max_area_this_color = max(areas) if areas else 0
        if max_area_this_color > max_border_area:
            max_border_area = max_area_this_color
            border_color = color
        elif max_area_this_color == max_border_area and max_area_this_color > 0:
            # Tie: pick smallest color
            border_color = min(border_color, color)

    # Compute CC areas on interior
    interior_cc_areas = _compute_cc_areas_on_mask(X0, interior_mask)

    # Find interior_color: min area >0, tie→min color (A2)
    interior_color = 0  # Default background
    min_interior_area = float('inf')

    for color, areas in sorted(interior_cc_areas.items()):
        min_area_this_color = min(areas) if areas else 0
        if min_area_this_color > 0:
            if min_area_this_color < min_interior_area:
                min_interior_area = min_area_this_color
                interior_color = color
            elif min_area_this_color == min_interior_area:
                # Tie: pick smallest color
                interior_color = min(interior_color, color)

    # If no interior CCs, set to 0
    if min_interior_area == float('inf'):
        interior_color = 0

    # Build rule string
    rule = f"border_max_cc(area={max_border_area})→color_{border_color}; interior_min_cc(area={min_interior_area})→color_{interior_color}; tie→min_color"

    # Verify on ALL trainings
    fit_verified_on = []
    verification_failures = []

    for train_id, X_t, Y_t in train_pairs:
        H_y, W_y = Y_t.shape

        # Extract border and interior from Y
        border_mask_y = _extract_border_region(Y_t)
        interior_mask_y = _extract_interior_region(Y_t)

        # Check border: all border pixels = border_color
        border_pixels = Y_t[border_mask_y]
        border_ok = np.all(border_pixels == border_color)

        # Check interior: all interior pixels = interior_color (if interior exists)
        interior_pixels = Y_t[interior_mask_y]
        if len(interior_pixels) > 0:
            interior_ok = np.all(interior_pixels == interior_color)
        else:
            interior_ok = True  # No interior to check

        if border_ok and interior_ok:
            fit_verified_on.append(train_id)
        else:
            verification_failures.append({
                "train_id": train_id,
                "border_ok": border_ok,
                "interior_ok": interior_ok,
                "border_unique": list(np.unique(border_pixels)) if not border_ok else [],
                "interior_unique": list(np.unique(interior_pixels)) if not interior_ok and len(interior_pixels) > 0 else []
            })

    # Success if ALL trainings verified
    ok = (len(fit_verified_on) == len(train_pairs))

    # Build receipt
    counts = {
        "border_cc_areas": {int(k): v for k, v in border_cc_areas.items()},
        "interior_cc_areas": {int(k): v for k, v in interior_cc_areas.items()},
        "max_border_area": int(max_border_area),
        "min_interior_area": int(min_interior_area) if min_interior_area != float('inf') else 0,
        "verification_failures": verification_failures
    }

    receipt_str = f"{border_color}:{interior_color}:{rule}:{sorted(fit_verified_on)}"
    rc = BorderScalarFitRc(
        border_color=int(border_color),
        interior_color=int(interior_color),
        rule=rule,
        fit_verified_on=fit_verified_on,
        counts=counts,
        hash=hash_bytes(receipt_str.encode())
    )

    return ok, rc


def apply_border_scalar(
    test_Xt: np.ndarray,
    fit_rc: BorderScalarFitRc,
    C: List[int],
    expected_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Apply border-scalar engine to test input (emit native admits).

    Contract (WO-10B + WO-11A):
    Emit singleton admits for border (border_color) and interior (interior_color).

    Args:
        test_Xt: Test input in Π frame
        fit_rc: Fit receipt from fit_border_scalar
        C: Color universe (sorted unique colors)
        expected_shape: Optional (R, C) from WO-02

    Returns:
        (A, S, apply_rc): Admit bitmap, scope mask, and application receipt
    """
    # Determine output shape
    if expected_shape is not None:
        R, C_out = expected_shape
    else:
        # Use test input shape (identity size)
        R, C_out = test_Xt.shape

    # Build color index lookup
    color_to_idx = {c: i for i, c in enumerate(C)}

    # Initialize admits (all colors allowed) and scope (all silent)
    A = empty_domains(R, C_out, C)
    S = np.zeros((R, C_out), dtype=np.uint8)

    # Create temporary grid for mask extraction
    temp_grid = np.zeros((R, C_out), dtype=np.uint8)

    # Extract regions
    border_mask = _extract_border_region(temp_grid)
    interior_mask = _extract_interior_region(temp_grid)

    # Emit singleton admits for border pixels
    border_color = fit_rc.border_color
    if border_color in color_to_idx:
        border_idx = color_to_idx[border_color]
        for r in range(R):
            for c in range(C_out):
                if border_mask[r, c]:
                    # Clear all bits, then set only border_color
                    A[r, c, :] = 0
                    _set_bit(A[r, c], border_idx)
                    S[r, c] = 1  # Mark as scoped

    # Emit singleton admits for interior pixels
    interior_color = fit_rc.interior_color
    if interior_color in color_to_idx:
        interior_idx = color_to_idx[interior_color]
        for r in range(R):
            for c in range(C_out):
                if interior_mask[r, c]:
                    # Clear all bits, then set only interior_color
                    A[r, c, :] = 0
                    _set_bit(A[r, c], interior_idx)
                    S[r, c] = 1  # Mark as scoped

    # Normalize (if A[p]=C, set S[p]=0)
    _normalize_scope(A, S, C)

    # Compute stats for receipt
    scope_bits = int(S.sum())

    # Build receipt
    apply_rc = {
        "engine": "border_scalar",
        "border_color": border_color,
        "interior_color": interior_color,
        "output_shape": [R, C_out],
        "border_pixels": int(np.sum(border_mask)),
        "interior_pixels": int(np.sum(interior_mask)),
        "scope_bits": scope_bits,
        "bitmap_hash": hash_bytes(A.tobytes()),
        "scope_hash": hash_bytes(S.tobytes())
    }

    return A, S, apply_rc
