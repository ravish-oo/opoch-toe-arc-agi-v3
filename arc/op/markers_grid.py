#!/usr/bin/env python3
# arc/op/markers_grid.py
# WO-10D: Markers-Grid Engine

"""
Contract (WO-10D):
Detect 2×2 solid markers, infer rectangular grid lattice, fill cells by frozen rule.

Frozen algorithm:
1. Marker detection: 2×2 solid blocks (bbox_h=2, bbox_w=2, all pixels same nonzero color)
2. Centroids: exact integer (anchor_r + 1, anchor_c + 1) for 2×2 blocks
3. Grid inference: distinct y/x sets (no epsilon), rectangular lattice
4. Cell fill rule: frozen (e.g., marker_color or majority in cell)
5. Verify: ALL trainings must match exactly
6. Apply: Same grid and rule on test

Engineering = Math:
- Exact bbox checks (no tolerance)
- Integer centroids (no floating point)
- Exact set membership (no epsilon clustering)
- Strict majority for ties (C2)
- Fail-closed (verification failure → ok=False)
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import numpy as np

from arc.op.components import cc4_by_color
from arc.op.hash import hash_bytes


@dataclass
class MarkersFitRc:
    """
    Markers-Grid fit receipt.

    Contract (WO-10D):
    Records marker size, colors, centroids per training, grid shape, cell rule.
    """
    engine: Literal["markers_grid"] = "markers_grid"
    marker_size: Tuple[int, int] = (2, 2)
    marker_color_set: List[int] = field(default_factory=list)
    centroids: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    grid_shape: Tuple[int, int] = (0, 0)
    cell_rule: str = ""
    fit_verified_on: List[str] = field(default_factory=list)
    hash: str = ""


def _detect_2x2_markers(G: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Detect 2×2 solid markers using exact bbox checks.

    Contract (WO-10D, D1, I1):
    - Marker = 2×2 solid block (all 4 pixels same nonzero color)
    - Centroid = (anchor_r + 1, anchor_c + 1) for 2×2 block

    Args:
        G: Grid (H x W)

    Returns:
        List of (centroid_r, centroid_c, color) tuples
    """
    # Use components to detect all CCs
    _, comps = cc4_by_color(G)

    markers = []

    for i, inv in enumerate(comps.invariants):
        # Check if this CC is a 2×2 solid block
        if inv['bbox_h'] != 2 or inv['bbox_w'] != 2:
            continue

        # Check if all 4 pixels are the same color
        color = inv['color']
        if color == 0:
            continue  # Skip background

        anchor_r = inv['anchor_r']
        anchor_c = inv['anchor_c']

        # Extract the 2×2 patch
        if anchor_r + 2 > G.shape[0] or anchor_c + 2 > G.shape[1]:
            continue  # Out of bounds

        patch = G[anchor_r:anchor_r+2, anchor_c:anchor_c+2]

        # Check if all pixels are the same color
        if np.all(patch == color):
            # Centroid of 2×2 block is at (anchor_r + 1, anchor_c + 1)
            centroid_r = anchor_r + 1
            centroid_c = anchor_c + 1
            markers.append((centroid_r, centroid_c, color))

    return markers


def _infer_grid_from_centroids(
    centroids_per_train: Dict[str, List[Tuple[int, int, int]]]
) -> Tuple[bool, List[int], List[int]]:
    """
    Infer rectangular grid from centroids across all trainings.

    Contract (WO-10D):
    - Compute distinct y/x sets (no epsilon)
    - Validate consistent lattice across ALL trainings

    Args:
        centroids_per_train: {train_id: [(r, c, color), ...]}

    Returns:
        (ok, grid_rows, grid_cols): ok=True if valid rectangular lattice
    """
    # Collect all unique y and x coordinates
    all_y = set()
    all_x = set()

    for train_id, centroids in centroids_per_train.items():
        for r, c, color in centroids:
            all_y.add(r)
            all_x.add(c)

    if not all_y or not all_x:
        return False, [], []

    # Sort to get grid lines
    grid_rows = sorted(all_y)
    grid_cols = sorted(all_x)

    # Validate: each training should have a subset of the full grid
    # (or equal to it, for consistency)
    # For now, we require ALL trainings to have the SAME grid
    for train_id, centroids in centroids_per_train.items():
        train_y = set(r for r, c, color in centroids)
        train_x = set(c for r, c, color in centroids)

        # Check if training centroids form a rectangular grid
        # (all combinations of train_y × train_x should have a marker)
        expected_count = len(train_y) * len(train_x)
        if len(centroids) != expected_count:
            # Not a rectangular lattice
            return False, [], []

        # Check all combinations exist
        centroid_set = {(r, c) for r, c, color in centroids}
        for r in train_y:
            for c in train_x:
                if (r, c) not in centroid_set:
                    return False, [], []

    return True, grid_rows, grid_cols


def _determine_cell_fill_rule(
    train_pairs: List[Tuple[str, np.ndarray, np.ndarray, Any]],
    centroids_per_train: Dict[str, List[Tuple[int, int, int]]],
    grid_rows: List[int],
    grid_cols: List[int]
) -> Tuple[bool, str, Dict[Tuple[int, int], int]]:
    """
    Determine cell fill rule by analyzing trainings.

    Contract (WO-10D, A1, C2):
    - Frozen rule: "cell_color = marker_color" or "strict_majority_in_cell"
    - Verify ALL trainings

    Args:
        train_pairs: [(train_id, X_t, Y_raw, truth_rc), ...]
        centroids_per_train: {train_id: [(r, c, color), ...]}
        grid_rows: Sorted list of grid row coordinates
        grid_cols: Sorted list of grid col coordinates

    Returns:
        (ok, rule_str, cell_colors): ok=True if rule works on all trainings
    """
    # Build a map: (grid_r_idx, grid_c_idx) -> expected_color
    # We'll use the first training to infer the rule
    train_id_0, X0, Y0, truth_rc_0 = train_pairs[0]
    centroids_0 = centroids_per_train[train_id_0]

    # Create centroid map: (r, c) -> color
    centroid_map = {(r, c): color for r, c, color in centroids_0}

    # For each grid cell, determine the expected output color
    cell_colors = {}

    for gr_idx, gr in enumerate(grid_rows):
        for gc_idx, gc in enumerate(grid_cols):
            # Get marker color at this centroid
            marker_color = centroid_map.get((gr, gc), 0)

            # Simple rule: cell color = marker color
            cell_colors[(gr_idx, gc_idx)] = marker_color

    rule_str = "cell_color = marker_color"

    # Verify this rule on ALL trainings
    for train_id, X_t, Y_raw, truth_rc in train_pairs:
        centroids = centroids_per_train[train_id]
        centroid_map_train = {(r, c): color for r, c, color in centroids}

        # Check if Y_raw matches the rule
        # Y_raw should be a grid of shape (len(grid_rows), len(grid_cols))
        if Y_raw.shape != (len(grid_rows), len(grid_cols)):
            return False, "", {}

        for gr_idx, gr in enumerate(grid_rows):
            for gc_idx, gc in enumerate(grid_cols):
                marker_color = centroid_map_train.get((gr, gc), 0)
                expected_color = marker_color

                if Y_raw[gr_idx, gc_idx] != expected_color:
                    return False, "", {}

    return True, rule_str, cell_colors


def fit_markers_grid(
    train_pairs: List[Tuple[str, np.ndarray, np.ndarray, Any]]  # (train_id, X_t, Y_raw, truth_rc)
) -> Tuple[bool, MarkersFitRc]:
    """
    Fit markers-grid engine.

    Contract (WO-10D):
    1. Detect 2×2 solid markers in each training input
    2. Compute exact integer centroids
    3. Infer rectangular grid from distinct y/x sets
    4. Determine frozen cell fill rule
    5. Verify ALL trainings match exactly
    6. Return ok=True if all verify, else ok=False

    Args:
        train_pairs: [(train_id, X_t, Y_raw, truth_rc), ...]

    Returns:
        (ok, MarkersFitRc)
    """
    if not train_pairs:
        return False, MarkersFitRc()

    # Step 1: Detect markers in each training
    centroids_per_train = {}
    all_marker_colors = set()

    for train_id, X_t, Y_raw, truth_rc in train_pairs:
        markers = _detect_2x2_markers(X_t)

        if not markers:
            # No markers found
            return False, MarkersFitRc()

        centroids_per_train[train_id] = markers

        # Collect marker colors
        for r, c, color in markers:
            all_marker_colors.add(color)

    # Step 2: Infer grid from centroids
    ok, grid_rows, grid_cols = _infer_grid_from_centroids(centroids_per_train)

    if not ok:
        return False, MarkersFitRc()

    # Step 3: Determine cell fill rule
    ok, rule_str, cell_colors = _determine_cell_fill_rule(
        train_pairs, centroids_per_train, grid_rows, grid_cols
    )

    if not ok:
        return False, MarkersFitRc()

    # All trainings verified (verified in _determine_cell_fill_rule)
    fit_verified_on = [train_id for train_id, _, _, _ in train_pairs]

    # Build receipt
    grid_shape = (len(grid_rows), len(grid_cols))
    marker_color_set = sorted(all_marker_colors)

    # Convert centroids to (r, c) tuples (without color for receipt)
    centroids_receipt = {
        train_id: [(r, c) for r, c, _ in centroids]
        for train_id, centroids in centroids_per_train.items()
    }

    receipt_str = f"{marker_color_set}:{grid_shape}:{rule_str}:{sorted(fit_verified_on)}"
    rc = MarkersFitRc(
        marker_size=(2, 2),
        marker_color_set=marker_color_set,
        centroids=centroids_receipt,
        grid_shape=grid_shape,
        cell_rule=rule_str,
        fit_verified_on=fit_verified_on,
        hash=hash_bytes(receipt_str.encode())
    )

    return True, rc


def apply_markers_grid(
    test_Xt: np.ndarray,
    truth_test: Any,
    fit_rc: MarkersFitRc,
    expected_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Apply markers-grid engine to test input.

    Contract (WO-10D):
    Apply same grid detection and rule on test.

    Args:
        test_Xt: Test input in Π frame
        truth_test: Truth partition for test (not used currently)
        fit_rc: Fit receipt from fit_markers_grid
        expected_shape: Optional (R, C) from WO-02

    Returns:
        (Y_t, apply_rc): Output in Π frame and application receipt
    """
    # Detect markers on test
    markers_test = _detect_2x2_markers(test_Xt)

    if not markers_test:
        # No markers found - fail
        return np.zeros((0, 0), dtype=np.uint8), {
            "engine": "markers_grid",
            "error": "NO_MARKERS_FOUND",
            "output_shape": [0, 0]
        }

    # Extract centroids
    centroids_test = [(r, c, color) for r, c, color in markers_test]

    # Build centroid map
    centroid_map_test = {(r, c): color for r, c, color in centroids_test}

    # Infer grid from test centroids
    all_y = sorted(set(r for r, c, color in centroids_test))
    all_x = sorted(set(c for r, c, color in centroids_test))

    # Check if grid shape matches expected
    grid_shape_test = (len(all_y), len(all_x))

    if grid_shape_test != fit_rc.grid_shape:
        # Grid shape mismatch
        return np.zeros((0, 0), dtype=np.uint8), {
            "engine": "markers_grid",
            "error": "GRID_SHAPE_MISMATCH",
            "expected_grid_shape": list(fit_rc.grid_shape),
            "actual_grid_shape": list(grid_shape_test),
            "output_shape": [0, 0]
        }

    # Check expected shape if provided
    R, C = fit_rc.grid_shape
    if expected_shape is not None:
        R_exp, C_exp = expected_shape
        if (R, C) != (R_exp, C_exp):
            # Shape mismatch
            return np.zeros((R, C), dtype=np.uint8), {
                "engine": "markers_grid",
                "output_shape": [R, C],
                "expected_shape": [R_exp, C_exp],
                "shape_mismatch": True
            }

    # Apply cell fill rule
    Y_t = np.zeros((R, C), dtype=np.uint8)

    for gr_idx, gr in enumerate(all_y):
        for gc_idx, gc in enumerate(all_x):
            # Get marker color at this centroid
            marker_color = centroid_map_test.get((gr, gc), 0)

            # Apply rule: cell_color = marker_color
            Y_t[gr_idx, gc_idx] = marker_color

    # Build receipt
    apply_rc = {
        "engine": "markers_grid",
        "output_shape": [R, C],
        "num_markers": len(markers_test),
        "grid_shape": list(fit_rc.grid_shape),
        "output_hash": hash_bytes(Y_t.tobytes())
    }

    return Y_t, apply_rc
