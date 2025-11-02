#!/usr/bin/env python3
# arc/op/pooled_blocks.py
# WO-10C: Pooled-Blocks Engine

"""
Contract (WO-10C):
Two-stage block voting from band grids: block votes → pooled quadrants.

Frozen algorithm:
1. Bands: From WO-05 row/col_clusters (D1)
2. Stage-1: Per band cell (br,bc), compute foreground counts → strict majority
3. Stage-2: Pool stage-1 grid (e.g., 2×2) → strict majority per pooled cell
4. Verify: ALL trainings must match exactly
5. Apply: Same bands and rule on test

Engineering = Math:
- Counts exact integers
- Strict majority (NOT mode): count > total/2 (C2)
- No majority → background=0 (A2)
- Ties → smallest color (C2)
- Fail-closed (verification failure → ok=False)
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import numpy as np

from arc.op.hash import hash_bytes


@dataclass
class PooledBlocksFitRc:
    """
    Pooled-Blocks fit receipt.

    Contract (WO-10C):
    Records bands, block/pool shapes, foreground colors, stage-1/2 results.
    """
    engine: Literal["pooled_blocks"] = "pooled_blocks"
    row_bands: List[int] = field(default_factory=list)
    col_bands: List[int] = field(default_factory=list)
    block_shape: Tuple[int, int] = (0, 0)  # (num_row_bands, num_col_bands)
    pool_shape: Tuple[int, int] = (2, 2)   # Fixed for now
    foreground_colors: List[int] = field(default_factory=list)
    background_colors: List[int] = field(default_factory=list)
    stage1_counts: Dict[str, Dict[int, int]] = field(default_factory=dict)  # "(br,bc)" -> {color: count}
    stage2_pooled: List[List[int]] = field(default_factory=list)
    decision_rule: str = "strict_majority_foreground_fallback_0"
    fit_verified_on: List[str] = field(default_factory=list)
    hash: str = ""


def _extract_foreground_colors(train_Y_list: List[np.ndarray]) -> Tuple[List[int], List[int]]:
    """
    Extract foreground and background colors from training outputs.

    Contract (A1):
    Foreground = all non-zero colors appearing in ANY training output.
    Background = {0}.

    Returns:
        (foreground_colors, background_colors): sorted lists
    """
    all_colors = set()
    for Y in train_Y_list:
        all_colors.update(np.unique(Y))

    # Background is always 0
    background_colors = [0]

    # Foreground is all non-zero colors
    foreground_colors = sorted([c for c in all_colors if c != 0])

    return foreground_colors, background_colors


def _compute_stage1_votes(
    X: np.ndarray,
    row_bands: List[int],
    col_bands: List[int],
    foreground_colors: List[int]
) -> Tuple[np.ndarray, Dict[str, Dict[int, int]]]:
    """
    Compute stage-1 votes: per band cell, count foreground colors → strict majority.

    Contract (C2):
    - Strict majority: count > total/2 (NOT mode)
    - No majority → background=0 (A2)
    - Ties among majority winners → smallest color (C2)

    Args:
        X: Input grid (H x W)
        row_bands: Band boundaries [r0, r1, ...] (from truth row_clusters)
        col_bands: Band boundaries [c0, c1, ...] (from truth col_clusters)
        foreground_colors: List of valid foreground colors

    Returns:
        (stage1_grid, stage1_counts):
        - stage1_grid: (num_row_bands x num_col_bands) grid of voted colors
        - stage1_counts: {(br,bc): {color: count}} for all cells
    """
    H, W = X.shape
    num_row_bands = len(row_bands) - 1 if len(row_bands) > 1 else 1
    num_col_bands = len(col_bands) - 1 if len(col_bands) > 1 else 1

    stage1_grid = np.zeros((num_row_bands, num_col_bands), dtype=np.uint8)
    stage1_counts = {}

    for br in range(num_row_bands):
        r_start = row_bands[br] if br < len(row_bands) else 0
        r_end = row_bands[br + 1] if (br + 1) < len(row_bands) else H

        for bc in range(num_col_bands):
            c_start = col_bands[bc] if bc < len(col_bands) else 0
            c_end = col_bands[bc + 1] if (bc + 1) < len(col_bands) else W

            # Extract patch
            patch = X[r_start:r_end, c_start:c_end]

            # Count foreground colors only
            color_counts = {}
            total_pixels = patch.size

            for color in foreground_colors:
                count = int(np.sum(patch == color))
                if count > 0:
                    color_counts[color] = count

            # Store counts
            stage1_counts[f"({br},{bc})"] = color_counts

            # Decide winner: strict majority (count > total/2)
            winner = 0  # Default background
            majority_threshold = total_pixels / 2.0

            majority_winners = [c for c, cnt in color_counts.items() if cnt > majority_threshold]

            if len(majority_winners) == 1:
                winner = majority_winners[0]
            elif len(majority_winners) > 1:
                # Tie among majority winners: smallest color (C2)
                winner = min(majority_winners)
            # else: no majority → background=0

            stage1_grid[br, bc] = winner

    return stage1_grid, stage1_counts


def _compute_stage2_pooled(
    stage1_grid: np.ndarray,
    pool_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Compute stage-2 pooled grid: pool stage-1 votes using strict majority.

    Contract (B1, C2):
    - Pool fixed windows (e.g., 2×2) of stage-1 grid
    - Strict majority per pooled cell
    - Ties → smallest color (C2)

    Args:
        stage1_grid: (num_row_bands x num_col_bands) from stage-1
        pool_shape: (pool_rows, pool_cols) e.g., (2, 2)

    Returns:
        pooled_grid: (num_pool_rows x num_pool_cols) final output
    """
    num_row_bands, num_col_bands = stage1_grid.shape
    pool_rows, pool_cols = pool_shape

    # Determine pooled grid size
    num_pool_rows = (num_row_bands + pool_rows - 1) // pool_rows
    num_pool_cols = (num_col_bands + pool_cols - 1) // pool_cols

    pooled_grid = np.zeros((num_pool_rows, num_pool_cols), dtype=np.uint8)

    for pr in range(num_pool_rows):
        for pc in range(num_pool_cols):
            # Extract pool window
            r_start = pr * pool_rows
            r_end = min((pr + 1) * pool_rows, num_row_bands)
            c_start = pc * pool_cols
            c_end = min((pc + 1) * pool_cols, num_col_bands)

            pool_window = stage1_grid[r_start:r_end, c_start:c_end]

            # Count colors in window
            color_counts = {}
            total_cells = pool_window.size

            for color in np.unique(pool_window):
                count = int(np.sum(pool_window == color))
                color_counts[int(color)] = count

            # Decide winner: strict majority (count > total/2)
            winner = 0  # Default background
            majority_threshold = total_cells / 2.0

            majority_winners = [c for c, cnt in color_counts.items() if cnt > majority_threshold]

            if len(majority_winners) == 1:
                winner = majority_winners[0]
            elif len(majority_winners) > 1:
                # Tie: smallest color (C2)
                winner = min(majority_winners)
            # else: no majority → 0

            pooled_grid[pr, pc] = winner

    return pooled_grid


def fit_pooled_blocks(
    train_pairs: List[Tuple[str, np.ndarray, np.ndarray, Any]]  # (train_id, X_t, Y_raw, truth_rc)
) -> Tuple[bool, PooledBlocksFitRc]:
    """
    Fit pooled-blocks engine.

    Contract (WO-10C):
    1. Extract row/col bands from first training truth_rc
    2. Extract foreground colors from ALL training outputs (A1)
    3. Compute stage-1 votes for first training
    4. Compute stage-2 pooled grid
    5. Verify ALL trainings match exactly
    6. Return ok=True if all verify, else ok=False

    Args:
        train_pairs: [(train_id, X_t, Y_raw, truth_rc), ...]

    Returns:
        (ok, PooledBlocksFitRc)
    """
    if not train_pairs:
        return False, PooledBlocksFitRc()

    # Extract bands from first training truth
    train_id_0, X0, Y0, truth_rc_0 = train_pairs[0]
    row_bands = truth_rc_0.row_clusters
    col_bands = truth_rc_0.col_clusters

    if not row_bands or not col_bands:
        return False, PooledBlocksFitRc()

    # Extract foreground colors from ALL training outputs
    train_Y_list = [Y for _, _, Y, _ in train_pairs]
    foreground_colors, background_colors = _extract_foreground_colors(train_Y_list)

    # Compute stage-1 and stage-2 for first training
    stage1_grid, stage1_counts = _compute_stage1_votes(X0, row_bands, col_bands, foreground_colors)

    pool_shape = (2, 2)  # Fixed for now
    stage2_pooled_grid = _compute_stage2_pooled(stage1_grid, pool_shape)

    # Convert to list for receipt
    stage2_pooled = stage2_pooled_grid.tolist()

    # Verify ALL trainings
    fit_verified_on = []
    verification_failures = []

    for train_id, X_t, Y_raw, truth_rc in train_pairs:
        # Compute stage-1 for this training
        stage1_test, _ = _compute_stage1_votes(X_t, row_bands, col_bands, foreground_colors)

        # Compute stage-2 for this training
        stage2_test = _compute_stage2_pooled(stage1_test, pool_shape)

        # Check if stage-2 matches Y_raw
        if stage2_test.shape == Y_raw.shape and np.array_equal(stage2_test, Y_raw):
            fit_verified_on.append(train_id)
        else:
            verification_failures.append({
                "train_id": train_id,
                "stage2_shape": list(stage2_test.shape),
                "Y_raw_shape": list(Y_raw.shape),
                "match": False
            })

    # Success if ALL trainings verified
    ok = (len(fit_verified_on) == len(train_pairs))

    # Build receipt
    block_shape = (len(row_bands) - 1 if len(row_bands) > 1 else 1,
                   len(col_bands) - 1 if len(col_bands) > 1 else 1)

    receipt_str = f"{row_bands}:{col_bands}:{foreground_colors}:{stage2_pooled}:{sorted(fit_verified_on)}"
    rc = PooledBlocksFitRc(
        row_bands=row_bands,
        col_bands=col_bands,
        block_shape=block_shape,
        pool_shape=pool_shape,
        foreground_colors=foreground_colors,
        background_colors=background_colors,
        stage1_counts=stage1_counts,
        stage2_pooled=stage2_pooled,
        fit_verified_on=fit_verified_on,
        hash=hash_bytes(receipt_str.encode())
    )

    return ok, rc


def apply_pooled_blocks(
    test_Xt: np.ndarray,
    truth_test: Any,
    fit_rc: PooledBlocksFitRc,
    expected_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Apply pooled-blocks engine to test input.

    Contract (WO-10C):
    Apply same bands and rule on test.

    Args:
        test_Xt: Test input in Π frame
        truth_test: Truth partition for test (not used, bands from fit_rc)
        fit_rc: Fit receipt from fit_pooled_blocks
        expected_shape: Optional (R, C) from WO-02

    Returns:
        (Y_t, apply_rc): Output in Π frame and application receipt
    """
    # Compute stage-1 votes
    stage1_grid, stage1_counts_test = _compute_stage1_votes(
        test_Xt,
        fit_rc.row_bands,
        fit_rc.col_bands,
        fit_rc.foreground_colors
    )

    # Compute stage-2 pooled
    stage2_pooled_grid = _compute_stage2_pooled(stage1_grid, fit_rc.pool_shape)

    # Check shape
    R, C = stage2_pooled_grid.shape
    if expected_shape is not None:
        R_exp, C_exp = expected_shape
        if (R, C) != (R_exp, C_exp):
            # Shape mismatch - fail
            return stage2_pooled_grid, {
                "engine": "pooled_blocks",
                "output_shape": [R, C],
                "expected_shape": [R_exp, C_exp],
                "shape_mismatch": True,
                "output_hash": hash_bytes(stage2_pooled_grid.tobytes())
            }

    # Build receipt
    apply_rc = {
        "engine": "pooled_blocks",
        "output_shape": [R, C],
        "stage1_counts_sample": {k: v for i, (k, v) in enumerate(stage1_counts_test.items()) if i < 5},  # Sample
        "output_hash": hash_bytes(stage2_pooled_grid.tobytes())
    }

    return stage2_pooled_grid, apply_rc
