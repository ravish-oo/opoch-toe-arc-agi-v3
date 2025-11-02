#!/usr/bin/env python3
# arc/op/unanimity.py
# WO-07: Unanimity on truth blocks (MAJOR)
# Implements 00_math_spec.md §6: unanimous color u(B) per truth block

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from .hash import hash_bytes
from .receipts import BlockVote, UnanimityRc


def _pullback_pixel(
    p: Tuple[int, int],    # (r_*, c_*) in test Π frame
    H_star: int,           # test height in Π frame
    W_star: int,           # test width in Π frame
    R_i: int,              # training i output height
    C_i: int,              # training i output width
) -> Optional[Tuple[int, int]]:
    """
    Pull back test pixel p to training i output coordinates.

    Contract (WO-07 frozen pullback):
    1. Normalize to unit coordinates: u = r_*/H_*, v = c_*/W_*
    2. Map to training output: r_i = floor(u * R_i), c_i = floor(v * C_i)
    3. Bounds check: 0 ≤ r_i < R_i and 0 ≤ c_i < C_i

    Args:
        p: pixel in test Π frame
        H_star, W_star: test dimensions
        R_i, C_i: training output dimensions (from WO-02 S)

    Returns:
        (r_i, c_i): training output coordinates, or None if out of bounds
    """
    r_star, c_star = p

    # Normalize to unit coordinates [0, 1)
    # Contract: exact division
    u = r_star / H_star
    v = c_star / W_star

    # Map to training output size
    # Contract: floor division
    r_i = int(u * R_i)  # floor(u * R_i)
    c_i = int(v * C_i)  # floor(v * C_i)

    # Bounds check (define-domain rule)
    if 0 <= r_i < R_i and 0 <= c_i < C_i:
        return (r_i, c_i)
    else:
        # Undefined for this training
        return None


def compute_unanimity(
    truth_blocks: np.ndarray,  # H_* × W_*, int block IDs from WO-05
    test_shape: Tuple[int, int],  # (H_*, W_*) in Π frame
    train_infos: List[Tuple[str, Tuple[int, int], Tuple[int, int], np.ndarray]],
    # list of (train_id, (H_i, W_i) presented input, (R_i, C_i) output, Y_i array)
) -> Tuple[Dict[int, int], UnanimityRc]:
    """
    Compute unanimous color u(B) for each truth block B.

    Contract (00_math_spec.md §6 lines 92-98):
    "For each truth block B⊂Ω, compute unanimous color u(B) if for all trainings i
    and for all p∈B, letting p_i = Π_i∘Π_*^(-1)(p) (and applying shape pullback),
    the set {Y_i(p_i)|p∈B} is a singleton (same color across all i and all p∈B)."

    Contract (02_determinism_addendum.md §5 lines 183-188):
    "If the pullback is **empty for every training**, unanimity **does not apply**
    to B and may not be used to write it."

    Algorithm:
    1. For each block B (pixels with same truth_blocks value):
       a. For each training i:
          - Pull back each p∈B to training output via frozen Π+S
          - Collect colors: S_i = {Y_i[p_i] | p∈B, p_i defined}
       b. If defined_trainings = {i | |S_i| > 0} is empty → G1: no unanimity
       c. If any |S_i| > 1 → disagreement within training → no unanimity
       d. If all S_i are singletons with same value u → unanimous: u(B) = u
       e. Else → no unanimity

    Args:
        truth_blocks: Partition from WO-05 (H_* × W_*)
        test_shape: (H_*, W_*) dimensions
        train_infos: [(train_id, (H_i,W_i), (R_i,C_i), Y_i), ...]

    Returns:
        (block_color_map, UnanimityRc):
        - block_color_map: {block_id → color} for unanimous blocks only
        - UnanimityRc: full receipt with diagnostics
    """
    H_star, W_star = test_shape

    # Validate truth_blocks shape
    if truth_blocks.shape != (H_star, W_star):
        raise ValueError(
            f"truth_blocks shape {truth_blocks.shape} != test_shape {test_shape}"
        )

    # Validate each training output shape
    for train_id, (H_i, W_i), (R_i, C_i), Y_i in train_infos:
        if Y_i.shape != (R_i, C_i):
            raise ValueError(
                f"{train_id}: Y_i shape {Y_i.shape} != (R_i,C_i)=({R_i},{C_i})"
            )

    # Find all unique block IDs
    block_ids = np.unique(truth_blocks)

    # Accumulators
    blocks_total = len(block_ids)
    unanimous_count = 0
    empty_pullback_blocks = 0
    disagree_blocks = 0

    block_votes = []
    block_color_map = {}

    # Process each block
    for block_id in block_ids:
        # Find all pixels in this block
        coords = np.argwhere(truth_blocks == block_id)
        pixel_count = len(coords)

        # Collect colors from each training
        per_train_colors = {}
        defined_train_ids = []
        defined_pixel_counts = {}

        for train_id, (H_i, W_i), (R_i, C_i), Y_i in train_infos:
            # Collect colors for this training
            colors_seen = []

            for r_star, c_star in coords:
                # Pull back to training output
                p_i = _pullback_pixel(
                    (r_star, c_star),
                    H_star, W_star,
                    R_i, C_i
                )

                if p_i is not None:
                    r_i, c_i = p_i
                    color = int(Y_i[r_i, c_i])
                    colors_seen.append(color)

            # Get unique colors
            unique_colors = sorted(set(colors_seen))
            per_train_colors[train_id] = unique_colors
            defined_pixel_counts[train_id] = len(colors_seen)

            if len(unique_colors) > 0:
                defined_train_ids.append(train_id)

        # Check unanimity conditions
        color = None

        # G1: Empty pullback check
        if len(defined_train_ids) == 0:
            # No training defines any pixel in this block
            empty_pullback_blocks += 1
            # color remains None

        else:
            # At least one training defines pixels
            # Check if all defined trainings have singletons
            all_singletons = True
            singleton_value = None

            for train_id in defined_train_ids:
                colors = per_train_colors[train_id]

                if len(colors) != 1:
                    # Training has multiple colors in this block → disagreement
                    all_singletons = False
                    break

                if singleton_value is None:
                    singleton_value = colors[0]
                elif singleton_value != colors[0]:
                    # Different singleton values → disagreement
                    all_singletons = False
                    break

            if all_singletons and singleton_value is not None:
                # Unanimous!
                color = singleton_value
                unanimous_count += 1
                block_color_map[int(block_id)] = color
            else:
                # Disagreement
                disagree_blocks += 1
                # color remains None

        # Record block vote
        vote = BlockVote(
            block_id=int(block_id),
            color=color,
            defined_train_ids=defined_train_ids.copy(),
            per_train_colors=per_train_colors.copy(),
            pixel_count=pixel_count,
            defined_pixel_counts=defined_pixel_counts.copy(),
        )
        block_votes.append(vote)

    # Compute table hash for determinism
    # Hash: (block_id, color or -1, sorted defined_train_ids) for all blocks
    table_parts = []
    for vote in block_votes:
        color_val = vote.color if vote.color is not None else -1
        train_ids_str = ",".join(sorted(vote.defined_train_ids))
        part = f"{vote.block_id}:{color_val}:{train_ids_str}"
        table_parts.append(part)

    table_str = "|".join(table_parts)
    table_hash = hash_bytes(table_str.encode())

    # Build receipt
    receipt = UnanimityRc(
        blocks_total=blocks_total,
        unanimous_count=unanimous_count,
        empty_pullback_blocks=empty_pullback_blocks,
        disagree_blocks=disagree_blocks,
        table_hash=table_hash,
        blocks=block_votes,
    )

    return block_color_map, receipt
