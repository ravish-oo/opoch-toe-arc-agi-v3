#!/usr/bin/env python3
# arc/op/kronecker.py
# WO-10F: Kronecker Tiling Engine

"""
Contract (WO-10F):
Learn base tile T that when replicated (Kronecker product) produces exact training outputs.

Frozen algorithm:
1. Base tile inference: GCD-based divisors, smallest tile that replicates exactly
2. Cross-intersection: Single (r0, c0) must work for ALL trainings
3. Verify: ALL trainings must equal tile ⊗ 1_{k×k}
4. Apply: R*/r0 and C*/c0 must divide exactly, else fail-closed

Engineering = Math:
- Divisor computation is exact integer math
- Tile replication: np.tile(base_tile, (k_r, k_c))
- Verification: np.array_equal(replicated, Y_i)
- Tie-break: lex-smallest tile bytes (frozen)
- Fail-closed (no common tile, divisibility failure → ok=False)
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import numpy as np

from arc.op.hash import hash_bytes


@dataclass
class KroneckerFitRc:
    """
    Kronecker fit receipt.

    Contract (WO-10F):
    Records base tile, tile shape, repetition counts per training, verification status.
    """
    engine: Literal["kronecker"] = "kronecker"
    base_tile: str = ""  # hex-encoded tile bytes
    tile_shape: Tuple[int, int] = (0, 0)  # (r0, c0)
    reps: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # train_id → (k_r, k_c)
    fit_verified_on: List[str] = field(default_factory=list)
    hash: str = ""


def _compute_divisors(n: int) -> List[int]:
    """
    Compute all divisors of n.

    Contract (WO-10F):
    Exact integer divisors only, no floating point.

    Args:
        n: Positive integer

    Returns:
        Sorted list of divisors
    """
    if n <= 0:
        return []

    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)

    return sorted(divisors)


def _verify_tile_replication(Y: np.ndarray, tile: np.ndarray) -> bool:
    """
    Verify that Y equals tile replicated.

    Contract (WO-10F):
    Exact equality check, no tolerance.

    Args:
        Y: Output grid (r_i × c_i)
        tile: Base tile (r0 × c0)

    Returns:
        True if Y = tile ⊗ 1_{k×k}, False otherwise
    """
    r_i, c_i = Y.shape
    r0, c0 = tile.shape

    # Check divisibility
    if r_i % r0 != 0 or c_i % c0 != 0:
        return False

    k_r = r_i // r0
    k_c = c_i // c0

    # Replicate tile
    replicated = np.tile(tile, (k_r, k_c))

    # Check exact equality
    return np.array_equal(replicated, Y)


def _find_minimal_tile_sizes(Y: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find all minimal tile sizes that replicate to Y.

    Contract (WO-10F):
    - Compute divisors of Y shape
    - Test each candidate tile size
    - Return all tiles with minimal area

    Args:
        Y: Output grid (r_i × c_i)

    Returns:
        List of (r0, c0) tile sizes with minimal area
    """
    r_i, c_i = Y.shape

    # Compute divisors
    row_divisors = _compute_divisors(r_i)
    col_divisors = _compute_divisors(c_i)

    # Find all valid tile sizes
    valid_tiles = []

    for r0 in row_divisors:
        for c0 in col_divisors:
            # Extract candidate tile
            tile = Y[:r0, :c0]

            # Check if this tile replicates to Y
            if _verify_tile_replication(Y, tile):
                valid_tiles.append((r0, c0))

    if not valid_tiles:
        return []

    # Find minimal area
    min_area = min(r0 * c0 for r0, c0 in valid_tiles)

    # Return all tiles with minimal area
    minimal_tiles = [(r0, c0) for r0, c0 in valid_tiles if r0 * c0 == min_area]

    return minimal_tiles


def fit_kronecker(
    train_pairs: List[Tuple[str, np.ndarray, np.ndarray, Any]]  # (train_id, X_t, Y_raw, truth_rc)
) -> Tuple[bool, KroneckerFitRc]:
    """
    Fit Kronecker tiling engine.

    Contract (WO-10F):
    1. Find minimal tile sizes for each training output
    2. Intersect across ALL trainings to find common (r0, c0)
    3. Extract base tile from first training
    4. Verify ALL trainings replicate exactly
    5. Return ok=True if all verify, else ok=False

    Args:
        train_pairs: [(train_id, X_t, Y_raw, truth_rc), ...]

    Returns:
        (ok, KroneckerFitRc)
    """
    if not train_pairs:
        return False, KroneckerFitRc()

    # Step 1: Find minimal tile sizes for each training
    tile_sizes_per_training = {}

    for train_id, X_t, Y_raw, truth_rc in train_pairs:
        minimal_tiles = _find_minimal_tile_sizes(Y_raw)

        if not minimal_tiles:
            # No valid tile found for this training
            return False, KroneckerFitRc()

        tile_sizes_per_training[train_id] = set(minimal_tiles)

    # Step 2: Intersect across ALL trainings
    common_tile_sizes = None

    for train_id, tile_sizes in tile_sizes_per_training.items():
        if common_tile_sizes is None:
            common_tile_sizes = tile_sizes
        else:
            common_tile_sizes = common_tile_sizes.intersection(tile_sizes)

    if not common_tile_sizes:
        # No common tile size across trainings
        return False, KroneckerFitRc()

    # Step 3: Choose tile size (lex-smallest if multiple)
    # For ties, we need to extract tiles and compare bytes
    candidate_tiles = []

    for r0, c0 in common_tile_sizes:
        # Extract tile from first training
        train_id_0, X0, Y0, truth_rc_0 = train_pairs[0]
        tile = Y0[:r0, :c0].copy()
        tile_bytes = tile.tobytes()

        candidate_tiles.append(((r0, c0), tile, tile_bytes))

    # Sort by tile bytes (lex-smallest)
    candidate_tiles.sort(key=lambda x: x[2])

    # Choose first (lex-smallest)
    (r0, c0), base_tile, base_tile_bytes = candidate_tiles[0]

    # Step 4: Verify ALL trainings
    fit_verified_on = []
    reps = {}

    for train_id, X_t, Y_raw, truth_rc in train_pairs:
        # Check if tile replicates to Y_raw
        if _verify_tile_replication(Y_raw, base_tile):
            r_i, c_i = Y_raw.shape
            k_r = r_i // r0
            k_c = c_i // c0
            reps[train_id] = (k_r, k_c)
            fit_verified_on.append(train_id)

    # Success if ALL trainings verified
    ok = (len(fit_verified_on) == len(train_pairs))

    # Build receipt
    receipt_str = f"{base_tile_bytes.hex()}:{r0}:{c0}:{sorted(reps.items())}:{sorted(fit_verified_on)}"
    rc = KroneckerFitRc(
        base_tile=base_tile_bytes.hex(),
        tile_shape=(r0, c0),
        reps=reps,
        fit_verified_on=fit_verified_on,
        hash=hash_bytes(receipt_str.encode())
    )

    return ok, rc


def apply_kronecker(
    test_Xt: np.ndarray,
    truth_test: Any,
    fit_rc: KroneckerFitRc,
    expected_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Apply Kronecker tiling engine to test input.

    Contract (WO-10F):
    Replicate base tile to expected shape.
    Fail-closed if expected_shape not divisible by tile_shape.

    Args:
        test_Xt: Test input in Π frame (not used, output is tile replication)
        truth_test: Truth partition for test (not used)
        fit_rc: Fit receipt from fit_kronecker
        expected_shape: Optional (R, C) from WO-02

    Returns:
        (Y_t, apply_rc): Output in Π frame and application receipt
    """
    r0, c0 = fit_rc.tile_shape

    # Decode tile from hex
    tile_bytes = bytes.fromhex(fit_rc.base_tile)
    base_tile = np.frombuffer(tile_bytes, dtype=np.uint8).reshape((r0, c0))

    # Determine output shape
    if expected_shape is not None:
        R, C = expected_shape
    else:
        # No expected shape - cannot determine replication count
        return np.zeros((0, 0), dtype=np.uint8), {
            "engine": "kronecker",
            "error": "NO_EXPECTED_SHAPE",
            "output_shape": [0, 0]
        }

    # Check divisibility
    if R % r0 != 0 or C % c0 != 0:
        # Shape not divisible by tile
        return np.zeros((0, 0), dtype=np.uint8), {
            "engine": "kronecker",
            "error": "SHAPE_NOT_DIVISIBLE",
            "expected_shape": [R, C],
            "tile_shape": [r0, c0],
            "output_shape": [0, 0]
        }

    # Compute repetitions
    k_r = R // r0
    k_c = C // c0

    # Replicate tile
    Y_t = np.tile(base_tile, (k_r, k_c))

    # Build receipt
    apply_rc = {
        "engine": "kronecker",
        "tile_shape": [r0, c0],
        "reps": [k_r, k_c],
        "output_shape": [R, C],
        "output_hash": hash_bytes(Y_t.tobytes())
    }

    return Y_t, apply_rc
