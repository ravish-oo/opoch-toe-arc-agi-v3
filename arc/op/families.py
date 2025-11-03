#!/usr/bin/env python3
# arc/op/families.py
# WO-10: Family adapters (symbolic slice/band emitters)
# Implements Column-Dictionary engine (schema v1) and stubs for other engines

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .hash import hash_bytes
from .admit import empty_domains, _set_bit, _normalize_scope


@dataclass
class EngineFitRc:
    """
    Engine fit receipt.

    Contract (WO-10):
    Records what was learned from trainings and whether fit succeeded.
    """
    engine: str                         # "column_dict", "macro_tiling", etc.
    ok: bool                            # True if engine fits all trainings exactly
    receipt: Dict[str, Any]             # complete fit receipts (schema, dict, verification)
    # If ok, frozen spec/state needed for apply (e.g., dict, schema_id, bands)


@dataclass
class EngineApplyRc:
    """
    Engine apply receipt.

    Contract (WO-10):
    Records how test was rendered, with exact lookups/errors.
    """
    engine: str                         # engine identifier
    ok: bool                            # True if apply succeeded (no unseen sigs, etc.)
    Yt: Optional[np.ndarray]            # Π(test) output if ok, else None
    final_shape: Optional[Tuple[int, int]]  # (R, C) if ok
    receipt: Dict[str, Any]             # apply receipts (lookup, errors, final_shape)


def _compute_column_signatures_v1(Xt: np.ndarray) -> List[Tuple[int, int]]:
    """
    Compute column signatures (schema v1).

    Contract (WO-10):
    sig(j) = (has8, has5) where:
      has8 = 1[∃r: Xt[r,j]==8]
      has5 = 1[∃r: Xt[r,j]==5]

    Args:
        Xt: Π-presented grid (H, W)

    Returns:
        signatures: [(has8, has5), ...] for each column j
    """
    H, W = Xt.shape
    signatures = []

    for j in range(W):
        col = Xt[:, j]
        has8 = int(np.any(col == 8))
        has5 = int(np.any(col == 5))
        signatures.append((has8, has5))

    return signatures


def _rle_squash(signatures: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Run-length encode signatures (compress consecutive equal sigs).

    Contract (WO-10):
    Canonical compression: [s1, s1, s2, s2, s2] → [s1, s2]

    Args:
        signatures: [(has8, has5), ...] per column

    Returns:
        squashed: RLE compressed signatures
    """
    if not signatures:
        return []

    squashed = [signatures[0]]
    for sig in signatures[1:]:
        if sig != squashed[-1]:
            squashed.append(sig)

    return squashed


def fit_column_dict(
    train_Xt_list: List[np.ndarray],
    train_Y_list: List[np.ndarray],
) -> EngineFitRc:
    """
    Fit Column-Dictionary engine (schema v1).

    Contract (WO-10):
    1. Compute signatures sig(j) = (has8, has5) for each training column
    2. RLE squash signatures
    3. Build dict: sig → output column (exact bytewise equality)
    4. Detect conflicts: if sig maps to >1 distinct column → ok=false
    5. Verify reconstruction: concat(dict[s] for s in S_i) == Y_i exactly

    Algorithm:
    - Schema v1: (has8, has5) - frozen boolean functions
    - Dict building: exact map from sig to column bytes
    - Conflict: if dict[sig] has multiple distinct values across trainings
    - Verification: exact reconstruction or fail

    Args:
        train_Xt_list: [Π(X_i), ...] for i in trainings
        train_Y_list: [Y_i, ...] (raw outputs, not Π-presented)

    Returns:
        EngineFitRc with ok=True if fits all trainings exactly, else ok=False
    """
    schema_id = "col_sig_v1"
    engine = "column_dict"

    # Infer dtype from first training output (no assumptions)
    dtype = train_Y_list[0].dtype

    # Collect signatures and output columns from all trainings
    train_receipts = []
    dict_builder: Dict[Tuple[int, int], List[bytes]] = {}  # sig → [col_bytes, ...]

    for idx, (Xt, Y) in enumerate(zip(train_Xt_list, train_Y_list)):
        train_id = f"train{idx}"

        # Compute column signatures
        col_signatures = _compute_column_signatures_v1(Xt)

        # RLE squash
        squashed = _rle_squash(col_signatures)

        # Map squashed signatures to output columns
        W_in = Xt.shape[1]
        W_out = Y.shape[1]

        if len(squashed) != W_out:
            # Squashed signature count doesn't match output width
            receipt = {
                "engine": engine,
                "ok": False,
                "schema_id": schema_id,
                "error": "SQUASHED_WIDTH_MISMATCH",
                "squashed_count": len(squashed),
                "output_width": W_out,
            }
            return EngineFitRc(engine=engine, ok=False, receipt=receipt)

        # Build dict entries from this training
        for k, sig in enumerate(squashed):
            col_bytes = Y[:, k].tobytes()

            if sig not in dict_builder:
                dict_builder[sig] = []

            # Check if this column bytes is already in the list
            if col_bytes not in dict_builder[sig]:
                dict_builder[sig].append(col_bytes)

        # Record training receipt
        train_receipts.append({
            "train_id": train_id,
            "col_signatures": col_signatures,
            "squashed": squashed,
            "out_width": W_out,
            "fit_verified": False,  # will verify below
        })

    # Check for conflicts (sig → multiple distinct columns)
    conflicts = []
    for sig, cols_bytes in dict_builder.items():
        if len(cols_bytes) > 1:
            # Conflict: same sig maps to different columns
            conflicts.append({
                "sig": list(sig),
                "cols": [col_bytes.hex() for col_bytes in cols_bytes],
            })

    if conflicts:
        # Cannot build deterministic dict
        receipt = {
            "engine": engine,
            "ok": False,
            "schema_id": schema_id,
            "conflicts": conflicts,
            "error": "SIGNATURE_CONFLICTS",
        }
        return EngineFitRc(engine=engine, ok=False, receipt=receipt)

    # Build final dict (one column per sig)
    final_dict = {}
    R = None  # output height (should be constant)

    for sig, cols_bytes in dict_builder.items():
        # Take the only column (no conflicts)
        col_bytes = cols_bytes[0]
        final_dict[sig] = col_bytes

        # Infer R from first column
        if R is None:
            R = len(col_bytes) // dtype.itemsize

    # Verify reconstruction on all trainings
    fit_verified_on = []
    for idx, (Xt, Y) in enumerate(zip(train_Xt_list, train_Y_list)):
        train_id = f"train{idx}"
        squashed = train_receipts[idx]["squashed"]

        # Reconstruct output
        reconstructed_cols = []
        for sig in squashed:
            if tuple(sig) not in final_dict:
                # Signature from training not in dict (shouldn't happen)
                receipt = {
                    "engine": engine,
                    "ok": False,
                    "schema_id": schema_id,
                    "error": "TRAINING_SIG_NOT_IN_DICT",
                    "train_id": train_id,
                    "sig": sig,
                }
                return EngineFitRc(engine=engine, ok=False, receipt=receipt)

            col_bytes = final_dict[tuple(sig)]
            col = np.frombuffer(col_bytes, dtype=dtype)
            reconstructed_cols.append(col)

        # Concatenate columns
        if not reconstructed_cols:
            Y_reconstructed = np.zeros((Y.shape[0], 0), dtype=dtype)
        else:
            Y_reconstructed = np.column_stack(reconstructed_cols)

        # Verify exact equality
        if not np.array_equal(Y_reconstructed, Y):
            receipt = {
                "engine": engine,
                "ok": False,
                "schema_id": schema_id,
                "error": "RECONSTRUCTION_MISMATCH",
                "train_id": train_id,
            }
            return EngineFitRc(engine=engine, ok=False, receipt=receipt)

        train_receipts[idx]["fit_verified"] = True
        fit_verified_on.append(train_id)

    # Build final receipt (success)
    dict_serialized = {
        f"{sig[0]}{sig[1]}": col_bytes.hex()
        for sig, col_bytes in final_dict.items()
    }

    receipt = {
        "engine": engine,
        "ok": True,
        "schema_id": schema_id,
        "dtype": str(dtype),
        "train": train_receipts,
        "dict": dict_serialized,
        "conflicts": [],
        "fit_verified_on": fit_verified_on,
        "output_height": R,
    }

    return EngineFitRc(engine=engine, ok=True, receipt=receipt)


def apply_column_dict(
    test_Xt: np.ndarray,
    fit_rc: EngineFitRc,
    C: List[int]
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Apply Column-Dictionary to test (emit native admits).

    Contract (WO-10 + WO-11A):
    1. Compute signatures on Π(test)
    2. RLE squash → S_*
    3. For each s in S_*:
       - If s in dict → append dict[s]
       - Else → FAIL-CLOSED {"error":"UNSEEN_SIGNATURE","signature":s}
    4. final_shape = (R, len(S_*))
    5. Emit singleton admits

    Args:
        test_Xt: Π(test) grid (H*, W*)
        fit_rc: EngineFitRc from fit_column_dict
        C: Color universe (sorted unique colors)

    Returns:
        (A, S, receipt): Admit bitmap, scope mask, and receipt dict
    """
    engine = "column_dict"

    # Check fit succeeded
    if not fit_rc.ok:
        A = empty_domains(0, 0, C)
        S = np.zeros((0, 0), dtype=np.uint8)
        return A, S, {
            "engine": engine,
            "error": "FIT_FAILED",
            "fit_receipt": fit_rc.receipt,
            "output_shape": [0, 0],
            "scope_bits": 0,
            "bitmap_hash": hash_bytes(A.tobytes()) if A.size > 0 else "",
            "scope_hash": hash_bytes(S.tobytes()) if S.size > 0 else ""
        }

    schema_id = fit_rc.receipt["schema_id"]
    dict_hex = fit_rc.receipt["dict"]
    R = fit_rc.receipt["output_height"]
    dtype = np.dtype(fit_rc.receipt["dtype"])

    # Deserialize dict
    final_dict = {}
    for sig_str, col_hex in dict_hex.items():
        sig = (int(sig_str[0]), int(sig_str[1]))
        col_bytes = bytes.fromhex(col_hex)
        final_dict[sig] = col_bytes

    # Compute test signatures
    test_col_signatures = _compute_column_signatures_v1(test_Xt)

    # RLE squash
    test_squashed = _rle_squash(test_col_signatures)

    # Lookup each signature
    unseen = []
    lookup = []
    reconstructed_cols = []

    for k, sig in enumerate(test_squashed):
        sig_tuple = tuple(sig)

        if sig_tuple in final_dict:
            # Lookup succeeds
            col_bytes = final_dict[sig_tuple]
            col = np.frombuffer(col_bytes, dtype=dtype)
            reconstructed_cols.append(col)

            lookup.append({
                "sig": list(sig),
                "col_hex": col_bytes.hex(),
                "index": k,
            })
        else:
            # Unseen signature → fail-closed
            unseen.append({
                "sig": list(sig),
                "index": k,
            })

    # Check for unseen signatures
    if unseen:
        A = empty_domains(0, 0, C)
        S = np.zeros((0, 0), dtype=np.uint8)
        return A, S, {
            "engine": engine,
            "schema_id": schema_id,
            "error": "UNSEEN_SIGNATURE",
            "test_squashed": test_squashed,
            "lookup": lookup,
            "unseen": unseen,
            "output_shape": [0, 0],
            "scope_bits": 0,
            "bitmap_hash": hash_bytes(A.tobytes()) if A.size > 0 else "",
            "scope_hash": hash_bytes(S.tobytes()) if S.size > 0 else ""
        }

    # Concatenate columns
    if not reconstructed_cols:
        Yt = np.zeros((R, 0), dtype=dtype)
    else:
        Yt = np.column_stack(reconstructed_cols)

    final_shape = (R, len(test_squashed))

    # Build color index lookup
    color_to_idx = {c: i for i, c in enumerate(C)}

    # Initialize admits and scope
    R_out, C_out = final_shape
    A = empty_domains(R_out, C_out, C)
    S = np.zeros((R_out, C_out), dtype=np.uint8)

    # Emit singleton admits from reconstructed grid
    for r in range(R_out):
        for c in range(C_out):
            color = int(Yt[r, c])
            if color in color_to_idx:
                color_idx = color_to_idx[color]
                A[r, c, :] = 0
                _set_bit(A[r, c], color_idx)
                S[r, c] = 1

    # Normalize
    _normalize_scope(A, S, C)

    # Build receipt (success)
    receipt = {
        "engine": engine,
        "schema_id": schema_id,
        "test_squashed": test_squashed,
        "lookup": lookup,
        "final_shape": list(final_shape),  # Standardized key (was "output_shape")
        "unseen": [],
        "scope_bits": int(S.sum()),
        "bitmap_hash": hash_bytes(A.tobytes()),
        "scope_hash": hash_bytes(S.tobytes())
    }

    return A, S, receipt


# ============================================================================
# ENGINE STUBS (return ok=False with diagnostic receipts)
# ============================================================================

def fit_macro_tiling(
    train_Xt_list: List[np.ndarray],
    train_Y_list: List[np.ndarray],
    truth_list: List[Any],  # TruthRc from WO-05
) -> EngineFitRc:
    """
    Fit Macro-Tiling engine (WO-10A).

    Contract (WO-10A, common_mistakes.md A1/A2/C2/B1):
    1. Extract bands from Truth row_clusters/col_clusters (exact, no thresholds)
    2. For each training, iterate over tiles (band grid cells)
    3. Compute per-tile counts for all colors
    4. Apply strict majority rule: count > sum(other_counts) for foreground colors
    5. Handle ties: min(foreground_colors_with_max_count) (C2)
    6. Handle empty tiles: background color (A2)
    7. Verify exact reconstruction on all trainings
    8. Return full receipts: row_bands, col_bands, foreground_colors, background_colors, per-tile decisions (B1)

    Algorithm:
    - Bands from Truth: sorted(set(row_clusters)), sorted(set(col_clusters))
    - Candidate colors from trainings only (A1)
    - Strict majority: count > sum(other_counts), not mode
    - Verification: exact reconstruction or fail

    Args:
        train_Xt_list: [Π(X_i), ...] for i in trainings
        train_Y_list: [Y_i, ...] (raw outputs, not Π-presented)
        truth_list: [TruthRc_i, ...] for i in trainings

    Returns:
        EngineFitRc with ok=True if fits all trainings exactly, else ok=False
    """
    engine = "macro_tiling"

    # Check that we have at least one training
    if not train_Xt_list or not train_Y_list or not truth_list:
        receipt = {
            "engine": engine,
            "ok": False,
            "error": "NO_TRAININGS",
        }
        return EngineFitRc(engine=engine, ok=False, receipt=receipt)

    # Check that all lists have same length
    if not (len(train_Xt_list) == len(train_Y_list) == len(truth_list)):
        receipt = {
            "engine": engine,
            "ok": False,
            "error": "MISMATCHED_TRAINING_COUNTS",
        }
        return EngineFitRc(engine=engine, ok=False, receipt=receipt)

    # Infer dtype from first training output
    dtype = train_Y_list[0].dtype

    # Extract bands from first Truth (all trainings should have same bands)
    # Bands are boundary indices: row_clusters = [0, r1, r2, ...], col_clusters = [0, c1, c2, ...]
    row_clusters = truth_list[0].row_clusters
    col_clusters = truth_list[0].col_clusters

    # Bands: sorted unique boundary indices
    row_bands = sorted(set(row_clusters))
    col_bands = sorted(set(col_clusters))

    # Check that bands define at least one tile
    if len(row_bands) < 2 or len(col_bands) < 2:
        receipt = {
            "engine": engine,
            "ok": False,
            "error": "INSUFFICIENT_BANDS",
            "row_bands": row_bands,
            "col_bands": col_bands,
        }
        return EngineFitRc(engine=engine, ok=False, receipt=receipt)

    # Collect candidate colors from trainings only (A1 guard)
    # foreground_colors: all non-zero colors in outputs
    # background_colors: {0} (frozen)
    all_output_colors = set()
    for Y in train_Y_list:
        colors = np.unique(Y).tolist()
        all_output_colors.update(colors)

    # Split into foreground (nonzero) and background (zero)
    foreground_colors = sorted([c for c in all_output_colors if c != 0])
    background_colors = [0]  # frozen to 0 (H2 from common_mistakes.md)

    # Build decision rules per tile (band grid cell)
    # We'll learn a frozen rule: for each tile, what color to emit
    # Tile (i, j) spans row_bands[i]:row_bands[i+1], col_bands[j]:col_bands[j+1]

    num_row_tiles = len(row_bands) - 1
    num_col_tiles = len(col_bands) - 1

    # Dictionary to store per-tile decisions from trainings
    # Key: (tile_r, tile_c), Value: {color: count_across_trainings}
    tile_aggregates = {}
    for r_idx in range(num_row_tiles):
        for c_idx in range(num_col_tiles):
            tile_aggregates[(r_idx, c_idx)] = {}

    # Process each training
    train_receipts = []
    for train_idx, (Xt, Y) in enumerate(zip(train_Xt_list, train_Y_list)):
        train_id = f"train{train_idx}"

        # Check that Y dimensions match band grid
        H, W = Y.shape
        expected_H = row_bands[-1] if row_bands else 0
        expected_W = col_bands[-1] if col_bands else 0

        # For each tile, compute counts and majority decision
        tile_decisions = []
        for r_idx in range(num_row_tiles):
            r_start = row_bands[r_idx]
            r_end = row_bands[r_idx + 1]

            for c_idx in range(num_col_tiles):
                c_start = col_bands[c_idx]
                c_end = col_bands[c_idx + 1]

                # Extract tile from output
                tile = Y[r_start:r_end, c_start:c_end]

                # Compute per-color counts in this tile
                tile_colors, tile_counts = np.unique(tile, return_counts=True)
                counts_dict = {int(color): int(count) for color, count in zip(tile_colors, tile_counts)}

                # Total pixels in tile
                total_pixels = tile.size

                # Apply strict majority rule (C2 guard)
                # Strict majority: count > sum(other_counts) for foreground colors
                # If no strict majority, fallback to background (A2)

                decision_color = None
                decision_rule = None

                if total_pixels == 0:
                    # Empty tile (A2 guard: empty → background)
                    decision_color = background_colors[0]
                    decision_rule = "EMPTY_TILE_BACKGROUND"
                else:
                    # Check for strict majority among foreground colors
                    max_count = 0
                    max_color_candidates = []

                    for color in foreground_colors:
                        count = counts_dict.get(color, 0)
                        other_counts = sum(counts_dict.get(c, 0) for c in counts_dict if c != color)

                        # Strict majority: count > other_counts
                        if count > other_counts:
                            if count > max_count:
                                max_count = count
                                max_color_candidates = [color]
                            elif count == max_count:
                                max_color_candidates.append(color)

                    if max_color_candidates:
                        # Tie-break: min(candidates) (C2 guard)
                        decision_color = min(max_color_candidates)
                        decision_rule = "STRICT_MAJORITY_FOREGROUND"
                    else:
                        # No strict majority → fallback to background (C2 guard)
                        decision_color = background_colors[0]
                        decision_rule = "NO_STRICT_MAJORITY_FALLBACK_BACKGROUND"

                # Record decision
                tile_decisions.append({
                    "tile_r": r_idx,
                    "tile_c": c_idx,
                    "r_span": [r_start, r_end],
                    "c_span": [c_start, c_end],
                    "counts": counts_dict,
                    "decision": decision_color,
                    "rule": decision_rule,
                })

                # Aggregate for learning (accumulate across trainings)
                if decision_color not in tile_aggregates[(r_idx, c_idx)]:
                    tile_aggregates[(r_idx, c_idx)][decision_color] = 0
                tile_aggregates[(r_idx, c_idx)][decision_color] += 1

        # Record training receipt (B1: both stage-1 counts and final decisions)
        train_receipts.append({
            "train_id": train_id,
            "tile_decisions": tile_decisions,
            "fit_verified": False,  # will verify below
        })

    # Learn frozen tile rules: for each tile, unanimous decision across trainings
    tile_rules = {}
    for (r_idx, c_idx), decisions in tile_aggregates.items():
        if len(decisions) != 1:
            # Conflict: different trainings decided different colors for this tile
            receipt = {
                "engine": engine,
                "ok": False,
                "error": "TILE_DECISION_CONFLICT",
                "tile": [r_idx, c_idx],
                "decisions": {str(k): v for k, v in decisions.items()},
            }
            return EngineFitRc(engine=engine, ok=False, receipt=receipt)

        # Unanimous decision
        tile_rules[(r_idx, c_idx)] = list(decisions.keys())[0]

    # Verify reconstruction on all trainings
    fit_verified_on = []
    for train_idx, Y in enumerate(train_Y_list):
        train_id = f"train{train_idx}"

        # Reconstruct output from tile rules
        H_out, W_out = Y.shape
        Y_reconstructed = np.zeros((H_out, W_out), dtype=dtype)

        for r_idx in range(num_row_tiles):
            r_start = row_bands[r_idx]
            r_end = row_bands[r_idx + 1]

            for c_idx in range(num_col_tiles):
                c_start = col_bands[c_idx]
                c_end = col_bands[c_idx + 1]

                # Fill tile with learned color
                decision_color = tile_rules[(r_idx, c_idx)]
                Y_reconstructed[r_start:r_end, c_start:c_end] = decision_color

        # Verify exact equality
        if not np.array_equal(Y_reconstructed, Y):
            receipt = {
                "engine": engine,
                "ok": False,
                "error": "RECONSTRUCTION_MISMATCH",
                "train_id": train_id,
            }
            return EngineFitRc(engine=engine, ok=False, receipt=receipt)

        train_receipts[train_idx]["fit_verified"] = True
        fit_verified_on.append(train_id)

    # Build final receipt
    # Serialize tile rules
    tile_rules_serialized = {
        f"{r_idx},{c_idx}": int(color)
        for (r_idx, c_idx), color in tile_rules.items()
    }

    receipt = {
        "engine": engine,
        "ok": True,
        "dtype": str(dtype),
        "row_bands": row_bands,
        "col_bands": col_bands,
        "foreground_colors": foreground_colors,
        "background_colors": background_colors,
        "tile_rules": tile_rules_serialized,
        "train": train_receipts,
        "fit_verified_on": fit_verified_on,
    }

    return EngineFitRc(engine=engine, ok=True, receipt=receipt)


def apply_macro_tiling(
    test_Xt: np.ndarray,
    truth_test: Any,  # TruthRc from WO-05
    fit_rc: EngineFitRc,
    C: List[int],
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Apply Macro-Tiling to test (WO-10A + WO-11A).

    Contract (WO-10A + WO-11A):
    1. Use same bands and tile rules from fit_rc
    2. For each tile in test, apply learned color decision
    3. Emit singleton admits for each tile region
    4. Return (A, S, receipt) native admits

    Algorithm:
    - Extract row_bands, col_bands, and tile_rules from fit_rc
    - For each tile (r_idx, c_idx), emit singleton admits with tile_rules[(r_idx, c_idx)]
    - Build admit bitmap and scope mask
    - Return (A, S, receipt) with hashes

    Args:
        test_Xt: Π(test) grid (H*, W*)
        truth_test: TruthRc for test
        fit_rc: EngineFitRc from fit_macro_tiling
        C: Color universe (sorted unique colors)

    Returns:
        (A, S, receipt): Admit bitmap, scope mask, and application receipt
    """
    engine = "macro_tiling"

    # Check fit succeeded
    if not fit_rc.ok:
        A = empty_domains(0, 0, C)
        S = np.zeros((0, 0), dtype=np.uint8)
        receipt = {
            "engine": engine,
            "error": "FIT_FAILED",
            "fit_receipt": fit_rc.receipt,
            "scope_bits": 0,
            "bitmap_hash": hash_bytes(A.tobytes()) if A.size > 0 else "",
            "scope_hash": hash_bytes(S.tobytes()) if S.size > 0 else ""
        }
        return A, S, receipt

    # Extract learned parameters from fit_rc
    row_bands = fit_rc.receipt["row_bands"]
    col_bands = fit_rc.receipt["col_bands"]
    tile_rules_serialized = fit_rc.receipt["tile_rules"]
    dtype = np.dtype(fit_rc.receipt["dtype"])

    # Deserialize tile rules
    tile_rules = {}
    for key, color in tile_rules_serialized.items():
        r_idx, c_idx = map(int, key.split(","))
        tile_rules[(r_idx, c_idx)] = color

    # Check that test Truth has compatible bands
    # (In practice, test might have different bands, but for this engine we use fit bands)

    # Determine output shape from bands
    num_row_tiles = len(row_bands) - 1
    num_col_tiles = len(col_bands) - 1

    if num_row_tiles <= 0 or num_col_tiles <= 0:
        A = empty_domains(0, 0, C)
        S = np.zeros((0, 0), dtype=np.uint8)
        receipt = {
            "engine": engine,
            "error": "INVALID_BANDS",
            "row_bands": row_bands,
            "col_bands": col_bands,
            "scope_bits": 0,
            "bitmap_hash": hash_bytes(A.tobytes()) if A.size > 0 else "",
            "scope_hash": hash_bytes(S.tobytes()) if S.size > 0 else ""
        }
        return A, S, receipt

    H_out = row_bands[-1]
    W_out = col_bands[-1]

    # Build color index lookup
    color_to_idx = {c: i for i, c in enumerate(C)}

    # Initialize admits and scope
    A = empty_domains(H_out, W_out, C)
    S = np.zeros((H_out, W_out), dtype=np.uint8)

    # Apply tile rules and emit singleton admits
    tile_applications = []
    for r_idx in range(num_row_tiles):
        r_start = row_bands[r_idx]
        r_end = row_bands[r_idx + 1]

        for c_idx in range(num_col_tiles):
            c_start = col_bands[c_idx]
            c_end = col_bands[c_idx + 1]

            # Lookup tile rule
            if (r_idx, c_idx) not in tile_rules:
                # Missing tile rule (shouldn't happen if fit succeeded)
                A = empty_domains(0, 0, C)
                S = np.zeros((0, 0), dtype=np.uint8)
                receipt = {
                    "engine": engine,
                    "error": "MISSING_TILE_RULE",
                    "tile": [r_idx, c_idx],
                    "scope_bits": 0,
                    "bitmap_hash": hash_bytes(A.tobytes()) if A.size > 0 else "",
                    "scope_hash": hash_bytes(S.tobytes()) if S.size > 0 else ""
                }
                return A, S, receipt

            decision_color = tile_rules[(r_idx, c_idx)]

            # Emit singleton admits for this tile
            if decision_color in color_to_idx:
                color_idx = color_to_idx[decision_color]
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        A[r, c, :] = 0
                        _set_bit(A[r, c], color_idx)
                        S[r, c] = 1

            # Record application
            tile_applications.append({
                "tile_r": r_idx,
                "tile_c": c_idx,
                "r_span": [r_start, r_end],
                "c_span": [c_start, c_end],
                "decision": int(decision_color),
            })

    # Normalize
    _normalize_scope(A, S, C)

    final_shape = (H_out, W_out)

    # Build receipt (success)
    receipt = {
        "engine": engine,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "tile_applications": tile_applications,
        "final_shape": list(final_shape),
        "scope_bits": int(S.sum()),
        "bitmap_hash": hash_bytes(A.tobytes()),
        "scope_hash": hash_bytes(S.tobytes())
    }

    return A, S, receipt


def fit_pooled_blocks(
    train_Xt_list: List[np.ndarray],
    train_Y_list: List[np.ndarray],
) -> EngineFitRc:
    """Stub: Pooled-Blocks engine not implemented in WO-10 MVP."""
    receipt = {
        "engine": "pooled_blocks",
        "ok": False,
        "error": "NOT_IMPLEMENTED",
        "note": "WO-10 MVP implements Column-Dictionary only",
    }
    return EngineFitRc(engine="pooled_blocks", ok=False, receipt=receipt)


def fit_markers_grid(
    train_Xt_list: List[np.ndarray],
    train_Y_list: List[np.ndarray],
) -> EngineFitRc:
    """Stub: Markers-Grid engine not implemented in WO-10 MVP."""
    receipt = {
        "engine": "markers_grid",
        "ok": False,
        "error": "NOT_IMPLEMENTED",
        "note": "WO-10 MVP implements Column-Dictionary only",
    }
    return EngineFitRc(engine="markers_grid", ok=False, receipt=receipt)


def fit_slice_stack(
    train_Xt_list: List[np.ndarray],
    train_Y_list: List[np.ndarray],
) -> EngineFitRc:
    """Stub: SliceStack engine not implemented in WO-10 MVP."""
    receipt = {
        "engine": "slice_stack",
        "ok": False,
        "error": "NOT_IMPLEMENTED",
        "note": "WO-10 MVP implements Column-Dictionary only",
    }
    return EngineFitRc(engine="slice_stack", ok=False, receipt=receipt)


def fit_kronecker(
    train_Xt_list: List[np.ndarray],
    train_Y_list: List[np.ndarray],
) -> EngineFitRc:
    """Stub: Kronecker engine not implemented in WO-10 MVP."""
    receipt = {
        "engine": "kronecker",
        "ok": False,
        "error": "NOT_IMPLEMENTED",
        "note": "WO-10 MVP implements Column-Dictionary only",
    }
    return EngineFitRc(engine="kronecker", ok=False, receipt=receipt)
