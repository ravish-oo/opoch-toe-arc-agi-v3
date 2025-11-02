#!/usr/bin/env python3
# arc/op/families.py
# WO-10: Family adapters (symbolic slice/band emitters)
# Implements Column-Dictionary engine (schema v1) and stubs for other engines

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .hash import hash_bytes


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
) -> EngineApplyRc:
    """
    Apply Column-Dictionary to test.

    Contract (WO-10):
    1. Compute signatures on Π(test)
    2. RLE squash → S_*
    3. For each s in S_*:
       - If s in dict → append dict[s]
       - Else → FAIL-CLOSED {"error":"UNSEEN_SIGNATURE","signature":s}
    4. final_shape = (R, len(S_*))

    Args:
        test_Xt: Π(test) grid (H*, W*)
        fit_rc: EngineFitRc from fit_column_dict

    Returns:
        EngineApplyRc with ok=True if all lookups succeed, else ok=False
    """
    engine = "column_dict"

    # Check fit succeeded
    if not fit_rc.ok:
        receipt = {
            "engine": engine,
            "ok": False,
            "error": "FIT_FAILED",
            "fit_receipt": fit_rc.receipt,
        }
        return EngineApplyRc(
            engine=engine,
            ok=False,
            Yt=None,
            final_shape=None,
            receipt=receipt,
        )

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
        receipt = {
            "engine": engine,
            "ok": False,
            "schema_id": schema_id,
            "error": "UNSEEN_SIGNATURE",
            "test_squashed": test_squashed,
            "lookup": lookup,
            "unseen": unseen,
        }
        return EngineApplyRc(
            engine=engine,
            ok=False,
            Yt=None,
            final_shape=None,
            receipt=receipt,
        )

    # Concatenate columns
    if not reconstructed_cols:
        Yt = np.zeros((R, 0), dtype=dtype)
    else:
        Yt = np.column_stack(reconstructed_cols)

    final_shape = (R, len(test_squashed))

    # Build receipt (success)
    receipt = {
        "engine": engine,
        "ok": True,
        "schema_id": schema_id,
        "test_squashed": test_squashed,
        "lookup": lookup,
        "final_shape": list(final_shape),
        "unseen": [],
    }

    return EngineApplyRc(
        engine=engine,
        ok=True,
        Yt=Yt,
        final_shape=final_shape,
        receipt=receipt,
    )


# ============================================================================
# ENGINE STUBS (return ok=False with diagnostic receipts)
# ============================================================================

def fit_macro_tiling(
    train_Xt_list: List[np.ndarray],
    train_Y_list: List[np.ndarray],
    truth_list: List[Any],  # TruthRc from WO-05
) -> EngineFitRc:
    """Stub: Macro-Tiling engine not implemented in WO-10 MVP."""
    receipt = {
        "engine": "macro_tiling",
        "ok": False,
        "error": "NOT_IMPLEMENTED",
        "note": "WO-10 MVP implements Column-Dictionary only",
    }
    return EngineFitRc(engine="macro_tiling", ok=False, receipt=receipt)


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
