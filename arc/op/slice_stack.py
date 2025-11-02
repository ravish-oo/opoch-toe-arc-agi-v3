#!/usr/bin/env python3
# arc/op/slice_stack.py
# WO-10E: Slice-Stack Engine

"""
Contract (WO-10E):
Learn dictionary mapping input slices → output slices, concatenate to form output.

Frozen algorithm:
1. Axis selection: truth col_clusters → axis="cols", else axis="rows" (frozen rule)
2. Slice extraction: fixed-size slices with frozen signature schema (row-major uint8 bytes)
3. Dictionary: exact byte mapping from trainings, verify ALL trainings
4. Apply: lookup signatures, concatenate; fail-closed on UNSEEN_SIGNATURE

Engineering = Math:
- Dictionary is pure function: bytes → bytes
- Exact byte equality for signatures (no fuzzy matching)
- Frozen axis selection (no heuristics)
- Deterministic dict serialization for hashing
- Fail-closed (UNSEEN_SIGNATURE, conflicts → ok=False)
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import numpy as np

from arc.op.hash import hash_bytes


@dataclass
class SliceStackFitRc:
    """
    Slice-Stack fit receipt.

    Contract (WO-10E):
    Records axis, slice dimensions, dictionary, decision rule, verification status.
    """
    engine: Literal["slice_stack"] = "slice_stack"
    axis: Literal["cols", "rows"] = "cols"
    slice_height: int = 0
    slice_width: int = 0
    dict: Dict[str, str] = field(default_factory=dict)  # hex(input_sig) → hex(output_slice)
    argmax_windows: List[Tuple[int, int]] = field(default_factory=list)
    decision_rule: str = "exact_slice_match_only"
    fit_verified_on: List[str] = field(default_factory=list)
    hash: str = ""


def _select_axis(truth_rc: Any) -> Literal["cols", "rows"]:
    """
    Select axis for slicing based on truth features.

    Contract (WO-10E, D1):
    - If col_clusters exist and len > 1 → axis="cols"
    - Else → axis="rows"
    - No heuristics, frozen rule only

    Args:
        truth_rc: Truth receipt from WO-05

    Returns:
        axis: "cols" or "rows"
    """
    if truth_rc is not None and hasattr(truth_rc, 'col_clusters'):
        col_clusters = truth_rc.col_clusters
        if col_clusters and len(col_clusters) > 1:
            return "cols"

    # Default to rows
    return "rows"


def _extract_slice_bytes(G: np.ndarray, axis: Literal["cols", "rows"], index: int) -> bytes:
    """
    Extract slice bytes from grid at given index.

    Contract (WO-10E):
    - Frozen signature schema: row-major uint8 bytes
    - For axis="cols": extract column, serialize top-to-bottom
    - For axis="rows": extract row, serialize left-to-right

    Args:
        G: Grid (H x W)
        axis: "cols" or "rows"
        index: Slice index (column index or row index)

    Returns:
        slice_bytes: Serialized slice (uint8 row-major)
    """
    if axis == "cols":
        # Extract column
        if index < G.shape[1]:
            slice_data = G[:, index]  # (H,)
        else:
            slice_data = np.array([], dtype=np.uint8)
    else:  # axis == "rows"
        # Extract row
        if index < G.shape[0]:
            slice_data = G[index, :]  # (W,)
        else:
            slice_data = np.array([], dtype=np.uint8)

    return slice_data.tobytes()


def _determine_slice_dimensions(
    axis: Literal["cols", "rows"],
    first_output: np.ndarray
) -> Tuple[int, int]:
    """
    Determine slice dimensions from first training output.

    Contract (WO-10E):
    - For axis="cols": slice_height = output height, slice_width = 1
    - For axis="rows": slice_height = 1, slice_width = output width

    Args:
        axis: "cols" or "rows"
        first_output: First training output (R x C)

    Returns:
        (slice_height, slice_width): Dimensions of each slice
    """
    H, W = first_output.shape

    if axis == "cols":
        return (H, 1)
    else:  # axis == "rows"
        return (1, W)


def fit_slice_stack(
    train_pairs: List[Tuple[str, np.ndarray, np.ndarray, Any]]  # (train_id, X_t, Y_raw, truth_rc)
) -> Tuple[bool, SliceStackFitRc]:
    """
    Fit slice-stack engine.

    Contract (WO-10E):
    1. Select axis from truth features (frozen rule)
    2. Determine slice dimensions from first training output
    3. Build dictionary: input_slice_bytes → output_slice_bytes
    4. Verify ALL trainings (exact equality)
    5. Return ok=True if all verify, else ok=False

    Args:
        train_pairs: [(train_id, X_t, Y_raw, truth_rc), ...]

    Returns:
        (ok, SliceStackFitRc)
    """
    if not train_pairs:
        return False, SliceStackFitRc()

    # Step 1: Select axis from first training truth
    train_id_0, X0, Y0, truth_rc_0 = train_pairs[0]
    axis = _select_axis(truth_rc_0)

    # Step 2: Determine slice dimensions from first output
    slice_height, slice_width = _determine_slice_dimensions(axis, Y0)

    # Step 3: Build dictionary from trainings
    slice_dict = {}  # bytes → bytes
    signature_to_output = {}  # For conflict detection

    for train_id, X_t, Y_raw, truth_rc in train_pairs:
        # Determine number of slices
        if axis == "cols":
            num_input_slices = X_t.shape[1]
            num_output_slices = Y_raw.shape[1]
        else:  # axis == "rows"
            num_input_slices = X_t.shape[0]
            num_output_slices = Y_raw.shape[0]

        # Check if input and output have same number of slices
        if num_input_slices != num_output_slices:
            # Slice count mismatch - cannot build 1:1 mapping
            return False, SliceStackFitRc()

        # Extract slices and build mapping
        for i in range(num_input_slices):
            # Extract input slice signature
            input_sig = _extract_slice_bytes(X_t, axis, i)

            # Extract output slice
            output_slice = _extract_slice_bytes(Y_raw, axis, i)

            # Check for conflicts (same input → different outputs)
            if input_sig in signature_to_output:
                if signature_to_output[input_sig] != output_slice:
                    # Conflict: same input maps to different outputs
                    return False, SliceStackFitRc()
            else:
                signature_to_output[input_sig] = output_slice
                slice_dict[input_sig] = output_slice

    # Step 4: Verify ALL trainings
    fit_verified_on = []
    verification_failures = []

    for train_id, X_t, Y_raw, truth_rc in train_pairs:
        # Reconstruct output from dictionary
        if axis == "cols":
            num_slices = X_t.shape[1]
            reconstructed = np.zeros_like(Y_raw)

            for i in range(num_slices):
                input_sig = _extract_slice_bytes(X_t, axis, i)

                if input_sig not in slice_dict:
                    # Missing signature
                    verification_failures.append({
                        "train_id": train_id,
                        "error": "MISSING_SIGNATURE",
                        "slice_index": i
                    })
                    break

                # Get output slice from dict
                output_slice_bytes = slice_dict[input_sig]
                output_slice = np.frombuffer(output_slice_bytes, dtype=np.uint8)

                # Place in reconstructed grid
                if len(output_slice) == Y_raw.shape[0] and i < Y_raw.shape[1]:
                    reconstructed[:, i] = output_slice
                else:
                    verification_failures.append({
                        "train_id": train_id,
                        "error": "SLICE_SHAPE_MISMATCH",
                        "slice_index": i
                    })
                    break

        else:  # axis == "rows"
            num_slices = X_t.shape[0]
            reconstructed = np.zeros_like(Y_raw)

            for i in range(num_slices):
                input_sig = _extract_slice_bytes(X_t, axis, i)

                if input_sig not in slice_dict:
                    # Missing signature
                    verification_failures.append({
                        "train_id": train_id,
                        "error": "MISSING_SIGNATURE",
                        "slice_index": i
                    })
                    break

                # Get output slice from dict
                output_slice_bytes = slice_dict[input_sig]
                output_slice = np.frombuffer(output_slice_bytes, dtype=np.uint8)

                # Place in reconstructed grid
                if len(output_slice) == Y_raw.shape[1] and i < Y_raw.shape[0]:
                    reconstructed[i, :] = output_slice
                else:
                    verification_failures.append({
                        "train_id": train_id,
                        "error": "SLICE_SHAPE_MISMATCH",
                        "slice_index": i
                    })
                    break

        # Check if reconstruction matches
        if len(verification_failures) == 0 and np.array_equal(reconstructed, Y_raw):
            fit_verified_on.append(train_id)
        elif len(verification_failures) == 0:
            verification_failures.append({
                "train_id": train_id,
                "error": "RECONSTRUCTION_MISMATCH"
            })

    # Success if ALL trainings verified
    ok = (len(fit_verified_on) == len(train_pairs))

    # Build receipt with hex-encoded dict for serialization
    dict_hex = {
        input_sig.hex(): output_slice.hex()
        for input_sig, output_slice in sorted(slice_dict.items())  # Sort for determinism
    }

    receipt_str = f"{axis}:{slice_height}:{slice_width}:{sorted(dict_hex.items())}:{sorted(fit_verified_on)}"
    rc = SliceStackFitRc(
        axis=axis,
        slice_height=slice_height,
        slice_width=slice_width,
        dict=dict_hex,
        fit_verified_on=fit_verified_on,
        hash=hash_bytes(receipt_str.encode())
    )

    return ok, rc


def apply_slice_stack(
    test_Xt: np.ndarray,
    truth_test: Any,
    fit_rc: SliceStackFitRc,
    expected_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Apply slice-stack engine to test input.

    Contract (WO-10E):
    Apply dictionary lookup and concatenation on test.
    Fail-closed on UNSEEN_SIGNATURE.

    Args:
        test_Xt: Test input in Π frame
        truth_test: Truth partition for test (not used currently)
        fit_rc: Fit receipt from fit_slice_stack
        expected_shape: Optional (R, C) from WO-02

    Returns:
        (Y_t, apply_rc): Output in Π frame and application receipt
    """
    axis = fit_rc.axis

    # Decode dict from hex
    slice_dict = {
        bytes.fromhex(k): bytes.fromhex(v)
        for k, v in fit_rc.dict.items()
    }

    # Determine number of slices
    if axis == "cols":
        num_slices = test_Xt.shape[1]
    else:  # axis == "rows"
        num_slices = test_Xt.shape[0]

    # Extract test slices and look up in dictionary
    output_slices = []
    missing_signatures = []

    for i in range(num_slices):
        input_sig = _extract_slice_bytes(test_Xt, axis, i)

        if input_sig not in slice_dict:
            # UNSEEN_SIGNATURE - fail closed
            missing_signatures.append(i)
            # Use zero slice as fallback for shape inference
            if axis == "cols":
                output_slices.append(np.zeros(fit_rc.slice_height, dtype=np.uint8))
            else:
                output_slices.append(np.zeros(fit_rc.slice_width, dtype=np.uint8))
        else:
            # Look up output slice
            output_slice_bytes = slice_dict[input_sig]
            output_slice = np.frombuffer(output_slice_bytes, dtype=np.uint8)
            output_slices.append(output_slice)

    if missing_signatures:
        # Fail-closed on UNSEEN_SIGNATURE
        return np.zeros((0, 0), dtype=np.uint8), {
            "engine": "slice_stack",
            "error": "UNSEEN_SIGNATURE",
            "missing_slice_indices": missing_signatures,
            "output_shape": [0, 0]
        }

    # Concatenate slices to form output
    if axis == "cols":
        # Stack columns horizontally
        Y_t = np.column_stack(output_slices)
    else:  # axis == "rows"
        # Stack rows vertically
        Y_t = np.row_stack(output_slices)

    # Check expected shape
    R, C = Y_t.shape
    if expected_shape is not None:
        R_exp, C_exp = expected_shape
        if (R, C) != (R_exp, C_exp):
            # Shape mismatch
            return Y_t, {
                "engine": "slice_stack",
                "output_shape": [R, C],
                "expected_shape": [R_exp, C_exp],
                "shape_mismatch": True,
                "output_hash": hash_bytes(Y_t.tobytes())
            }

    # Build receipt
    apply_rc = {
        "engine": "slice_stack",
        "axis": axis,
        "num_slices": num_slices,
        "output_shape": [R, C],
        "output_hash": hash_bytes(Y_t.tobytes())
    }

    return Y_t, apply_rc
