#!/usr/bin/env python3
# arc/op/kronecker_mask.py
# WO-10K: Kronecker-Mask Engine

"""
Contract (WO-10K):
Learn mask-Kronecker law: Y = (X ≠ 0) ⊗ X

This is DIFFERENT from base-tile Kronecker (in kronecker.py).
Mask-Kronecker: wherever input has non-zero, place entire input grid at that block.

Frozen algorithm:
1. Fit: For each training, verify Y_train == kron((X_train != 0), X_train)
2. Apply: Compute Y_test = kron((X_test != 0), X_test)
3. Verify: ALL trainings must match exactly
4. Shape: Output is (H*H, W*W) for (H, W) input

Engineering = Math:
- Exact NumPy kron() for Kronecker product
- Exact equality verification (no tolerance)
- Binary mask (X != 0) computed exactly
- Fail-closed (any training fails → ok=False)

Receipts (WO-11):
- fit_verified_on: List of training IDs
- proof: Hashes for X, mask, kron result, Y_train
- final_shape: [R, C] for test output
- failure_reason/failure_details: If fit fails
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
import numpy as np

from arc.op.hash import hash_bytes


@dataclass
class KroneckerMaskFitRc:
    """
    Kronecker-mask fit receipt.

    Contract (WO-10K):
    Records which trainings verified, proof hashes, output shape.
    """
    engine: Literal["kronecker_mask"] = "kronecker_mask"
    fit_verified_on: List[str] = field(default_factory=list)
    final_shape: Tuple[int, int] = (0, 0)  # (R, C) for test output
    proof_hashes: Dict[str, str] = field(default_factory=dict)  # X_hash, M_hash, Y_hash
    hash: str = ""
    failure_reason: str = ""  # WO-11: "verification_failed", "shape_mismatch"
    failure_details: Dict[str, Any] = field(default_factory=dict)


def fit_kronecker_mask(
    train_pairs: List[Tuple[str, np.ndarray, np.ndarray, Any]]  # (train_id, X_t, Y_raw, truth_rc)
) -> Tuple[bool, KroneckerMaskFitRc]:
    """
    Fit Kronecker-mask engine.

    Contract (WO-10K):
    1. For each training: verify Y_train == (X_train != 0) ⊗ X_train
    2. All trainings must verify (fail-closed)
    3. Record proof hashes for audit

    Args:
        train_pairs: [(train_id, X_t, Y_raw, truth_rc), ...]

    Returns:
        (ok, KroneckerMaskFitRc)
    """
    print(f"[KRONECKER_MASK] fit called with {len(train_pairs)} trainings")
    if not train_pairs:
        return False, KroneckerMaskFitRc()

    fit_verified = []
    proof_hashes = {}

    for train_id, X_t, Y_raw, truth_rc in train_pairs:
        # Compute mask-Kronecker: Y_pred = (X != 0) ⊗ X
        M = (X_t != 0).astype(np.uint8)
        Y_pred = np.kron(M, X_t)

        print(f"[KRONECKER_MASK] {train_id}: X {X_t.shape} → Y_pred {Y_pred.shape} vs Y_raw {Y_raw.shape}")

        # Verify exact equality
        if np.array_equal(Y_pred, Y_raw):
            fit_verified.append(train_id)
            print(f"[KRONECKER_MASK] {train_id}: ✅ VERIFIED")

            # Store proof hashes for this training
            proof_hashes[train_id] = {
                "X_hash": hash_bytes(X_t.tobytes()),
                "M_hash": hash_bytes(M.tobytes()),
                "Y_pred_hash": hash_bytes(Y_pred.tobytes()),
                "Y_train_hash": hash_bytes(Y_raw.tobytes())
            }
        else:
            # Verification failed for this training
            # Check if it's a shape mismatch or value mismatch
            shape_match = (Y_pred.shape == Y_raw.shape)

            return False, KroneckerMaskFitRc(
                failure_reason="verification_failed",
                failure_details={
                    "failed_train_id": train_id,
                    "expected_shape": Y_raw.shape,
                    "predicted_shape": Y_pred.shape,
                    "shape_match": shape_match,
                    "input_shape": X_t.shape
                }
            )

    # All trainings verified
    ok = (len(fit_verified) == len(train_pairs))

    print(f"[KRONECKER_MASK] fit result: ok={ok}, verified {len(fit_verified)}/{len(train_pairs)}")

    if ok:
        # Infer output shape from first training
        H, W = train_pairs[0][1].shape  # Input shape
        final_shape = (H * H, W * W)

        # Build receipt hash
        receipt_str = f"kronecker_mask:{sorted(fit_verified)}:{final_shape}"

        rc = KroneckerMaskFitRc(
            fit_verified_on=fit_verified,
            final_shape=final_shape,
            proof_hashes=proof_hashes,
            hash=hash_bytes(receipt_str.encode())
        )
    else:
        # Partial verification (shouldn't reach here due to early return)
        failed_trains = [tid for tid, _, _, _ in train_pairs if tid not in fit_verified]
        rc = KroneckerMaskFitRc(
            fit_verified_on=fit_verified,
            failure_reason="verification_failed",
            failure_details={"failed_train_ids": failed_trains}
        )

    return ok, rc


def apply_kronecker_mask(
    test_Xt: np.ndarray,
    truth_test: Any,
    fit_rc: KroneckerMaskFitRc,
    expected_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Kronecker-mask law to test input.

    Contract (WO-10K):
    - Compute Y_test = (X_test != 0) ⊗ X_test
    - Verify shape matches expected (from shape synthesis) if provided
    - Return error if shape mismatch

    Args:
        test_Xt: Test input (H, W) in Π frame
        truth_test: Truth partition for test (unused for this engine)
        fit_rc: Fit receipt from fit_kronecker_mask
        expected_shape: Expected output shape (R, C) from shape synthesis

    Returns:
        (Y_test, apply_rc) where:
        - Y_test: Predicted output (R, C) in Π frame
        - apply_rc: Application receipt {"error": str | None, "shape": [R, C]}
    """
    # Compute mask-Kronecker
    M_test = (test_Xt != 0).astype(np.uint8)
    Y_test = np.kron(M_test, test_Xt)

    # Verify shape if expected_shape provided
    if expected_shape is not None:
        if Y_test.shape != expected_shape:
            return np.zeros(expected_shape, dtype=np.uint8), {
                "error": f"SHAPE_MISMATCH: Kronecker-mask produced {Y_test.shape}, expected {expected_shape}",
                "shape": Y_test.shape
            }

    # Success
    apply_rc = {
        "error": None,
        "shape": list(Y_test.shape),
        "test_input_hash": hash_bytes(test_Xt.tobytes()),
        "test_mask_hash": hash_bytes(M_test.tobytes()),
        "test_output_hash": hash_bytes(Y_test.tobytes())
    }

    return Y_test, apply_rc
