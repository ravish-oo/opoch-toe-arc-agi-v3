#!/usr/bin/env python3
"""WO-10E Slice-Stack Engine Tests"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from arc.op.slice_stack import fit_slice_stack, apply_slice_stack


@dataclass
class MockTruthRc:
    col_clusters: List[int] = field(default_factory=list)
    row_clusters: List[int] = field(default_factory=list)


def test_basic_column_slices():
    """Test basic column-wise slice stacking."""
    print("Testing basic column slices...")

    # Input: 3 columns, each distinct pattern
    X0 = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]
    ], dtype=np.uint8)

    # Output: 3 columns, transformed
    Y0 = np.array([
        [4, 5, 6],
        [4, 5, 6],
        [4, 5, 6]
    ], dtype=np.uint8)

    truth_rc = MockTruthRc(col_clusters=[0, 1, 2, 3])  # col_clusters exist → axis="cols"
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_slice_stack(train_pairs)

    assert ok, f"Fit should succeed, got ok={ok}"
    assert fit_rc.axis == "cols", f"Expected axis='cols', got {fit_rc.axis}"
    assert fit_rc.slice_height == 3, f"Expected slice_height=3, got {fit_rc.slice_height}"
    assert fit_rc.slice_width == 1, f"Expected slice_width=1, got {fit_rc.slice_width}"
    assert len(fit_rc.dict) == 3, f"Expected 3 dict entries, got {len(fit_rc.dict)}"

    print("  ✓ Basic column slices work")


def test_basic_row_slices():
    """Test basic row-wise slice stacking."""
    print("Testing basic row slices...")

    # Input: 3 rows
    X0 = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ], dtype=np.uint8)

    # Output: 3 rows, transformed
    Y0 = np.array([
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6]
    ], dtype=np.uint8)

    truth_rc = MockTruthRc(col_clusters=[])  # No col_clusters → axis="rows"
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_slice_stack(train_pairs)

    assert ok, f"Fit should succeed, got ok={ok}"
    assert fit_rc.axis == "rows", f"Expected axis='rows', got {fit_rc.axis}"
    assert fit_rc.slice_height == 1, f"Expected slice_height=1, got {fit_rc.slice_height}"
    assert fit_rc.slice_width == 3, f"Expected slice_width=3, got {fit_rc.slice_width}"

    print("  ✓ Basic row slices work")


def test_dictionary_conflict_detection():
    """Test conflict detection: same input → different outputs."""
    print("Testing dictionary conflict detection...")

    # Training 1: column [1,1,1] → [4,4,4]
    X0 = np.array([[1], [1], [1]], dtype=np.uint8)
    Y0 = np.array([[4], [4], [4]], dtype=np.uint8)

    # Training 2: same column [1,1,1] → [5,5,5] (CONFLICT)
    X1 = np.array([[1], [1], [1]], dtype=np.uint8)
    Y1 = np.array([[5], [5], [5]], dtype=np.uint8)

    truth_rc = MockTruthRc(col_clusters=[0, 1])
    train_pairs = [("train0", X0, Y0, truth_rc), ("train1", X1, Y1, truth_rc)]

    ok, fit_rc = fit_slice_stack(train_pairs)

    # Should FAIL due to conflict (same input → different outputs)
    assert not ok, "Fit should fail on conflict"

    print("  ✓ Dictionary conflict detection works")


def test_slice_count_mismatch():
    """Test fail on input/output slice count mismatch."""
    print("Testing slice count mismatch...")

    # Input: 3 columns
    X0 = np.array([[1, 2, 3]], dtype=np.uint8)
    # Output: 2 columns (MISMATCH)
    Y0 = np.array([[4, 5]], dtype=np.uint8)

    truth_rc = MockTruthRc(col_clusters=[0, 1, 2, 3])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_slice_stack(train_pairs)

    # Should FAIL (can't build 1:1 mapping)
    assert not ok, "Fit should fail on slice count mismatch"

    print("  ✓ Slice count mismatch detection works")


def test_verification_on_all_trainings():
    """Test verification on all trainings."""
    print("Testing verification on all trainings...")

    # Training 1
    X0 = np.array([[1, 2]], dtype=np.uint8)
    Y0 = np.array([[3, 4]], dtype=np.uint8)

    # Training 2 (consistent mapping)
    X1 = np.array([[2, 1]], dtype=np.uint8)  # Reordered columns
    Y1 = np.array([[4, 3]], dtype=np.uint8)  # Consistent: 1→3, 2→4

    truth_rc = MockTruthRc(col_clusters=[0, 1, 2])
    train_pairs = [("train0", X0, Y0, truth_rc), ("train1", X1, Y1, truth_rc)]

    ok, fit_rc = fit_slice_stack(train_pairs)

    assert ok, "Fit should succeed with consistent mapping"
    assert len(fit_rc.fit_verified_on) == 2, "Should verify both trainings"

    print("  ✓ Verification on all trainings works")


def test_apply_with_dictionary():
    """Test apply using learned dictionary."""
    print("Testing apply with dictionary...")

    # Training
    X0 = np.array([[1, 2]], dtype=np.uint8)
    Y0 = np.array([[5, 6]], dtype=np.uint8)

    truth_rc = MockTruthRc(col_clusters=[0, 1, 2])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_slice_stack(train_pairs)

    # Test (same pattern)
    test_X = np.array([[1, 2]], dtype=np.uint8)
    Y_out, apply_rc = apply_slice_stack(test_X, truth_rc, fit_rc, expected_shape=(1, 2))

    assert Y_out.shape == (1, 2), f"Expected shape (1,2), got {Y_out.shape}"
    assert np.array_equal(Y_out, Y0), "Output should match expected"

    print("  ✓ Apply with dictionary works")


def test_unseen_signature_fail_closed():
    """Test UNSEEN_SIGNATURE fail-closed."""
    print("Testing UNSEEN_SIGNATURE fail-closed...")

    # Training with specific columns
    X0 = np.array([[1, 2]], dtype=np.uint8)
    Y0 = np.array([[5, 6]], dtype=np.uint8)

    truth_rc = MockTruthRc(col_clusters=[0, 1, 2])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_slice_stack(train_pairs)

    # Test with UNSEEN column
    test_X = np.array([[1, 9]], dtype=np.uint8)  # Column "9" not in training
    Y_out, apply_rc = apply_slice_stack(test_X, truth_rc, fit_rc)

    # Should fail-closed on UNSEEN_SIGNATURE
    assert "error" in apply_rc, "Should have error in apply_rc"
    assert apply_rc["error"] == "UNSEEN_SIGNATURE", f"Expected UNSEEN_SIGNATURE, got {apply_rc['error']}"
    assert Y_out.shape == (0, 0), "Should return empty array on fail"

    print("  ✓ UNSEEN_SIGNATURE fail-closed works")


def test_determinism():
    """Test determinism: fit twice, same result."""
    print("Testing determinism...")

    X0 = np.array([[1, 2]], dtype=np.uint8)
    Y0 = np.array([[3, 4]], dtype=np.uint8)

    truth_rc = MockTruthRc(col_clusters=[0, 1, 2])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok1, fit_rc1 = fit_slice_stack(train_pairs)
    ok2, fit_rc2 = fit_slice_stack(train_pairs)

    assert ok1 == ok2, "ok should be deterministic"
    assert fit_rc1.axis == fit_rc2.axis, "axis should be deterministic"
    assert fit_rc1.dict == fit_rc2.dict, "dict should be deterministic"
    assert fit_rc1.hash == fit_rc2.hash, "hash should be deterministic"

    print("  ✓ Determinism verified")


def run_tests():
    print("\n" + "="*60)
    print("WO-10E Slice-Stack Engine Tests")
    print("="*60 + "\n")

    test_basic_column_slices()
    test_basic_row_slices()
    test_dictionary_conflict_detection()
    test_slice_count_mismatch()
    test_verification_on_all_trainings()
    test_apply_with_dictionary()
    test_unseen_signature_fail_closed()
    test_determinism()

    print("\n" + "="*60)
    print("✓ All WO-10E tests passed")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_tests()
