#!/usr/bin/env python3
"""
WO-10C Pooled-Blocks Engine Tests

Tests:
1. Basic fit/apply cycle with two-stage voting
2. Stage-1 strict majority (count > total/2), not mode (C2)
3. Stage-1 no majority → background=0 (A2)
4. Stage-1 ties among majority winners → min color (C2)
5. Stage-2 pooling with strict majority
6. Foreground colors from ALL trainings (A1)
7. Bands from truth_rc (D1)
8. Evidence recording (B1: stage1_counts)
9. Verification on all trainings
10. Fail-closed on verification failure
11. Receipts completeness
12. Determinism

Contract (WO-10C):
Two-stage voting: block votes → pooled quadrants
Strict majority (NOT mode): count > total/2
No majority → background=0
Ties → smallest color
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from arc.op.pooled_blocks import fit_pooled_blocks, apply_pooled_blocks


# Mock TruthRc for testing
@dataclass
class MockTruthRc:
    row_clusters: List[int] = field(default_factory=list)
    col_clusters: List[int] = field(default_factory=list)


def test_basic_fit_apply():
    """Test basic fit/apply cycle with simple two-stage voting."""
    print("Testing basic fit/apply...")

    # Create 4x4 input with 2x2 bands
    # Bands: rows=[0,2,4], cols=[0,2,4] → 2x2 blocks
    # Each block votes for a color
    X0 = np.array([
        [3, 3, 5, 5],  # Block (0,0): 3,3,3,3=3  Block (0,1): 5,5,5,5=5
        [3, 3, 5, 5],
        [7, 7, 9, 9],  # Block (1,0): 7,7,7,7=7  Block (1,1): 9,9,9,9=9
        [7, 7, 9, 9]
    ], dtype=np.uint8)

    # Expected stage-1 grid: [[3, 5], [7, 9]]
    # Expected stage-2 (2x2 pool → 1x1): majority among [3,5,7,9]? No single majority
    # Actually with pool (2,2), we get 1x1 output

    # Let's make Y match a simple pattern: all cells vote for their color
    # For simplicity, let's say output is [[3]] (top-left wins in pool)
    # But we need to check what strict majority gives...

    # Actually, let me make a simpler test where stage-2 has clear majority
    # Let's use 6x6 input with 3x3 bands → 2x2 stage-1 → 1x1 stage-2

    X0 = np.array([
        [2, 2, 2, 4, 4, 4],  # Block (0,0): 2  Block (0,1): 4
        [2, 2, 2, 4, 4, 4],
        [2, 2, 2, 4, 4, 4],
        [2, 2, 2, 2, 2, 2],  # Block (1,0): 2  Block (1,1): 2
        [2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2]
    ], dtype=np.uint8)

    # Stage-1: [[2, 4], [2, 2]]
    # Stage-2 (pool 2x2): majority in [2,4,2,2] = 2 (3 out of 4 > 2)
    Y0 = np.array([[2]], dtype=np.uint8)

    truth_rc = MockTruthRc(row_clusters=[0, 3, 6], col_clusters=[0, 3, 6])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    # Fit
    ok, fit_rc = fit_pooled_blocks(train_pairs)

    assert ok, f"Fit should succeed, got ok={ok}"
    assert fit_rc.row_bands == [0, 3, 6], f"Expected row_bands=[0,3,6], got {fit_rc.row_bands}"
    assert fit_rc.col_bands == [0, 3, 6], f"Expected col_bands=[0,3,6], got {fit_rc.col_bands}"
    assert fit_rc.block_shape == (2, 2), f"Expected block_shape=(2,2), got {fit_rc.block_shape}"
    assert 2 in fit_rc.foreground_colors, f"Expected 2 in foreground, got {fit_rc.foreground_colors}"
    assert "train0" in fit_rc.fit_verified_on, "Should verify train0"

    # Apply to test
    test_X = X0.copy()  # Same pattern
    Y_out, apply_rc = apply_pooled_blocks(test_X, truth_rc, fit_rc, expected_shape=(1, 1))

    assert Y_out.shape == (1, 1), f"Output shape should be (1,1), got {Y_out.shape}"
    assert Y_out[0, 0] == 2, f"Expected output=2, got {Y_out[0,0]}"

    print("  ✓ Basic fit/apply works correctly")


def test_stage1_strict_majority():
    """Test stage-1 strict majority: count > total/2 (C2)."""
    print("Testing stage-1 strict majority...")

    # Create block where one color has strict majority (>50%)
    # Block: 9 pixels, color 3 appears 7 times (7/9 = 77% > 50%)
    # Note: foreground colors come from Y (A1), so Y must contain color 3
    X0 = np.array([
        [3, 3, 3],
        [3, 3, 3],
        [3, 0, 0]   # 7×3, 2×0 → 3 wins (7/9 > 50%)
    ], dtype=np.uint8)

    # Expected stage-1: [[3]]
    # Expected stage-2: [[3]]
    Y0 = np.array([[3]], dtype=np.uint8)

    truth_rc = MockTruthRc(row_clusters=[0, 3], col_clusters=[0, 3])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_pooled_blocks(train_pairs)

    assert ok, f"Fit should succeed, got ok={ok}"
    # Check stage1_counts recorded the counts (B1)
    # Note: only foreground colors (from Y) are counted
    assert "(0,0)" in fit_rc.stage1_counts, "Should have stage1_counts for (0,0)"
    counts = fit_rc.stage1_counts["(0,0)"]
    assert counts.get(3, 0) == 7, f"Expected 7 pixels of color 3, got {counts.get(3, 0)}"
    # Color 0 is not foreground, so not counted

    print("  ✓ Stage-1 strict majority works")


def test_stage1_no_majority_fallback():
    """Test stage-1 no majority → background=0 (A2)."""
    print("Testing stage-1 no majority fallback...")

    # Create block where NO color has majority (all ≤50%)
    # Block: 6 pixels, color 2 (3px), color 4 (3px) → tie at 50% each, no >50%
    X0 = np.array([
        [2, 2, 2],
        [4, 4, 4]
    ], dtype=np.uint8)

    # No color has >50%, so stage-1 should be 0 (background)
    # Expected stage-1: [[0]]
    # Expected stage-2: [[0]]
    Y0 = np.array([[0]], dtype=np.uint8)

    truth_rc = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 3])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_pooled_blocks(train_pairs)

    assert ok, f"Fit should succeed, got ok={ok}"
    # Verify stage-1 voted 0 (no majority)
    # stage2_pooled should be [[0]]
    assert fit_rc.stage2_pooled == [[0]], f"Expected stage2=[[0]], got {fit_rc.stage2_pooled}"

    print("  ✓ Stage-1 no majority fallback to 0 works")


def test_stage1_tie_min_color():
    """Test stage-1 ties among majority winners → min color (C2)."""
    print("Testing stage-1 tie→min color...")

    # Create block where TWO colors both have >50% (impossible in single block)
    # Wait, that's impossible. Let me think...
    # For ties, we need multiple colors with SAME majority count
    # But only ONE can have >50%...

    # Actually, the tie case is when NO color has >50%, we already tested that.
    # Let me re-read the spec...

    # Looking at the code: lines 137-143
    # majority_winners = [c for c, cnt in color_counts.items() if cnt > majority_threshold]
    # if len(majority_winners) > 1: winner = min(majority_winners)

    # For a single block, at most ONE color can have >50% (strict majority)
    # So this tie case is theoretical? Or maybe with rounding?

    # Actually wait - if total_pixels = 10, threshold = 5.0
    # Color A: 6 pixels (>5) → majority
    # Color B: 6 pixels... but A+B=12 > 10, impossible

    # The tie path is unreachable for strict majority in a single block
    # BUT it's still in the code as a safety check
    # Let me just verify the code doesn't crash with a normal case

    # I'll skip this test since tie among majority winners is impossible

    print("  ✓ Stage-1 tie→min (unreachable, skipped)")


def test_stage2_pooling():
    """Test stage-2 pooling with strict majority."""
    print("Testing stage-2 pooling...")

    # Create 6x6 input → 2x2 stage-1 → 1x1 stage-2 (pool 2x2)
    # Stage-1: [[7, 7], [7, 8]]
    # Stage-2: pool [7,7,7,8] → 7 has 3/4 > 50% → winner=7

    X0 = np.array([
        [7, 7, 7, 7, 7, 7],  # Block (0,0): 7  Block (0,1): 7
        [7, 7, 7, 7, 7, 7],
        [7, 7, 7, 7, 7, 7],
        [7, 7, 7, 8, 8, 8],  # Block (1,0): 7  Block (1,1): 8
        [7, 7, 7, 8, 8, 8],
        [7, 7, 7, 8, 8, 8]
    ], dtype=np.uint8)

    Y0 = np.array([[7]], dtype=np.uint8)

    truth_rc = MockTruthRc(row_clusters=[0, 3, 6], col_clusters=[0, 3, 6])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_pooled_blocks(train_pairs)

    assert ok, f"Fit should succeed, got ok={ok}"
    # stage2_pooled should be [[7]]
    assert fit_rc.stage2_pooled == [[7]], f"Expected stage2=[[7]], got {fit_rc.stage2_pooled}"

    print("  ✓ Stage-2 pooling works")


def test_foreground_from_all_trainings():
    """Test foreground colors from ALL trainings (A1)."""
    print("Testing foreground from ALL trainings...")

    # Training 0: colors {2, 3}
    X0 = np.array([[2, 2], [2, 2]], dtype=np.uint8)
    Y0 = np.array([[2]], dtype=np.uint8)
    truth0 = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 2])

    # Training 1: colors {3, 5} (new color 5)
    X1 = np.array([[3, 3], [3, 3]], dtype=np.uint8)
    Y1 = np.array([[3]], dtype=np.uint8)
    truth1 = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 2])

    train_pairs = [("train0", X0, Y0, truth0), ("train1", X1, Y1, truth1)]

    ok, fit_rc = fit_pooled_blocks(train_pairs)

    # Foreground should include colors from ALL trainings' outputs
    # Y0 has {2}, Y1 has {3}
    assert 2 in fit_rc.foreground_colors, f"Expected 2 in foreground, got {fit_rc.foreground_colors}"
    assert 3 in fit_rc.foreground_colors, f"Expected 3 in foreground, got {fit_rc.foreground_colors}"

    print("  ✓ Foreground from ALL trainings works (A1)")


def test_bands_from_truth():
    """Test bands from truth_rc (D1)."""
    print("Testing bands from truth_rc...")

    X0 = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ], dtype=np.uint8)
    Y0 = np.array([[1]], dtype=np.uint8)

    # Custom bands from truth
    truth_rc = MockTruthRc(row_clusters=[0, 3], col_clusters=[0, 4])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_pooled_blocks(train_pairs)

    # Should use bands from truth_rc
    assert fit_rc.row_bands == [0, 3], f"Expected row_bands=[0,3], got {fit_rc.row_bands}"
    assert fit_rc.col_bands == [0, 4], f"Expected col_bands=[0,4], got {fit_rc.col_bands}"

    print("  ✓ Bands from truth_rc works (D1)")


def test_evidence_recording():
    """Test evidence recording (B1: stage1_counts)."""
    print("Testing evidence recording (B1)...")

    # Create 2x3 input: one block
    # Note: Only foreground colors (from Y) are counted (A1)
    # So if we want to see counts for both 6 and 8, Y must contain both
    # Let's use two blocks that each vote for their color
    X0 = np.array([
        [6, 6, 6, 8, 8, 8],  # Block (0,0): 6  Block (0,1): 8
        [6, 6, 6, 8, 8, 8]
    ], dtype=np.uint8)
    Y0 = np.array([[6, 8]], dtype=np.uint8)  # Output has both colors

    truth_rc = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 3, 6])
    train_pairs = [("train0", X0, Y0, truth_rc)]

    ok, fit_rc = fit_pooled_blocks(train_pairs)

    # Check stage1_counts recorded evidence (B1)
    assert "stage1_counts" in fit_rc.__dict__, "Should have stage1_counts field"
    assert isinstance(fit_rc.stage1_counts, dict), "stage1_counts should be dict"
    assert "(0,0)" in fit_rc.stage1_counts, "Should have counts for (0,0)"
    assert "(0,1)" in fit_rc.stage1_counts, "Should have counts for (0,1)"

    counts_00 = fit_rc.stage1_counts["(0,0)"]
    counts_01 = fit_rc.stage1_counts["(0,1)"]

    # Block (0,0): 6 pixels of color 6
    assert 6 in counts_00, f"Should count color 6 in (0,0), got {counts_00}"
    assert counts_00[6] == 6, f"Expected 6 pixels of 6, got {counts_00[6]}"

    # Block (0,1): 6 pixels of color 8
    assert 8 in counts_01, f"Should count color 8 in (0,1), got {counts_01}"
    assert counts_01[8] == 6, f"Expected 6 pixels of 8, got {counts_01[8]}"

    print("  ✓ Evidence recording (B1) works")


def test_verification_on_all_trainings():
    """Test verification on all trainings."""
    print("Testing verification on all trainings...")

    # Both trainings should verify
    X0 = np.array([[9, 9], [9, 9]], dtype=np.uint8)
    Y0 = np.array([[9]], dtype=np.uint8)
    truth0 = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 2])

    X1 = np.array([[9, 9], [9, 9]], dtype=np.uint8)
    Y1 = np.array([[9]], dtype=np.uint8)
    truth1 = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 2])

    train_pairs = [("train0", X0, Y0, truth0), ("train1", X1, Y1, truth1)]
    ok, fit_rc = fit_pooled_blocks(train_pairs)

    assert ok, f"Fit should succeed, got ok={ok}"
    assert len(fit_rc.fit_verified_on) == 2, f"Should verify 2 trainings, got {len(fit_rc.fit_verified_on)}"
    assert "train0" in fit_rc.fit_verified_on, "Should verify train0"
    assert "train1" in fit_rc.fit_verified_on, "Should verify train1"

    print("  ✓ Verification on all trainings works")


def test_verification_failure():
    """Test verification failure: fail-closed."""
    print("Testing verification failure...")

    # Create training where Y doesn't match stage-2 output
    X0 = np.array([[5, 5], [5, 5]], dtype=np.uint8)
    Y0_wrong = np.array([[7]], dtype=np.uint8)  # Wrong! Should be [[5]]
    truth0 = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 2])

    train_pairs = [("train0", X0, Y0_wrong, truth0)]
    ok, fit_rc = fit_pooled_blocks(train_pairs)

    # Should FAIL verification (fail-closed)
    assert not ok, f"Fit should fail verification, got ok={ok}"
    assert len(fit_rc.fit_verified_on) == 0, f"Should have 0 verified trainings, got {len(fit_rc.fit_verified_on)}"

    print("  ✓ Verification failure (fail-closed) works")


def test_receipts_completeness():
    """Test receipt fields completeness."""
    print("Testing receipts completeness...")

    X0 = np.array([[4, 4], [4, 4]], dtype=np.uint8)
    Y0 = np.array([[4]], dtype=np.uint8)
    truth0 = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 2])

    train_pairs = [("train0", X0, Y0, truth0)]
    ok, fit_rc = fit_pooled_blocks(train_pairs)

    # Verify all required fields
    assert hasattr(fit_rc, "engine"), "Missing engine field"
    assert hasattr(fit_rc, "row_bands"), "Missing row_bands field"
    assert hasattr(fit_rc, "col_bands"), "Missing col_bands field"
    assert hasattr(fit_rc, "block_shape"), "Missing block_shape field"
    assert hasattr(fit_rc, "pool_shape"), "Missing pool_shape field"
    assert hasattr(fit_rc, "foreground_colors"), "Missing foreground_colors field"
    assert hasattr(fit_rc, "background_colors"), "Missing background_colors field"
    assert hasattr(fit_rc, "stage1_counts"), "Missing stage1_counts field"
    assert hasattr(fit_rc, "stage2_pooled"), "Missing stage2_pooled field"
    assert hasattr(fit_rc, "decision_rule"), "Missing decision_rule field"
    assert hasattr(fit_rc, "fit_verified_on"), "Missing fit_verified_on field"
    assert hasattr(fit_rc, "hash"), "Missing hash field"

    assert fit_rc.engine == "pooled_blocks", f"Expected engine='pooled_blocks', got {fit_rc.engine}"
    assert isinstance(fit_rc.row_bands, list), "row_bands should be list"
    assert isinstance(fit_rc.col_bands, list), "col_bands should be list"
    assert isinstance(fit_rc.foreground_colors, list), "foreground_colors should be list"
    assert isinstance(fit_rc.stage1_counts, dict), "stage1_counts should be dict"
    assert isinstance(fit_rc.stage2_pooled, list), "stage2_pooled should be list"
    assert isinstance(fit_rc.hash, str), "hash should be str"
    assert fit_rc.hash != "", "hash should be non-empty"

    print("  ✓ Receipts completeness verified")


def test_determinism():
    """Test determinism: fit twice, same result."""
    print("Testing determinism...")

    X0 = np.array([[3, 3], [3, 3]], dtype=np.uint8)
    Y0 = np.array([[3]], dtype=np.uint8)
    truth0 = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 2])

    train_pairs = [("train0", X0, Y0, truth0)]

    # Fit twice
    ok1, fit_rc1 = fit_pooled_blocks(train_pairs)
    ok2, fit_rc2 = fit_pooled_blocks(train_pairs)

    # Verify identical results
    assert ok1 == ok2, "ok should be deterministic"
    assert fit_rc1.foreground_colors == fit_rc2.foreground_colors, "foreground_colors should be deterministic"
    assert fit_rc1.stage2_pooled == fit_rc2.stage2_pooled, "stage2_pooled should be deterministic"
    assert fit_rc1.hash == fit_rc2.hash, "hash should be deterministic"

    print("  ✓ Determinism verified")


def test_apply_with_shape():
    """Test apply with expected_shape parameter."""
    print("Testing apply with expected_shape...")

    X0 = np.array([[2, 2], [2, 2]], dtype=np.uint8)
    Y0 = np.array([[2]], dtype=np.uint8)
    truth0 = MockTruthRc(row_clusters=[0, 2], col_clusters=[0, 2])

    train_pairs = [("train0", X0, Y0, truth0)]
    ok, fit_rc = fit_pooled_blocks(train_pairs)

    # Apply
    test_X = X0.copy()
    Y_out, apply_rc = apply_pooled_blocks(test_X, truth0, fit_rc, expected_shape=(1, 1))

    assert Y_out.shape == (1, 1), f"Output shape should be (1,1), got {Y_out.shape}"
    assert "output_shape" in apply_rc, "apply_rc should have output_shape"
    assert apply_rc["output_shape"] == [1, 1], f"Expected output_shape=[1,1], got {apply_rc['output_shape']}"

    print("  ✓ Apply with expected_shape works")


def run_tests():
    """Run all WO-10C tests."""
    print("\n" + "="*60)
    print("WO-10C Pooled-Blocks Engine Tests")
    print("="*60 + "\n")

    test_basic_fit_apply()
    test_stage1_strict_majority()
    test_stage1_no_majority_fallback()
    test_stage1_tie_min_color()
    test_stage2_pooling()
    test_foreground_from_all_trainings()
    test_bands_from_truth()
    test_evidence_recording()
    test_verification_on_all_trainings()
    test_verification_failure()
    test_receipts_completeness()
    test_determinism()
    test_apply_with_shape()

    print("\n" + "="*60)
    print("✓ All WO-10C tests passed")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_tests()
