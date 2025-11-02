#!/usr/bin/env python3
# Test kronecker engine with a synthetic example

import numpy as np
import sys
sys.path.insert(0, '.')

from arc.op import kronecker

# Create a simple test case: 2×2 tile replicated to 4×6

def test_divisors():
    """Test divisor computation"""
    print("Testing divisor computation...")

    divisors_12 = kronecker._compute_divisors(12)
    print(f"  Divisors of 12: {divisors_12}")
    assert divisors_12 == [1, 2, 3, 4, 6, 12], f"Expected [1,2,3,4,6,12], got {divisors_12}"

    divisors_8 = kronecker._compute_divisors(8)
    print(f"  Divisors of 8: {divisors_8}")
    assert divisors_8 == [1, 2, 4, 8], f"Expected [1,2,4,8], got {divisors_8}"

    print("✓ Divisor computation test passed!")


def test_tile_verification():
    """Test tile replication verification"""
    print("\nTesting tile replication verification...")

    # Create a 2×2 tile
    tile = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    # Create replicated grid (2×3 repetitions = 4×6)
    Y = np.tile(tile, (2, 3))

    print(f"  Tile shape: {tile.shape}")
    print(f"  Replicated shape: {Y.shape}")

    # Verify replication
    is_valid = kronecker._verify_tile_replication(Y, tile)
    print(f"  Verification: {is_valid}")

    assert is_valid, "Tile verification failed"

    # Test with non-matching grid
    Y_bad = np.zeros((4, 6), dtype=np.uint8)
    is_valid_bad = kronecker._verify_tile_replication(Y_bad, tile)
    print(f"  Verification (bad): {is_valid_bad}")

    assert not is_valid_bad, "Should fail for non-matching grid"

    print("✓ Tile verification test passed!")


def test_minimal_tile_sizes():
    """Test finding minimal tile sizes"""
    print("\nTesting minimal tile size detection...")

    # Create a 2×2 tile replicated to 4×6
    tile = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    Y = np.tile(tile, (2, 3))

    minimal_tiles = kronecker._find_minimal_tile_sizes(Y)
    print(f"  Minimal tile sizes: {minimal_tiles}")

    # Should find (2, 2) as the minimal tile
    assert (2, 2) in minimal_tiles, f"Expected (2,2) in {minimal_tiles}"

    print("✓ Minimal tile size detection test passed!")


def test_fit_apply():
    """Test full fit and apply pipeline"""
    print("\nTesting fit/apply pipeline...")

    # Create training pairs with 2×2 tile
    base_tile = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    train_pairs = []

    # Training 1: 2×3 repetitions = 4×6
    Y1 = np.tile(base_tile, (2, 3))
    X1 = np.zeros_like(Y1)  # Input not used by kronecker
    train_pairs.append(("train_0", X1, Y1, None))

    # Training 2: 3×2 repetitions = 6×4
    Y2 = np.tile(base_tile, (3, 2))
    X2 = np.zeros_like(Y2)
    train_pairs.append(("train_1", X2, Y2, None))

    # Fit
    ok, fit_rc = kronecker.fit_kronecker(train_pairs)

    print(f"  Fit OK: {ok}")
    print(f"  Tile shape: {fit_rc.tile_shape}")
    print(f"  Reps: {fit_rc.reps}")
    print(f"  Verified on: {fit_rc.fit_verified_on}")

    assert ok, "Fit failed"
    assert fit_rc.tile_shape == (2, 2), f"Expected tile_shape=(2,2), got {fit_rc.tile_shape}"
    assert fit_rc.reps["train_0"] == (2, 3), f"Expected reps=(2,3), got {fit_rc.reps['train_0']}"
    assert fit_rc.reps["train_1"] == (3, 2), f"Expected reps=(3,2), got {fit_rc.reps['train_1']}"

    # Apply - replicate to 4×4 (2×2 repetitions)
    test_Xt = np.zeros((1, 1), dtype=np.uint8)  # Not used
    Y_t, apply_rc = kronecker.apply_kronecker(test_Xt, None, fit_rc, expected_shape=(4, 4))

    print(f"\n  Apply result:")
    print(f"  Output shape: {Y_t.shape}")
    print(f"  Output:\n{Y_t}")
    print(f"  Apply receipt: {apply_rc}")

    expected_Y = np.tile(base_tile, (2, 2))
    assert np.array_equal(Y_t, expected_Y), f"Output mismatch:\n{Y_t}\nvs\n{expected_Y}"

    print("✓ Fit/apply test passed!")


def test_divisibility_failure():
    """Test fail-closed on shape not divisible by tile"""
    print("\nTesting divisibility failure...")

    # Create training pairs
    base_tile = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    Y1 = np.tile(base_tile, (2, 2))
    X1 = np.zeros_like(Y1)

    train_pairs = [("train_0", X1, Y1, None)]

    # Fit
    ok, fit_rc = kronecker.fit_kronecker(train_pairs)
    assert ok, "Fit failed"

    # Apply with non-divisible shape (5×5 not divisible by 2×2)
    test_Xt = np.zeros((1, 1), dtype=np.uint8)
    Y_t, apply_rc = kronecker.apply_kronecker(test_Xt, None, fit_rc, expected_shape=(5, 5))

    print(f"  Apply result: {apply_rc}")

    assert apply_rc.get("error") == "SHAPE_NOT_DIVISIBLE", f"Expected SHAPE_NOT_DIVISIBLE error, got {apply_rc}"

    print("✓ Divisibility failure test passed!")


def test_lex_smallest_tie():
    """Test lex-smallest tile selection for ties"""
    print("\nTesting lex-smallest tile selection...")

    # Create a uniform grid that can be tiled with multiple minimal tiles
    Y = np.ones((4, 4), dtype=np.uint8) * 5

    # All of these should work: (1,1), (1,2), (2,1), (2,2), (1,4), (4,1), (2,4), (4,2), (4,4)
    # But (1,1) should be chosen (minimal area, and lex-smallest bytes)

    train_pairs = [("train_0", np.zeros_like(Y), Y, None)]

    ok, fit_rc = kronecker.fit_kronecker(train_pairs)

    print(f"  Fit OK: {ok}")
    print(f"  Tile shape: {fit_rc.tile_shape}")

    assert ok, "Fit failed"
    assert fit_rc.tile_shape == (1, 1), f"Expected (1,1) for uniform grid, got {fit_rc.tile_shape}"

    print("✓ Lex-smallest tile selection test passed!")


if __name__ == "__main__":
    test_divisors()
    test_tile_verification()
    test_minimal_tile_sizes()
    test_fit_apply()
    test_divisibility_failure()
    test_lex_smallest_tie()
    print("\n✅ All kronecker tests passed!")
