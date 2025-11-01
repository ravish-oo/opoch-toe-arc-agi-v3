# arc/op/d4.py
# WO-01: D4 dihedral group operations
# Implements 02_determinism_addendum.md §1.2

from __future__ import annotations
import numpy as np

# D4 pose IDs (frozen)
# Contract (02_determinism_addendum.md §1.2 lines 53-57):
# 0=identity, 1=rot90, 2=rot180, 3=rot270,
# 4=flipH, 5=flipH∘rot90, 6=flipH∘rot180, 7=flipH∘rot270
POSES = (0, 1, 2, 3, 4, 5, 6, 7)

# Inverse map for D4 group
# Used in unpresentation to invert pose transformations
INVERSE_POSE = {
    0: 0,  # identity inverse is identity
    1: 3,  # R90 inverse is R270
    2: 2,  # R180 inverse is R180 (self-inverse)
    3: 1,  # R270 inverse is R90
    4: 4,  # flipH inverse is flipH (self-inverse)
    5: 5,  # flipH∘R90 is self-inverse (involution)
    6: 6,  # flipH∘R180 is self-inverse (involution)
    7: 7,  # flipH∘R270 is self-inverse (involution)
}


def apply_pose(G: np.ndarray, pose_id: int) -> np.ndarray:
    """
    Apply D4 pose transformation to grid.

    Contract (02_determinism_addendum.md §1.2):
    D4 ids: 0=I, 1=R90, 2=R180, 3=R270, 4=FH, 5=FH∘R90, 6=FH∘R180, 7=FH∘R270

    Note: numpy's rot90(k=1) is 90° counterclockwise.
    For 90° clockwise (R90), use rot90(k=3).

    Args:
        G: input grid (H×W)
        pose_id: D4 pose identifier (0-7)

    Returns:
        Transformed grid

    Raises:
        ValueError: if pose_id not in {0..7}
    """
    if pose_id not in POSES:
        raise ValueError(f"Invalid pose_id {pose_id}, must be in {POSES}")

    if pose_id == 0:
        return G.copy()
    elif pose_id == 1:
        # R90 clockwise = rot90(k=3) counterclockwise
        return np.rot90(G, k=3)
    elif pose_id == 2:
        # R180 = rot90(k=2)
        return np.rot90(G, k=2)
    elif pose_id == 3:
        # R270 clockwise = rot90(k=1) counterclockwise
        return np.rot90(G, k=1)
    elif pose_id == 4:
        # Flip horizontal (left-right)
        return np.fliplr(G)
    elif pose_id == 5:
        # FH ∘ R90: flip then rotate 90° clockwise
        return np.rot90(np.fliplr(G), k=3)
    elif pose_id == 6:
        # FH ∘ R180: flip then rotate 180°
        return np.rot90(np.fliplr(G), k=2)
    elif pose_id == 7:
        # FH ∘ R270: flip then rotate 270° clockwise
        return np.rot90(np.fliplr(G), k=1)
    else:
        raise ValueError(f"Unhandled pose_id {pose_id}")


def raster_lex_key(G: np.ndarray) -> tuple:
    """
    Compute raster lexicographic key for grid.

    Contract: direct tuple of ints, row-major, for total order comparison.
    Used in D4 lex pose selection (choose pose with minimal lex key).

    Args:
        G: input grid

    Returns:
        Tuple of integers (row-major flattened)
    """
    # Row-major flattened as tuple of ints
    # This gives stable total order for comparison
    return tuple(int(x) for x in G.reshape(-1))


def get_inverse_pose(pose_id: int) -> int:
    """
    Get inverse pose for a given D4 pose.

    Args:
        pose_id: D4 pose identifier (0-7)

    Returns:
        Inverse pose identifier

    Raises:
        ValueError: if pose_id not in {0..7}
    """
    if pose_id not in INVERSE_POSE:
        raise ValueError(f"Invalid pose_id {pose_id}")
    return INVERSE_POSE[pose_id]
