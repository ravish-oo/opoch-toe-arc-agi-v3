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

# D4 composition table (frozen)
# Contract (WO-04C): compose_d4(p1, p2) = result of applying p2 then p1
# COMPOSE_D4[p1][p2] = p1 ∘ p2
# Computed from D4 group multiplication table
COMPOSE_D4 = [
    # p2:  0  1  2  3  4  5  6  7
    [0, 1, 2, 3, 4, 5, 6, 7],  # p1=0 (identity): 0∘p2 = p2
    [1, 2, 3, 0, 5, 6, 7, 4],  # p1=1 (R90)
    [2, 3, 0, 1, 6, 7, 4, 5],  # p1=2 (R180)
    [3, 0, 1, 2, 7, 4, 5, 6],  # p1=3 (R270)
    [4, 7, 6, 5, 0, 3, 2, 1],  # p1=4 (FH)
    [5, 4, 7, 6, 1, 0, 3, 2],  # p1=5 (FH∘R90)
    [6, 5, 4, 7, 2, 1, 0, 3],  # p1=6 (FH∘R180)
    [7, 6, 5, 4, 3, 2, 1, 0],  # p1=7 (FH∘R270)
]


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


def compose_pose(p1: int, p2: int) -> int:
    """
    Compose two D4 poses.

    Contract (WO-04C): p1 ∘ p2 = result of applying p2 then p1

    Args:
        p1: first pose (applied after p2)
        p2: second pose (applied first)

    Returns:
        Composed pose identifier

    Raises:
        ValueError: if p1 or p2 not in {0..7}
    """
    if p1 not in POSES or p2 not in POSES:
        raise ValueError(f"Invalid pose_id: p1={p1}, p2={p2}")
    return COMPOSE_D4[p1][p2]


def transform_vector(dr: int, dc: int, pose_id: int) -> tuple[int, int]:
    """
    Apply D4 pose to a translation vector.

    Contract (WO-04C): Apply rotation/reflection to (dr, dc) vector.

    For a grid of size (H, W), the transformation depends on the pose:
    - Rotations change the direction and potentially swap coordinates
    - Reflections negate coordinates

    Args:
        dr: row translation
        dc: column translation
        pose_id: D4 pose to apply

    Returns:
        (dr', dc') transformed translation vector

    Raises:
        ValueError: if pose_id not in {0..7}
    """
    if pose_id not in POSES:
        raise ValueError(f"Invalid pose_id {pose_id}")

    if pose_id == 0:
        # Identity: (dr, dc) unchanged
        return (dr, dc)
    elif pose_id == 1:
        # R90 clockwise: (r,c) → (c, H-1-r) → dr'=dc, dc'=-dr
        return (dc, -dr)
    elif pose_id == 2:
        # R180: (r,c) → (H-1-r, W-1-c) → dr'=-dr, dc'=-dc
        return (-dr, -dc)
    elif pose_id == 3:
        # R270 clockwise: (r,c) → (W-1-c, r) → dr'=-dc, dc'=dr
        return (-dc, dr)
    elif pose_id == 4:
        # Flip horizontal: (r,c) → (r, W-1-c) → dr'=dr, dc'=-dc
        return (dr, -dc)
    elif pose_id == 5:
        # FH ∘ R90: flip then rot90 → (r,c) → (r, W-1-c) → (W-1-c, H-1-r)
        # dr'=-dc, dc'=-dr
        return (-dc, -dr)
    elif pose_id == 6:
        # FH ∘ R180: flip then rot180 → (r,c) → (r, W-1-c) → (H-1-r, c)
        # dr'=-dr, dc'=dc
        return (-dr, dc)
    elif pose_id == 7:
        # FH ∘ R270: flip then rot270 → (r,c) → (r, W-1-c) → (c, r)
        # dr'=dc, dc'=dr
        return (dc, dr)
    else:
        raise ValueError(f"Unhandled pose_id {pose_id}")
