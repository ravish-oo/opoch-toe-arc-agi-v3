#!/usr/bin/env python3
# arc/op/components.py
# WO-03: Components + stable matching (BLOCKER)
# Implements 4-connected components per color, canonical invariants, and stable matching

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
from .hash import hash_bytes
from .receipts import ComponentsRc


@dataclass
class CompInv:
    """
    Component invariant tuple (pose/color-independent).

    Contract (02_determinism_addendum.md §3 lines 115-122):
    Invariants tuple (lex key): (area, bbox_h, bbox_w, perim4, outline_hash, anchor_rc)
    where outline_hash = BLAKE3 of the D4-min raster of the component's mask.
    """
    color: int
    area: int
    bbox_h: int
    bbox_w: int
    perim4: int
    outline_hash: str  # BLAKE3 of D4-min mask raster (pose/color-independent)
    anchor_r: int      # top-left row of bbox in Π frame
    anchor_c: int      # top-left col of bbox in Π frame


@dataclass
class MatchRc:
    """
    Stable matching receipt.

    Contract:
    - pairs: indices into left/right invariants lists
    - left_only: unmatched components in left
    - right_only: unmatched components in right
    - verified_pixelwise: whether all matched pairs verified
    """
    pairs: List[Tuple[int, int]]  # [(left_idx, right_idx), ...]
    left_only: List[int]
    right_only: List[int]
    verified_pixelwise: bool


def _d4_poses(B: np.ndarray) -> List[np.ndarray]:
    """
    Generate all 8 D4 poses of a boolean array.

    Contract (02_determinism_addendum.md §1.2):
    D4 ids: 0=identity, 1=rot90, 2=rot180, 3=rot270, 4=flipH, 5=flipH∘rot90, 6=flipH∘rot180, 7=flipH∘rot270

    Args:
        B: boolean array

    Returns:
        List of 8 D4 poses
    """
    poses = [
        B,                           # 0: identity
        np.rot90(B, k=3),           # 1: rot90 (k=3 rotates CCW 90° = CW 270°)
        np.rot90(B, k=2),           # 2: rot180
        np.rot90(B, k=1),           # 3: rot270 (k=1 rotates CCW 270° = CW 90°)
        np.fliplr(B),               # 4: flipH (horizontal flip)
        np.rot90(np.fliplr(B), k=3),  # 5: flipH∘rot90
        np.rot90(np.fliplr(B), k=2),  # 6: flipH∘rot180
        np.rot90(np.fliplr(B), k=1),  # 7: flipH∘rot270
    ]
    return poses


def _outline_hash(B: np.ndarray) -> str:
    """
    Compute D4-minimal outline hash of a component mask.

    Contract (02_determinism_addendum.md §3 line 118):
    "outline_hash = BLAKE3 of the D4-min raster of the component's mask
    (independent of pose/color)"

    Args:
        B: boolean bbox mask

    Returns:
        BLAKE3 hex hash of lex-min D4 pose bytes
    """
    candidates = []
    for pose in _d4_poses(B):
        # Convert to uint8 row-major bytes
        pose_bytes = pose.astype(np.uint8).tobytes(order='C')
        candidates.append(pose_bytes)

    # Lex-min bytes across all D4 poses
    min_bytes = min(candidates)

    return hash_bytes(min_bytes)


def _safe_slice(G: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    """
    Safe bbox slicing with bounds checks (WO-11G).

    Raises IndexError with precise message if bbox is out of bounds.

    Args:
        G: grid to slice
        r0, c0: top-left (inclusive)
        r1, c1: bottom-right (exclusive)

    Returns:
        G[r0:r1, c0:c1]

    Raises:
        IndexError: if bbox out of bounds
    """
    H, W = G.shape
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        raise IndexError(
            f"bbox_oob: req=({r0}:{r1},{c0}:{c1}) vs grid=({H},{W})"
        )
    return G[r0:r1, c0:c1]


def _perimeter4(B: np.ndarray) -> int:
    """
    Compute 4-connected perimeter.

    Perimeter = count of edges where component neighbor is background or out-of-bounds.

    Args:
        B: boolean bbox mask

    Returns:
        4-connected perimeter count
    """
    H, W = B.shape
    per = 0

    for r in range(H):
        for c in range(W):
            if not B[r, c]:
                continue

            # Count edges to background/boundary
            # Up
            if r == 0 or not B[r-1, c]:
                per += 1
            # Down
            if r == H-1 or not B[r+1, c]:
                per += 1
            # Left
            if c == 0 or not B[r, c-1]:
                per += 1
            # Right
            if c == W-1 or not B[r, c+1]:
                per += 1

    return per


def _label_cc4(M: np.ndarray) -> np.ndarray:
    """
    Two-pass 4-connected component labeling.

    Contract: Deterministic labeling with union-find.

    Args:
        M: boolean mask

    Returns:
        Label array (0 = background, 1..N = component labels)
    """
    H, W = M.shape
    labels = np.zeros((H, W), dtype=np.int32)

    # Union-find parent array
    parent = [0]  # parent[0] unused

    def find(x: int) -> int:
        """Find with path compression."""
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            next_x = parent[x]
            parent[x] = root
            x = next_x
        return root

    def union(a: int, b: int) -> None:
        """Union by attaching b's root to a's root."""
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    next_label = 1

    # First pass: assign provisional labels
    for r in range(H):
        for c in range(W):
            if not M[r, c]:
                continue

            neighbors = []
            # Check up
            if r > 0 and labels[r-1, c] > 0:
                neighbors.append(labels[r-1, c])
            # Check left
            if c > 0 and labels[r, c-1] > 0:
                neighbors.append(labels[r, c-1])

            if not neighbors:
                # New component
                parent.append(next_label)
                labels[r, c] = next_label
                next_label += 1
            else:
                # Use minimum neighbor label
                min_label = min(neighbors)
                labels[r, c] = min_label
                # Union all neighbor labels
                for nb in neighbors:
                    union(min_label, nb)

    # Second pass: relabel with root labels (compacted)
    label_map = {}
    new_id = 1

    for r in range(H):
        for c in range(W):
            if labels[r, c] > 0:
                root = find(labels[r, c])
                if root not in label_map:
                    label_map[root] = new_id
                    new_id += 1
                labels[r, c] = label_map[root]

    return labels


def cc4_by_color(G: np.ndarray) -> Tuple[List[np.ndarray], ComponentsRc]:
    """
    Extract 4-connected components per color with canonical invariants.

    Contract (02_determinism_addendum.md §3):
    - connectivity = 4 (frozen)
    - Invariants: (area, bbox_h, bbox_w, perim4, outline_hash, anchor_r, anchor_c)
    - Lex sort by invariant tuple

    Args:
        G: grid in Π frame

    Returns:
        (component_masks, ComponentsRc) where masks are bbox-relative boolean arrays
        sorted by lex order of invariant tuples
    """
    H, W = G.shape
    component_masks: List[np.ndarray] = []
    invariants: List[CompInv] = []
    counts: Dict[int, int] = {}

    # Process each color
    for color in sorted(np.unique(G).tolist()):
        M = (G == color)
        if not M.any():
            continue

        # Label components for this color
        L = _label_cc4(M)
        num_comps = int(L.max())

        if num_comps == 0:
            continue

        counts[color] = num_comps

        # Extract each component
        for label_id in range(1, num_comps + 1):
            # Find bbox
            rows, cols = np.where(L == label_id)
            r0, r1 = int(rows.min()), int(rows.max()) + 1
            c0, c1 = int(cols.min()), int(cols.max()) + 1

            # Extract bbox mask with bounds check (WO-11G)
            bbox_slice = _safe_slice(L, r0, c0, r1, c1)
            bbox_mask = (bbox_slice == label_id)

            # Compute invariants
            area = int(bbox_mask.sum())
            bbox_h = r1 - r0
            bbox_w = c1 - c0
            perim4 = _perimeter4(bbox_mask)
            outline = _outline_hash(bbox_mask)

            inv = CompInv(
                color=color,
                area=area,
                bbox_h=bbox_h,
                bbox_w=bbox_w,
                perim4=perim4,
                outline_hash=outline,
                anchor_r=r0,
                anchor_c=c0,
            )

            invariants.append(inv)
            component_masks.append(bbox_mask)

    # Lex sort by invariant tuple
    # Key: (area, bbox_h, bbox_w, perim4, outline_hash, anchor_r, anchor_c, color)
    order = sorted(
        range(len(invariants)),
        key=lambda i: (
            invariants[i].area,
            invariants[i].bbox_h,
            invariants[i].bbox_w,
            invariants[i].perim4,
            invariants[i].outline_hash,
            invariants[i].anchor_r,
            invariants[i].anchor_c,
            invariants[i].color,
        )
    )

    # Reorder
    component_masks = [component_masks[i] for i in order]
    invariants = [invariants[i] for i in order]

    # Create receipt with serialized invariants
    rc = ComponentsRc(
        connectivity="4",
        per_color_counts=counts,
        invariants=[asdict(inv) for inv in invariants],
    )

    return component_masks, rc


def stable_match(
    left_invs: List[CompInv],
    right_invs: List[CompInv]
) -> Tuple[List[Tuple[int, int]], MatchRc]:
    """
    Stable matching by lex-sorted invariant tuple equality.

    Contract (02_determinism_addendum.md §3 lines 119-120):
    "Sort both lists by the same lex key; pair left-to-right by equality of the lex key;
    if unequal, mark as unmatched"

    Args:
        left_invs: left component invariants (lex sorted)
        right_invs: right component invariants (lex sorted)

    Returns:
        (pairs, MatchRc) where pairs = [(left_idx, right_idx), ...]
    """
    i = 0
    j = 0
    pairs: List[Tuple[int, int]] = []
    left_only: List[int] = []
    right_only: List[int] = []

    def inv_key(inv: CompInv) -> tuple:
        """Canonical invariant comparison key."""
        return (
            inv.area,
            inv.bbox_h,
            inv.bbox_w,
            inv.perim4,
            inv.outline_hash,
            inv.anchor_r,
            inv.anchor_c,
            inv.color,
        )

    while i < len(left_invs) and j < len(right_invs):
        left_key = inv_key(left_invs[i])
        right_key = inv_key(right_invs[j])

        if left_key == right_key:
            # Match
            pairs.append((i, j))
            i += 1
            j += 1
        elif left_key < right_key:
            # Left has extra component
            left_only.append(i)
            i += 1
        else:
            # Right has extra component
            right_only.append(j)
            j += 1

    # Remaining unmatched
    while i < len(left_invs):
        left_only.append(i)
        i += 1

    while j < len(right_invs):
        right_only.append(j)
        j += 1

    rc = MatchRc(
        pairs=pairs,
        left_only=left_only,
        right_only=right_only,
        verified_pixelwise=False,  # Will be set by caller if verification performed
    )

    return pairs, rc


def verify_pixelwise_equal(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_G: np.ndarray,
    right_G: np.ndarray,
    left_inv: CompInv,
    right_inv: CompInv,
) -> bool:
    """
    Verify pixelwise equality of matched component pair.

    Contract: Within aligned bboxes, verify mask shapes match and values equal
    on masked pixels.

    Args:
        left_mask: left component bbox mask
        right_mask: right component bbox mask
        left_G: left full grid
        right_G: right full grid
        left_inv: left component invariants
        right_inv: right component invariants

    Returns:
        True if masks identical in shape
    """
    # For WO-03, we verify mask shape equality
    # (Full pixelwise value verification happens in WO-04 witness solving)
    if left_mask.shape != right_mask.shape:
        return False

    # Verify masks are identical
    return bool(np.array_equal(left_mask, right_mask))
