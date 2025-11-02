#!/usr/bin/env python3
# arc/op/copy.py
# WO-06: Free copy S(p) (BLOCKER)
# Implements 00_math_spec.md §5: S(p) = ⋂_i {φ_i^*(p)}

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from .hash import hash_bytes
from .witness import PhiRc, PhiPiece, _apply_d4_pose
from .receipts import CopyRc


def _eval_phi_star_at_pixel(
    p: Tuple[int, int],
    phi_star: PhiRc,
    comp_masks: List[Tuple[np.ndarray, int, int]],  # (mask, r0, c0) per component
) -> Optional[Tuple[int, int]]:
    """
    Evaluate φ_i^*(p) → s or None if undefined.

    Contract (00_math_spec.md §5):
    φ_i^* is piecewise by components; within each component bbox,
    φ_i^*(p) applies (pose, dr, dc, residue).

    Args:
        p: pixel (r, c) in test Π frame
        phi_star: conjugated φ_i^* for training i
        comp_masks: list of (mask, r0, c0) for each component in test grid
                    mask is bbox-local boolean array
                    (r0, c0) is top-left anchor of bbox

    Returns:
        (s_r, s_c): source pixel in test grid, or None if p not in any component
    """
    r, c = p

    # Check each piece (component)
    for piece_idx, piece in enumerate(phi_star.pieces):
        if piece_idx >= len(comp_masks):
            # Component doesn't exist in test grid (shouldn't happen if properly conjugated)
            continue

        mask, r0, c0 = comp_masks[piece_idx]
        bbox_h, bbox_w = mask.shape

        # Check if p is in this component's bbox
        if not (r0 <= r < r0 + bbox_h and c0 <= c < c0 + bbox_w):
            continue

        # Convert to bbox-local coordinates
        r_local = r - r0
        c_local = c - c0

        # Check if p is actually in the component (not just bbox)
        if not mask[r_local, c_local]:
            continue

        # Apply inverse φ: given p in destination, find source s
        # φ maps source → destination via (pose, dr, dc)
        # So φ^(-1) maps destination → source via (pose^(-1), -dr, -dc)

        # For piecewise geometric φ:
        # If φ(s) = pose(s) + (dr, dc) = p
        # Then s = pose^(-1)(p - (dr, dc))

        # Note: In the witness solver, φ was defined as X_i → Y_i
        # Here φ_i^* is conjugated to map test → test
        # For free copy, we want: given test output pixel p, find test input pixel s

        # Actually, let me reconsider the math:
        # In WO-04, φ_i maps X_i → Y_i (input → output)
        # φ_i^* is conjugated to test frame
        # For free copy S(p), we want: given output pixel p, where does it copy from?
        # This is the INVERSE: φ_i^*(-1)(p) → s

        # But wait, the spec says S(p) = ⋂_i {φ_i^*(p)}
        # This means φ_i^* maps test output pixel p → test input pixel s

        # So φ_i^* is already the map we need (output → input)

        # The PhiPiece encodes how output relates to input:
        # output_bbox is transformed by pose, then translated by (dr, dc)
        # To invert: source = pose^(-1)(dest - (dr, dc))

        # Actually, let me check the witness.py encoding more carefully.
        # The PhiPiece stores (pose_id, dr, dc) for the transformation.
        # Looking at witness solver: it finds (pose, dr, dc) such that
        # Y_bbox = pose(X_bbox) when translated by (dr, dc)

        # So: Y[r', c'] = X[pose^(-1)(r' - dr, c' - dc)]
        # For free copy: we want the source pixel s given destination p
        # s = pose^(-1)(p - (dr, dc))

        # Wait, I need to be more careful about the bbox framing.
        # The (dr, dc) is in bbox-local coordinates.

        # Let me think step by step:
        # 1. p = (r, c) is in test grid (global coordinates)
        # 2. Convert to bbox-local: p_local = (r - r0, c - c0)
        # 3. Apply inverse transform: s_local = pose^(-1)(p_local - (dr, dc))
        # 4. Convert back to global: s = (s_local[0] + r0, s_local[1] + c0)

        # Subtract translation
        r_shifted = r_local - piece.dr
        c_shifted = c_local - piece.dc

        # Apply inverse pose
        # For D4 group, inverse of each pose:
        # 0 (identity) → 0
        # 1 (rot90) → 3 (rot270)
        # 2 (rot180) → 2
        # 3 (rot270) → 1 (rot90)
        # 4 (flipH) → 4
        # 5 (flipH∘rot90) → 5 (self-inverse)
        # 6 (flipH∘rot180) → 6 (self-inverse)
        # 7 (flipH∘rot270) → 7 (self-inverse)

        # Actually, let me use the inverse pose directly by applying the pose transform
        # to a grid with a single 1 at the shifted position, then finding where it maps

        # Simpler approach: create a coordinate grid and apply pose
        H_bbox, W_bbox = bbox_h, bbox_w

        # Check bounds after shift
        if not (0 <= r_shifted < H_bbox and 0 <= c_shifted < W_bbox):
            continue  # Out of bounds after translation

        # Create point at shifted location
        # Apply inverse pose to find source

        # Inverse pose mappings:
        inv_pose = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
        inv_pose_id = inv_pose[piece.pose_id]

        # Apply inverse pose to single point
        # For this, we need to think of the transformation in reverse
        # If pose transforms (r_src, c_src) → (r_dst, c_dst)
        # Then we want (r_shifted, c_shifted) → (r_src, c_src)

        # Let me use a different approach: create indicator grid and transform
        indicator = np.zeros((H_bbox, W_bbox), dtype=bool)
        indicator[r_shifted, c_shifted] = True

        # Apply inverse pose
        from .witness import _apply_d4_pose
        transformed = _apply_d4_pose(indicator, inv_pose_id)

        # Find where the 1 is now
        coords = np.argwhere(transformed)
        if len(coords) != 1:
            # Shouldn't happen
            continue

        r_src_local, c_src_local = coords[0]

        # Handle residues (if periodic tiling)
        if piece.r_per > 1 or piece.c_per > 1:
            # Apply residue class constraint
            if piece.r_per > 1:
                # Source must satisfy: r_src_local ≡ r_res (mod r_per)
                if r_src_local % piece.r_per != piece.r_res:
                    continue
            if piece.c_per > 1:
                if c_src_local % piece.c_per != piece.c_res:
                    continue

        # Convert to global coordinates
        s_r = r_src_local + r0
        s_c = c_src_local + c0

        return (s_r, s_c)

    # p not in any component's domain
    return None


def build_free_copy_mask(
    test_grid: np.ndarray,  # Π-presented X_*
    phi_stars: List[Optional[PhiRc]],  # conjugated per-train φ_i^*, or None if summary
    comp_masks: List[Tuple[np.ndarray, int, int]],  # (mask, r0, c0) per component in test
) -> Tuple[bytes, Optional[np.ndarray], CopyRc]:
    """
    Compute S(p) = ⋂_i {φ_i^*(p)}.

    Contract (00_math_spec.md §5 lines 82-88):
    "For each test pixel p, define the free candidate source set by intersection
    of transported witnesses: S(p) = ⋂_i {φ_i^*(p)}. If |S(p)|=1, p is a free copy site."

    Contract (02_determinism_addendum.md §4 lines 173-176):
    "Each conjugated φ_i^* is a partial function.
    Define S(p) = ⋂_i dom_images(φ_i^*, p) where dom_images is {φ_i^*(p)} if defined, else ∅.
    If *any* i is undefined at p, intersection is ∅. **Never** copy by majority or union."

    Args:
        test_grid: Π-presented test input X_* (H×W)
        phi_stars: list of conjugated φ_i^* (one per training), None if summary
        comp_masks: list of (mask, r0, c0) for each component in test grid

    Returns:
        (mask_bitset, copy_values, CopyRc):
        - mask_bitset: bytes (row-major, LSB-first) marking singleton pixels
        - copy_values: np.ndarray (H×W) with X_*[S(p)] for singleton pixels, 0 elsewhere
        - CopyRc: receipt with counts and hash
    """
    H, W = test_grid.shape
    m = len(phi_stars)  # number of trainings

    # Allocate bitset mask (row-major, LSB-first)
    n_pixels = H * W
    n_bytes = (n_pixels + 7) // 8
    mask_bits = bytearray(n_bytes)

    # Allocate copy values (optional, for WO-09)
    copy_values = np.zeros((H, W), dtype=np.int64)

    # Diagnostic counters
    singleton_count = 0
    undefined_count = 0
    disagree_count = 0
    multi_hit_count = 0

    # Process each pixel
    for r in range(H):
        for c in range(W):
            p = (r, c)
            idx = r * W + c

            # Collect images from each training
            images = []
            has_undefined = False

            for i, phi_star in enumerate(phi_stars):
                if phi_star is None:
                    # Summary witness: φ_i^* is undefined everywhere
                    has_undefined = True
                    break

                # Evaluate φ_i^*(p)
                s = _eval_phi_star_at_pixel(p, phi_star, comp_masks)

                if s is None:
                    # p ∉ dom(φ_i^*)
                    has_undefined = True
                    break

                images.append(s)

            # Check if undefined for any training
            if has_undefined:
                undefined_count += 1
                continue  # S(p) = ∅

            # Check if all images agree (intersection is singleton)
            if len(images) < m:
                # Shouldn't happen (logic error)
                continue

            # Check if all equal
            s_first = images[0]
            all_equal = all(s == s_first for s in images)

            if not all_equal:
                disagree_count += 1
                continue  # S(p) = ∅

            # Singleton! Set bit and copy value
            mask_bits[idx >> 3] |= (1 << (idx & 7))  # LSB-first
            singleton_count += 1

            # Copy value from source
            s_r, s_c = s_first
            if 0 <= s_r < H and 0 <= s_c < W:
                copy_values[r, c] = test_grid[s_r, s_c]
            # else: source out of bounds (shouldn't happen if φ properly defined)

    # Compute mask hash
    mask_hash = hash_bytes(bytes(mask_bits))

    receipt = CopyRc(
        singleton_count=singleton_count,
        singleton_mask_hash=mask_hash,
        undefined_count=undefined_count,
        disagree_count=disagree_count,
        multi_hit_count=multi_hit_count,
        H=H,
        W=W,
    )

    return bytes(mask_bits), copy_values, receipt
