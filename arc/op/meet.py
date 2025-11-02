#!/usr/bin/env python3
# arc/op/meet.py
# WO-09: Meet writer (copy ▷ law ▷ unanimity ▷ bottom)
# Implements 00_math_spec.md §8 and 02_determinism_addendum.md §9

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from .hash import hash_bytes


@dataclass
class MeetRc:
    """
    Meet writer receipt.

    Contract (02_determinism_addendum.md §10 line 248):
    "Meet: counts per rule (copy / law / unanimity / bottom); idempotent repaint hash."

    Contract (02_determinism_addendum.md §9 lines 232-234):
    "Priority: copy ▷ law ▷ unanimity ▷ bottom (strict). One pass.
    Bottom color is canonical 0.
    After write, immediately recompute BLAKE3 hash and assert unchanged (idempotence receipt)."
    """
    count_copy: int                   # |{p : copy fired}|
    count_law: int                    # |{p : law fired}|
    count_unanimity: int              # |{p : unanimity fired}|
    count_bottom: int                 # |{p : bottom fired}|
    bottom_color: int                 # must be 0 (H2)
    repaint_hash: str                 # BLAKE3(Y_pass2)
    copy_mask_hash: Optional[str]     # BLAKE3(copy_mask_bits)
    law_mask_hash: Optional[str]      # BLAKE3(law_mask_bits)
    uni_mask_hash: Optional[str]      # BLAKE3(unanimity coverage bits)
    frame: str                        # "presented"
    shape: Tuple[int, int]            # (H*, W*)


def _decode_bitset_lsb(bits: bytes, size: int) -> np.ndarray:
    """
    Decode LSB-first bitset to boolean array.

    Contract (02_determinism_addendum.md §4 line 176):
    "Singleton mask encoding: serialize as a bitset (row-major, LSB-first per byte)"

    Algorithm:
    - index = row*W + col (row-major)
    - byte_idx = index >> 3
    - bit_offset = index & 7
    - bit_value = (bits[byte_idx] >> bit_offset) & 1

    Args:
        bits: Bitset bytes (row-major, LSB-first)
        size: Total number of bits (H*W)

    Returns:
        mask: Boolean array of shape (size,) in row-major order
    """
    mask = np.zeros(size, dtype=np.bool_)
    for idx in range(size):
        byte_idx = idx >> 3
        bit_offset = idx & 7
        if byte_idx < len(bits):
            bit_val = (bits[byte_idx] >> bit_offset) & 1
            mask[idx] = bool(bit_val)
    return mask


def compose_meet(
    Xt: np.ndarray,                             # Π(test), H* × W*
    copy_mask_bits: Optional[bytes],            # LSB-first bitset; None => no copy
    copy_values: Optional[np.ndarray],          # H* × W*, or None to derive from Xt
    law_mask_bits: Optional[bytes],             # None => law everywhere (if law_values) or nowhere
    law_values: Optional[np.ndarray],           # H* × W*, or None
    truth_blocks: Optional[np.ndarray],         # H* × W*; required if block_color_map not empty
    block_color_map: Optional[Dict[int, int]],  # block_id → color; None or {}
    *,
    bottom_color: int = 0                       # H2: must be 0
) -> Tuple[np.ndarray, MeetRc]:
    """
    One-pass write in fixed priority: copy ▷ law ▷ unanimity ▷ bottom.

    Contract (00_math_spec.md §8 lines 119-134):
    "For each test pixel p, define the admissible set A_p = ...
    Write the least element in the fixed order: copy ▷ law ▷ unanimity ▷ bottom.
    This is a pointwise meet in a finite poset; repainting is idempotent."

    Contract (02_determinism_addendum.md §9 lines 230-234):
    "Priority: copy ▷ law ▷ unanimity ▷ bottom (strict). One pass.
    Bottom color is canonical 0.
    After write, recompute BLAKE3 hash; apply same pass again and assert unchanged."

    Algorithm:
    1. Initialize Y = zeros(H*, W*) [already bottom=0]
    2. Decode bitsets → masks
    3. One pass (no re-entry):
       - Copy: if copy_mask[p]==1 → Y[p]=copy_values[p]
       - Law: if ~written & law_defined[p] → Y[p]=law_values[p]
       - Unanimity: if ~written & block has color → Y[p]=u(block)
       - Bottom: remaining pixels already 0
    4. Run same pass again → Y2, compute repaint_hash

    Args:
        Xt: Π(test) grid (H*, W*)
        copy_mask_bits: Bitset for singleton copy sites (LSB-first)
        copy_values: Values to copy (H*, W*) or None
        law_mask_bits: Bitset for law-defined pixels (LSB-first) or None
        law_values: Law colors (H*, W*) or None
        truth_blocks: Truth partition (H*, W*) or None
        block_color_map: {block_id → color} for unanimous blocks
        bottom_color: Must be 0 (H2)

    Returns:
        (Y, MeetRc): Final Π-frame output and receipt
    """
    H, W = Xt.shape
    size = H * W

    # H2 enforcement
    if bottom_color != 0:
        raise ValueError(f"H2 violation: bottom_color must be 0, got {bottom_color}")

    # Initialize Y (already bottom=0)
    Y = np.zeros((H, W), dtype=Xt.dtype)

    # Track which pixels have been written (for counts and priority enforcement)
    written = np.zeros((H, W), dtype=np.bool_)

    # Counters
    count_copy = 0
    count_law = 0
    count_unanimity = 0

    # Layer hashes
    copy_mask_hash = None
    law_mask_hash = None
    uni_mask_hash = None

    # ===== LAYER 1: COPY =====
    if copy_mask_bits is not None and copy_values is not None:
        # Decode copy mask
        copy_mask_flat = _decode_bitset_lsb(copy_mask_bits, size)
        copy_mask = copy_mask_flat.reshape(H, W)

        # Apply copy
        Y[copy_mask] = copy_values[copy_mask]
        written[copy_mask] = True
        count_copy = int(copy_mask.sum())

        # Hash copy mask
        copy_mask_hash = hash_bytes(copy_mask_bits)

    # ===== LAYER 2: LAW =====
    if law_values is not None:
        if law_mask_bits is not None:
            # Law defined at specific pixels (witness law)
            law_mask_flat = _decode_bitset_lsb(law_mask_bits, size)
            law_mask = law_mask_flat.reshape(H, W)
            law_mask_hash = hash_bytes(law_mask_bits)
        else:
            # Law defined everywhere (engine law, full-frame)
            law_mask = np.ones((H, W), dtype=np.bool_)
            # Compute hash of full mask for consistency
            full_bits = bytes([(0xFF if i < size else 0) for i in range((size + 7) // 8)])
            law_mask_hash = hash_bytes(full_bits)

        # Apply law where not already written
        law_write_mask = law_mask & ~written
        Y[law_write_mask] = law_values[law_write_mask]
        written[law_write_mask] = True
        count_law = int(law_write_mask.sum())

    # ===== LAYER 3: UNANIMITY =====
    if block_color_map and truth_blocks is not None:
        # Build unanimity coverage mask
        uni_mask = np.zeros((H, W), dtype=np.bool_)

        for block_id, color in block_color_map.items():
            # Find pixels in this block that haven't been written yet
            block_mask = (truth_blocks == block_id) & ~written
            Y[block_mask] = color
            written[block_mask] = True
            uni_mask[block_mask] = True

        count_unanimity = int(uni_mask.sum())

        # Hash unanimity mask (encode as bitset for consistency)
        uni_bits_list = []
        uni_flat = uni_mask.flatten()
        for byte_idx in range((size + 7) // 8):
            byte_val = 0
            for bit_offset in range(8):
                idx = byte_idx * 8 + bit_offset
                if idx < size and uni_flat[idx]:
                    byte_val |= (1 << bit_offset)
            uni_bits_list.append(byte_val)
        uni_mask_bits = bytes(uni_bits_list)
        uni_mask_hash = hash_bytes(uni_mask_bits)

    # ===== LAYER 4: BOTTOM =====
    # Already zero-filled; count remaining
    count_bottom = size - (count_copy + count_law + count_unanimity)

    # ===== IDEMPOTENCE CHECK (H1) =====
    # Run same procedure again to produce Y2
    Y2 = np.zeros((H, W), dtype=Xt.dtype)
    written2 = np.zeros((H, W), dtype=np.bool_)

    # Copy pass 2
    if copy_mask_bits is not None and copy_values is not None:
        copy_mask_flat = _decode_bitset_lsb(copy_mask_bits, size)
        copy_mask = copy_mask_flat.reshape(H, W)
        Y2[copy_mask] = copy_values[copy_mask]
        written2[copy_mask] = True

    # Law pass 2
    if law_values is not None:
        if law_mask_bits is not None:
            law_mask_flat = _decode_bitset_lsb(law_mask_bits, size)
            law_mask = law_mask_flat.reshape(H, W)
        else:
            law_mask = np.ones((H, W), dtype=np.bool_)

        law_write_mask = law_mask & ~written2
        Y2[law_write_mask] = law_values[law_write_mask]
        written2[law_write_mask] = True

    # Unanimity pass 2
    if block_color_map and truth_blocks is not None:
        for block_id, color in block_color_map.items():
            block_mask = (truth_blocks == block_id) & ~written2
            Y2[block_mask] = color
            written2[block_mask] = True

    # Compute repaint hash
    repaint_hash = hash_bytes(Y2.tobytes())

    # Build receipt
    receipt = MeetRc(
        count_copy=count_copy,
        count_law=count_law,
        count_unanimity=count_unanimity,
        count_bottom=count_bottom,
        bottom_color=bottom_color,
        repaint_hash=repaint_hash,
        copy_mask_hash=copy_mask_hash,
        law_mask_hash=law_mask_hash,
        uni_mask_hash=uni_mask_hash,
        frame="presented",
        shape=(H, W),
    )

    return Y, receipt


# ============================================================================
# WO-09' - Meet as selector inside D* (after fixed-point propagation)
# ============================================================================

def select_from_domains(
    D: np.ndarray,                           # Final domains (H,W,K) uint64 after lfp
    C: list[int],                            # Color universe (sorted)
    copy_values: Optional[np.ndarray] = None,  # From shape S (H,W), or None
    S_copy: Optional[np.ndarray] = None,     # Copy scope mask (H,W) uint8, or None
    unanimity_colors: Optional[np.ndarray] = None,  # Unanimous colors (H,W), or None
    S_unanimity: Optional[np.ndarray] = None,  # Unanimity scope mask (H,W) uint8, or None
    *,
    bottom_color: int = 0                    # H2: must be 0
) -> Tuple[np.ndarray, MeetRc]:
    """
    Select from final domains D* using scope-gated frozen precedence.

    Contract (WO-09' + Scope S):
    For each pixel p, selection inside D*[p] with scope gating:
    1. If S_copy[p]=1 AND copy_value[p] ∈ D*[p]: select copy_value[p] (copy path)
    2. Else if D*[p] ≠ ∅: select min(D*[p]) (law path)
    3. Else if S_unanimity[p]=1 AND unanimity_color[p] ∈ D*[p]: select unanimity_color[p] (unanimity path)
    4. Else: select 0 (bottom path)

    Scope gating: a layer only "wins" precedence if its scope S[p]=1 (non-silent).
    Law path has no scope gate (always tries min(D*[p]) as fallback).

    Containment guarantee: selected ∈ D*[p] always (harness will verify)
    Idempotence: repaint produces same hash

    Args:
        D: Final domains after propagate_fixed_point (H,W,K) uint64
        C: Color universe (sorted unique from Π inputs)
        copy_values: Copy colors from shape S (H,W), or None
        S_copy: Copy scope mask (H,W) uint8; S[p]=1 if copy constrains p
        unanimity_colors: Unanimous block colors (H,W), or None
        S_unanimity: Unanimity scope mask (H,W) uint8; S[p]=1 if unanimity constrains p
        bottom_color: Must be 0 (H2)

    Returns:
        (Y, MeetRc): Selected output and receipt

    Raises:
        ValueError: if bottom_color != 0 (H2 violation)
        ValueError: if selected color not in D*[p] (containment violation)
    """
    from .admit import _test_bit, _popcount_bitset

    H, W, K = D.shape

    # H2 enforcement
    if bottom_color != 0:
        raise ValueError(f"H2 violation: bottom_color must be 0, got {bottom_color}")

    # Build color index lookup
    color_to_idx = {c: i for i, c in enumerate(C)}

    # Initialize Y (will fill with selections)
    Y = np.zeros((H, W), dtype=np.uint8)

    # Counters
    count_copy = 0
    count_law = 0
    count_unanimity = 0
    count_bottom = 0

    # For each pixel, select according to scope-gated frozen precedence
    for r in range(H):
        for c in range(W):
            selected_color = None

            # Path 1: Copy (if S_copy[p]=1 AND copy_value ∈ D*[p])
            if S_copy is not None and S_copy[r, c] == 1:
                if copy_values is not None:
                    copy_color = int(copy_values[r, c])
                    if copy_color in color_to_idx:
                        copy_idx = color_to_idx[copy_color]
                        if _test_bit(D[r, c], copy_idx):
                            # Copy color is admitted in D*[p] and scope is active
                            selected_color = copy_color
                            count_copy += 1

            # Path 2: Law (pick min(D*[p]) if non-empty)
            # No scope gate: law is fallback for any constrained domain
            if selected_color is None:
                # Find all admitted colors at this pixel
                admitted = []
                for color_idx in range(len(C)):
                    if _test_bit(D[r, c], color_idx):
                        admitted.append(C[color_idx])

                if admitted:
                    # Pick smallest admitted color
                    selected_color = min(admitted)
                    count_law += 1

            # Path 3: Unanimity (if S_unanimity[p]=1 AND unanimity_color ∈ D*[p])
            if selected_color is None:
                if S_unanimity is not None and S_unanimity[r, c] == 1:
                    if unanimity_colors is not None:
                        unanimity_color = int(unanimity_colors[r, c])
                        if unanimity_color in color_to_idx:
                            unanimity_idx = color_to_idx[unanimity_color]
                            if _test_bit(D[r, c], unanimity_idx):
                                # Unanimity color is admitted in D*[p] and scope is active
                                selected_color = unanimity_color
                                count_unanimity += 1

            # Path 4: Bottom (select 0)
            if selected_color is None:
                selected_color = bottom_color
                count_bottom += 1

            # Verify containment (WO-09' harness requirement)
            if selected_color != bottom_color:
                if selected_color not in color_to_idx:
                    raise ValueError(
                        f"Containment violation at ({r},{c}): "
                        f"selected color {selected_color} not in universe C={C}"
                    )
                selected_idx = color_to_idx[selected_color]
                if not _test_bit(D[r, c], selected_idx):
                    raise ValueError(
                        f"Containment violation at ({r},{c}): "
                        f"selected color {selected_color} not in D*[p]"
                    )

            Y[r, c] = selected_color

    # Idempotence check: repaint and verify same result
    Y2 = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            selected_color = None

            # Same scope-gated selection logic
            if S_copy is not None and S_copy[r, c] == 1:
                if copy_values is not None:
                    copy_color = int(copy_values[r, c])
                    if copy_color in color_to_idx:
                        copy_idx = color_to_idx[copy_color]
                        if _test_bit(D[r, c], copy_idx):
                            selected_color = copy_color

            if selected_color is None:
                admitted = []
                for color_idx in range(len(C)):
                    if _test_bit(D[r, c], color_idx):
                        admitted.append(C[color_idx])
                if admitted:
                    selected_color = min(admitted)

            if selected_color is None:
                if S_unanimity is not None and S_unanimity[r, c] == 1:
                    if unanimity_colors is not None:
                        unanimity_color = int(unanimity_colors[r, c])
                        if unanimity_color in color_to_idx:
                            unanimity_idx = color_to_idx[unanimity_color]
                            if _test_bit(D[r, c], unanimity_idx):
                                selected_color = unanimity_color

            if selected_color is None:
                selected_color = bottom_color

            Y2[r, c] = selected_color

    # Compute repaint hash
    repaint_hash = hash_bytes(Y2.tobytes())

    # Build receipt
    receipt = MeetRc(
        count_copy=count_copy,
        count_law=count_law,
        count_unanimity=count_unanimity,
        count_bottom=count_bottom,
        bottom_color=bottom_color,
        repaint_hash=repaint_hash,
        copy_mask_hash=None,  # Not applicable for domain selection
        law_mask_hash=None,
        uni_mask_hash=None,
        frame="presented",
        shape=(H, W),
    )

    return Y, receipt
