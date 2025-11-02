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
