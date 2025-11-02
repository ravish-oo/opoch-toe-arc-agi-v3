#!/usr/bin/env python3
# arc/op/tiebreak.py
# WO-08: Tie-break L (argmin over frozen cost tuple)
# Implements 02_determinism_addendum.md §6: fixed L-tuple lex ordering

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Tuple
from .hash import hash_bytes

TieContextKind = Literal["generic_placement", "skyline", "none"]

# Frozen pose rank (02_determinism_addendum.md line 202)
POSE_RANK = {"REF": 0, "ROT": 1, "TRANS": 2}


def lehmer_to_perm(lehmer: List[int]) -> List[int]:
    """
    Convert Lehmer code to permutation in one-line notation.

    Contract (WO-08 patch):
    Standard factorial-number-system reconstruction.
    Precondition: 0 <= lehmer[i] <= n-1-i for all i, n = len(lehmer).

    Algorithm:
    1. Start with pool = [0, 1, ..., n-1] (available images)
    2. For each position i, assign pool[lehmer[i]] and remove it
    3. Result: perm[i] = image of index i under permutation σ

    Fail-closed (02_determinism_addendum.md §0 line 14):
    If any digit out of range → return worst-case permutation (all moved)
    to make invalid inputs visible in receipts with maximal recolor_bits.

    Args:
        lehmer: Lehmer code (0-indexed)

    Returns:
        perm: One-line notation where perm[i] = σ(i)
    """
    if not lehmer:
        return []

    n = len(lehmer)

    # Validate digits (fail-closed)
    for i, d in enumerate(lehmer):
        if d < 0 or d > (n - 1 - i):
            # Invalid digit: return worst-case permutation (reverse order = all moved)
            return list(range(n - 1, -1, -1))

    # Reconstruct permutation
    pool = list(range(n))  # available images in ascending order
    perm = [0] * n

    for i, d in enumerate(lehmer):
        perm[i] = pool.pop(d)  # assign d-th smallest remaining to position i

    return perm


@dataclass
class Candidate:
    """
    Candidate law for tie-breaking.

    Contract (WO-08):
    All fields must be pre-computed by upstream (WO-04 witness or WO-10 engine).
    Encodings must be canonical (ZigZag LEB128, Lehmer, D4 ids).
    """
    phi_bytes: bytes                                      # piecewise φ encoding (ZigZag LEB128 framed), or b"" if engine law
    sigma_domain_colors: List[int]                        # active palette in σ domain
    sigma_lehmer: List[int]                               # σ Lehmer code
    anchor_displacements: List[Tuple[int, int]]           # [(dr, dc) per piece/coframe anchor]
    component_count_before: int                           # #components in X
    component_count_after: int                            # #components in X∘φ
    residue_list: List[Tuple[int, int, int, int]]         # [(r_per, r_res, c_per, c_res) per piece]
    pose_classes: List[Literal["REF", "ROT", "TRANS"]]    # per piece
    placement_refs: Optional[List[Tuple[int, int]]] = None  # anchor(s) or centroid(s) in Π-frame
    skyline_keys: Optional[Tuple[int, int, int]] = None     # (first_appearance_idx, no_overlap_flag, maxrect_rank)
    meta: Dict[str, Any] = field(default_factory=dict)      # extra audit fields (H, W, etc.)


@dataclass
class CandidateCost:
    """
    Per-candidate cost tuple.

    Contract (02_determinism_addendum.md §6 lines 194-210):
    Lex order over (1..7) with placement_keys only if tie_context != "none".
    """
    idx: int                                               # index in original candidate list
    l1_disp_anchors: int                                   # Σ|Δr|+|Δc| over component anchors
    param_len_bytes: int                                   # len(phi_bytes)
    recolor_bits: int                                      # #moved colors in σ domain
    object_breaks: int                                     # max(0, #comp_after - #comp_before)
    tie_code: int                                          # REF=0 < ROT=1 < TRANS=2
    residue_key: Tuple[Tuple[int, int, int, int], ...]     # lex-sorted residue tuples
    placement_keys: Optional[Tuple[int, int, int]]         # (center_L1, topmost, leftmost) or skyline
    cand_hash: str                                         # BLAKE3(phi_bytes + lehmer)


@dataclass
class TieBreakRc:
    """
    Tie-break receipt.

    Contract (02_determinism_addendum.md §10 line 247):
    "Tie-break: candidates with cost tuples; chosen_idx."
    """
    costs: List[Dict[str, Any]]     # serialized CandidateCost (for JSON)
    chosen_idx: int                 # index of lex-min candidate
    table_hash: str                 # BLAKE3 of serialized cost table
    tie_context: TieContextKind     # which C1 chain was used


def _sum_l1(displacements: List[Tuple[int, int]]) -> int:
    """
    Compute L1 displacement sum: Σ|Δr|+|Δc|.

    Contract (02_determinism_addendum.md line 198):
    "L1_disp_anchors(φ): sum of |Δr|+|Δc| over component anchors only"
    """
    return sum(abs(dr) + abs(dc) for dr, dc in displacements)


def _residue_key(residues: List[Tuple[int, int, int, int]]) -> Tuple[Tuple[int, int, int, int], ...]:
    """
    Compute residue key: lex-sorted residue tuples.

    Contract (02_determinism_addendum.md line 209):
    "Residue preference: prefer smaller residues (r mod per_r, c mod per_c) by lex"
    """
    return tuple(sorted(residues))


def _placement_keys_generic(
    placements: List[Tuple[int, int]],
    H: int,
    W: int
) -> Tuple[int, int, int]:
    """
    Compute C1 placement keys: (center_L1, topmost_row, leftmost_col).

    Contract (WO-08):
    1. center_L1: L1 distance to grid center (H-1)/2, (W-1)/2
    2. topmost_row: min row index
    3. leftmost_col: min col index

    If no placements, return (10^9, 10^9, 10^9) for lex ordering.
    """
    if not placements:
        return (10**9, 10**9, 10**9)

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    # Center L1: compute for all placements, take min
    dists = sorted(int(abs(r - cy) + abs(c - cx)) for r, c in placements)
    center_l1 = dists[0]

    # Topmost and leftmost
    topmost = min(r for r, _ in placements)
    leftmost = min(c for _, c in placements)

    return (center_l1, topmost, leftmost)


def _placement_keys_skyline(skyline_keys: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Pass through skyline keys (already encoded).

    Contract (WO-08):
    Skyline keys: (first_appearance_idx, no_overlap_flag, maxrect_rank)
    """
    return skyline_keys


def resolve(
    cands: List[Candidate],
    *,
    tie_context: TieContextKind = "none"
) -> Tuple[int, TieBreakRc]:
    """
    Compute frozen L-tuple for each candidate and return (chosen_idx, receipt).

    Contract (02_determinism_addendum.md §6):
    Cost tuple: (L1_disp, param_len, recolor_bits, object_breaks, tie_code, residue_key, placement_keys?)
    Lex-min over tuple; placement_keys only if tie_context != "none".

    Algorithm:
    1. For each candidate, compute 7-component cost tuple
    2. Sort by lex order (include placement_keys only if tie_context != "none")
    3. Select candidate with lex-min cost
    4. Compute table_hash over serialized costs
    5. Return (chosen_idx, TieBreakRc)

    Args:
        cands: List of Candidate objects with all fields populated
        tie_context: "none" | "generic_placement" | "skyline"

    Returns:
        (chosen_idx, TieBreakRc): chosen_idx is index in original cands list
    """
    if not cands:
        raise ValueError("resolve: empty candidate list")

    # Extract H, W from first candidate's meta (if placement keys needed)
    H = cands[0].meta.get("H")
    W = cands[0].meta.get("W")

    costs: List[CandidateCost] = []

    for idx, c in enumerate(cands):
        # 1. L1_disp_anchors
        l1_disp = _sum_l1(c.anchor_displacements)

        # 2. param_len_bytes
        param_len = len(c.phi_bytes)

        # 3. recolor_bits (moved colors count)
        # Contract (02_determinism_addendum.md §1.4 line 114):
        # "recolor_bits(σ) = number of *moved* colors in domain (k)"
        # Proper support size: reconstruct σ from Lehmer code, count i where σ(i) ≠ i
        perm = lehmer_to_perm(c.sigma_lehmer or [])
        moved = sum(1 for i, p in enumerate(perm) if p != i)

        # 4. object_breaks (only increases)
        obj_breaks = max(0, c.component_count_after - c.component_count_before)

        # 5. tie_code (max pose class across pieces)
        if c.pose_classes:
            tie_code_val = max(POSE_RANK[p] for p in c.pose_classes)
        else:
            tie_code_val = POSE_RANK["REF"]  # default if no pieces

        # 6. residue_key (lex-sorted residue tuples)
        residue_k = _residue_key(c.residue_list or [])

        # 7. placement_keys (C1 chain, only if tie_context != "none")
        placement_k: Optional[Tuple[int, int, int]] = None
        if tie_context == "generic_placement":
            if H is None or W is None:
                raise ValueError("resolve: H,W required in meta for generic_placement")
            placement_k = _placement_keys_generic(c.placement_refs or [], H, W)
        elif tie_context == "skyline":
            if c.skyline_keys is None:
                placement_k = (10**9, 10**9, 10**9)  # fallback
            else:
                placement_k = _placement_keys_skyline(c.skyline_keys)

        # Compute candidate hash
        cand_hash = hash_bytes((c.phi_bytes or b"") + bytes(c.sigma_lehmer))

        cost = CandidateCost(
            idx=idx,
            l1_disp_anchors=l1_disp,
            param_len_bytes=param_len,
            recolor_bits=moved,
            object_breaks=obj_breaks,
            tie_code=tie_code_val,
            residue_key=residue_k,
            placement_keys=placement_k,
            cand_hash=cand_hash,
        )
        costs.append(cost)

    # Lex order key function
    def lex_key(cc: CandidateCost):
        base = (
            cc.l1_disp_anchors,
            cc.param_len_bytes,
            cc.recolor_bits,
            cc.object_breaks,
            cc.tie_code,
            cc.residue_key,
        )
        if tie_context != "none":
            return base + (cc.placement_keys,)
        else:
            return base

    # Sort costs by lex key
    costs_sorted = sorted(costs, key=lex_key)

    # Choose lex-min (first in sorted list)
    chosen_idx = costs_sorted[0].idx

    # Compute table hash
    table_lines = []
    for cc in costs_sorted:
        # Serialize cost tuple for hashing
        key_str = f"{cc.idx}:{lex_key(cc)}"
        table_lines.append(key_str)
    table_str = "|".join(table_lines)
    table_hash = hash_bytes(table_str.encode())

    # Build receipt (serialize costs as dicts for JSON)
    costs_serialized = []
    for cc in costs_sorted:
        costs_serialized.append({
            "idx": cc.idx,
            "l1_disp_anchors": cc.l1_disp_anchors,
            "param_len_bytes": cc.param_len_bytes,
            "recolor_bits": cc.recolor_bits,
            "object_breaks": cc.object_breaks,
            "tie_code": cc.tie_code,
            "residue_key": cc.residue_key,
            "placement_keys": cc.placement_keys,
            "cand_hash": cc.cand_hash,
        })

    receipt = TieBreakRc(
        costs=costs_serialized,
        chosen_idx=chosen_idx,
        table_hash=table_hash,
        tie_context=tie_context,
    )

    return chosen_idx, receipt
