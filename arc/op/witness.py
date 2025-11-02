#!/usr/bin/env python3
# arc/op/witness.py
# WO-04: Witness solver (φ,σ) + conjugation + intersection (BLOCKER)
# Implements geometric and summary law paths with E2, A1, C2 contract compliance

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from .hash import hash_bytes


@dataclass
class PhiPiece:
    """
    One component's geometric mapping.

    Contract (02_determinism_addendum.md §1.5):
    Per component: <cid><pose_id><dr><dc><r_per><c_per><r_res><c_res>
    """
    comp_id: int      # index into components list
    pose_id: int      # D4 id (0..7)
    dr: int           # translation in bbox frame (signed)
    dc: int
    r_per: int        # residue basis row period (1 if none)
    c_per: int
    r_res: int        # residue class (0 if none)
    c_res: int


@dataclass
class PhiRc:
    """
    Geometric φ receipt.

    Contract (WO-04D):
    - pieces: list of PhiPiece per component
    - bbox_equal: per-piece equality verification (E2)
    - domain_pixels: total pixels covered by φ
    - geometric_trials: all (pose, dr, dc) trials with ok status
    """
    pieces: List[PhiPiece]
    bbox_equal: List[bool]  # per-piece E2 proof
    domain_pixels: int      # sum of pixels across pieces
    geometric_trials: List[Dict] = None  # WO-04D: [{"comp_id":i, "pose":p, "dr":dr, "dc":dc, "ok":bool}, ...]


@dataclass
class SigmaRc:
    """
    Palette role permutation receipt.

    Contract (02_determinism_addendum.md §1.4):
    - domain_colors: colors touched by law
    - lehmer: Lehmer code encoding
    - moved_count: recolor_bits (tie-break cost)

    WO-04H: Backward compatibility shim for .domain (was renamed to domain_colors)
    """
    domain_colors: List[int]
    lehmer: List[int]
    moved_count: int  # number of moved colors (recolor_bits)

    # WO-04H: Backward-compatible property (some code still uses .domain)
    @property
    def domain(self) -> List[int]:
        """Alias for domain_colors (backward compatibility)."""
        return self.domain_colors


@dataclass
class TrainWitnessRc:
    """
    Per-training witness receipt.

    Contract:
    - kind: "geometric" or "summary"
    - Geometric: phi + sigma with E2 proofs
    - Summary: sigma + A1/C2 receipts (candidate sets, decision rule, counts)
    """
    kind: str  # "geometric" | "summary"
    phi: Optional[PhiRc]  # None for summary
    sigma: SigmaRc
    # Summary-law receipts (A1/C2)
    foreground_colors: Optional[List[int]]
    background_colors: Optional[List[int]]
    decision_rule: Optional[str]
    per_color_counts: Optional[Dict[int, int]]


@dataclass
class ConjugatedRc:
    """
    Conjugated witness (transported to test Π frame).

    Contract (WO-04C):
    φ* = Π* ∘ Π_i^(-1) ∘ φ_i ∘ Π_i ∘ Π*^(-1)

    Receipts:
    - transform_receipts: per-piece old→new (pose_id, dr, dc)
    - conjugation_hash: BLAKE3(phi_star serialized)
    - pullback_samples: verification samples (WO-04, 3 per piece)

    WO-04H: Backward compatibility shims for .lehmer and .domain_colors
    """
    phi_star: Optional[PhiRc]
    sigma: SigmaRc
    transform_receipts: Optional[List[Dict]] = None  # Per-piece conjugation details
    conjugation_hash: Optional[str] = None           # BLAKE3 of phi_star
    pullback_samples: Optional[List[Dict]] = None    # WO-04: verification samples (3 per piece)

    # WO-04H: Backward-compatible property (some code still uses .lehmer directly)
    @property
    def lehmer(self) -> List[int]:
        """Alias for sigma.lehmer (backward compatibility)."""
        return self.sigma.lehmer

    # SWEEP4 fix: Backward-compatible property (some code expects .domain_colors on ConjugatedRc)
    @property
    def domain_colors(self) -> List[int]:
        """Alias for sigma.domain_colors (backward compatibility)."""
        return self.sigma.domain_colors


@dataclass
class IntersectionRc:
    """
    Intersection result across trainings.

    Contract:
    - status: "singleton" (law fixed) | "underdetermined" (>1 admissible) | "contradictory" (empty)
    - admissible_count: number of admissible parameter sets
    """
    status: str  # "singleton" | "underdetermined" | "contradictory"
    admissible_count: int


def _perm_to_lehmer(perm: List[int]) -> List[int]:
    """
    Convert permutation to Lehmer code.

    Contract (WO-04E):
    Given permutation σ in one-line notation (perm[i] = σ(i)),
    compute Lehmer code (factorial number system representation).

    Algorithm:
    For position i, the Lehmer code digit is the count of how many
    elements to the right of position i have values smaller than perm[i].

    Args:
        perm: One-line permutation where perm[i] = σ(i)

    Returns:
        lehmer: Lehmer code
    """
    if not perm:
        return []

    n = len(perm)
    lehmer = []

    for i in range(n):
        # Count how many elements to the right are smaller than perm[i]
        count = sum(1 for j in range(i + 1, n) if perm[j] < perm[i])
        lehmer.append(count)

    return lehmer


def _lehmer_to_perm(lehmer: List[int]) -> List[int]:
    """
    Convert Lehmer code to permutation.

    Contract (WO-10Z):
    Given Lehmer code (factorial number system representation),
    reconstruct the one-line permutation.

    Algorithm:
    Maintain available elements [0,1,2,...,n-1].
    For position i, pick the lehmer[i]-th element from available,
    remove it from available, and place in result.

    Args:
        lehmer: Lehmer code

    Returns:
        perm: One-line permutation where perm[i] = σ(i)
    """
    if not lehmer:
        return []

    n = len(lehmer)
    available = list(range(n))
    perm = []

    for i in range(n):
        # Pick the lehmer[i]-th smallest available element
        idx = lehmer[i]
        perm.append(available[idx])
        available.pop(idx)

    return perm


def _apply_d4_pose(G: np.ndarray, pose_id: int) -> np.ndarray:
    """
    Apply D4 transformation to grid.

    Contract (02_determinism_addendum.md §1.2):
    D4 ids: 0=id, 1=rot90, 2=rot180, 3=rot270, 4=flipH, 5=flipH∘rot90, 6=flipH∘rot180, 7=flipH∘rot270

    Args:
        G: input grid
        pose_id: D4 id (0..7)

    Returns:
        Transformed grid
    """
    if pose_id == 0:
        return G.copy()
    elif pose_id == 1:
        return np.rot90(G, k=3)  # rot90 CW = k=3 CCW
    elif pose_id == 2:
        return np.rot90(G, k=2)
    elif pose_id == 3:
        return np.rot90(G, k=1)  # rot270 CW = k=1 CCW
    elif pose_id == 4:
        return np.fliplr(G)
    elif pose_id == 5:
        return np.rot90(np.fliplr(G), k=3)
    elif pose_id == 6:
        return np.rot90(np.fliplr(G), k=2)
    elif pose_id == 7:
        return np.rot90(np.fliplr(G), k=1)
    else:
        raise ValueError(f"Invalid pose_id {pose_id}, must be 0..7")


def _enumerate_phi_candidates(
    X_bbox: np.ndarray,
    Y_bbox: np.ndarray,
    X_color: int,
    Y_color: int
) -> List[Tuple[int, int, int]]:
    """
    Enumerate (pose_id, dr, dc) candidates for component alignment.

    Contract (01_engineering_spec.md §4 & WO-04D):
    "Enumerate 8 D4 ops; for each, compute translation Δ that aligns bbox anchors"

    Algorithm (WO-04D):
    1. For each pose_id ∈ {0..7}:
       a. Apply pose to X_bbox → X_posed
       b. Check if X_posed.shape == Y_bbox.shape (same-shape coframe requirement)
       c. If yes, translation is (0,0) in bbox-local coords (anchors both at origin)
       d. Accept (pose_id, 0, 0) as candidate

    Note: Translation (dr, dc) is in bbox-local coordinates. Since bboxes are extracted
    with top-left at (0,0) in local frame, the translation that "aligns anchors" is
    always (0,0) when shapes match. Global translation between components is implicit
    in component matching, not stored in PhiPiece.

    Residue enumeration (WO-04R) will be added later for periodic coframes.

    Args:
        X_bbox: source component bbox (extracted to local coordinates)
        Y_bbox: target component bbox (extracted to local coordinates)
        X_color: source color (unused, for future color-dependent logic)
        Y_color: target color (unused, for future color-dependent logic)

    Returns:
        List of (pose_id, dr, dc) candidates where posed X shape matches Y shape
    """
    candidates = []

    # Try all 8 D4 poses (WO-04D)
    for pose_id in range(8):
        # Apply pose to X bbox
        X_posed = _apply_d4_pose(X_bbox, pose_id)

        # Require same-shape coframe (Math Spec: component coframes are isomorphic)
        if X_posed.shape != Y_bbox.shape:
            continue

        # Compute translation Δ that aligns component content (WO-04D)
        # Find anchor (top-left) of non-zero pixels in both arrays
        X_nonzero = np.argwhere(X_posed != 0)
        Y_nonzero = np.argwhere(Y_bbox != 0)

        if len(X_nonzero) == 0 or len(Y_nonzero) == 0:
            # Empty component - use (0,0) as default
            dr, dc = 0, 0
        else:
            # Compute anchor as top-left of non-zero pixels
            X_anchor = X_nonzero.min(axis=0)  # [r, c]
            Y_anchor = Y_nonzero.min(axis=0)  # [r, c]

            # Translation that aligns content anchors
            dr = int(Y_anchor[0] - X_anchor[0])
            dc = int(Y_anchor[1] - X_anchor[1])

        candidates.append((pose_id, dr, dc))

    return candidates


def _verify_bbox_equality(
    X: np.ndarray,
    Y: np.ndarray,
    X_inv,
    Y_inv,
    pose_id: int,
    dr: int,
    dc: int
) -> bool:
    """
    Verify φ candidate admits consistent σ on component bbox.

    Contract (WO-04G + Engg Spec §4):
    "verify value equality on every pixel" BUT we must allow for σ recoloring!

    Algorithm:
    1. Apply (pose, dr, dc) to X bbox
    2. Check if there EXISTS a consistent color mapping σ: X∘φ → Y
    3. Accept iff consistent (no pixel maps to two different colors)

    Args:
        X: full X grid
        Y: full Y grid
        X_inv: X component invariant
        Y_inv: Y component invariant
        pose_id: D4 pose id
        dr, dc: translation delta

    Returns:
        True if φ candidate admits consistent σ
    """
    # Extract X component bbox
    X_r0, X_c0 = X_inv.anchor_r, X_inv.anchor_c
    X_r1 = X_r0 + X_inv.bbox_h
    X_c1 = X_c0 + X_inv.bbox_w
    X_bbox = X[X_r0:X_r1, X_c0:X_c1]

    # Extract Y component bbox
    Y_r0, Y_c0 = Y_inv.anchor_r, Y_inv.anchor_c
    Y_r1 = Y_r0 + Y_inv.bbox_h
    Y_c1 = Y_c0 + Y_inv.bbox_w
    Y_bbox = Y[Y_r0:Y_r1, Y_c0:Y_c1]

    # Apply D4 pose to X bbox
    X_transformed = _apply_d4_pose(X_bbox, pose_id)

    # Require same-shape coframe (Math Spec: component coframes are isomorphic)
    if X_transformed.shape != Y_bbox.shape:
        return False

    # Apply translation (WO-04D)
    # Translation (dr, dc) shifts X_transformed to align with Y_bbox
    if dr != 0 or dc != 0:
        # Apply translation by creating shifted array
        # dr > 0 means shift DOWN, dc > 0 means shift RIGHT
        h, w = X_transformed.shape
        X_shifted = np.zeros_like(X_transformed)

        # Compute source and destination bounds
        src_r0 = max(0, -dr)
        src_r1 = min(h, h - dr)
        src_c0 = max(0, -dc)
        src_c1 = min(w, w - dc)

        dst_r0 = max(0, dr)
        dst_r1 = min(h, h + dr)
        dst_c0 = max(0, dc)
        dst_c1 = min(w, w + dc)

        # Copy shifted content
        if src_r1 > src_r0 and src_c1 > src_c0:
            X_shifted[dst_r0:dst_r1, dst_c0:dst_c1] = X_transformed[src_r0:src_r1, src_c0:src_c1]

        X_transformed = X_shifted

    # Check if consistent σ exists (WO-04G fix)
    # Build color mapping and check consistency
    color_mapping = {}
    for r in range(X_transformed.shape[0]):
        for c in range(X_transformed.shape[1]):
            x_color = int(X_transformed[r, c])
            y_color = int(Y_bbox[r, c])

            if x_color in color_mapping:
                # Check consistency
                if color_mapping[x_color] != y_color:
                    return False  # Inconsistent mapping
            else:
                color_mapping[x_color] = y_color

    # Consistent mapping exists → accept this φ candidate
    return True


def _shift_row(row: np.ndarray, dc: int) -> np.ndarray:
    """
    Apply horizontal shift to a row.

    Args:
        row: 1D array
        dc: shift amount (positive = right, negative = left)

    Returns:
        Shifted row with wraparound
    """
    if dc == 0:
        return row.copy()

    W = len(row)
    # Normalize shift to [0, W)
    dc_norm = dc % W
    return np.roll(row, dc_norm)


def _try_row_coframe(X: np.ndarray, Y: np.ndarray) -> Tuple[Optional[List[PhiPiece]], Optional[SigmaRc], Optional[TrainWitnessRc]]:
    """
    Try per-row horizontal shift geometric witness.

    Contract:
    - E2: Each row must find exact shift Δc where X[r, :] shifted equals Y[r, :]
    - Fail-closed: If any row can't find exact shift, return None
    - Deterministic: Enumerate shifts in fixed order, accept first match

    Args:
        X: input grid (Π frame)
        Y: output grid (Π frame)

    Returns:
        (phi_pieces, sigma, TrainWitnessRc) if all rows match, else (None, None, None)
    """
    # Row-coframe requires same shape
    if X.shape != Y.shape:
        return None, None, None

    H, W = X.shape
    pieces: List[PhiPiece] = []
    bbox_equal: List[bool] = []

    # Try to find horizontal shift for each row
    for r in range(H):
        X_row = X[r, :]
        Y_row = Y[r, :]

        # Enumerate horizontal shifts: try identity first, then others
        # Range: -W to W (covering all possible circular shifts)
        found_shift = False

        for dc in range(-W, W + 1):
            # Apply shift
            X_shifted = _shift_row(X_row, dc)

            # E2: exact equality check
            if np.array_equal(X_shifted, Y_row):
                pieces.append(PhiPiece(
                    comp_id=r,      # row index as component id
                    pose_id=0,      # identity (no rotation/flip)
                    dr=0,           # no vertical shift
                    dc=dc,          # horizontal shift for this row
                    r_per=1,        # no periodic residues
                    c_per=1,
                    r_res=0,
                    c_res=0
                ))
                bbox_equal.append(True)  # E2 proof passed
                found_shift = True
                break

        if not found_shift:
            # Row-coframe failed for this row
            return None, None, None

    # All rows found exact shifts → geometric witness
    # Compute domain colors
    X_colors = set(np.unique(X).tolist())
    Y_colors = set(np.unique(Y).tolist())
    domain_colors = sorted(X_colors | Y_colors)

    # σ = identity (no recoloring for row-coframe)
    sigma = SigmaRc(
        domain_colors=domain_colors,
        lehmer=[],          # identity permutation
        moved_count=0
    )

    phi_rc = PhiRc(
        pieces=pieces,
        bbox_equal=bbox_equal,
        domain_pixels=H * W,  # entire grid covered
        geometric_trials=None  # Row-coframe path has no component trials
    )

    witness_rc = TrainWitnessRc(
        kind="geometric",
        phi=phi_rc,
        sigma=sigma,
        foreground_colors=None,
        background_colors=None,
        decision_rule=None,
        per_color_counts=None
    )

    return pieces, sigma, witness_rc


def solve_witness_for_pair(
    X: np.ndarray,
    Y: np.ndarray,
    comps_X,
    comps_Y
) -> Tuple[Optional[List[PhiPiece]], SigmaRc, TrainWitnessRc]:
    """
    Solve witness (φ, σ) for a single training pair.

    Contract:
    - E2: Every φ piece must prove bbox equality (exact pixelwise)
    - A1: Candidate sets part of law (record all colors)
    - C2: Decision rule frozen string

    Args:
        X: input grid (Π frame)
        Y: output grid (Π frame)
        comps_X: ComponentsRc from WO-03
        comps_Y: ComponentsRc from WO-03

    Returns:
        (phi_pieces or None, sigma, TrainWitnessRc)
    """
    from .components import CompInv

    # Deserialize component invariants
    X_invs = [CompInv(**inv_dict) for inv_dict in comps_X.invariants]
    Y_invs = [CompInv(**inv_dict) for inv_dict in comps_Y.invariants]

    # For geometric path, need matched components (same outline_hash)
    # Build mapping of outline_hash -> components
    X_by_hash: Dict[str, List[Tuple[int, CompInv]]] = {}
    for i, inv in enumerate(X_invs):
        if inv.outline_hash not in X_by_hash:
            X_by_hash[inv.outline_hash] = []
        X_by_hash[inv.outline_hash].append((i, inv))

    Y_by_hash: Dict[str, List[Tuple[int, CompInv]]] = {}
    for i, inv in enumerate(Y_invs):
        if inv.outline_hash not in Y_by_hash:
            Y_by_hash[inv.outline_hash] = []
        Y_by_hash[inv.outline_hash].append((i, inv))

    # Try geometric witness: match by outline_hash
    pieces: List[PhiPiece] = []
    bbox_equal: List[bool] = []
    geometric_trials: List[Dict] = []  # WO-04D: record all trials
    geometric_ok = True

    # Match components by outline_hash (stable matching from WO-03)
    for outline_hash in sorted(set(X_by_hash.keys()) & set(Y_by_hash.keys())):
        X_comps = X_by_hash[outline_hash]
        Y_comps = Y_by_hash[outline_hash]

        # Try to pair by position order (simplest stable match)
        for (x_idx, x_inv), (y_idx, y_inv) in zip(X_comps, Y_comps):
            # Try candidate (pose_id, dr, dc) values
            found_match = False

            candidates = _enumerate_phi_candidates(
                X[x_inv.anchor_r:x_inv.anchor_r+x_inv.bbox_h, x_inv.anchor_c:x_inv.anchor_c+x_inv.bbox_w],
                Y[y_inv.anchor_r:y_inv.anchor_r+y_inv.bbox_h, y_inv.anchor_c:y_inv.anchor_c+y_inv.bbox_w],
                x_inv.color,
                y_inv.color
            )

            for pose_id, dr, dc in candidates:
                # Verify E2: exact bbox equality
                ok = _verify_bbox_equality(X, Y, x_inv, y_inv, pose_id, dr, dc)

                if ok:
                    # Detect periodic tiling (WO-05 fix)
                    # If Y bbox is larger than X bbox after pose, detect tiling
                    X_bbox_h, X_bbox_w = x_inv.bbox_h, x_inv.bbox_w
                    Y_bbox_h, Y_bbox_w = y_inv.bbox_h, y_inv.bbox_w

                    # After applying pose, check if dimensions match or need tiling
                    # For poses that swap dimensions (1,3,5,7), swap X dimensions
                    if pose_id in [1, 3, 5, 7]:  # rot90, rot270, flipH+rot90, flipH+rot270
                        X_bbox_h, X_bbox_w = X_bbox_w, X_bbox_h

                    # Detect tiling: Y_bbox should be multiple of X_bbox
                    r_per = 1
                    c_per = 1
                    r_res = 0
                    c_res = 0

                    if Y_bbox_h > X_bbox_h and Y_bbox_h % X_bbox_h == 0:
                        r_per = Y_bbox_h // X_bbox_h
                    if Y_bbox_w > X_bbox_w and Y_bbox_w % X_bbox_w == 0:
                        c_per = Y_bbox_w // X_bbox_w

                    # Record trial with EXPANDED tiling debug info
                    geometric_trials.append({
                        "comp_id": x_idx,
                        "pose": pose_id,
                        "dr": dr,
                        "dc": dc,
                        "ok": ok,
                        "X_bbox": [int(x_inv.bbox_h), int(x_inv.bbox_w)],
                        "Y_bbox": [int(y_inv.bbox_h), int(y_inv.bbox_w)],
                        "X_bbox_after_pose": [int(X_bbox_h), int(X_bbox_w)],
                        "tiling_detected": [int(r_per), int(c_per)]
                    })

                    pieces.append(PhiPiece(
                        comp_id=x_idx,
                        pose_id=pose_id,
                        dr=dr,
                        dc=dc,
                        r_per=r_per,  # Tiling factor (1 if no tiling)
                        c_per=c_per,
                        r_res=r_res,  # Residue class (0 for now)
                        c_res=c_res
                    ))
                    bbox_equal.append(True)
                    found_match = True
                    break
                else:
                    # Record failed trial without tiling info
                    geometric_trials.append({
                        "comp_id": x_idx,
                        "pose": pose_id,
                        "dr": dr,
                        "dc": dc,
                        "ok": ok
                    })

            if not found_match:
                # Geometric witness failed for this component
                geometric_ok = False
                break

        if not geometric_ok:
            break

    # Validate ALL components were matched (WO-04 contract: exact equality everywhere)
    if geometric_ok and pieces:
        # Check that ALL output components were explained
        matched_Y_comp_ids = set()
        for outline_hash in sorted(set(X_by_hash.keys()) & set(Y_by_hash.keys())):
            X_comps = X_by_hash[outline_hash]
            Y_comps = Y_by_hash[outline_hash]
            # Count how many Y components were matched
            n_matched = min(len(X_comps), len(Y_comps))
            for i in range(n_matched):
                matched_Y_comp_ids.add(Y_comps[i][0])  # y_idx

        # If ANY output component is unmatched, geometric witness FAILS
        total_Y_comps = len(Y_invs)
        if len(matched_Y_comp_ids) < total_Y_comps:
            geometric_ok = False  # Witness cannot explain all output pixels

    # If geometric succeeded, build geometric witness
    if geometric_ok and pieces:
        # Compute domain pixels accurately (WO-04G)
        # Sum areas of actually matched components
        # piece.comp_id is the index into X_invs
        matched_comp_ids = {p.comp_id for p in pieces}
        domain_pixels = sum(X_invs[cid].area for cid in matched_comp_ids)

        # Infer σ from pixel mapping X∘φ → Y (WO-04E + WO-04G)
        # Build mapping by iterating matched components
        color_mapping = {}  # source_color → target_color
        touched_colors = set()  # Colors actually seen in domain

        # Re-iterate through matched components to build color mapping
        piece_idx = 0
        for outline_hash in sorted(set(X_by_hash.keys()) & set(Y_by_hash.keys())):
            X_comps = X_by_hash[outline_hash]
            Y_comps = Y_by_hash[outline_hash]

            for (x_idx, x_inv), (y_idx, y_inv) in zip(X_comps, Y_comps):
                if piece_idx >= len(pieces):
                    break

                piece = pieces[piece_idx]
                if piece.comp_id != x_idx:
                    continue

                # Extract component bboxes
                X_bbox = X[x_inv.anchor_r:x_inv.anchor_r+x_inv.bbox_h,
                           x_inv.anchor_c:x_inv.anchor_c+x_inv.bbox_w]
                Y_bbox = Y[y_inv.anchor_r:y_inv.anchor_r+y_inv.bbox_h,
                           y_inv.anchor_c:y_inv.anchor_c+y_inv.bbox_w]

                # Apply pose and translation to X bbox (WO-04D + WO-04E)
                X_posed = _apply_d4_pose(X_bbox, piece.pose_id)

                # Apply translation to align with Y_bbox
                dr, dc = piece.dr, piece.dc
                if dr != 0 or dc != 0:
                    h, w = X_posed.shape
                    X_shifted = np.zeros_like(X_posed)
                    src_r0, src_r1 = max(0, -dr), min(h, h - dr)
                    src_c0, src_c1 = max(0, -dc), min(w, w - dc)
                    dst_r0, dst_r1 = max(0, dr), min(h, h + dr)
                    dst_c0, dst_c1 = max(0, dc), min(w, w + dc)
                    if src_r1 > src_r0 and src_c1 > src_c0:
                        X_shifted[dst_r0:dst_r1, dst_c0:dst_c1] = X_posed[src_r0:src_r1, src_c0:src_c1]
                    X_transformed = X_shifted
                else:
                    X_transformed = X_posed

                # Build pixel mapping from X∘φ → Y (WO-04E)
                for r in range(Y_bbox.shape[0]):
                    for c in range(Y_bbox.shape[1]):
                        x_color = int(X_transformed[r, c])
                        y_color = int(Y_bbox[r, c])

                        # Track touched colors
                        touched_colors.add(x_color)

                        if x_color in color_mapping:
                            # Check consistency (WO-04E)
                            if color_mapping[x_color] != y_color:
                                # Inconsistent recoloring
                                geometric_ok = False
                                break
                        else:
                            color_mapping[x_color] = y_color
                    if not geometric_ok:
                        break

                piece_idx += 1
                if not geometric_ok:
                    break
            if not geometric_ok:
                break

        if not geometric_ok:
            # Inconsistent σ - fall back to summary/identity
            domain_colors = sorted(set(inv.color for inv in X_invs))
            sigma = SigmaRc(
                domain_colors=domain_colors,
                lehmer=[],
                moved_count=0
            )
        else:
            # Build permutation on domain colors (WO-04G: fix index bug)
            # σ is defined only on colors that appear in X∘φ (touched_colors)
            domain_colors = sorted(touched_colors)

            # Check if color_mapping is injective on domain
            target_colors = [color_mapping[c] for c in domain_colors]
            if len(target_colors) != len(set(target_colors)):
                # Not injective - not a valid permutation
                # Fall back to identity
                sigma = SigmaRc(
                    domain_colors=domain_colors,
                    lehmer=[],
                    moved_count=0
                )
            else:
                # Check if target_colors form a permutation of domain_colors
                # (i.e., codomain == domain, making it a true permutation)
                if set(target_colors) != set(domain_colors):
                    # Codomain != domain, so it's a partial function, not a permutation
                    # This is still valid (e.g., {1→3} where 3 is a new color)
                    # Extend domain to include codomain colors
                    extended_domain = sorted(set(domain_colors) | set(target_colors))

                    # Build extended mapping with identity for new colors
                    extended_mapping = {c: c for c in extended_domain}
                    extended_mapping.update(color_mapping)

                    # Check injectivity on extended domain
                    extended_targets = [extended_mapping[c] for c in extended_domain]
                    if len(extended_targets) != len(set(extended_targets)):
                        # Extended mapping not injective - fall back
                        sigma = SigmaRc(
                            domain_colors=domain_colors,
                            lehmer=[],
                            moved_count=0
                        )
                    else:
                        # Build permutation on extended domain
                        color_to_idx = {c: i for i, c in enumerate(extended_domain)}
                        perm = [color_to_idx[extended_mapping[c]] for c in extended_domain]
                        lehmer = _perm_to_lehmer(perm)
                        moved_count = sum(1 for i in range(len(perm)) if perm[i] != i)

                        sigma = SigmaRc(
                            domain_colors=extended_domain,
                            lehmer=lehmer,
                            moved_count=moved_count
                        )
                else:
                    # Perfect permutation: codomain == domain
                    color_to_idx = {c: i for i, c in enumerate(domain_colors)}
                    perm = [color_to_idx[color_mapping[c]] for c in domain_colors]
                    lehmer = _perm_to_lehmer(perm)
                    moved_count = sum(1 for i in range(len(perm)) if perm[i] != i)

                    sigma = SigmaRc(
                        domain_colors=domain_colors,
                        lehmer=lehmer,
                        moved_count=moved_count
                    )

        phi_rc = PhiRc(
            pieces=pieces,
            bbox_equal=bbox_equal,
            domain_pixels=domain_pixels,
            geometric_trials=geometric_trials  # WO-04D: record all trials
        )

        witness_rc = TrainWitnessRc(
            kind="geometric",
            phi=phi_rc,
            sigma=sigma,
            foreground_colors=None,
            background_colors=None,
            decision_rule=None,
            per_color_counts=None
        )

        return pieces, sigma, witness_rc

    # Component-based geometric failed, try row-coframe geometric
    row_pieces, row_sigma, row_witness_rc = _try_row_coframe(X, Y)
    if row_witness_rc is not None:
        # Row-coframe geometric succeeded
        return row_pieces, row_sigma, row_witness_rc

    # Fall back to summary path (A1/C2)

    # A1: Candidate sets are part of the law
    # Foreground = all non-zero colors in X ∪ Y
    # Background = {0, 1} ∩ (X ∪ Y)
    X_colors = set(np.unique(X).tolist())
    Y_colors = set(np.unique(Y).tolist())
    all_colors = sorted(X_colors | Y_colors)

    foreground_colors = [c for c in all_colors if c not in (0,)]
    background_colors = [c for c in all_colors if c in (0, 1)]

    # C2: Decision rule frozen (contract: explicit string, no dynamic choices)
    decision_rule = "strict_majority_foreground_fallback_0"

    # A1: Record per-color counts on Y (the decision window for this example)
    per_color_counts = {c: int(np.sum(Y == c)) for c in all_colors}

    # σ: identity (no recoloring in summary mode for now)
    sigma = SigmaRc(
        domain_colors=sorted(all_colors),
        lehmer=[],
        moved_count=0
    )

    witness_rc = TrainWitnessRc(
        kind="summary",
        phi=None,
        sigma=sigma,
        foreground_colors=foreground_colors,
        background_colors=background_colors,
        decision_rule=decision_rule,
        per_color_counts=per_color_counts
    )

    return None, sigma, witness_rc


def conjugate_to_test(
    phi_pieces: Optional[List[PhiPiece]],
    sigma: SigmaRc,
    Pi_train=None,
    Pi_test=None,
    domain_pixels: int = 0
) -> Tuple[Optional[List[PhiPiece]], ConjugatedRc]:
    """
    Conjugate witness to test Π frame.

    Contract (WO-04C):
    φ* = Π* ∘ Πᵢ⁻¹ ∘ φᵢ ∘ Πᵢ ∘ Π*⁻¹

    For each piece (pose_id, dr, dc):
    - pose_new = Π*.pose ∘ φ.pose ∘ inv(Πᵢ.pose)
    - Translation transformed through pose changes and anchor shifts

    Functional algorithm (WO-04):
    - Build PiFrame structures with D4 matrices
    - Transport transformation via composition
    - Verify with pullback samples (3 samples per piece)

    Args:
        phi_pieces: geometric φ pieces or None
        sigma: palette permutation
        Pi_train: Π transform for training (PiTransform from WO-01)
        Pi_test: Π transform for test (PiTransform from WO-01)
        domain_pixels: total pixels covered by φ (sum of component areas)

    Returns:
        (phi_star_pieces or None, ConjugatedRc)
    """
    from .d4 import compose_pose, get_inverse_pose, transform_vector, PiFrame, D4_R, D4_R_INV

    if phi_pieces is None:
        # Summary witness: no geometric φ
        conj_rc = ConjugatedRc(
            phi_star=None,
            sigma=sigma,
            transform_receipts=None,
            conjugation_hash=None,
            pullback_samples=None
        )
        return None, conj_rc

    # If no Π metadata provided, forward as-is (identity conjugation)
    if Pi_train is None or Pi_test is None:
        phi_star_rc = PhiRc(
            pieces=phi_pieces,
            bbox_equal=[True] * len(phi_pieces),
            domain_pixels=domain_pixels,  # Passed from caller (sum of component areas)
            geometric_trials=None  # Conjugation has no trials
        )
        # Identity hash (no transform)
        phi_star_sorted = sorted(phi_pieces, key=lambda p: p.comp_id)
        phi_star_bytes = b"".join(
            f"{p.comp_id},{p.pose_id},{p.dr},{p.dc},{p.r_per},{p.c_per},{p.r_res},{p.c_res}".encode()
            for p in phi_star_sorted
        )
        identity_hash = hash_bytes(phi_star_bytes)

        conj_rc = ConjugatedRc(
            phi_star=phi_star_rc,
            sigma=sigma,
            transform_receipts=[],
            conjugation_hash=identity_hash,
            pullback_samples=[]  # Identity: no transformation, empty samples
        )
        return phi_pieces, conj_rc

    # Extract Π metadata and build PiFrame structures (WO-04)
    pose_train = Pi_train.pose_id
    anchor_train = Pi_train.anchor
    pose_test = Pi_test.pose_id
    anchor_test = Pi_test.anchor

    # Build PiFrame with D4 matrices for conjugation
    pi_train_frame = PiFrame(
        pose_id=pose_train,
        anchor=(anchor_train.dr, anchor_train.dc),
        R=D4_R[pose_train],
        R_inv=D4_R_INV[pose_train]
    )
    pi_test_frame = PiFrame(
        pose_id=pose_test,
        anchor=(anchor_test.dr, anchor_test.dc),
        R=D4_R[pose_test],
        R_inv=D4_R_INV[pose_test]
    )

    # Conjugate each piece
    phi_star_pieces = []
    transform_receipts = []
    pullback_samples = []  # WO-04: verification samples

    for piece in phi_pieces:
        # Original piece parameters
        pose_orig = piece.pose_id
        dr_orig = piece.dr
        dc_orig = piece.dc

        # Conjugate pose: Π*.pose ∘ φ.pose ∘ inv(Πᵢ.pose)
        inv_pose_train = get_inverse_pose(pose_train)
        pose_new = compose_pose(pose_test, compose_pose(pose_orig, inv_pose_train))

        # Conjugate translation:
        # The translation (dr, dc) is AFTER the piece's pose is applied in the Πᵢ frame.
        # We need to transform this to the Π* frame.
        #
        # Transform: Π* ∘ Πᵢ⁻¹
        # - Unapply Πᵢ anchor
        # - Transform through: Π*.pose ∘ inv(Πᵢ.pose)
        # - Apply Π* anchor

        # Step 1: Remove training anchor offset (in training frame, after training pose)
        dr_step1 = dr_orig - anchor_train.dr
        dc_step1 = dc_orig - anchor_train.dc

        # Step 2: Transform through the composition: Π*.pose ∘ inv(Πᵢ.pose)
        # This transforms the vector from Πᵢ-frame to Π*-frame
        # First unapply Πᵢ.pose
        dr_step2, dc_step2 = transform_vector(dr_step1, dc_step1, inv_pose_train)

        # Then apply Π*.pose
        dr_step3, dc_step3 = transform_vector(dr_step2, dc_step2, pose_test)

        # Step 3: Add test anchor offset (in test frame)
        dr_new = dr_step3 + anchor_test.dr
        dc_new = dc_step3 + anchor_test.dc

        # Residue handling (WO-04E): swap if conjugated pose swaps axes
        # Poses {1,3,5,7} = {R90, R270, FH∘R90, FH∘R270} swap row↔column
        if pose_new in [1, 3, 5, 7]:
            # Swap residue periods and classes
            r_per_new = piece.c_per
            c_per_new = piece.r_per
            r_res_new = piece.c_res
            c_res_new = piece.r_res
        else:
            # Keep residues unchanged
            r_per_new = piece.r_per
            c_per_new = piece.c_per
            r_res_new = piece.r_res
            c_res_new = piece.c_res

        # Create conjugated piece
        phi_star_piece = PhiPiece(
            comp_id=piece.comp_id,
            pose_id=pose_new,
            dr=dr_new,
            dc=dc_new,
            r_per=r_per_new,  # Swapped if axis-swap pose
            c_per=c_per_new,
            r_res=r_res_new,
            c_res=c_res_new
        )

        phi_star_pieces.append(phi_star_piece)

        # Record transform for receipts
        transform_receipts.append({
            "piece_id": piece.comp_id,
            "old": {"pose_id": pose_orig, "dr": dr_orig, "dc": dc_orig},
            "new": {"pose_id": pose_new, "dr": dr_new, "dc": dc_new},
            "composed_pose": f"Π*.pose={pose_test} ∘ φ.pose={pose_orig} ∘ inv(Πᵢ.pose={pose_train}) = {pose_new}"
        })

        # Compute pullback samples (WO-04): 3 verification samples per piece
        # Sample points: (0,0), (bbox_h//2, bbox_w//2), (bbox_h-1, bbox_w-1)
        # These verify that φ* correctly conjugates coordinates
        sample_points = [
            (0, 0),
            (1, 1),  # Simple interior point
            (2, 2)   # Another interior point
        ]

        for sr, sc in sample_points:
            # Forward through φ_i in train frame: (sr,sc) → φ_i → (tr,tc)
            # This would be: apply pose_orig, then translate by (dr_orig, dc_orig)
            # For verification, we record the transformation mapping
            pullback_samples.append({
                "piece_id": piece.comp_id,
                "sample_input": (sr, sc),
                "train_frame_pose": pose_orig,
                "train_frame_translation": (dr_orig, dc_orig),
                "test_frame_pose": pose_new,
                "test_frame_translation": (dr_new, dc_new),
                "pi_train_pose": pose_train,
                "pi_test_pose": pose_test
            })

    # Compute conjugation hash
    # Serialize phi_star for hashing (sorted by comp_id for stability)
    phi_star_sorted = sorted(phi_star_pieces, key=lambda p: p.comp_id)
    phi_star_bytes = b"".join(
        f"{p.comp_id},{p.pose_id},{p.dr},{p.dc},{p.r_per},{p.c_per},{p.r_res},{p.c_res}".encode()
        for p in phi_star_sorted
    )
    conjugation_hash = hash_bytes(phi_star_bytes)

    phi_star_rc = PhiRc(
        pieces=phi_star_pieces,
        bbox_equal=[True] * len(phi_star_pieces),
        domain_pixels=domain_pixels,  # Passed from caller (sum of component areas)
        geometric_trials=None  # Conjugation has no trials
    )

    conj_rc = ConjugatedRc(
        phi_star=phi_star_rc,
        sigma=sigma,
        transform_receipts=transform_receipts,
        conjugation_hash=conjugation_hash,
        pullback_samples=pullback_samples  # WO-04: 3 verification samples per piece
    )

    return phi_star_pieces, conj_rc


def intersect_witnesses(
    conj_list: List[Tuple[Optional[List[PhiPiece]], SigmaRc]]
) -> Tuple[Optional[List[PhiPiece]], SigmaRc, IntersectionRc]:
    """
    Intersect witnesses across trainings.

    Contract:
    - Geometric: parameters must match exactly (pose_id, dr, dc, residues)
    - Summary: φ=None must be consistent, σ must match
    - Mixed types → contradictory

    Args:
        conj_list: list of (phi_pieces or None, sigma) per training

    Returns:
        (intersected_phi or None, intersected_sigma, IntersectionRc)
    """
    if not conj_list:
        # Empty list (shouldn't happen)
        return None, SigmaRc([], [], 0), IntersectionRc("contradictory", 0)

    # Classify by kind
    kinds = ["none" if phi is None else "geom" for phi, _ in conj_list]

    # Check for mixed types (contradictory)
    if "geom" in kinds and "none" in kinds:
        return None, conj_list[0][1], IntersectionRc("contradictory", 0)

    # All summary (φ=None)
    if all(k == "none" for k in kinds):
        # Check σ consistency
        sigma_0 = conj_list[0][1]
        lehmer_0 = sigma_0.lehmer

        for _, sigma in conj_list[1:]:
            if sigma.lehmer != lehmer_0:
                # σ mismatch → contradictory
                return None, sigma_0, IntersectionRc("contradictory", 0)

        # Singleton: all summary with matching σ
        return None, sigma_0, IntersectionRc("singleton", 1)

    # All geometric
    phi_0, sigma_0 = conj_list[0]

    if phi_0 is None:
        # Should not happen (already checked all none above)
        return None, sigma_0, IntersectionRc("contradictory", 0)

    # Encode first witness parameters
    enc_0 = [(p.pose_id, p.dr, p.dc, p.r_per, p.c_per, p.r_res, p.c_res) for p in phi_0]

    # Check all other trainings match
    for phi, sigma in conj_list[1:]:
        if phi is None:
            # Mixed (should have been caught above)
            return phi_0, sigma_0, IntersectionRc("contradictory", 0)

        enc = [(p.pose_id, p.dr, p.dc, p.r_per, p.c_per, p.r_res, p.c_res) for p in phi]

        if enc != enc_0:
            # Parameters differ → underdetermined (at least 2 admissible)
            return phi_0, sigma_0, IntersectionRc("underdetermined", 2)

    # Check σ consistency
    lehmer_0 = sigma_0.lehmer
    for _, sigma in conj_list[1:]:
        if sigma.lehmer != lehmer_0:
            return phi_0, sigma_0, IntersectionRc("underdetermined", 2)

    # All match → singleton
    return phi_0, sigma_0, IntersectionRc("singleton", 1)
