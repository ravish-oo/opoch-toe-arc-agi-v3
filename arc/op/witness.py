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

    Contract:
    - pieces: list of PhiPiece per component
    - bbox_equal: per-piece equality verification (E2)
    - domain_pixels: total pixels covered by φ
    """
    pieces: List[PhiPiece]
    bbox_equal: List[bool]  # per-piece E2 proof
    domain_pixels: int      # sum of pixels across pieces


@dataclass
class SigmaRc:
    """
    Palette role permutation receipt.

    Contract (02_determinism_addendum.md §1.4):
    - domain_colors: colors touched by law
    - lehmer: Lehmer code encoding
    - moved_count: recolor_bits (tie-break cost)
    """
    domain_colors: List[int]
    lehmer: List[int]
    moved_count: int  # number of moved colors (recolor_bits)


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
    """
    phi_star: Optional[PhiRc]
    sigma: SigmaRc
    transform_receipts: Optional[List[Dict]] = None  # Per-piece conjugation details
    conjugation_hash: Optional[str] = None           # BLAKE3 of phi_star


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

    Contract (01_engineering_spec.md §4):
    "Enumerate 8 D4 ops; for each, compute translation Δ that aligns bbox anchors"

    For now: simple version trying identity translation.
    TODO: expand with lattice residues when WO-05 provides periods.

    Args:
        X_bbox: source component bbox
        Y_bbox: target component bbox
        X_color: source color
        Y_color: target color

    Returns:
        List of (pose_id, dr, dc) candidates
    """
    candidates = []

    # Try all 8 D4 poses
    for pose_id in range(8):
        # For now, try identity translation (0, 0)
        # In full implementation, would enumerate translations that align masks
        candidates.append((pose_id, 0, 0))

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
    Verify exact pixelwise equality on component bbox.

    Contract (02_determinism_addendum.md §3 line 121):
    "Verification: after applying candidate (pose,Δ,residue), compare pixelwise across bbox; must be equal"

    Contract (E2): Every φ piece must pass per-component bbox equality

    Args:
        X: full X grid
        Y: full Y grid
        X_inv: X component invariant
        Y_inv: Y component invariant
        pose_id: D4 pose id
        dr, dc: translation delta

    Returns:
        True if exact equality holds on bbox
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

    # Apply translation (dr, dc)
    # For now, require shapes match after pose (simplified)
    if X_transformed.shape != Y_bbox.shape:
        return False

    # Exact pixelwise equality check (E2)
    return bool(np.array_equal(X_transformed, Y_bbox))


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
        domain_pixels=H * W  # entire grid covered
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
    geometric_ok = True

    # Match components by outline_hash (stable matching from WO-03)
    for outline_hash in sorted(set(X_by_hash.keys()) & set(Y_by_hash.keys())):
        X_comps = X_by_hash[outline_hash]
        Y_comps = Y_by_hash[outline_hash]

        # Try to pair by position order (simplest stable match)
        for (x_idx, x_inv), (y_idx, y_inv) in zip(X_comps, Y_comps):
            # Try candidate (pose_id, dr, dc) values
            found_match = False

            for pose_id, dr, dc in _enumerate_phi_candidates(
                X[x_inv.anchor_r:x_inv.anchor_r+x_inv.bbox_h, x_inv.anchor_c:x_inv.anchor_c+x_inv.bbox_w],
                Y[y_inv.anchor_r:y_inv.anchor_r+y_inv.bbox_h, y_inv.anchor_c:y_inv.anchor_c+y_inv.bbox_w],
                x_inv.color,
                y_inv.color
            ):
                # Verify E2: exact bbox equality
                if _verify_bbox_equality(X, Y, x_inv, y_inv, pose_id, dr, dc):
                    pieces.append(PhiPiece(
                        comp_id=x_idx,
                        pose_id=pose_id,
                        dr=dr,
                        dc=dc,
                        r_per=1,  # No periodic residues for now (WO-05)
                        c_per=1,
                        r_res=0,
                        c_res=0
                    ))
                    bbox_equal.append(True)
                    found_match = True
                    break

            if not found_match:
                # Geometric witness failed for this component
                geometric_ok = False
                break

        if not geometric_ok:
            break

    # If geometric succeeded, build geometric witness
    if geometric_ok and pieces:
        # Compute domain pixels
        domain_pixels = sum(inv.area for inv in X_invs[:len(pieces)])

        # Compute σ (for now, identity permutation)
        domain_colors = sorted(set(inv.color for inv in X_invs))
        sigma = SigmaRc(
            domain_colors=domain_colors,
            lehmer=[],  # Identity permutation
            moved_count=0
        )

        phi_rc = PhiRc(
            pieces=pieces,
            bbox_equal=bbox_equal,
            domain_pixels=domain_pixels
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

    # C2: Decision rule frozen
    # For now: generic "strict_majority_foreground_fallback_0"
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
    Pi_test=None
) -> Tuple[Optional[List[PhiPiece]], ConjugatedRc]:
    """
    Conjugate witness to test Π frame.

    Contract (WO-04C):
    φ* = Π* ∘ Πᵢ⁻¹ ∘ φᵢ ∘ Πᵢ ∘ Π*⁻¹

    For each piece (pose_id, dr, dc):
    - pose_new = Π*.pose ∘ φ.pose ∘ inv(Πᵢ.pose)
    - Translation transformed through pose changes and anchor shifts

    Args:
        phi_pieces: geometric φ pieces or None
        sigma: palette permutation
        Pi_train: Π transform for training (PiTransform from WO-01)
        Pi_test: Π transform for test (PiTransform from WO-01)

    Returns:
        (phi_star_pieces or None, ConjugatedRc)
    """
    from .d4 import compose_pose, get_inverse_pose, transform_vector

    if phi_pieces is None:
        # Summary witness: no geometric φ
        conj_rc = ConjugatedRc(
            phi_star=None,
            sigma=sigma,
            transform_receipts=None,
            conjugation_hash=None
        )
        return None, conj_rc

    # If no Π metadata provided, forward as-is (identity conjugation)
    if Pi_train is None or Pi_test is None:
        phi_star_rc = PhiRc(
            pieces=phi_pieces,
            bbox_equal=[True] * len(phi_pieces),
            domain_pixels=0
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
            conjugation_hash=identity_hash
        )
        return phi_pieces, conj_rc

    # Extract Π metadata
    pose_train = Pi_train.pose_id
    anchor_train = Pi_train.anchor
    pose_test = Pi_test.pose_id
    anchor_test = Pi_test.anchor

    # Conjugate each piece
    phi_star_pieces = []
    transform_receipts = []

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

        # Create conjugated piece
        phi_star_piece = PhiPiece(
            comp_id=piece.comp_id,
            pose_id=pose_new,
            dr=dr_new,
            dc=dc_new,
            r_per=piece.r_per,  # Residues unchanged (lattice structure preserved)
            c_per=piece.c_per,
            r_res=piece.r_res,
            c_res=piece.c_res
        )

        phi_star_pieces.append(phi_star_piece)

        # Record transform for receipts
        transform_receipts.append({
            "piece_id": piece.comp_id,
            "old": {"pose_id": pose_orig, "dr": dr_orig, "dc": dc_orig},
            "new": {"pose_id": pose_new, "dr": dr_new, "dc": dc_new},
            "composed_pose": f"Π*.pose={pose_test} ∘ φ.pose={pose_orig} ∘ inv(Πᵢ.pose={pose_train}) = {pose_new}"
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
        domain_pixels=sum(1 for _ in phi_star_pieces)  # Placeholder
    )

    conj_rc = ConjugatedRc(
        phi_star=phi_star_rc,
        sigma=sigma,
        transform_receipts=transform_receipts,
        conjugation_hash=conjugation_hash
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
