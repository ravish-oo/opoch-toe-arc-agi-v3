# arc/op/pi.py
# WO-01: Presentation Π and unpresentation U⁻¹
# Implements 00_math_spec.md §1, 01_engineering_spec.md §2

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .palette import build_palette_canon, apply_palette_map, invert_palette_map, PaletteRc
from .d4 import apply_pose, raster_lex_key, get_inverse_pose, POSES
from .anchor import anchor_to_origin, unanchor_from_origin, AnchorRc
from .hash import hash_grid


@dataclass
class PiRc:
    """
    Presentation Π receipt.

    Contract (02_determinism_addendum.md §10 line 194):
    Π receipts must include: palette_hash, palette_freqs, pose_id, anchor_drdc, roundtrip_hash

    Contract (docs/common_mistakes.md F2):
    Per-grid receipts to prove Π²=Π and U⁻¹∘Π=id for EVERY grid
    """
    palette: PaletteRc
    per_grid: list[dict]  # [{"kind":"train"/"test", "pose_id":int, "anchor":{"dr":int,"dc":int}, "pi2_hash":str, "roundtrip_hash":str}, ...]
    test_pose_id: int     # Pose for test grid (used in PiTransform)
    test_anchor: AnchorRc  # Anchor for test grid (used in PiTransform)


@dataclass
class PiTransform:
    """
    Π transformation data sufficient for inversion.

    Contains:
    - Palette maps (forward and inverse)
    - D4 pose ID
    - Anchor offset
    """
    map: dict[int, int]
    inv_map: dict[int, int]
    pose_id: int
    anchor: AnchorRc


def choose_pose_and_anchor(G: np.ndarray) -> tuple[np.ndarray, int, AnchorRc]:
    """
    Choose D4 pose via lexicographic minimum (after anchoring).

    Algorithm (01_engineering_spec.md §2):
    1. For each pose in D4:
       - Apply pose to G
       - Anchor result to origin
       - Compute raster lex key
    2. Choose pose with minimal lex key

    Args:
        G: canonical palette-mapped grid

    Returns:
        (presented_grid, pose_id, anchor)
        - presented_grid: G after chosen pose + anchor
        - pose_id: chosen D4 pose
        - anchor: anchor offset applied
    """
    best_key = None
    best_pose = 0
    best_anchor = AnchorRc(0, 0)
    best_grid = G

    for pose_id in POSES:
        # Apply pose
        posed = apply_pose(G, pose_id)

        # Anchor to origin
        anchored, anchor_rc = anchor_to_origin(posed)

        # Compute lex key
        key = raster_lex_key(anchored)

        # Choose minimal key
        if best_key is None or key < best_key:
            best_key = key
            best_pose = pose_id
            best_anchor = anchor_rc
            best_grid = anchored

    return best_grid, best_pose, best_anchor


def present(G: np.ndarray, mapping: dict[int, int]) -> tuple[np.ndarray, int, AnchorRc]:
    """
    Apply Π to a single grid: palette map → choose pose+anchor.

    Args:
        G: input grid
        mapping: palette mapping (original → canonical)

    Returns:
        (presented_grid, pose_id, anchor)
    """
    # Apply palette mapping
    G_pal = apply_palette_map(G, mapping)

    # Choose pose + anchor via lex-min
    G_presented, pose_id, anchor = choose_pose_and_anchor(G_pal)

    return G_presented, pose_id, anchor


def present_all(
    train_inputs: list[np.ndarray], test_input: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray, PiTransform, PiRc]:
    """
    Build Π for all inputs (train + test) and present them.

    Algorithm (01_engineering_spec.md §2):
    1. Build palette canon over ALL inputs (train + test)
    2. Apply Π to EVERY grid (train + test)
    3. Prove Π²=Π for EVERY grid
    4. Prove U⁻¹∘Π=id for EVERY grid
    5. Return presented grids, transform for inversion, and per-grid receipts

    Contract (02_determinism_addendum.md §1.3):
    Palette built on inputs only (never outputs)

    Contract (docs/common_mistakes.md F1, F2):
    - F1: Palette canon with forbid_outputs=True guard
    - F2: Per-grid proofs of idempotence and invertibility

    Args:
        train_inputs: list of training input grids
        test_input: test input grid

    Returns:
        (train_presented, test_presented, transform, receipt)
        - train_presented: list of presented training inputs
        - test_presented: presented test input
        - transform: PiTransform for unpresentation
        - receipt: PiRc with per-grid proofs
    """
    # 1. Build palette canon (inputs only, F1 guard)
    all_inputs = train_inputs + [test_input]
    mapping, inv_map, pal_rc = build_palette_canon(all_inputs, forbid_outputs=True)

    # 2. Apply Π to EVERY grid and prove properties (F2)
    per_grid_receipts = []
    train_presented = []

    # Process train grids
    for i, G_orig in enumerate(train_inputs):
        # Apply palette
        G_pal = apply_palette_map(G_orig, mapping)

        # Choose pose + anchor (first application of Π)
        G_pi1, pose_id, anchor = choose_pose_and_anchor(G_pal)

        # Prove Π²=Π: apply Π again and verify hash equality
        G_pi2, pose_id2, anchor2 = choose_pose_and_anchor(G_pi1)
        pi2_hash = hash_grid(G_pi2)
        pi1_hash = hash_grid(G_pi1)

        # F2 proof: Π²=Π
        if pi2_hash != pi1_hash:
            raise ValueError(
                f"Train grid {i}: Π²≠Π! Idempotence violated (F2).\n"
                f"  Π(G) hash: {pi1_hash}\n"
                f"  Π(Π(G)) hash: {pi2_hash}"
            )
        # After first Π, grid is canonical, so second Π should return identity transform
        if pose_id2 != 0 or anchor2.dr != 0 or anchor2.dc != 0:
            raise ValueError(
                f"Train grid {i}: Π(Π(G)) should return identity (pose=0, anchor=0,0) but got pose={pose_id2}, anchor=({anchor2.dr},{anchor2.dc}) (F2).\n"
                f"  This means the first Π did not produce a canonical grid."
            )

        # Prove U⁻¹∘Π=id: unpresent and verify hash equality
        T_train = PiTransform(mapping, inv_map, pose_id, anchor)
        G_roundtrip = unpresent(T_train, G_pi1)
        roundtrip_hash = hash_grid(G_roundtrip)
        original_hash = hash_grid(G_orig)

        # F2 proof: U⁻¹∘Π=id
        if roundtrip_hash != original_hash:
            raise ValueError(
                f"Train grid {i}: U⁻¹∘Π≠id! Invertibility violated (F2).\n"
                f"  Original hash: {original_hash}\n"
                f"  Roundtrip hash: {roundtrip_hash}"
            )

        # Record per-grid receipt
        per_grid_receipts.append({
            "kind": "train",
            "index": i,
            "pose_id": pose_id,
            "anchor": {"dr": anchor.dr, "dc": anchor.dc},
            "pi2_hash": pi2_hash,
            "roundtrip_hash": roundtrip_hash,
        })

        train_presented.append(G_pi1)

    # Process test grid
    G_orig_test = test_input
    G_pal_test = apply_palette_map(G_orig_test, mapping)

    # Choose pose + anchor (first application of Π)
    G_pi1_test, pose_id_test, anchor_test = choose_pose_and_anchor(G_pal_test)

    # Prove Π²=Π
    G_pi2_test, pose_id2_test, anchor2_test = choose_pose_and_anchor(G_pi1_test)
    pi2_hash_test = hash_grid(G_pi2_test)
    pi1_hash_test = hash_grid(G_pi1_test)

    # F2 proof: Π²=Π
    if pi2_hash_test != pi1_hash_test:
        raise ValueError(
            f"Test grid: Π²≠Π! Idempotence violated (F2).\n"
            f"  Π(G) hash: {pi1_hash_test}\n"
            f"  Π(Π(G)) hash: {pi2_hash_test}"
        )
    # After first Π, grid is canonical, so second Π should return identity transform
    if pose_id2_test != 0 or anchor2_test.dr != 0 or anchor2_test.dc != 0:
        raise ValueError(
            f"Test grid: Π(Π(G)) should return identity (pose=0, anchor=0,0) but got pose={pose_id2_test}, anchor=({anchor2_test.dr},{anchor2_test.dc}) (F2).\n"
            f"  This means the first Π did not produce a canonical grid."
        )

    # Prove U⁻¹∘Π=id
    T_test = PiTransform(mapping, inv_map, pose_id_test, anchor_test)
    G_roundtrip_test = unpresent(T_test, G_pi1_test)
    roundtrip_hash_test = hash_grid(G_roundtrip_test)
    original_hash_test = hash_grid(G_orig_test)

    # F2 proof: U⁻¹∘Π=id
    if roundtrip_hash_test != original_hash_test:
        raise ValueError(
            f"Test grid: U⁻¹∘Π≠id! Invertibility violated (F2).\n"
            f"  Original hash: {original_hash_test}\n"
            f"  Roundtrip hash: {roundtrip_hash_test}"
        )

    # Record test grid receipt
    per_grid_receipts.append({
        "kind": "test",
        "pose_id": pose_id_test,
        "anchor": {"dr": anchor_test.dr, "dc": anchor_test.dc},
        "pi2_hash": pi2_hash_test,
        "roundtrip_hash": roundtrip_hash_test,
    })

    # Build transform (uses test grid's pose/anchor)
    T = PiTransform(
        map=mapping,
        inv_map=inv_map,
        pose_id=pose_id_test,
        anchor=anchor_test,
    )

    # Build receipt with per-grid proofs
    rc = PiRc(
        palette=pal_rc,
        per_grid=per_grid_receipts,
        test_pose_id=pose_id_test,
        test_anchor=anchor_test,
    )

    return train_presented, G_pi1_test, T, rc


def unpresent(T: PiTransform, Yt: np.ndarray) -> np.ndarray:
    """
    Apply U⁻¹ (inverse presentation) to recover original grid.

    Algorithm (01_engineering_spec.md §2):
    1. Inverse anchor: shift back by (dr, dc)
    2. Inverse D4: apply inverse pose
    3. Inverse palette: map canonical codes → original colors

    Contract (00_math_spec.md line 24):
    U⁻¹ is exact inverse of Π

    Args:
        T: PiTransform with inversion data
        Yt: presented grid (output after Π operations)

    Returns:
        Original grid (before Π)
    """
    # 1. Inverse anchor: shift back
    Y_unanchored = unanchor_from_origin(Yt, T.anchor)

    # 2. Inverse D4: apply inverse pose
    inv_pose = get_inverse_pose(T.pose_id)
    Y_unposed = apply_pose(Y_unanchored, inv_pose)

    # 3. Inverse palette: restore original colors
    Y_orig = invert_palette_map(Y_unposed, T.inv_map)

    return Y_orig
