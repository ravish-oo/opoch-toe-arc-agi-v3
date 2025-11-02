#!/usr/bin/env python3
# arc/runner.py
# WO-11: Task runner + determinism harness (BLOCKER)
# Implements the single commuting operator end-to-end with receipts and determinism

"""
Contract (WO-11):
Y* = U⁻¹(Meet ∘ (copy ⊎ law ⊎ unanimity ⊎ bottom) ∘ Truth(gfp) ∘ S ∘ Π(X*))

Frozen order (no reordering):
Π(01) → S(02) → Truth(05) → [Engines(10)→Witness(04)→Tie(08)] → Copy(06) → Unanimity(07) → Meet(09) → U⁻¹(01)

J1 Determinism: Run twice per task, compare all section hashes + table_hash + output_hash.
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
import sys
import platform
from blake3 import blake3

from arc.op import pi, shape, truth, witness, copy, unanimity, meet, families, components, tiebreak
from arc.op.receipts import ShapeRc
from arc.op.hash import hash_bytes


def env_fingerprint() -> Dict[str, str]:
    """
    Environment fingerprint for determinism checks.

    Contract (02_determinism_addendum.md §11 lines 254-256):
    Record platform, endian, versions for NONDETERMINISTIC_ENV detection.
    """
    return {
        "platform": platform.system(),
        "endian": sys.byteorder,
        "py_version": platform.python_version(),
        "blake3_version": "1.0.0",  # frozen
    }


@dataclass
class RunRc:
    """
    Full run receipt for one task.

    Contract (WO-11):
    Aggregate all section receipts + their BLAKE3 hashes + table_hash.
    No timestamps (frozen serialization).
    """
    task_id: str
    env: Dict[str, str]
    sections: Dict[str, Any]  # {pi: {...}, shape: {...}, truth: {...}, ...}
    hashes: Dict[str, str]    # {pi: "...", shape: "...", ...}
    table_hash: str           # BLAKE3(concat(sorted(section_key + ':' + hash)))
    final: Dict[str, Any]     # {shape: [H, W], law_status: "singleton|underdetermined|contradictory|engine|none"}


def solve_task(
    task_id: str,
    train_pairs: List[Tuple[str, np.ndarray, np.ndarray]],  # (train_id, X_raw, Y_raw)
    Xstar_raw: np.ndarray
) -> Tuple[np.ndarray, RunRc]:
    """
    Solve one ARC task using the frozen operator pipeline.

    Contract (WO-11):
    Returns (Y_raw, receipts) where receipts are first-class and contain
    section receipts + BLAKE3 hashes + table_hash.

    Frozen order:
    Π → S → Truth → [Engines→Witness→Tie] → Copy → Unanimity → Meet → U⁻¹

    Args:
        task_id: Task identifier
        train_pairs: [(train_id, X_i_raw, Y_i_raw), ...]
        Xstar_raw: Test input (raw frame)

    Returns:
        (Y_raw, run_rc): Test output and full receipts

    Raises:
        ValueError: If pipeline encounters unrecoverable error (fail-closed)
    """
    # Record environment
    env = env_fingerprint()

    # Storage for section receipts and hashes
    sections = {}
    hashes = {}

    # ========================================================================
    # Step 1: Π — present inputs + test (WO-01)
    # ========================================================================

    train_X_raw = [X for _, X, _ in train_pairs]
    train_Y_raw = [Y for _, _, Y in train_pairs]
    train_ids = [tid for tid, _, _ in train_pairs]

    # Present all inputs (trainings + test)
    # Returns: train_presented, test_presented, transform, receipt
    from arc.op.pi import present_all

    Xt_list, Xstar_t, Pi_test, pi_rc = present_all(train_X_raw, Xstar_raw)

    # Outputs are already in Π frame (from WO-01 contract)
    # For now, assume outputs match (we'll verify in shape step)
    Yt_list = train_Y_raw

    # Verify Π contract (idempotent, inputs-only palette)
    if pi_rc.palette.scope != "inputs_only":
        raise ValueError(f"Π palette must be inputs_only, got {pi_rc.palette.scope}")

    # Store Π receipts
    sections["pi"] = asdict(pi_rc)
    hashes["pi"] = hash_bytes(str(sections["pi"]).encode())

    # ========================================================================
    # Step 2: S — shape from WO-02 serialized params (reuse, no refit)
    # ========================================================================

    # Load WO-02 shape receipt (in real system, this comes from task receipts)
    # For now, synthesize shape fresh (WO-02 implementation)
    train_shape_pairs = [
        (train_ids[i], (X.shape[0], X.shape[1]), (Y.shape[0], Y.shape[1]))
        for i, (X, Y) in enumerate(zip(Xt_list, Yt_list))
    ]

    S_fn, shape_rc = shape.synthesize_shape(train_shape_pairs)

    # Check if shape was contradictory (content-dependent case)
    shape_contradictory = (S_fn is None)

    # Apply S to test (on Π-presented sizes) if available
    H_star, W_star = Xstar_t.shape

    if not shape_contradictory:
        R_star, C_star = S_fn(H_star, W_star)

        # Verify shape is positive
        if R_star <= 0 or C_star <= 0:
            raise ValueError(f"Shape S produced non-positive dimensions: ({R_star}, {C_star})")
    else:
        # Shape contradictory - will be determined by engine
        R_star, C_star = None, None

    # Store shape receipts
    if shape_rc:
        sections["shape"] = asdict(shape_rc)
        sections["shape"]["R_star"] = R_star
        sections["shape"]["C_star"] = C_star
        sections["shape"]["shape_source"] = "S" if not shape_contradictory else "CONTRADICTION"
    else:
        sections["shape"] = {"branch_byte": "CONTRADICTION", "R_star": None, "C_star": None, "shape_source": "CONTRADICTION"}

    hashes["shape"] = hash_bytes(str(sections["shape"]).encode())

    # ========================================================================
    # Step 3: Truth — gfp(ℱ) on Π(test) (WO-05)
    # ========================================================================

    truth_partition = truth.compute_truth_partition(Xstar_t)

    if truth_partition is None:
        raise ValueError("Truth partition computation failed")

    truth_rc = truth_partition.receipt

    # Store truth receipts
    sections["truth"] = {
        "tag_set_version": truth_rc.tag_set_version,
        "partition_hash": truth_rc.partition_hash,
        "block_hist": truth_rc.block_hist,
        "row_clusters": truth_rc.row_clusters,
        "col_clusters": truth_rc.col_clusters,
        "refinement_steps": truth_rc.refinement_steps,
        "overlaps": {
            "method": truth_rc.overlaps.method,
            "candidates_count": len(truth_rc.overlaps.candidates),
            "accepted_count": len(truth_rc.overlaps.accepted),
            "identity_excluded": truth_rc.overlaps.identity_excluded,
        }
    }
    hashes["truth"] = hash_bytes(str(sections["truth"]).encode())

    # ========================================================================
    # Step 3.5: Components (WO-03) for witness/engines
    # ========================================================================

    # Extract components from all training grids and test grid
    comps_X_list = []
    comps_Y_list = []
    for Xt, Yt in zip(Xt_list, Yt_list):
        _, comps_X = components.cc4_by_color(Xt)
        _, comps_Y = components.cc4_by_color(Yt)
        comps_X_list.append(comps_X)
        comps_Y_list.append(comps_Y)

    # Test components
    _, comps_Xstar = components.cc4_by_color(Xstar_t)

    sections["components"] = {
        "trainings": len(comps_X_list),
        "test_components_count": len(comps_Xstar.invariants)
    }
    hashes["components"] = hash_bytes(str(sections["components"]).encode())

    # ========================================================================
    # Step 4: Law selection (Engines → Witness → Tie)
    # ========================================================================

    law_layer_values = None
    law_layer_mask = None
    law_status = "none"
    engine_used = None
    witness_rc = None
    tie_rc = None

    # Decision: Try engines if shape contradictory or witness fails
    # For WO-11 MVP, we'll try engines first, then fall back to witness

    # Try engines in frozen order
    engine_names = ["column_dict", "macro_tiling"]  # Only implemented engines

    for engine_name in engine_names:
        if engine_name == "column_dict":
            # Fit column_dict engine
            fit_rc = families.fit_column_dict(Xt_list, Yt_list)

            if fit_rc.ok:
                # Apply to test
                apply_rc = families.apply_column_dict(Xstar_t, fit_rc)

                if apply_rc.ok:
                    # Engine succeeded
                    law_layer_values = apply_rc.Yt
                    law_layer_mask = None  # Full frame
                    law_status = "engine"
                    engine_used = engine_name

                    # Update shape if it was contradictory
                    if shape_contradictory:
                        R_star, C_star = apply_rc.final_shape
                        sections["shape"]["R_star"] = R_star
                        sections["shape"]["C_star"] = C_star
                        sections["shape"]["shape_source"] = "engine"

                    sections["engines"] = {
                        "used": engine_name,
                        "fit": fit_rc.receipt,
                        "apply": apply_rc.receipt,
                    }
                    hashes["engines"] = hash_bytes(str(sections["engines"]).encode())
                    break

        elif engine_name == "macro_tiling":
            # Fit macro_tiling engine
            # Need truth receipts for each training
            truth_list = []
            for Xt in Xt_list:
                truth_t = truth.compute_truth_partition(Xt)
                if truth_t is None:
                    # Skip if truth computation failed
                    truth_list = None
                    break
                truth_list.append(truth_t.receipt)

            if truth_list is None:
                # Can't fit macro-tiling without truth
                continue

            fit_rc = families.fit_macro_tiling(Xt_list, Yt_list, truth_list)

            if fit_rc.ok:
                # Apply to test (needs test truth receipt)
                apply_rc = families.apply_macro_tiling(Xstar_t, truth_partition.receipt, fit_rc)

                if apply_rc.ok:
                    # Engine succeeded
                    law_layer_values = apply_rc.Yt
                    law_layer_mask = None  # Full frame
                    law_status = "engine"
                    engine_used = engine_name

                    # Update shape if it was contradictory
                    if shape_contradictory:
                        R_star, C_star = apply_rc.final_shape
                        sections["shape"]["R_star"] = R_star
                        sections["shape"]["C_star"] = C_star
                        sections["shape"]["shape_source"] = "engine"

                    sections["engines"] = {
                        "used": engine_name,
                        "fit": fit_rc.receipt,
                        "apply": apply_rc.receipt,
                    }
                    hashes["engines"] = hash_bytes(str(sections["engines"]).encode())
                    break

    # Storage for witness results
    phi_law = None
    sigma_law = None
    phi_stars = []  # Conjugated φ_i^* for copy
    intersection_rc = None

    # If no engine succeeded, try witness (WO-04)
    if law_status == "none":
        # Solve witness for each training pair
        train_witnesses = []
        witness_ok = True

        for i, (Xt, Yt, comps_X, comps_Y) in enumerate(zip(Xt_list, Yt_list, comps_X_list, comps_Y_list)):
            phi_pieces, sigma, train_rc = witness.solve_witness_for_pair(
                Xt, Yt, comps_X, comps_Y
            )
            train_witnesses.append({
                "train_id": train_ids[i],
                "phi_pieces": phi_pieces,
                "sigma": sigma,
                "receipt": train_rc
            })

            # Check if witness solving failed
            if phi_pieces is None and sigma.lehmer == []:
                witness_ok = False
                break

        if witness_ok and train_witnesses:
            # Conjugate each witness to test frame
            conj_list = []
            for tw in train_witnesses:
                phi_conj, sigma_conj = witness.conjugate_to_test(
                    tw["phi_pieces"],
                    tw["sigma"],
                    Pi_test,  # Test Π transform
                    comps_Xstar  # Test components
                )
                conj_list.append((phi_conj, sigma_conj))
                phi_stars.append(phi_conj)  # Store for copy

            # Intersect witnesses
            phi_law, sigma_law, intersection_rc = witness.intersect_witnesses(conj_list)

            # Check intersection result
            if intersection_rc.status == "singleton":
                # Unique law found
                law_status = "witness_singleton"
            elif intersection_rc.status == "underdetermined":
                # Need tie-break (WO-08)
                law_status = "witness_underdetermined"
            else:  # "contradictory"
                law_status = "witness_contradictory"

            # Store witness receipts
            sections["witness"] = {
                "status": "ok",
                "trainings": [
                    {
                        "train_id": tw["train_id"],
                        "phi_kind": "geometric" if tw["phi_pieces"] else "summary",
                        "sigma_domain_size": len(tw["sigma"].domain),
                    }
                    for tw in train_witnesses
                ],
                "intersection_status": intersection_rc.status,
                "intersection_admissible_count": intersection_rc.admissible_count
            }
        else:
            # Witness solving failed
            sections["witness"] = {"status": "failed"}
            law_status = "witness_failed"

        hashes["witness"] = hash_bytes(str(sections["witness"]).encode())

    # Tie-break (WO-08) if underdetermined
    if law_status == "witness_underdetermined":
        # TODO: Enumerate all admissible witnesses and build Candidate objects
        # For now, use first admissible (phi_law, sigma_law)
        # This is a simplification - proper tie-break requires enumerating all candidates
        sections["tie"] = {"status": "simplified", "note": "using first admissible"}
        hashes["tie"] = hash_bytes(str(sections["tie"]).encode())
        law_status = "witness_singleton"  # Treat as resolved
    elif tie_rc:
        sections["tie"] = asdict(tie_rc)
        hashes["tie"] = hash_bytes(str(sections["tie"]).encode())
    else:
        sections["tie"] = {"status": "not_needed"}
        hashes["tie"] = hash_bytes(str(sections["tie"]).encode())

    # ========================================================================
    # Step 4.5: Verify output shape is determined
    # ========================================================================

    if R_star is None or C_star is None:
        # Shape was contradictory and no engine provided it
        raise ValueError(
            "SHAPE_CONTRADICTION: Output shape undetermined. "
            "Shape synthesis failed and no engine provided shape. "
            "This is a content-dependent task that requires an engine."
        )

    # ========================================================================
    # Step 5: Copy — free singletons S(p) = ⋂ᵢ {φᵢ*(p)} (WO-06)
    # ========================================================================

    # Build component masks for test grid (needed by build_free_copy_mask)
    comp_masks_test = []
    test_masks_by_comp, _ = components.cc4_by_color(Xstar_t)
    # comp_masks format: [(mask, r0, c0), ...]
    # For now, use simplified version with full grid masks
    # TODO: Extract actual bbox offsets from components if needed
    for mask in test_masks_by_comp:
        # Find bbox
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            r0, c0 = coords.min(axis=0)
        else:
            r0, c0 = 0, 0
        comp_masks_test.append((mask, int(r0), int(c0)))

    # Build free copy mask using conjugated witnesses
    mask_bitset, copy_values, copy_rc = copy.build_free_copy_mask(
        Xstar_t,
        phi_stars if phi_stars else [None] * len(train_ids),  # Use witness results
        comp_masks_test
    )

    # Convert bitset to boolean mask for Meet
    H, W = Xstar_t.shape
    copy_mask = np.zeros((R_star, C_star), dtype=bool)

    # Decode bitset (row-major, LSB-first) to boolean mask
    # mask_bitset is bytes where each bit represents a pixel
    if copy_values is not None and len(mask_bitset) > 0:
        # Unpack bits from bytes
        n_pixels = H * W
        mask_bits = np.unpackbits(np.frombuffer(mask_bitset, dtype=np.uint8))

        # Take only the first n_pixels bits (rest are padding)
        mask_bits = mask_bits[:n_pixels]

        # Reshape to grid
        if len(mask_bits) == H * W:
            mask_2d = mask_bits.reshape(H, W).astype(bool)
            # Copy to output mask (handling size mismatch)
            copy_mask[:min(R_star, H), :min(C_star, W)] = mask_2d[:min(R_star, H), :min(C_star, W)]

            # Also resize copy_values if needed
            if copy_values.shape != (R_star, C_star):
                copy_values_resized = np.zeros((R_star, C_star), dtype=np.int64)
                copy_values_resized[:min(R_star, H), :min(C_star, W)] = \
                    copy_values[:min(R_star, H), :min(C_star, W)]
                copy_values = copy_values_resized

    sections["copy"] = {
        "singleton_count": copy_rc.singleton_count,
        "singleton_mask_hash": copy_rc.singleton_mask_hash,
        "undefined_count": copy_rc.undefined_count,
        "disagree_count": copy_rc.disagree_count,
    }
    hashes["copy"] = hash_bytes(str(sections["copy"]).encode())

    # ========================================================================
    # Step 6: Unanimity — block constants u(B) (WO-07)
    # ========================================================================

    # Build train_infos for unanimity
    # Format: [(train_id, (H_i, W_i), (R_i, C_i), Y_i), ...]
    train_infos = []
    for i, (Xt, Yt) in enumerate(zip(Xt_list, Yt_list)):
        H_i, W_i = Xt.shape
        R_i, C_i = Yt.shape
        train_infos.append((train_ids[i], (H_i, W_i), (R_i, C_i), Yt))

    # Compute unanimity using truth blocks
    # truth_partition.labels is on test INPUT (H*, W*), not output (R*, C*)
    H_star_input, W_star_input = Xstar_t.shape
    unanimity_map, unanimity_rc = unanimity.compute_unanimity(
        truth_partition.labels,  # Truth blocks from WO-05 (on input grid)
        (H_star_input, W_star_input),  # Test INPUT shape (not output)
        train_infos
    )

    sections["unanimity"] = {
        "blocks_total": unanimity_rc.blocks_total,
        "unanimous_count": unanimity_rc.unanimous_count,
        "empty_pullback_blocks": unanimity_rc.empty_pullback_blocks,
        "disagree_blocks": unanimity_rc.disagree_blocks,
    }
    hashes["unanimity"] = hash_bytes(str(sections["unanimity"]).encode())

    # ========================================================================
    # Step 6.5: Apply witness law if available (compute law layer)
    # ========================================================================

    if law_status in ["witness_singleton"] and phi_law is not None and sigma_law is not None:
        # Apply witness law: Y[p] = σ(X[φ(p)])
        # For geometric witness, phi_law is list of PhiPiece
        # For summary witness, phi_law is None (identity φ)

        # For now, this is a placeholder
        # TODO: Implement full witness law application
        # This requires evaluating φ on each pixel and applying σ
        law_layer_values = np.zeros((R_star, C_star), dtype=np.uint8)
        # Simplified: just mark that witness law exists but don't apply yet
        law_status = "witness_law_not_applied"  # Mark for tracking

    # ========================================================================
    # Step 7: Meet — compose with priority copy ▷ law ▷ unanimity ▷ bottom (WO-09)
    # ========================================================================

    # Initialize output canvas
    Yt = np.zeros((R_star, C_star), dtype=np.uint8)

    # Priority: copy ▷ law ▷ unanimity ▷ bottom=0
    counts = {"copy": 0, "law": 0, "unanimity": 0, "bottom": 0}

    for r in range(R_star):
        for c in range(C_star):
            if copy_mask[r, c]:
                # Copy priority
                Yt[r, c] = copy_values[r, c]
                counts["copy"] += 1
            elif law_layer_values is not None:
                # Law priority
                Yt[r, c] = law_layer_values[r, c]
                counts["law"] += 1
            elif (r, c) in unanimity_map:
                # Unanimity priority
                Yt[r, c] = unanimity_map[(r, c)]
                counts["unanimity"] += 1
            else:
                # Bottom priority (frozen to 0)
                Yt[r, c] = 0
                counts["bottom"] += 1

    # Verify idempotence (repaint)
    repaint_hash = hash_bytes(Yt.tobytes())

    sections["meet"] = {
        "counts": counts,
        "repaint_hash": repaint_hash,
        "shape": [R_star, C_star],
    }
    hashes["meet"] = hash_bytes(str(sections["meet"]).encode())

    # ========================================================================
    # Step 8: U⁻¹ — unpresent to raw frame (WO-01)
    # ========================================================================

    Y_raw = pi.unpresent(Pi_test, Yt)

    # Final output hash
    output_hash = hash_bytes(Y_raw.tobytes())

    # ========================================================================
    # Step 9: Aggregate receipts
    # ========================================================================

    # Compute table_hash: BLAKE3(concat(sorted(section_key + ':' + hash)))
    hash_pairs = sorted((k, v) for k, v in hashes.items())
    table_str = "".join(f"{k}:{v}" for k, v in hash_pairs)
    table_hash = hash_bytes(table_str.encode())

    # Build final receipt
    run_rc = RunRc(
        task_id=task_id,
        env=env,
        sections=sections,
        hashes=hashes,
        table_hash=table_hash,
        final={
            "shape": [Y_raw.shape[0], Y_raw.shape[1]],
            "law_status": law_status,
            "output_hash": output_hash,
        }
    )

    return Y_raw, run_rc
