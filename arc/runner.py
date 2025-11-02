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

from arc.op import pi, shape, truth, witness, copy, unanimity, meet, families, components, tiebreak, border_scalar, pooled_blocks, markers_grid, slice_stack, kronecker
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

    # WO-02Z: Check shape status
    shape_ok = (shape_rc.status == "OK")
    # WO-04H/WO-11D: Define shape_contradictory for backward compatibility with engines
    shape_contradictory = not shape_ok

    # Apply S to test (on Π-presented sizes) if available
    H_star, W_star = Xstar_t.shape

    if shape_ok and S_fn is not None:
        R_star, C_star = S_fn(H_star, W_star)

        # Verify shape is positive
        if R_star <= 0 or C_star <= 0:
            raise ValueError(f"Shape S produced non-positive dimensions: ({R_star}, {C_star})")
    else:
        # WO-02Z: Shape status="NONE" - defer to engines
        # Engines will compute (R, C) from content
        R_star, C_star = None, None

    # Store shape receipts (WO-02Z)
    if shape_rc:
        sections["shape"] = asdict(shape_rc)
        # Update with computed shape if available
        if shape_ok:
            sections["shape"]["R"] = R_star
            sections["shape"]["C"] = C_star
            if "extras" not in sections["shape"]:
                sections["shape"]["extras"] = {}
            sections["shape"]["extras"]["shape_source"] = "pi_inputs"  # WO-02Z: from Π
        else:
            # status="NONE": will be filled by engines
            sections["shape"]["extras"]["shape_source"] = "engine"  # WO-02Z: deferred
    else:
        # Shouldn't happen, but handle gracefully
        sections["shape"] = {"status": "NONE", "branch_byte": None, "R": None, "C": None, "extras": {"shape_source": "engine"}}

    hashes["shape"] = hash_bytes(str(sections["shape"]).encode())

    # ========================================================================
    # Step 3: Truth — gfp(ℱ) on Π(test) (WO-05)
    # ========================================================================

    truth_partition = truth.compute_truth_partition(Xstar_t)

    if truth_partition is None:
        raise ValueError("Truth partition computation failed")

    truth_rc = truth_partition.receipt

    # WO-05T: Refine truth using training signatures
    # Build training data for refinement (need shapes for pullback)
    train_refine_infos = []
    for i, (Xt, Yt) in enumerate(zip(Xt_list, Yt_list)):
        H_i, W_i = Xt.shape
        R_i, C_i = Yt.shape
        train_refine_infos.append(((H_i, W_i), (R_i, C_i), Yt))

    refined_labels, refine_rc = truth.refine_truth_with_training(
        truth_partition.labels,
        Xstar_t.shape,
        train_refine_infos
    )

    # Update truth partition with refined labels
    # Recompute partition hash and histogram for refined partition
    refined_partition_hash = truth._partition_hash(refined_labels)
    uniq, counts = np.unique(refined_labels, return_counts=True)
    refined_block_hist = counts.tolist()

    # Create updated receipt with refined data
    from dataclasses import replace
    refined_truth_rc = replace(
        truth_rc,
        partition_hash=refined_partition_hash,
        block_hist=refined_block_hist
    )

    # Store truth receipts (including refinement)
    sections["truth"] = {
        "tag_set_version": refined_truth_rc.tag_set_version,
        "partition_hash": refined_truth_rc.partition_hash,
        "block_hist": refined_truth_rc.block_hist,
        "row_clusters": refined_truth_rc.row_clusters,
        "col_clusters": refined_truth_rc.col_clusters,
        "refinement_steps": refined_truth_rc.refinement_steps,
        "overlaps": {
            "method": refined_truth_rc.overlaps.method,
            "candidates_count": len(refined_truth_rc.overlaps.candidates),
            "accepted_count": len(refined_truth_rc.overlaps.accepted),
            "identity_excluded": refined_truth_rc.overlaps.identity_excluded,
        },
        "training_refinement": refine_rc  # WO-05T receipt
    }
    hashes["truth"] = hash_bytes(str(sections["truth"]).encode())

    # Create new TruthPartition with refined labels (frozen dataclass)
    truth_partition = truth.TruthPartition(
        labels=refined_labels,
        receipt=refined_truth_rc
    )

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

    # Try engines in frozen order (WO-11.md:66)
    # Frozen spec: ["border_scalar", "window_dict.column", "macro_tiling", "pooled_blocks", "markers_grid", "slice_stack", "kronecker"]
    # Note: "window_dict.column" implemented as "column_dict" per WO-10
    engine_names = ["border_scalar", "column_dict", "macro_tiling", "pooled_blocks", "markers_grid", "slice_stack", "kronecker"]

    for engine_name in engine_names:
        if engine_name == "border_scalar":
            # Fit border_scalar engine
            train_pairs_for_fit = [(train_ids[i], Xt, Yt) for i, (Xt, Yt) in enumerate(zip(Xt_list, Yt_list))]
            ok, fit_rc = border_scalar.fit_border_scalar(train_pairs_for_fit)

            if ok:
                # Apply to test
                Yt_border, apply_rc = border_scalar.apply_border_scalar(
                    Xstar_t,
                    fit_rc,
                    expected_shape=(R_star, C_star) if not shape_contradictory else None
                )

                # Engine succeeded
                law_layer_values = Yt_border
                law_layer_mask = None  # Full frame
                law_status = "engine"
                engine_used = engine_name

                # Update shape if it was contradictory
                if shape_contradictory:
                    R_star, C_star = apply_rc["output_shape"]
                    sections["shape"]["R_star"] = R_star
                    sections["shape"]["C_star"] = C_star
                    sections["shape"]["shape_source"] = "engine"

                sections["engines"] = {
                    "used": engine_name,
                    "fit": {
                        "border_color": fit_rc.border_color,
                        "interior_color": fit_rc.interior_color,
                        "rule": fit_rc.rule,
                        "fit_verified_on": fit_rc.fit_verified_on,
                        "hash": fit_rc.hash
                    },
                    "apply": apply_rc,
                }
                hashes["engines"] = hash_bytes(str(sections["engines"]).encode())
                break

        elif engine_name == "pooled_blocks":
            # Fit pooled_blocks engine
            # Need truth for each training input
            train_truth_list = []
            for Xt in Xt_list:
                truth_t = truth.compute_truth_partition(Xt)
                if truth_t is None:
                    train_truth_list = None
                    break
                train_truth_list.append(truth_t.receipt)

            if train_truth_list is not None:
                train_pairs_for_fit = [
                    (train_ids[i], Xt, Yt, truth_rc)
                    for i, (Xt, Yt, truth_rc) in enumerate(zip(Xt_list, Yt_list, train_truth_list))
                ]
                ok, fit_rc = pooled_blocks.fit_pooled_blocks(train_pairs_for_fit)

                if ok:
                    # Apply to test
                    Yt_pooled, apply_rc = pooled_blocks.apply_pooled_blocks(
                        Xstar_t,
                        truth_partition.receipt,
                        fit_rc,
                        expected_shape=(R_star, C_star) if not shape_contradictory else None
                    )

                    if not apply_rc.get("shape_mismatch", False):
                        # Engine succeeded
                        law_layer_values = Yt_pooled
                        law_layer_mask = None  # Full frame
                        law_status = "engine"
                        engine_used = engine_name

                        # Update shape if it was contradictory
                        if shape_contradictory:
                            R_star, C_star = apply_rc["output_shape"]
                            sections["shape"]["R_star"] = R_star
                            sections["shape"]["C_star"] = C_star
                            sections["shape"]["shape_source"] = "engine"

                        sections["engines"] = {
                            "used": engine_name,
                            "fit": {
                                "row_bands": fit_rc.row_bands,
                                "col_bands": fit_rc.col_bands,
                                "block_shape": list(fit_rc.block_shape),
                                "pool_shape": list(fit_rc.pool_shape),
                                "foreground_colors": fit_rc.foreground_colors,
                                "decision_rule": fit_rc.decision_rule,
                                "fit_verified_on": fit_rc.fit_verified_on,
                                "hash": fit_rc.hash
                            },
                            "apply": apply_rc,
                        }
                        hashes["engines"] = hash_bytes(str(sections["engines"]).encode())
                        break

        elif engine_name == "markers_grid":
            # Fit markers_grid engine
            # Need truth for each training input
            train_truth_list = []
            for Xt in Xt_list:
                truth_t = truth.compute_truth_partition(Xt)
                if truth_t is None:
                    train_truth_list = None
                    break
                train_truth_list.append(truth_t.receipt)

            if train_truth_list is not None:
                train_pairs_for_fit = [
                    (train_ids[i], Xt, Yt, truth_rc)
                    for i, (Xt, Yt, truth_rc) in enumerate(zip(Xt_list, Yt_list, train_truth_list))
                ]
                ok, fit_rc = markers_grid.fit_markers_grid(train_pairs_for_fit)

                if ok:
                    # Apply to test
                    Yt_markers, apply_rc = markers_grid.apply_markers_grid(
                        Xstar_t,
                        truth_partition.receipt,
                        fit_rc,
                        expected_shape=(R_star, C_star) if not shape_contradictory else None
                    )

                    if not apply_rc.get("error") and not apply_rc.get("shape_mismatch", False):
                        # Engine succeeded
                        law_layer_values = Yt_markers
                        law_layer_mask = None  # Full frame
                        law_status = "engine"
                        engine_used = engine_name

                        # Update shape if it was contradictory
                        if shape_contradictory:
                            R_star, C_star = apply_rc["output_shape"]
                            sections["shape"]["R_star"] = R_star
                            sections["shape"]["C_star"] = C_star
                            sections["shape"]["shape_source"] = "engine"

                        sections["engines"] = {
                            "used": engine_name,
                            "fit": {
                                "marker_size": list(fit_rc.marker_size),
                                "marker_color_set": fit_rc.marker_color_set,
                                "grid_shape": list(fit_rc.grid_shape),
                                "cell_rule": fit_rc.cell_rule,
                                "fit_verified_on": fit_rc.fit_verified_on,
                                "hash": fit_rc.hash
                            },
                            "apply": apply_rc,
                        }
                        hashes["engines"] = hash_bytes(str(sections["engines"]).encode())
                        break

        elif engine_name == "slice_stack":
            # Fit slice_stack engine
            # Need truth for each training input
            train_truth_list = []
            for Xt in Xt_list:
                truth_t = truth.compute_truth_partition(Xt)
                if truth_t is None:
                    train_truth_list = None
                    break
                train_truth_list.append(truth_t.receipt)

            if train_truth_list is not None:
                train_pairs_for_fit = [
                    (train_ids[i], Xt, Yt, truth_rc)
                    for i, (Xt, Yt, truth_rc) in enumerate(zip(Xt_list, Yt_list, train_truth_list))
                ]
                ok, fit_rc = slice_stack.fit_slice_stack(train_pairs_for_fit)

                if ok:
                    # Apply to test
                    Yt_slices, apply_rc = slice_stack.apply_slice_stack(
                        Xstar_t,
                        truth_partition.receipt,
                        fit_rc,
                        expected_shape=(R_star, C_star) if not shape_contradictory else None
                    )

                    if not apply_rc.get("error") and not apply_rc.get("shape_mismatch", False):
                        # Engine succeeded
                        law_layer_values = Yt_slices
                        law_layer_mask = None  # Full frame
                        law_status = "engine"
                        engine_used = engine_name

                        # Update shape if it was contradictory
                        if shape_contradictory:
                            R_star, C_star = apply_rc["output_shape"]
                            sections["shape"]["R_star"] = R_star
                            sections["shape"]["C_star"] = C_star
                            sections["shape"]["shape_source"] = "engine"

                        sections["engines"] = {
                            "used": engine_name,
                            "fit": {
                                "axis": fit_rc.axis,
                                "slice_height": fit_rc.slice_height,
                                "slice_width": fit_rc.slice_width,
                                "dict_size": len(fit_rc.dict),
                                "decision_rule": fit_rc.decision_rule,
                                "fit_verified_on": fit_rc.fit_verified_on,
                                "hash": fit_rc.hash
                            },
                            "apply": apply_rc,
                        }
                        hashes["engines"] = hash_bytes(str(sections["engines"]).encode())
                        break

        elif engine_name == "kronecker":
            # Fit kronecker engine
            # Kronecker doesn't need truth, just trainings
            train_pairs_for_fit = [
                (train_ids[i], Xt, Yt, None)
                for i, (Xt, Yt) in enumerate(zip(Xt_list, Yt_list))
            ]
            ok, fit_rc = kronecker.fit_kronecker(train_pairs_for_fit)

            if ok:
                # Apply to test
                Yt_kronecker, apply_rc = kronecker.apply_kronecker(
                    Xstar_t,
                    None,
                    fit_rc,
                    expected_shape=(R_star, C_star) if not shape_contradictory else None
                )

                if not apply_rc.get("error"):
                    # Engine succeeded
                    law_layer_values = Yt_kronecker
                    law_layer_mask = None  # Full frame
                    law_status = "engine"
                    engine_used = engine_name

                    # Update shape if it was contradictory
                    if shape_contradictory:
                        R_star, C_star = apply_rc["output_shape"]
                        sections["shape"]["R_star"] = R_star
                        sections["shape"]["C_star"] = C_star
                        sections["shape"]["shape_source"] = "engine"

                    sections["engines"] = {
                        "used": engine_name,
                        "fit": {
                            "tile_shape": list(fit_rc.tile_shape),
                            "reps": {k: list(v) for k, v in fit_rc.reps.items()},
                            "fit_verified_on": fit_rc.fit_verified_on,
                            "hash": fit_rc.hash
                        },
                        "apply": apply_rc,
                    }
                    hashes["engines"] = hash_bytes(str(sections["engines"]).encode())
                    break

        elif engine_name == "column_dict":
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
                # Extract domain_pixels from original witness receipt
                domain_pixels = tw["receipt"].phi.domain_pixels if tw["receipt"].phi else 0

                # Conjugate witness from training frame to test frame (WO-04C.1)
                # Since present_all applies same Π to all grids, Pi_train = Pi_test
                phi_conj, sigma_conj = witness.conjugate_to_test(
                    tw["phi_pieces"],
                    tw["sigma"],
                    Pi_test,  # Pi_train (same Π used for all grids)
                    Pi_test,  # Pi_test
                    domain_pixels=domain_pixels
                )
                conj_list.append((phi_conj, sigma_conj))
                phi_stars.append(sigma_conj.phi_star)  # WO-04H: Store PhiRc, not list

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
        # Build Candidate from current phi_law and sigma_law
        from arc.op.tiebreak import Candidate, resolve

        # Serialize phi_law to bytes
        if phi_law is None:
            phi_bytes = b""
            anchor_displacements = []
            residue_list = []
            pose_classes = []
        else:
            # Serialize phi pieces deterministically
            phi_strs = []
            anchor_displacements = []
            residue_list = []
            pose_classes = []

            for piece in phi_law:
                phi_strs.append(f"{piece.comp_id},{piece.pose_id},{piece.dr},{piece.dc},{piece.r_per},{piece.c_per},{piece.r_res},{piece.c_res}")
                anchor_displacements.append((piece.dr, piece.dc))
                residue_list.append((piece.r_per, piece.c_per, piece.r_res, piece.c_res))

                # Classify pose: REF=0 < ROT=1 < TRANS=2
                # TRANS if has translation, REF if identity or reflection, ROT if pure rotation
                has_translation = (piece.dr != 0 or piece.dc != 0)
                if has_translation:
                    pose_classes.append("TRANS")
                elif piece.pose_id in [1, 2, 3]:  # Pure rotations
                    pose_classes.append("ROT")
                else:  # Identity (0) or reflections (4,5,6,7)
                    pose_classes.append("REF")

            phi_bytes = "|".join(phi_strs).encode()

        # Extract sigma info
        sigma_domain_colors = sigma_law.domain_colors if sigma_law else []
        sigma_lehmer = sigma_law.lehmer if sigma_law else []

        # Count components in test input
        test_masks_by_comp, _ = components.cc4_by_color(Xstar_t)
        component_count_before = len(test_masks_by_comp)
        component_count_after = component_count_before  # Neutral assumption (can't compute without applying law)

        # Build Candidate
        candidate = Candidate(
            phi_bytes=phi_bytes,
            sigma_domain_colors=sigma_domain_colors,
            sigma_lehmer=sigma_lehmer,
            anchor_displacements=anchor_displacements,
            component_count_before=component_count_before,
            component_count_after=component_count_after,
            residue_list=residue_list,
            pose_classes=pose_classes,
            placement_refs=None,  # Not used for witness tie-break
            skyline_keys=None,    # Not used for witness tie-break
            meta={"H": R_star, "W": C_star}  # Store shape for potential use
        )

        # Call tiebreak.resolve() with single candidate to get proper receipts
        chosen_idx, tie_rc = resolve([candidate], tie_context="none")

        # Store tie-break receipt
        sections["tie"] = asdict(tie_rc)
        hashes["tie"] = hash_bytes(str(sections["tie"]).encode())

        # Mark as resolved
        law_status = "witness_singleton"
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
        # WO-02Z: Shape status="NONE" and no engine provided it
        # This error only fires when BOTH S and engines fail
        raise ValueError(
            "SHAPE_CONTRADICTION: Output shape undetermined. "
            f"Shape synthesis returned status={shape_rc.status} and no engine provided shape. "
            "This is a content-dependent task that requires an engine. "
            f"Attempted families: {shape_rc.attempts if shape_rc.attempts else 'none'}"
        )

    # ========================================================================
    # Step 5: Copy — free singletons S(p) = ⋂ᵢ {φᵢ*(p)} (WO-06)
    # ========================================================================

    # Build component masks for test grid (needed by build_free_copy_mask)
    # comp_masks format: [(mask, r0, c0), ...] where (r0, c0) is bbox top-left
    comp_masks_test = []
    test_masks_by_comp, _ = components.cc4_by_color(Xstar_t)

    for mask in test_masks_by_comp:
        # Extract bbox top-left corner
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

    if law_status in ["witness_singleton"] and sigma_law is not None:
        # Apply witness law: Y[p] = σ(X[φ(p)])
        # phi_law can be:
        # - List[PhiPiece] for geometric witness (component-wise transforms)
        # - None for summary witness (identity φ, just apply σ globally)

        from arc.op.copy import _eval_phi_star_at_pixel
        from arc.op.tiebreak import lehmer_to_perm

        # Convert Lehmer code to permutation array
        # sigma_perm[c] = σ(c) for color c
        sigma_perm = lehmer_to_perm(sigma_law.lehmer) if sigma_law.lehmer else []

        # Create permutation lookup (handle missing colors with identity)
        max_color = max(sigma_law.domain_colors) if sigma_law.domain_colors else 9
        sigma_lookup = list(range(max_color + 1))  # Identity by default
        for i, c in enumerate(sigma_law.domain_colors):
            if i < len(sigma_perm):
                sigma_lookup[c] = sigma_law.domain_colors[sigma_perm[i]]

        law_layer_values = np.zeros((R_star, C_star), dtype=np.uint8)
        law_mask = np.zeros((R_star, C_star), dtype=bool)

        if phi_law is None:
            # Summary witness: identity φ, apply σ globally
            for r in range(R_star):
                for c in range(C_star):
                    source_color = Xstar_t[r, c]
                    if source_color < len(sigma_lookup):
                        law_layer_values[r, c] = sigma_lookup[source_color]
                    else:
                        law_layer_values[r, c] = source_color  # Identity fallback
                    law_mask[r, c] = True
        else:
            # Geometric witness: apply φ component-wise
            # phi_law is List[PhiPiece], comp_masks_test has component info
            from arc.op.witness import PhiRc

            # Build PhiRc for evaluation
            phi_rc = PhiRc(
                pieces=phi_law,
                bbox_equal=[True] * len(phi_law),  # Already verified
                domain_pixels=0  # Not needed for evaluation
            )

            # Evaluate φ for each output pixel
            for r in range(R_star):
                for c in range(C_star):
                    # Evaluate φ*(p) → s (source pixel)
                    source = _eval_phi_star_at_pixel((r, c), phi_rc, comp_masks_test)

                    if source is not None:
                        s_r, s_c = source
                        # Check bounds
                        if 0 <= s_r < Xstar_t.shape[0] and 0 <= s_c < Xstar_t.shape[1]:
                            source_color = Xstar_t[s_r, s_c]
                            # Apply σ
                            if source_color < len(sigma_lookup):
                                law_layer_values[r, c] = sigma_lookup[source_color]
                            else:
                                law_layer_values[r, c] = source_color  # Identity fallback
                            law_mask[r, c] = True

    # ========================================================================
    # Step 7: Meet — compose with priority copy ▷ law ▷ unanimity ▷ bottom (WO-09)
    # ========================================================================

    # Encode copy_mask as LSB-first bitset
    def _encode_bitset_lsb(mask: np.ndarray) -> bytes:
        """Encode boolean mask as LSB-first bitset."""
        flat = mask.flatten()
        size = len(flat)
        byte_count = (size + 7) // 8
        bits = []
        for byte_idx in range(byte_count):
            byte_val = 0
            for bit_offset in range(8):
                idx = byte_idx * 8 + bit_offset
                if idx < size and flat[idx]:
                    byte_val |= (1 << bit_offset)
            bits.append(byte_val)
        return bytes(bits)

    copy_mask_bits = _encode_bitset_lsb(copy_mask)

    # Encode law_mask as LSB-first bitset if it exists
    law_mask_bits = None
    if law_layer_values is not None and 'law_mask' in locals():
        # Witness law with explicit mask
        law_mask_bits = _encode_bitset_lsb(law_mask)
    # else: law_mask_bits = None means law defined everywhere (for engines) or nowhere

    # Call compose_meet from WO-09
    # Note: Xt parameter is only used for shape; all layers are on output grid (R*, C*)
    Yt_dummy = np.zeros((R_star, C_star), dtype=np.uint8)  # Output shape in Π frame

    # Unanimity: only applies when output shape == input shape
    # Contract: truth blocks are on INPUT grid; if S changes shape, blocks don't map to output pixels
    truth_blocks_for_meet = None
    block_color_map_for_meet = None
    H_star_input, W_star_input = Xstar_t.shape

    if (R_star == H_star_input) and (C_star == W_star_input):
        # Shapes match: output pixels align with input blocks
        truth_blocks_for_meet = truth_partition.labels
        block_color_map_for_meet = unanimity_map
    # else: shapes differ → unanimity doesn't apply (output pixels have no block assignment)

    Yt, meet_rc = meet.compose_meet(
        Xt=Yt_dummy,  # Dummy grid with output shape for size reference
        copy_mask_bits=copy_mask_bits,
        copy_values=copy_values,
        law_mask_bits=law_mask_bits,
        law_values=law_layer_values,
        truth_blocks=truth_blocks_for_meet,  # None if shapes differ
        block_color_map=block_color_map_for_meet,  # None if shapes differ
        bottom_color=0  # H2: frozen to 0
    )

    # Store meet receipt
    sections["meet"] = {
        "count_copy": meet_rc.count_copy,
        "count_law": meet_rc.count_law,
        "count_unanimity": meet_rc.count_unanimity,
        "count_bottom": meet_rc.count_bottom,
        "repaint_hash": meet_rc.repaint_hash,
        "copy_mask_hash": meet_rc.copy_mask_hash,
        "law_mask_hash": meet_rc.law_mask_hash,
        "uni_mask_hash": meet_rc.uni_mask_hash,
        "shape": list(meet_rc.shape),
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
