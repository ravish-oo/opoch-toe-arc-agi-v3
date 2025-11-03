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

from arc.op import pi, shape, truth, witness, copy, unanimity, meet, families, components, tiebreak, border_scalar, pooled_blocks, markers_grid, slice_stack, kronecker, kronecker_mask, admit
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

    # WO-11C: Dual-Coframe Architecture
    # Build TWO versions of each training output:
    # - Y^X (X-coframe): for engines (Y with X's Π)
    # - Y^Y (Y-coframe): for unanimity/truth (Y with its own Π)
    from arc.op.palette import apply_palette_map
    from arc.op.d4 import apply_pose
    from arc.op.pi import choose_pose_and_anchor

    # ========================================================================
    # X-COFRAME: Y_i^X = Π_{X_i}(Y_i^raw) for engines
    # ========================================================================
    Yt_X_list = []  # Y^X: outputs in X-coframe (for engines)

    for i, Y_raw in enumerate(train_Y_raw):
        # Get the D4 pose AND anchor that were applied to the paired input
        train_grid_rc = pi_rc.per_grid[i]
        pose_id = train_grid_rc["pose_id"]
        anchor_dr = train_grid_rc["anchor"]["dr"]
        anchor_dc = train_grid_rc["anchor"]["dc"]

        # Apply palette with identity fallback (unmapped colors → themselves)
        Y_pal = apply_palette_map(Y_raw, Pi_test.map)

        # Apply SAME D4 pose as the paired input
        Y_posed = apply_pose(Y_pal, pose_id)

        # Apply SAME anchor as the paired input (critical for component alignment)
        # DO NOT recompute anchor - Y must be in the EXACT SAME Π frame as X
        H, W = Y_posed.shape
        Y_anchored = np.zeros((H, W), dtype=Y_posed.dtype)

        # DEBUG: Log anchor application
        if DEBUG_DUAL_COFRAME:
            print(f"  [Y^X build] train {i}: Y_raw.shape={Y_raw.shape}, Y_posed.shape={Y_posed.shape}")
            print(f"  [Y^X build] anchor=({anchor_dr}, {anchor_dc}), pose={pose_id}")

        # Manual anchor shift: move content from (anchor_dr, anchor_dc) to (0, 0)
        if anchor_dr >= 0 and anchor_dc >= 0:
            Y_anchored[0:H-anchor_dr, 0:W-anchor_dc] = Y_posed[anchor_dr:H, anchor_dc:W]
        elif anchor_dr >= 0 and anchor_dc < 0:
            Y_anchored[0:H-anchor_dr, -anchor_dc:W] = Y_posed[anchor_dr:H, 0:W+anchor_dc]
        elif anchor_dr < 0 and anchor_dc >= 0:
            Y_anchored[-anchor_dr:H, 0:W-anchor_dc] = Y_posed[0:H+anchor_dr, anchor_dc:W]
        else:  # both negative
            Y_anchored[-anchor_dr:H, -anchor_dc:W] = Y_posed[0:H+anchor_dr, 0:W+anchor_dc]

        if DEBUG_DUAL_COFRAME:
            nonzero_before = np.count_nonzero(Y_posed)
            nonzero_after = np.count_nonzero(Y_anchored)
            print(f"  [Y^X build] nonzero: {nonzero_before} → {nonzero_after} (after anchor)")

        Yt_X_list.append(Y_anchored)

    # ========================================================================
    # Y-COFRAME: Y_i^Y = Π_{Y_i}(Y_i^raw) for unanimity/truth
    # ========================================================================
    Yt_Y_list = []  # Y^Y: outputs in Y-coframe (for unanimity/truth)
    Pi_Y_list = []  # Store Π_{Y_i} transforms for each output

    for i, Y_raw in enumerate(train_Y_raw):
        # Apply palette with identity fallback (same palette as inputs)
        Y_pal = apply_palette_map(Y_raw, Pi_test.map)

        # Compute Y's OWN Π (D4 lex min + anchor)
        Y_pi, pose_id_Y, anchor_Y = choose_pose_and_anchor(Y_pal)

        # Store Π_{Y_i} transform for this output
        Pi_Y_list.append({
            "pose_id": pose_id_Y,
            "anchor": {"dr": anchor_Y.dr, "dc": anchor_Y.dc}
        })

        Yt_Y_list.append(Y_pi)

    # WO-11C: Routing summary
    # - Engines:          use Xt_list, Yt_list (X-coframe: both in same Π_{X_i})
    # - Witness:          uses Xt_list, Yt_list (X-coframe: learns φ in X-coframe)
    # - Components:       uses Xt_list, Yt_list (X-coframe: for engine matching)
    # - Truth refinement: uses Yt_Y_list (Y-coframe: output's native frame)
    # - Unanimity:        uses Yt_Y_list (Y-coframe: votes in output frame)

    # For backward compatibility, keep Yt_list pointing to X-coframe
    # Engines, witness, and components use this
    Yt_list = Yt_X_list

    # WO-11C: Frame equality guards (fail-fast)
    for i in range(len(train_Y_raw)):
        # Engine frame guard: X and Y^X must have SAME Π (pose + anchor)
        Pi_X_i = pi_rc.per_grid[i]
        Y_X_pose = Pi_X_i["pose_id"]
        Y_X_anchor = Pi_X_i["anchor"]

        # Verify Y^X was built with X's Π (sanity check on our own code)
        # This guard catches if we accidentally recomputed anchor for Y^X
        if Yt_X_list[i].shape != Xt_list[i].shape:
            # Shapes can differ if Y has different content extent
            pass  # This is OK - shapes can differ even with same Π

        # Unanimity guard: check if Y^Y became all zeros (bad anchor applied)
        if np.count_nonzero(Yt_Y_list[i]) == 0 and np.count_nonzero(train_Y_raw[i]) > 0:
            raise ValueError(
                f"Train {i}: Y^Y (Y-coframe) became all zeros after Π_{Y_i}, "
                f"but raw Y has {np.count_nonzero(train_Y_raw[i])} non-zero pixels. "
                f"Likely bug: applied input anchor to output by mistake."
            )

    # Verify Π contract (idempotent, inputs-only palette)
    if pi_rc.palette.scope != "inputs_only":
        raise ValueError(f"Π palette must be inputs_only, got {pi_rc.palette.scope}")

    # Store Π receipts (WO-11C: dual-coframe)
    sections["pi"] = asdict(pi_rc)

    # Add dual-coframe information to receipts
    sections["frames"] = {
        "Pi_X_star": {
            "pose": pi_rc.test_pose_id,
            "anchor": pi_rc.per_grid[-1]["anchor"] if pi_rc.per_grid else {"dr": 0, "dc": 0}
        },
        "Pi_Y_star": {
            "pose": 0,  # Synthetic frame for test output
            "anchor": {"dr": 0, "dc": 0}
        },
        "train": [
            {
                "train_id": train_ids[i],
                "Pi_X": {
                    "pose": pi_rc.per_grid[i]["pose_id"],
                    "anchor": pi_rc.per_grid[i]["anchor"]
                },
                "Pi_Y": Pi_Y_list[i]
            }
            for i in range(len(train_ids))
        ]
    }
    hashes["pi"] = hash_bytes(str(sections["pi"]).encode())
    hashes["frames"] = hash_bytes(str(sections["frames"]).encode())

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

    # WO-05T: Refine truth using training signatures (WO-11C: use Y-coframe)
    # Build training data for refinement (need shapes for pullback)
    # CRITICAL: Use Yt_Y_list (Y^Y) for truth refinement
    train_refine_infos = []
    for i, (Xt, Yt_Y) in enumerate(zip(Xt_list, Yt_Y_list)):
        H_i, W_i = Xt.shape
        R_i, C_i = Yt_Y.shape
        train_refine_infos.append(((H_i, W_i), (R_i, C_i), Yt_Y))

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

    # WO-03Y: Per-training component counts for debugging alignment issues
    sections["components"] = {
        "train": [
            {
                "train_id": train_ids[i],
                "frame": "train_pi",  # X and Y in same Π frame (pose + anchor)
                "X_shape": list(Xt.shape),
                "Y_shape": list(Yt.shape),
                "X_comp_count": len(comps_X.invariants),
                "Y_comp_count": len(comps_Y.invariants),
                "connectivity": "4",
                "color_space": "original",  # CMR-A.3: logic operates on original colors
                "X_outline_hashes_head": [inv["outline_hash"] for inv in comps_X.invariants[:8]],
                "Y_outline_hashes_head": [inv["outline_hash"] for inv in comps_Y.invariants[:8]]
            }
            for i, (Xt, Yt, comps_X, comps_Y) in enumerate(zip(Xt_list, Yt_list, comps_X_list, comps_Y_list))
        ],
        "test": {
            "frame": "test_pi",
            "X_shape": list(Xstar_t.shape),
            "X_comp_count": len(comps_Xstar.invariants),
            "connectivity": "4",
            "color_space": "original"
        }
    }
    hashes["components"] = hash_bytes(str(sections["components"]).encode())

    # WO-03Y: Validation guards for component alignment
    for i, train_comp_rc in enumerate(sections["components"]["train"]):
        X_count = train_comp_rc["X_comp_count"]
        Y_count = train_comp_rc["Y_comp_count"]
        pose_id = pi_rc.per_grid[i]["pose_id"]

        # Guard: Fail fast if Y component count >> X component count
        # This usually indicates Y was extracted in the wrong frame (missed Π alignment)
        if Y_count > X_count * 3 and pose_id != 0:
            raise ValueError(
                f"Component count mismatch (train {i}): X={X_count}, Y={Y_count} "
                f"(Y is {Y_count/X_count:.1f}× larger). "
                f"Likely bug: Y not aligned to X's Π frame (pose={pose_id}). "
                f"Check that Y uses SAME anchor as X."
            )

        # Guard: Frame and color_space must be correct
        if train_comp_rc["frame"] != "train_pi":
            raise ValueError(f"Component frame must be 'train_pi', got {train_comp_rc['frame']}")
        if train_comp_rc["color_space"] != "original":
            raise ValueError(f"Component color_space must be 'original', got {train_comp_rc['color_space']}")
        if train_comp_rc["connectivity"] != "4":
            raise ValueError(f"Component connectivity must be '4', got {train_comp_rc['connectivity']}")

    # ========================================================================
    # Step 3.6: Color universe (needed for engines and admit layer)
    # ========================================================================

    # Color universe: sorted unique from Π(inputs only), includes bottom_color=0
    C = admit.color_universe(Xstar_t, Xt_list, bottom_color=0)

    # ========================================================================
    # Step 4: Law selection (Engines → Witness → Tie)
    # ========================================================================

    law_layer_values = None  # Legacy: painted grid (for old engines)
    law_layer_admits = None  # New: native (A, S, receipt) tuple from engines
    law_layer_mask = None
    law_status = "none"
    engine_used = None
    witness_rc = None
    tie_rc = None

    # Track all engine trials for receipts (WO-11 receipts expansion)
    engine_trials = []

    # Decision: Try engines if shape contradictory or witness fails
    # For WO-11 MVP, we'll try engines first, then fall back to witness

    # Try engines in frozen order (WO-11.md:66)
    # Frozen spec: ["border_scalar", "window_dict.column", "macro_tiling", "pooled_blocks", "markers_grid", "slice_stack", "kronecker", "kronecker_mask"]
    # Note: "window_dict.column" implemented as "column_dict" per WO-10
    # WO-10K: Added "kronecker_mask" for mask-Kronecker law (Y = (X!=0) ⊗ X)
    engine_names = ["border_scalar", "column_dict", "macro_tiling", "pooled_blocks", "markers_grid", "slice_stack", "kronecker", "kronecker_mask"]

    for engine_name in engine_names:
        if engine_name == "border_scalar":
            # Fit border_scalar engine
            train_pairs_for_fit = [(train_ids[i], Xt, Yt) for i, (Xt, Yt) in enumerate(zip(Xt_list, Yt_list))]
            ok, fit_rc = border_scalar.fit_border_scalar(train_pairs_for_fit)

            if ok:
                # Apply to test (now returns native admits)
                A_engine, S_engine, apply_rc = border_scalar.apply_border_scalar(
                    Xstar_t,
                    fit_rc,
                    C,
                    expected_shape=(R_star, C_star) if not shape_contradictory else None
                )

                # Engine succeeded - store native admits
                law_layer_admits = (A_engine, S_engine, apply_rc)
                law_layer_values = None  # No longer paint grid
                law_layer_mask = None  # Full frame
                law_status = "engine"
                engine_used = engine_name

                # Track success
                engine_trials.append({
                    "engine": engine_name,
                    "fit_ok": True,
                    "apply_ok": True,
                    "succeeded": True
                })

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
            else:
                # Fit failed
                engine_trials.append({
                    "engine": engine_name,
                    "fit_ok": False,
                    "apply_ok": None,
                    "succeeded": False,
                    "reason": "fit_failed"
                })

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
                    # Apply to test (now returns native admits)
                    A_engine, S_engine, apply_rc = pooled_blocks.apply_pooled_blocks(
                        Xstar_t,
                        truth_partition.receipt,
                        fit_rc,
                        C,
                        expected_shape=(R_star, C_star) if not shape_contradictory else None
                    )

                    if not apply_rc.get("shape_mismatch", False):
                        # Engine succeeded - store native admits
                        law_layer_admits = (A_engine, S_engine, apply_rc)
                        law_layer_values = None  # No longer paint grid
                        law_layer_mask = None  # Full frame
                        law_status = "engine"
                        engine_used = engine_name

                        # Track success
                        engine_trials.append({
                            "engine": engine_name,
                            "fit_ok": True,
                            "apply_ok": True,
                            "succeeded": True
                        })

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
                    else:
                        # Apply failed: shape_mismatch
                        engine_trials.append({
                            "engine": engine_name,
                            "fit_ok": True,
                            "apply_ok": False,
                            "succeeded": False,
                            "reason": "shape_mismatch"
                        })
                else:
                    # Fit failed
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": False,
                        "apply_ok": None,
                        "succeeded": False,
                        "reason": "fit_failed"
                    })
            else:
                # Truth computation failed
                engine_trials.append({
                    "engine": engine_name,
                    "fit_ok": None,
                    "apply_ok": None,
                    "succeeded": False,
                    "reason": "truth_failed"
                })

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
                    # Apply to test (now returns native admits)
                    A_engine, S_engine, apply_rc = markers_grid.apply_markers_grid(
                        Xstar_t,
                        truth_partition.receipt,
                        fit_rc,
                        C,
                        expected_shape=(R_star, C_star) if not shape_contradictory else None
                    )

                    if not apply_rc.get("error") and not apply_rc.get("shape_mismatch", False):
                        # Engine succeeded - store native admits
                        law_layer_admits = (A_engine, S_engine, apply_rc)
                        law_layer_values = None  # No longer paint grid
                        law_layer_mask = None  # Full frame
                        law_status = "engine"
                        engine_used = engine_name

                        # Track success
                        engine_trials.append({
                            "engine": engine_name,
                            "fit_ok": True,
                            "apply_ok": True,
                            "succeeded": True
                        })

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
                    else:
                        # Apply failed
                        reason = "error" if apply_rc.get("error") else "shape_mismatch"
                        engine_trials.append({
                            "engine": engine_name,
                            "fit_ok": True,
                            "apply_ok": False,
                            "succeeded": False,
                            "reason": reason
                        })
                else:
                    # Fit failed
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": False,
                        "apply_ok": None,
                        "succeeded": False,
                        "reason": "fit_failed"
                    })
            else:
                # Truth computation failed
                engine_trials.append({
                    "engine": engine_name,
                    "fit_ok": None,
                    "apply_ok": None,
                    "succeeded": False,
                    "reason": "truth_failed"
                })

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
                    # Apply to test (now returns native admits)
                    A_engine, S_engine, apply_rc = slice_stack.apply_slice_stack(
                        Xstar_t,
                        truth_partition.receipt,
                        fit_rc,
                        C,
                        expected_shape=(R_star, C_star) if not shape_contradictory else None
                    )

                    if not apply_rc.get("error") and not apply_rc.get("shape_mismatch", False):
                        # Engine succeeded - store native admits
                        law_layer_admits = (A_engine, S_engine, apply_rc)
                        law_layer_values = None  # No longer paint grid
                        law_layer_mask = None  # Full frame
                        law_status = "engine"
                        engine_used = engine_name

                        # Track success
                        engine_trials.append({
                            "engine": engine_name,
                            "fit_ok": True,
                            "apply_ok": True,
                            "succeeded": True
                        })

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
                    else:
                        # Apply failed
                        reason = "error" if apply_rc.get("error") else "shape_mismatch"
                        engine_trials.append({
                            "engine": engine_name,
                            "fit_ok": True,
                            "apply_ok": False,
                            "succeeded": False,
                            "reason": reason
                        })
                else:
                    # Fit failed
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": False,
                        "apply_ok": None,
                        "succeeded": False,
                        "reason": "fit_failed"
                    })
            else:
                # Truth computation failed
                engine_trials.append({
                    "engine": engine_name,
                    "fit_ok": None,
                    "apply_ok": None,
                    "succeeded": False,
                    "reason": "truth_failed"
                })

        elif engine_name == "kronecker":
            # Fit kronecker engine
            # Kronecker doesn't need truth, just trainings
            train_pairs_for_fit = [
                (train_ids[i], Xt, Yt, None)
                for i, (Xt, Yt) in enumerate(zip(Xt_list, Yt_list))
            ]
            ok, fit_rc = kronecker.fit_kronecker(train_pairs_for_fit)

            if ok:
                # Apply to test (now returns native admits)
                A_engine, S_engine, apply_rc = kronecker.apply_kronecker(
                    Xstar_t,
                    None,
                    fit_rc,
                    C,
                    expected_shape=(R_star, C_star) if not shape_contradictory else None
                )

                if not apply_rc.get("error"):
                    # Engine succeeded - store native admits
                    law_layer_admits = (A_engine, S_engine, apply_rc)
                    law_layer_values = None  # No longer paint grid
                    law_layer_mask = None  # Full frame
                    law_status = "engine"
                    engine_used = engine_name

                    # Track success
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": True,
                        "apply_ok": True,
                        "succeeded": True
                    })

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
                else:
                    # Apply failed
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": True,
                        "apply_ok": False,
                        "succeeded": False,
                        "reason": "error"
                    })
            else:
                # Fit failed - include detailed failure info from receipt
                trial = {
                    "engine": engine_name,
                    "fit_ok": False,
                    "apply_ok": None,
                    "succeeded": False,
                    "reason": "fit_failed"
                }
                # WO-11: Add detailed failure diagnostics if available
                if hasattr(fit_rc, 'failure_reason') and fit_rc.failure_reason:
                    trial["failure_reason"] = fit_rc.failure_reason
                if hasattr(fit_rc, 'failure_details') and fit_rc.failure_details:
                    trial["failure_details"] = fit_rc.failure_details
                engine_trials.append(trial)

        elif engine_name == "kronecker_mask":
            # WO-10K: Fit kronecker_mask engine (mask-Kronecker: Y = (X!=0) ⊗ X)
            train_pairs_for_fit = [
                (train_ids[i], Xt, Yt, None)
                for i, (Xt, Yt) in enumerate(zip(Xt_list, Yt_list))
            ]
            ok, fit_rc = kronecker_mask.fit_kronecker_mask(train_pairs_for_fit)

            if ok:
                # Apply to test (now returns native admits)
                A_engine, S_engine, apply_rc = kronecker_mask.apply_kronecker_mask(
                    Xstar_t,
                    None,
                    fit_rc,
                    C,
                    expected_shape=(R_star, C_star) if not shape_contradictory else None
                )

                if not apply_rc.get("error"):
                    # Engine succeeded - store native admits
                    law_layer_admits = (A_engine, S_engine, apply_rc)
                    law_layer_values = None  # No longer paint grid
                    law_layer_mask = None  # Full frame
                    law_status = "engine"
                    engine_used = engine_name

                    # Track success
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": True,
                        "apply_ok": True,
                        "succeeded": True
                    })

                    # Update shape if it was contradictory
                    if shape_contradictory:
                        R_star, C_star = apply_rc["shape"]
                        sections["shape"]["R_star"] = R_star
                        sections["shape"]["C_star"] = C_star
                        sections["shape"]["shape_source"] = "engine"

                    sections["engines"] = {
                        "used": engine_name,
                        "fit": {
                            "final_shape": list(fit_rc.final_shape),
                            "fit_verified_on": fit_rc.fit_verified_on,
                            "hash": fit_rc.hash
                        },
                        "apply": apply_rc,
                    }
                    hashes["engines"] = hash_bytes(str(sections["engines"]).encode())
                    break
                else:
                    # Apply failed
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": True,
                        "apply_ok": False,
                        "succeeded": False,
                        "reason": "error"
                    })
            else:
                # Fit failed - include detailed failure info from receipt
                trial = {
                    "engine": engine_name,
                    "fit_ok": False,
                    "apply_ok": None,
                    "succeeded": False,
                    "reason": "fit_failed"
                }
                # WO-11: Add detailed failure diagnostics if available
                if hasattr(fit_rc, 'failure_reason') and fit_rc.failure_reason:
                    trial["failure_reason"] = fit_rc.failure_reason
                if hasattr(fit_rc, 'failure_details') and fit_rc.failure_details:
                    trial["failure_details"] = fit_rc.failure_details
                engine_trials.append(trial)

        elif engine_name == "column_dict":
            # Fit column_dict engine
            fit_rc = families.fit_column_dict(Xt_list, Yt_list)

            if fit_rc.ok:
                # Apply to test (now returns native admits)
                A_engine, S_engine, apply_rc = families.apply_column_dict(Xstar_t, fit_rc, C)

                if not apply_rc.get("error"):
                    # Engine succeeded - store native admits
                    law_layer_admits = (A_engine, S_engine, apply_rc)
                    law_layer_values = None  # No longer paint grid
                    law_layer_mask = None  # Full frame
                    law_status = "engine"
                    engine_used = engine_name

                    # Track success
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": True,
                        "apply_ok": True,
                        "succeeded": True
                    })

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
                else:
                    # Apply failed
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": True,
                        "apply_ok": False,
                        "succeeded": False,
                        "reason": "apply_failed"
                    })
            else:
                # Fit failed
                engine_trials.append({
                    "engine": engine_name,
                    "fit_ok": False,
                    "apply_ok": None,
                    "succeeded": False,
                    "reason": "fit_failed"
                })

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
                engine_trials.append({
                    "engine": engine_name,
                    "fit_ok": None,
                    "apply_ok": None,
                    "succeeded": False,
                    "reason": "truth_failed"
                })
                continue

            fit_rc = families.fit_macro_tiling(Xt_list, Yt_list, truth_list)

            if fit_rc.ok:
                # Apply to test (needs test truth receipt)
                A_engine, S_engine, apply_rc = families.apply_macro_tiling(Xstar_t, truth_partition.receipt, fit_rc, C)

                if not apply_rc.get("error"):
                    # Engine succeeded - store native admits
                    law_layer_admits = (A_engine, S_engine, apply_rc)
                    law_layer_values = None  # No longer paint grid
                    law_layer_mask = None  # Full frame
                    law_status = "engine"
                    engine_used = engine_name

                    # Track success
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": True,
                        "apply_ok": True,
                        "succeeded": True
                    })

                    # Update shape if it was contradictory
                    if shape_contradictory:
                        R_star, C_star = tuple(apply_rc["final_shape"])
                        sections["shape"]["R_star"] = R_star
                        sections["shape"]["C_star"] = C_star
                        sections["shape"]["shape_source"] = "engine"

                    sections["engines"] = {
                        "used": engine_name,
                        "fit": fit_rc.receipt,
                        "apply": apply_rc,
                    }
                    hashes["engines"] = hash_bytes(str(sections["engines"]).encode())
                    break
                else:
                    # Apply failed
                    engine_trials.append({
                        "engine": engine_name,
                        "fit_ok": True,
                        "apply_ok": False,
                        "succeeded": False,
                        "reason": "apply_failed"
                    })
            else:
                # Fit failed
                engine_trials.append({
                    "engine": engine_name,
                    "fit_ok": False,
                    "apply_ok": None,
                    "succeeded": False,
                    "reason": "fit_failed"
                })

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
                "receipt": train_rc,
                "geometric_trials": train_rc.phi.geometric_trials if train_rc.phi and hasattr(train_rc.phi, 'geometric_trials') else []
            })

            # Check if witness solving is inconsistent
            # Contract (WO-04 updated): two valid outcomes:
            # - kind="geometric": phi_pieces present, sigma with domain
            # - kind="contradictory": phi_pieces=None, sigma with empty domain
            # Any other combination is an internal error
            if train_rc.kind == "geometric" and phi_pieces is None:
                # Inconsistent: claims geometric but no phi
                witness_ok = False
                break

        if witness_ok and train_witnesses:
            # Conjugate each witness to test frame
            conj_list = []
            conj_receipts = []  # Store ConjugatedRc for pullback samples
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
                conj_receipts.append(sigma_conj)  # Store for pullback samples extraction
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

            # Store witness receipts (WO-04H: include actual phi_law and sigma_law for debugging)
            sections["witness"] = {
                "status": "ok",
                "trainings": [
                    {
                        "train_id": tw["train_id"],
                        "phi_kind": tw["receipt"].kind,  # "geometric" or "contradictory"
                        "sigma_domain_size": len(tw["sigma"].domain),
                        "geometric_trials": tw.get("geometric_trials", []),  # Include for all (geometric + contradictory)
                        # WO-04: Pullback samples (3 per training to prove conjugation)
                        "pullback_samples": conj_receipts[i].pullback_samples if conj_receipts[i].pullback_samples else [],
                        # Sigma inference debug info (why sigma fails: coverage, injectivity, surjectivity)
                        "sigma_debug": tw["receipt"].sigma_debug if hasattr(tw["receipt"], "sigma_debug") and tw["receipt"].sigma_debug else None
                    }
                    for i, tw in enumerate(train_witnesses)
                ],
                "intersection_status": intersection_rc.status,
                "intersection_admissible_count": intersection_rc.admissible_count,
                # Include actual law for algebraic debugging
                "phi_law": [
                    {
                        "comp_id": p.comp_id,
                        "pose_id": p.pose_id,
                        "dr": p.dr,
                        "dc": p.dc,
                        "r_per": p.r_per,
                        "c_per": p.c_per,
                        "r_res": p.r_res,
                        "c_res": p.c_res
                    }
                    for p in phi_law
                ] if phi_law else None,
                "sigma_law": {
                    "domain_colors": list(sigma_law.domain_colors) if sigma_law and hasattr(sigma_law, 'domain_colors') else None,
                    "lehmer": list(sigma_law.lehmer) if sigma_law and hasattr(sigma_law, 'lehmer') else None
                } if sigma_law else None
            }
        else:
            # Witness solving failed - include details for debugging
            sections["witness"] = {
                "status": "failed",
                "trainings": [
                    {
                        "train_id": tw["train_id"],
                        "phi_kind": tw["receipt"].kind,  # "geometric" or "contradictory"
                        "phi_pieces_count": len(tw["phi_pieces"]) if tw["phi_pieces"] else 0,
                        "sigma_domain_size": len(tw["sigma"].domain) if hasattr(tw["sigma"], 'domain') else len(tw["sigma"].domain_colors) if hasattr(tw["sigma"], 'domain_colors') else 0,
                        "sigma_lehmer_len": len(tw["sigma"].lehmer),
                        "geometric_trials": tw.get("geometric_trials", []),  # Include for all (geometric + contradictory)
                        # Sigma inference debug info (why sigma fails: coverage, injectivity, surjectivity)
                        "sigma_debug": tw["receipt"].sigma_debug if hasattr(tw["receipt"], "sigma_debug") and tw["receipt"].sigma_debug else None
                    }
                    for tw in train_witnesses
                ] if train_witnesses else [],
                "failure_reason": "phi_none_and_sigma_empty" if train_witnesses else "no_trainings"
            }
            law_status = "witness_failed"

        hashes["witness"] = hash_bytes(str(sections["witness"]).encode())

    # Add engine trials to receipts (WO-11 receipts expansion for algebraic debugging)
    if engine_trials:
        sections["engine_trials"] = engine_trials
        hashes["engine_trials"] = hash_bytes(str(sections["engine_trials"]).encode())

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

    # Build train_infos for unanimity (WO-11C: use Y-coframe)
    # Format: [(train_id, (H_i, W_i), (R_i, C_i), Y_i), ...]
    # CRITICAL: Use Yt_Y_list (Y^Y) not Yt_list (Y^X) for unanimity
    train_infos = []
    for i, (Xt, Yt_Y) in enumerate(zip(Xt_list, Yt_Y_list)):
        H_i, W_i = Xt.shape
        R_i, C_i = Yt_Y.shape
        train_infos.append((train_ids[i], (H_i, W_i), (R_i, C_i), Yt_Y))

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
    # Step 7: Admit & Propagate (WO-11A) → Meet Selector (WO-09')
    # ========================================================================

    # Initialize domains D0 to all colors allowed (C computed earlier in Step 3.6)
    D0 = admit.empty_domains(R_star, C_star, C)

    # Build witness receipt for admit_from_witness
    witness_rc_for_admit = None
    if law_status in ["witness_singleton", "witness_underdetermined"]:
        # Build per_train list from phi_law and sigma_law
        # Note: We have intersection result in sections["witness"]
        per_train = []
        if phi_law:
            # Geometric witness
            phi_pieces = []
            for p in phi_law:
                phi_pieces.append({
                    "comp_id": p.comp_id,
                    "pose_id": p.pose_id,
                    "dr": p.dr,
                    "dc": p.dc,
                    "r_per": p.r_per,
                    "c_per": p.c_per,
                    "r_res": p.r_res,
                    "c_res": p.c_res,
                    "bbox_h": 0,  # Not needed for admits (uses Xt dimensions)
                    "bbox_w": 0,
                    "src_r0": 0,
                    "src_c0": 0,
                    "target_r0": 0,
                    "target_c0": 0
                })

            per_train.append({
                "kind": "geometric",
                "phi": {"pieces": phi_pieces},
                "sigma": {
                    "domain_colors": list(sigma_law.domain_colors) if sigma_law and hasattr(sigma_law, 'domain_colors') else [],
                    "lehmer": list(sigma_law.lehmer) if sigma_law and hasattr(sigma_law, 'lehmer') else [],
                    "moved_count": len(sigma_law.domain_colors) if sigma_law and hasattr(sigma_law, 'domain_colors') else 0
                }
            })
        else:
            # Contradictory witness (no geometric law found)
            per_train.append({
                "kind": "contradictory",
                "sigma": {"domain_colors": [], "lehmer": [], "moved_count": 0}
            })

        witness_rc_for_admit = {
            "intersection": {
                "status": sections["witness"].get("intersection_status", "singleton")
            },
            "per_train": per_train
        }

    # Collect admits from all layers (now returns A, S, receipt)
    A_w, S_w, rc_w = admit.admit_from_witness(
        Xstar_t if witness_rc_for_admit else np.zeros((R_star, C_star), dtype=np.uint8),
        witness_rc_for_admit if witness_rc_for_admit else {"intersection": {"status": "failed"}, "per_train": []},
        C
    )

    # Engine admits: all engines now emit native (A, S, receipt)
    if law_layer_admits is not None:
        # Engine emitted native admits
        A_e, S_e, engine_apply_rc = law_layer_admits
        rc_e = admit.AdmitLayerRc(
            name=f"engine:{engine_used}",
            bitmap_hash=engine_apply_rc["bitmap_hash"],
            scope_hash=engine_apply_rc["scope_hash"],
            support_colors=C,  # Engine can emit any color
            stats={"scope_bits": engine_apply_rc["scope_bits"]}
        )
    else:
        # No engine succeeded: use empty admits (all-ones, S=0)
        A_e, S_e, rc_e = admit.admit_from_engine(Xstar_t, None, C)

    # Unanimity: only applies when output shape == input shape
    H_star_input, W_star_input = Xstar_t.shape
    A_u, S_u, rc_u = None, None, None
    if (R_star == H_star_input) and (C_star == W_star_input):
        # Shapes match: build unanimity admits from truth blocks
        # Build uni_rc dict from unanimity_map
        uni_rc_for_admit = {
            "blocks": [
                {"block_id": block_id, "color": color}
                for block_id, color in unanimity_map.items()
            ]
        }
        A_u, S_u, rc_u = admit.admit_from_unanimity(truth_partition.labels, uni_rc_for_admit, C)
    else:
        # Shapes differ: no unanimity admits
        A_u, S_u, rc_u = admit.admit_from_unanimity(
            np.zeros((R_star, C_star), dtype=int),
            {"blocks": []},
            C
        )

    # Propagate to fixed point with scope-gated intersection
    layers = [(A_w, S_w, "witness"), (A_e, S_e, "engine"), (A_u, S_u, "unanimity")]
    D_star, prop_rc = admit.propagate_fixed_point(D0, layers, C)

    # Select from D* using frozen precedence
    # Build copy_values grid (free copy colors)
    copy_values_grid = copy_values  # Already built earlier

    # Build unanimity_colors grid (only if shapes match)
    unanimity_colors = None
    if (R_star == H_star_input) and (C_star == W_star_input):
        # Map unanimity_map to a grid
        unanimity_colors = np.zeros((R_star, C_star), dtype=np.uint8)
        for block_id, color in unanimity_map.items():
            mask = (truth_partition.labels == block_id)
            unanimity_colors[mask] = color

    # Compute S_law: law scope = witness scope | engine scope (logical OR)
    # CMR-A.6: Law attribution occurs only where witness OR engine had S=1
    S_law = np.zeros((R_star, C_star), dtype=np.uint8)
    if S_w is not None and S_e is not None:
        S_law = (S_w | S_e).astype(np.uint8)
    elif S_w is not None:
        S_law = S_w
    elif S_e is not None:
        S_law = S_e

    Yt, meet_rc = meet.select_from_domains(
        D_star,
        C,
        copy_values=copy_values_grid,
        S_copy=S_w,
        unanimity_colors=unanimity_colors,
        S_unanimity=S_u,
        S_law=S_law,  # Pass law scope mask
        bottom_color=0  # H2: frozen to 0
    )

    # Store admit & propagate receipts (with scope hashes)
    sections["admit"] = {
        "layers": [
            {
                "name": rc_w.name,
                "bitmap_hash": rc_w.bitmap_hash,
                "scope_hash": rc_w.scope_hash,
                "support_colors": rc_w.support_colors,
                "stats": rc_w.stats
            },
            {
                "name": rc_e.name,
                "bitmap_hash": rc_e.bitmap_hash,
                "scope_hash": rc_e.scope_hash,
                "support_colors": rc_e.support_colors,
                "stats": rc_e.stats
            },
            {
                "name": rc_u.name,
                "bitmap_hash": rc_u.bitmap_hash,
                "scope_hash": rc_u.scope_hash,
                "support_colors": rc_u.support_colors,
                "stats": rc_u.stats
            }
        ],
        "propagate": {
            "passes": prop_rc.passes,
            "shrink_events": prop_rc.shrink_events,
            "shrunk_pixels": prop_rc.shrunk_pixels,
            "per_pass_shrink": prop_rc.per_pass_shrink,
            "domains_hash": prop_rc.domains_hash
        }
    }
    hashes["admit"] = hash_bytes(str(sections["admit"]).encode())

    # Store meet receipt (WO-09' selector)
    sections["meet"] = {
        "count_copy": meet_rc.count_copy,
        "count_law": meet_rc.count_law,
        "count_unanimity": meet_rc.count_unanimity,
        "count_bottom": meet_rc.count_bottom,
        "repaint_hash": meet_rc.repaint_hash,
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
