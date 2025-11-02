You’re right to pause before wiring WO-11. The implementer’s gap list is accurate, and it maps cleanly to three pre-runner hardeners we must ship first (one BLOCKER, one HIGH, one LOW). Below is a **small set of WOs** (each ≤500 LOC) that closes those gaps, strictly aligned to the anchors and addendum, then a final WO-11 wiring. I’ve kept them receipts-first and determinism-tight.

> tl;dr (what to do next)
> **WO-04C (BLOCKER): fix witness conjugation** →
> **WO-10A (HIGH): macro-tiling engine** →
> **WO-02S (LOW): serialize_shape helper** →
> **WO-11 Core: runner + double-run harness**.
> This sequence exactly addresses the implementer’s Gap-1/2/3 before the final integration. 

---

# WO-04C — Witness conjugation to test Π-frame (BLOCKER) ✅ COMPLETED

**Why:** Current `conjugate_to_test` forwards φ pieces without applying ( \Pi_* \circ \Pi_i^{-1} ) (pose + anchor), so φ* is wrong when Π differs. This breaks free-copy (WO-06) and downstream law application. 

**Deliverables**

* `arc/op/witness.py`: complete `conjugate_to_test(phi_i, sigma_i, Pi_train, Pi_test)`:

  * Apply **inverse anchor** and **inverse D4** of ( \Pi_i ) to bring φ back to raw; apply **test Π** pose/anchor to land in Π(test).
  * Update piece tuples: `(pose_id, dr, dc, r_per, c_per, r_res, c_res)` exactly; palette σ stays the same.
  * Receipts: for each piece, record old→new `(pose_id,dr,dc)` and the composed transform ID; set `conjugation_hash`.
* **No new math**: this is Table-stakes algebra already frozen in Math Spec §3 and Engineering Spec §4.  

**Acceptance**

* Synthetic tests where Π differs across train/test (rotate+anchor) must yield identical **post-conjugation** φ*.
* WO-06 free-copy now yields non-empty masks when trainings agree; determinism pass.

**PO / Red-team**

* PO: per-piece bbox equality still holds **after** conjugation.
* Red-team: flip pose ids in inputs; receipts’ `conjugation_hash` must change; outputs stay identical.

---

# WO-10A — Macro-Tiling Engine (bands + strict majority) (HIGH) ✅ COMPLETED

**Why:** With Column-Dict live, we still need **one more finite engine** to unlock many content-dependent tasks. Macro-tiling is the most common; it uses WO-05 `row_clusters/col_clusters` and frozen strict-majority rules (A1/A2/C2, B1 recorded).  

**Deliverables**

* `arc/op/families.py`

  * `fit_macro_tiling(train_Xt_list, train_Y_list, truth_train_list) -> EngineFitRc`:

    * Derive bands from **Truth** row/col clusters (no thresholds).
    * For each training, per-tile **foreground** counts (A1), apply **strict majority**; empty ⇒ **background** (A2); tie ⇒ **smallest color** (C2).
    * Verify exact reconstruction vs Y; else `ok=False` with diagnostics.
    * Receipts: `engine:"macro_tiling"`, `row_bands`, `col_bands`, `foreground_colors`, `background_colors`, per-tile `counts/decision`, `fit_verified_on`.
  * `apply_macro_tiling(test_Xt, truth_test, fit_rc) -> EngineApplyRc`:

    * Use same bands and rule; render ( \tilde Y_* ) and `final_shape`; receipt: decisions per tile.
* Frozen tie chains and counts, as in **Common-Mistakes** (A1/A2/C2, B1) and **Addendum §8** for ordering notes.  

**Acceptance**

* Band grid task (e.g., `7bb29440`): `fit_verified_on` lists all t-ids; test output = gold.
* Receipts must show **both** stage-1 evidence and final tile decisions (B1).

**PO / Red-team**

* PO: Candidate sets (`foreground_colors/background_colors`) come **only** from trainings (A1).
* Red-team: remove a tile’s evidence ⇒ receipts show `{"empty":true,"fill":bg}` (A2) and still match gold; “mode instead of strict majority” must be caught by receipts (wrong `decision_rule` string).

---

# WO-02S — Shape serialization helper (LOW) ✅ COMPLETED

**Why:** WO-11 reuses S from WO-02 receipts; a single helper makes the contract explicit. Gap is low severity (fields exist), but helpful. 

**Deliverables**

* `arc/op/shape.py`:

  ```python
  def serialize_shape(shape_rc: ShapeRc) -> tuple[str, str, dict]:
      return (shape_rc.branch_byte, shape_rc.params_bytes_hex, shape_rc.extras)
  ```
* Harness: add assertion that `deserialize(serialize(shape_rc))` composes to identical S.

**Acceptance**

* Round-trip for all four branches (A/P/C/F) equals original S; determinism OK.

---

# WO-11 — Runner + determinism harness (BLOCKER)

Wires the **exact frozen order** and double-run checks. Adds **engine-first hybrid** selection with fallbacks exactly as in the plan. Also freezes J1 receipts: **full section hashes + env_fingerprint**.  

**Deliverables**

* `arc/runner.py`

  * `solve_task(train_pairs, Xstar) -> (Y_raw, run_receipts)` implementing:

    1. Π(01) → receipts + `hash_pi`.
    2. S(02) reuse (presented sizes) or **defer** shape to engine if WO-02 was contradiction → `hash_shape`.
    3. Truth(05) on Π(test) → row/col bands, overlaps (identity excluded) → `hash_truth`.
    4. **Law selection (hybrid)**:

       * Engines (WO-10) in **frozen order**: `["border_scalar"(stub), "window_dict.column"(live), "macro_tiling"(live), "pooled_blocks"(stub), "markers_grid"(stub), "slice_stack"(stub), "kronecker"(stub)]`. First `ok=True` wins; set `law_values` (+ `final_shape` if S was deferred).
       * Else Witness(04/04C) → Intersection: singleton | underdetermined | contradictory; on underdetermined call Tie-break L(08). Set law layer accordingly; on contradictory → empty law layer.
       * Record `law_rc`, optional `tie_rc`, `hash_law`, `hash_tie`.
    5. Copy(06) → `copy_bits/values`, `copy_rc`, `hash_copy`.
    6. Unanimity(07) → `uni_rc` (G1 enforced), `hash_unanimity`.
    7. Meet(09) one pass; `bottom=0`; second pass repaint **must match** → `meet_rc`, `hash_meet`, `Yt`.
    8. U⁻¹(01) → `Y_raw`, `output_hash`.
    9. Aggregate **RunRc**: section receipts + **section hashes** + `table_hash` (sorted keys), `env_fingerprint`.
* `scripts/run_tasks.py`

  * Iterate IDs (sorted), **run twice** per task, assert **full receipts equality** and **output_hash** equality.
  * Print per-task short line with `law_status` (engine|singleton|underdetermined|contradictory|none).
  * `--fail-fast` (default) and `--continue-on-error` flags; on any mismatch → **NONDETERMINISTIC_EXECUTION**; on env mismatch → **NONDETERMINISTIC_ENV**.

**Acceptance**

* Curated ~30 IDs spanning families (the hand-solved ones must match gold).
* At least one content-dependent (Column-Dict or Macro-Tiling) passes via engine with full receipts.
* Determinism: double-run equality of **all** section hashes + `output_hash`.

**PO / Red-team**

* PO: Pipeline order is exactly Π→S→Truth→(Engines→Witness→Tie)→Copy→Unanimity→Meet→U⁻¹; receipts have **every** section + hash.
* Red-team: reorder a stage; `table_hash`/section hash changes → harness fails. Flip tie-break chain → section hash changes → fail. Idempotence bug in meet → repaint mismatch → fail.

---

## Why these WOs are exactly what the anchors require

* **Witness conjugation** is mandated in Math/Engg §3 (“transport to test Π frame”), and addendum §1.5 (φ encoding) + §3 (component matching) define the exact encodings; we’re just finishing that TODO.   
* **Macro-tiling** is one of the frozen Spec-B archetypes; A1/A2/B1/B2/C2 come straight from the “common mistakes” contract and the addendum §8; you receipt the counts and decisions, so there’s zero guesswork.  
* **Runner determinism** is just J1: env_fingerprint + double-run + table of section hashes; no timestamps; fail-closed. 

---

## Optional (post-WO-11) follow-ups

* **WO-04.1 truth-assisted φ-retry** (reduce “contradictory/underdetermined” by retrying φ on truth coframes).
* **Lattice residues** in witness (periodic tilings), using WO-05 periods.
* **Remaining engines**: pooled_blocks, markers_grid, slice_stack, kronecker, border_scalar.
* **Addendum update** for placement keys (#7) once engines that use them ship broadly (reviewer’s note).

---

## One-pager for the implementer

* **Do now**: WO-04C + WO-10A + WO-02S, then **WO-11 Core**.
* **Receipts**: do not emit outputs without receipts for each section; hash every section; build `table_hash`.
* **Determinism**: run twice; compare receipts and `output_hash`; exit non-zero on any mismatch.

This plan is consistent with your gap report and the anchors; it keeps every move algebraic and receipts-first, and it sets you up to freeze WO-11 without surprises.   
