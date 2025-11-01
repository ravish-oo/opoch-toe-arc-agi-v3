# High-level Work Orders (bottoms-up, receipts-first)

## WO-00 — Repo scaffold + receipts kernel (BLOCKER) ✅ COMPLETED

**Goal:** one place for canonical encodings, hashing, and receipts plumbing.
**Delivers:**

* `arc/op/bytes.py`: uint32_le grid serialization; ZigZag LEB128 varints; param framing.
* `arc/op/hash.py`: BLAKE3 helpers (`hash_grid`, `hash_bytes`).
* `arc/op/receipts.py`: typed receipts structs + aggregation; env_fingerprint.
* `scripts/run_wo.py`: harness shell (see “Harness” below).
  **Receipts:** env_fingerprint (platform, endian, blake3_version, …).
  **Acceptance (ARC data):** none functional yet; script prints env_fingerprint and passes determinism self-check (run twice, equal).
  **Proof obligations (PO):**

  * `hash_grid(Π(G)) == hash_grid(Π(Π(G)))` (metamorphic placeholder until Π lands).
  * `env_fingerprint` identical across the double run.
    **Red-team tests:**
  * Simulate endianness change or env tweak → harness must raise `NONDETERMINISTIC_ENV`.
    **Pitfalls to call out:** byte order; ZigZag for signed deltas; no floats anywhere.

## WO-01 — Π presentation (palette, D4 lex, anchor) (BLOCKER) ✅ COMPLETED

**Goal:** idempotent Π, U⁻¹ with round-trip receipts.
**Delivers:**

* `arc/op/pi.py`: `present(G) -> (tildeG, T)`, `unpresent(T, tildeY)`.
* `arc/op/palette.py`: global input-only palette canon (freq ↓, value ↑) — outputs are not included, unseen output colors use identity fallback.
* `arc/op/d4.py`: 8 poses, lex-min raster; pose ids frozen.
* `arc/op/anchor.py`: top-left nonzero to (0,0).
  **Receipts:** palette_freqs+map, pose_id, anchor_drdc, roundtrip_hash.
  **Acceptance (real tasks):** run Π on 20 diverse ARC inputs (5 families we already solved by hand) and assert Π²=Π, U⁻¹(Π(G))=G, hashes stable across runs.
  **Proof obligations (PO):** Π²=Π; U⁻¹∘Π=id; palette canon uses inputs-only (assert outputs not counted).
  **Red-team tests:** lex-close poses differing only after anchor — Π must resolve uniquely; forcing outputs into palette must be detected by receipts diff.
  **Gotchas:** never canonize outputs; identity fallback for unseen output colors.

## WO-02 — Shape S synthesizer (BLOCKER)

**Goal:** exact + least size law selection.
**Delivers:**

* `arc/op/shape.py`: families AFFINE, PERIOD_MULTIPLE, COUNT, FRAME; param serialization; key ordering (branch_byte, params_bytes, R, C); **period axis tie rule** (pick axis with more runs; if equal, rows).
  **Receipts:** branch_byte, params_bytes_hex, R,C, verified_train_ids; if PERIOD_MULTIPLE, record chosen `axis_code`.

  **Acceptance:** pick 12 ARC tasks where Y sizes are known (e.g., 23b5c85d, 3cd86f4f, 995c5fa3). Assert S fits all trainings and reproduces known R×C for test.
  **Proof obligations (PO):** S fits *all* trainings (not just test size); ordering key obeyed; axis tie rule applied and receipted.
  **Red-team tests:** construct both-axes-admissible case → rows win (or "more runs" rule), receipts include axis.
  **Gotchas:** “fit” means equalities verified for *all* trainings.

## WO-03 — Components + stable matching (BLOCKER)

**Goal:** deterministic component pairing across X and Y.
**Delivers:**

* `arc/op/components.py`: CC4 per color; invariant tuple `(area,bbox_h,bbox_w,perim4,outline_hash,anchor_rc)`; D4-min mask hashing (color/pose-independent); stable one-to-one match.
  **Receipts:** list of matched pairs with invariant tuples.
  **Acceptance:** use tasks with multiple identical shapes (652646ff family); verify stable pairing doesn’t flip across runs.
  **Proof obligations (PO):** invariant tuple strictly orders equal shapes; matching stable across runs; pixelwise equality verification after (pose,Δ,residue).
  **Red-team tests:** recolor duplicates; ensure outline hash remains invariant and matching remains stable.
  **Gotchas:** anchor_rc after Π; outline hash independent of color and pose.

## WO-04 — Witness solver (φ,σ) + conjugation + intersection (BLOCKER)

**Goal:** exact equality witness per training; intersect laws.
**Delivers:**

* `arc/op/witness.py`: per-component (pose,Δ,[residues]) search (finite: D4×translations in bbox or lattice residues); σ via unique permutation; φ piecewise; conjugation with Π; intersection; tie flags.
  **Receipts:** per-train φ/σ encodings; conjugated (φ*,σ*); intersection result or flags.
  **Acceptance:** run on known “geometric” tasks (rot/flip/translate; e.g., 652646ff style and simpler ones) and on “summary” tasks where φ=∅; verify encodings stable and intersection singleton.
  **Proof obligations (PO):** per-component equality on every bbox pixel; σ unique (Lehmer) over the law domain; intersection is singleton or yields a tie-table for §08.
  **Red-team tests:** underdetermined microcase with two admissible φ; tie-table present; chosen_idx matches fixed L.
  **Gotchas:** φ is *partial*; never infer by majority; σ encoding = Lehmer.

## WO-05 — Truth compiler (gfp) with frozen tags (BLOCKER)

**Goal:** coarsest bisimulation on fixed tag alphabet.
**Delivers:**

* `arc/op/truth.py`: tag extraction (local + global); integer FFT/NTT overlaps with post-verify; KMP minimal periods; Paige–Tarjan refine; partition hash.
  **Receipts:** tag_set_version (canonical string), refinement_steps, block_hist, partition_hash; transform method for overlaps (FFT/NTT) with method details; identity Δ excluded.
  **Acceptance:** band/sieve tasks (7bb29440, 2037f2c7). Assert partition hashes match expected band grids; identity overlap excluded.
  **Proof obligations (PO):** tag set equals frozen string; fixed point reached; Δ candidates verified by exact equality; identity Δ excluded; record `(method, modulus/root or verified=true)`.
  **Gotchas:** no dynamic tags; must verify overlaps by pixel equality.

## WO-06 — Free copy S(p) (BLOCKER)

**Goal:** compute singleton free copies from φ_i^*.
**Delivers:**

* `arc/op/copy.py`: intersect partial images; bitset mask encoding.
  **Receipts:** singleton_count, singleton_mask_hash (bitset LSB-first).
  **Acceptance:** a task with clean copy sites (mirror/translation); confirm only singletons are “free”.
  **Proof obligations (PO):** S(p) has a value iff every φ_i^* defines it and images are equal for all i; any undefined ⇒ S(p)=∅.
  **Red-team tests:** one training undefined at p; ensure no copy occurs.
  **Gotchas:** any undefined ⇒ S(p)=∅.

## WO-07 — Unanimity on truth blocks (MAJOR)

**Goal:** constant color per block if all trainings agree where defined.
**Delivers:**

* `arc/op/unanimity.py`: pullback via Π and S; define-domain rule.
  **Receipts:** list (block_id, color, defined_train_ids).
  **Acceptance:** tasks where borders/blocks are unanimous (23b5c85d type); confirm empty pullback doesn’t apply.
  **Proof obligations (PO):** unanimity applies only where some trainings define p_i and all defined agree; empty pullback across all trainings ⇒ unanimity must not fire.
  **Red-team tests:** construct empty-pullback case; verify no unanimity write.
  **Gotchas:** never treat “no data anywhere” as unanimous.

## WO-08 — Tie-break L (if needed) (MAJOR)

**Goal:** fixed argmin on admissible law set.
**Delivers:**

* `arc/op/tiebreak.py`: compute cost tuple `(L1_disp_anchors, param_len_bytes, recolor_bits, object_breaks, tie_code)`; apply lex; residue preference; Lehmer lex if still equal.
  **Receipts:** candidate table + chosen_idx.
  **Acceptance:** synth underdetermined mini-cases; assert argmin stable.
  **Proof obligations (PO):** cost tuple logged for all candidates; chosen_idx equals lex argmin; `object_breaks` counts only increases.
  **Red-team tests:** reorder candidate enumeration; chosen_idx unchanged.
  **Gotchas:** object_breaks counts *increases* only; merges ignored.

## WO-09 — Meet writer (copy ▷ law ▷ unanimity ▷ bottom) (BLOCKER)

**Goal:** one-pass, idempotent write; bottom=0.
**Delivers:**

* `arc/op/meet.py`: assemble A_p; pick by fixed priority; repaint idempotence check.
  **Receipts:** counts {copy,law,unanimity,bottom}, repaint_hash.
  **Acceptance:** on the hand-solved tasks (d5c634a2, 995c5fa3, 3cd86f4f, 23b5c85d, 2037f2c7, ccd554ac), the runner reproduces known outputs; second pass hash identical.
  **Proof obligations (PO):** single pass (no re-entry); repaint idempotence holds (same hash on second pass); bottom=0 only.
  **Red-team tests:** inject a second write path after law → repaint hash must change and harness must fail.
  **Gotchas:** never re-enter write order; idempotence must pass.

## WO-10 — Family adapters (symbolic slice/band emitters) (MINOR)

**Goal:** encode canonical ordering rules for symbolic outputs (when witness law is “summary”).
**Delivers:**

* `arc/op/families.py`: helpers for band compression (axis pick, vector orientation), slice stacks (color selection & ordering = first Π-raster appearance; max-rect anchor preferred for skyline/slices).
  **Receipts:** order list, `(axis, band_edges, band_colors)` when used; or x-offsets for slices.

  **Acceptance:** 7bb29440 (band grid), 652646ff (6×6 slices), ccd554ac (Kronecker grid).
  **Proof obligations (PO):** axis/orientation frozen; colors come only from witness/unanimity; skyline/slices left→right (first Π-raster appearance) with max-rect preference; receipts include ordering/edges/offsets.
  **Red-team tests:** attempt “most frequent in test” color selection → receipts must expose invalid provenance.
  **Gotchas:** selection must come from witness/unanimity, not “test guess”.

## WO-11 — Task runner + determinism harness (BLOCKER)

**Goal:** pull all modules together into the single commuting operator.
**Delivers:**

* `arc/runner.py`: `solve_task(train_pairs, X*) -> (Y, receipts)` implementing Π→S→Witness/Truth→S(p)/Unanimity→Tie→Meet→U⁻¹ (exact order).
* `scripts/run_tasks.py`: iterate over data/, write outputs + receipts, determinism check (double-run comparison and env_fingerprint gate).
  **Receipts:** aggregate all section receipts + final output_hash.
  **Acceptance:** run on a curated mini-suite of ~30 ARC tasks spanning families; must match known outputs for the ones we already verified by hand; determinism pass.
  **Proof obligations (PO):** pipeline order exact; double-run determinism including env; receipts aggregation stable.
  **Red-team tests:** simulate env flip; reorder pipeline in a branch to prove harness catches receipt drift.

---

# Harness (so every WO tests on real ARC, not unit tests)

* `scripts/run_wo.py --wo WO-XX --data data/ --subset ids.txt --out out/ --receipts out/receipts/`
  Behavior per WO:

  * Loads only what’s needed for that WO (e.g., WO-01 runs Π, WO-05 computes Truth partitions), writes receipts, and (when meaningful) asserts invariants or known results for selected tasks.
  * Always runs **twice** and compares receipt hashes (determinism).
* `scripts/make_subset.py` to build family-focused ID lists (banding, slices, geometric, min-rect, Kronecker, codebook).
* `scripts/check_receipts.py` to diff receipts between runs/branches.

\

---

## Audit deltas (addendum; keeps WO details intact)

### Cross-cutting freezes (recap)

1. Byte/varint encodings; palette canon inputs-only.
2. Frozen truth tag set string (BLAKE3); no dynamic tags.
3. FFT/NTT overlaps re-verified; identity Δ excluded; method recorded.
4. Mode ties (smallest color) recorded.
5. Band-axis tie rule (more runs, else rows) recorded.
6. Partial φ intersection; no majority copy.
7. Unanimity empty pullback forbidden.
8. Tie-break L tuple; lex; residues; Lehmer lex.
9. Slice/band ordering rules; max-rect preference for skyline.
10. Bottom=0; repaint idempotence required.
11. Determinism harness with env_fingerprint.

### Red-team tests to add when reviewing each WO

* **WO-00**: simulate env change → NONDETERMINISTIC_ENV.
* **WO-01**: build lex-close poses; anchor must disambiguate; palette must ignore outputs.
* **WO-02**: construct both-axes-admissible case; verify axis pick and receipts.
* **WO-03**: recolor duplicates; D4-min mask hash keeps matching stable.
* **WO-04**: two admissible φ; tie-table + argmin L must be stable.
* **WO-05**: attempt float FFT without equality verify → fail; with verify → pass; identity Δ excluded.
* **WO-06**: make one training undefined at p; S(p) must be ∅.
* **WO-07**: empty pullback across trainings; unanimity cannot fire.
* **WO-08**: reorder candidate enumeration; chosen_idx unchanged.
* **WO-09**: insert spurious second write; repaint hash must change and harness must fail.
* **WO-10**: attempt “most frequent in test” color selection; receipts must expose invalid provenance.
* **WO-11**: simulate env flip; pipeline reorder should change receipts and be caught.

### WO notes and things to take care
Where to place NTT details (Python)
Put NTT/FFT method selection and receipts in WO-05 — Truth compiler (gfp). That WO owns per_color_overlap and must:
implement integer FFT or NTT,
exclude identity Δ,
verify overlaps by pixel equality,
record (method, modulus/root or verified=true) in receipts.

COUNT-based S qualifiers
Add under WO-02 — Shape S as an optional sub-family that can be turned on when needed. Keep it trivial until a task actually requires it:
Implement the parameter encoding (<4><alpha1><beta1><alpha2><beta2>) and “qual id” hashing exactly as frozen in the addendum.
Only activate the COUNT candidate when a clear equality proof across all trainings exists.

from docs/common_mistakes.md
WO-00: J1 (determinism), no timestamps in receipts.
WO-01: F1, F2.
WO-02: E1 (+ the period-axis tie clarification).
WO-03: D2.
WO-04: E2, A1, C2.
WO-05: 0 fundamentals, D1, I1.
WO-06: G2.
WO-07: G1.
WO-08: C1 (plus tie tuple receipts).
WO-09: H1, H2.
WO-10: A1, A2, B1, B2, C1, C2.
WO-11: J1 (double-run determinism).

