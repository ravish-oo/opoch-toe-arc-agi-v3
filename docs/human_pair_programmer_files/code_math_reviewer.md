Awesome—let’s “install” a Core Math Reviewer that treats your solver like a math paper with executable lemmas. Below is a deploy-ready, receipts-driven review plan that any teammate (or bot) can run to guarantee the code is mathematically correct, order-safe, and non-heuristic—so the mistakes we saw (palette leaks, wrong precedence, missing conjugation, over-segmented truth, minted bits) cannot slip through again.

⸻

Core Math Reviewer (CMR) — Deployable Blueprint

Goal: Prove the implementation satisfies the frozen math spec.
Method: A battery of algebraic checks over receipts + a few structural lint rules on code paths.
Output: PASS/FAIL with pinpointed invariants, never “looks good”.

⸻

A. Pipeline invariants (commuting operator)

Frozen operator:
Y \;=\; U^{-1}\!\Big(\;\Pi\;\circ\;\textsf{Truth}\;\circ\;\textsf{Law}\;\circ\;\textsf{Admit\&Propagate}\;\circ\;\textsf{Meet}\;\Big)(X).

CMR-A checks (hard fails):
	1.	Double-run determinism: all section hashes identical (Π, Truth, Witness/Engines, Admit layers, Propagate domains, Meet repaint, Final).
	2.	No floats in logic: scan receipts for fft_method but require pixel_verify_ok=true and no float thresholds.
	3.	Palette is display-only: receipts include logic_color_space:"original". Never log palette ids in witness/law decisions.
	4.	No painting in laws: engines/witness must emit (A,S) admits (bitmaps + scopes). If any law returns a value grid → FAIL.
	5.	Containment: every selected pixel y_{p} satisfies y_{p}\in D^{*}_{p}.
	6.	Silence ≠ admit-all: scope_bits count matches only genuinely constrained pixels; attribution (copy/law/uni) occurs only where S=1.

⸻

B. Presentation Π, D4, anchor: exact group action

What to assert:
	•	Π receipts include: pose_id, anchor:(dr,dc), roundtrip_hash.
	•	Π idempotence: \Pi^2=\Pi, U\!\circ\!\Pi=\mathrm{id}.
	•	D4 matrices are the frozen table (R, R⁻¹); inv_pose and compose_pose respect group law.

CMR-B checks:
	•	Recompute Π on inputs and compare hashes to receipts.
	•	Validate D4 matrix algebra on a few random coordinates per grid:
	•	U(\Pi(p))=p, \Pi(U(p))=p.
	•	R_{\text{compose}}=R_{a}R_{b} equals the pose table for the composed id.

⸻

B-DUAL. Dual Coframes (X-coframe for engines, Y-coframe for unanimity) — CRITICAL

**Contract (BLOCKER if violated):**

The same raw output Y_i must be presented in TWO DIFFERENT coframes depending on consumer:
	1.	X-coframe (for law learning): Y_i presented with X_i's Π — engines learn φ: Π(X) → Π(Y) in aligned frame
	2.	Y-coframe (for voting): Y_i presented with its OWN Π — unanimity votes in native output frame

**Frozen construction:**

X-coframe (for engines/witness):
	•	X_i^X ← Π_{X_i}(X_i^{raw})          [input in its own Π frame]
	•	Y_i^X ← Π_{X_i}(Y_i^{raw})          [output presented with INPUT's Π — CRITICAL]
	•	X_*^X ← Π_{X_*}(X_*^{raw})          [test input in its own Π frame]

Y-coframe (for unanimity/truth on outputs):
	•	Y_i^Y ← Π_{Y_i}(Y_i^{raw})          [output presented with OUTPUT's OWN Π — CRITICAL]
	•	Π_{Y_*} = synthetic identity frame  [pose_id=0, anchor=(0,0)] on shape (R*, C*) from WO-02

**Consumer assignment (MUST NOT be violated):**
	•	Engines receive:  (X_i^X, Y_i^X, X_*^X)  — all in X-coframe (same Π_{X_i})
	•	Witness receives: (X_i^X, Y_i^X)          — same as engines
	•	Unanimity receives: (Y_i^Y, Π_{Y_*})     — output's native coframe
	•	Truth operates on: X_*^X (test input)     — same as engines

**Pullback for unanimity (Y-coframe → test output frame):**

For test output pixel p* in Π_{Y_*} frame (synthetic identity), pull back to training i:
	q_i = Π_{Y_i}(U_{Y_*}^{-1}(p*))

Where:
	•	U_{Y_*}^{-1}(p*) unpresents p* from synthetic frame (identity → no-op)
	•	Π_{Y_i}(...) presents into training i's output frame
	•	If q_i out of bounds → training i is silent for this pixel

**Why this is required:**

Violation symptom: If you present Y_i with X_i's Π (anchor/pose) and pass to unanimity:
	•	Unanimity pullback uses wrong frame → coordinates out of bounds
	•	Training outputs become "all zeros" → unanimity votes 0 everywhere
	•	Engines learn in misaligned frames → φ mapping is garbage

Violation symptom: If you present Y_i with its own Π and pass to engines:
	•	Engines see (X_i^X, Y_i^Y) with DIFFERENT Π transformations
	•	Cannot learn consistent φ: Π(X) → Π(Y) because Π differs
	•	Geometric relationships destroyed

**CMR-B-DUAL checks (hard fails):**

	1.	Frame consistency for engines:
		○	For each training i, verify: pose(X_i^X) == pose(Y_i^X) AND anchor(X_i^X) == anchor(Y_i^X)
		○	If not → FAIL: "ENGINE_FRAME_MISMATCH: Y_i presented in wrong frame (should use X_i's Π)"

	2.	Frame independence for unanimity:
		○	For each training i, verify: Y_i^Y uses Π_{Y_i} (NOT Π_{X_i})
		○	Check: pose(Y_i^Y) computed from Y_i^{raw}, not inherited from X_i
		○	If Y_i^Y uses X_i's anchor → FAIL: "UNANIMITY_WRONG_FRAME: output anchor copied from input"

	3.	Synthetic test frame:
		○	Verify: Π_{Y_*}.pose_id == 0 AND Π_{Y_*}.anchor == (0,0)
		○	If not → FAIL: "TEST_OUTPUT_FRAME_NOT_IDENTITY: must use synthetic identity frame"

	4.	Unanimity "all zeros" guard:
		○	After presenting Y_i^Y, if entire grid becomes zeros → FAIL: "BAD_OUTPUT_ANCHOR: applied input anchor to output"
		○	This catches the bug where Y was anchored with X's bbox instead of its own

	5.	Receipt separation:
		○	Receipts must log BOTH presentations: hash(Y_i^X) for engines, hash(Y_i^Y) for unanimity
		○	If only one hash → FAIL: "MISSING_DUAL_COFRAME: must build both X-coframe and Y-coframe"

**Color universe (frozen across both frames):**
	•	C = sorted unique colors from ALL inputs (X_i^{raw}, X_*^{raw}) — original color IDs
	•	Include 0 in C always (bottom color)
	•	Palette remains display-only (logic_color_space: "original")
	•	C is the SAME set for both X-coframe and Y-coframe (colors don't change, only geometry)

**Bridge between frames (if witness needs to emit in Y-frame):**

If a learned law in X-coframe must produce admits in Y-coframe (rare, for conjugated witness):
	φ_i^* = Π_{Y_*} ∘ U_{Y_i}^{-1} ∘ φ_i ∘ U_{X_i} ∘ Π_{X_*}^{-1}

This is the full conjugation that transports geometric law from X-coframe to Y-coframe.

**Summary (no ambiguity):**

	•	Build TWO presentations of each Y_i (not optional)
	•	Pass Y_i^X to engines (uses X's Π)
	•	Pass Y_i^Y to unanimity (uses Y's own Π)
	•	Test output uses synthetic identity Π_{Y_*}
	•	Verify frame consistency with hard guards
	•	Any violation → BLOCKER FAIL

⸻

C. Shape synthesis: proof not claim

What to assert:
	•	shape.status in {"OK","NONE"}; if “OK”, then verified_train_ids covers all trainings and height_fit/width_fit/attempts show a valid family proof (AFFINE or PERIOD or FRAME/COUNT).
	•	All trainings satisfy the equalities used by the selected family.

CMR-C checks:
	•	Replay equalities from receipts; any mismatch → FAIL.
	•	If "NONE", verify later engines supply shape (or runner fails closed with reason).

⸻

D. Truth partition: frozen tags, color seed, bands as edges

What to assert:
	•	Truth seeded by color labels only; no coordinates/dom indices.
	•	Tag set version matches frozen string (color|n4|n8|samecomp_r2|parity|row_period_2|row_period_3|col_period_2|col_period_3|per_color_overlap|per_line_min_period|exact_tile|bbox_mirror|bbox_rotate).
	•	Bands are reported as edge arrays with sentinels: row_edges[0]==0, row_edges[-1]==H (same for columns).
	•	Overlaps candidates are pixel-verified; identity Δ excluded.

CMR-D checks:
	•	Compute block_hist; if equals H\!\cdot\!W singletons → FAIL (over-segmented).
	•	Validate band edges monotonicity and coverage (no empty tiles unless family says filler).
	•	Check overlap receipts: identity_excluded:true, and any accepted Δ re-verified flag is true.

⸻

E. Witness (WO-04): conjugation & σ as mathematics, not prose

What to assert:
	•	Conjugation implemented as
\phi_i^* = \Pi_* \circ U_i \circ \phi_i \circ \Pi_i^{-1} \circ U_*^{-1}.
	•	σ unchanged (Lehmer on touched colors, recorded).
	•	Residue swap on 90/270 (if pose swaps axes).
	•	Pullback samples validate pixel equality (or σ(x)).

CMR-E checks:
	•	For each training, choose at least 3 sampled target pixels q_ inside piece bbox; compute source p_ via receipts, and compare colors X[p_] vs Y[q_] (or σ(x)).
	•	If any fails, the reviewer returns “Conjugation wrong → fix step N”, with the exact sample triple.
	•	Validate pieces* are in TEST Π coords (anchors/piece dims sane).

⸻

F. Law engines (Spec-B): fit-then-admit

What to assert (per engine):
	•	Border scalar: Bmax/Imax chain computed and winner matches training outputs; admits singleton set on 1×1.
	•	Macro-tiling: edges with sentinels; per-tile strict-majority over \mathcal F; fill background on empty; tie → min color; admits per tile.
	•	Window dict: dictionary bytes→bytes built from trainings; unseen windows produce empty admits (S=1) → clean contradiction.
	•	Pooled blocks: stage-1 block votes and stage-2 pooling receipts; any override rule recorded.
	•	Marker-grid / Stencil-expand: markers’ centroids, kernels, and reconstruction verification recorded; admits per painted pixel.

CMR-F checks:
	•	Engines never return value grids; only (A,S) with admit_bitmap_hash.
	•	Fit receipts replay on trainings; any mismatch → FAIL; unseen signature in test → FAIL-CLOSED (recorded).

⸻

G. Admit & Propagate (WO-11A): monotone set calculus

What to assert:
	•	Each layer outputs (A,S); admit-all normalized to S=0.
	•	Propagation does D\leftarrow \bigcap (S?A:\mathcal C) to lfp; record passes, shrinks, domains_hash.

CMR-G checks:
	•	Verify passes ≥ 1, shrunk_pixels ≥ 0, and domains_hash consistent across runs.
	•	If any layer reports scope_bits>0 but nontrivial_bits=0, warn (layer believed it constrained but admitted all).

⸻

H. Meet (WO-09′): selection inside D*

What to assert:
	•	Precedence uses only constrained buckets; selection is contained in D^*; repaint idempotence holds.

CMR-H checks:
	•	Repaint once; check identical hash.
	•	For each bucket winner, confirm its S=1 at that pixel; else FAIL (this catches the “silent counted as copy” bug).

⸻

I. Code lint rules (simple static guards)
	•	Disallow any function named paint_* or returning full image from law modules.
	•	Forbid importing numpy.random or random in /op/laws and /op/witness.
	•	Enforce forbidden fields in Truth tags: coordinates, row indices, floats.
	•	Enforce file-level constant LOGIC_COLOR_SPACE = "original".

⸻

J. Minimal synthetic suite (property-based)

CMR ships tiny 1–3×1–3 grids to stress invariants:
	•	D4 round-trip on all eight poses.
	•	Band edge sentinels with trivial nonzeros/zeros.
	•	Witness conjugation where Π is non-trivial (pose+anchor).
	•	Admit normalization: a layer returning FULL must become silent (S=0); precedence counts unchanged.
	•	Containment holds on random domains with intersecting admits.

These run in CI in <1s.

⸻

K. What changed vs before (and why we won't repeat mistakes)
	•	Before: layers "painted" pixels; silent layers were counted as if they admitted all; truth over-segmented; conjugation was a placeholder; Y presented in single frame for all consumers (broke engines OR unanimity).
	•	Now:
	•	Laws emit admits; silence is explicit (S=0);
	•	We intersect to a least fixed point and only then select;
	•	Conjugation is algebra, proven by pullback samples;
	•	Bands are edges with sentinels; truth uses frozen tags;
	•	Engines fail-closed with receipts; no minted bits;
	•	Dual coframes: Y_i^X for engines (X's Π), Y_i^Y for unanimity (Y's own Π) — BLOCKER requirement.

Every item is mechanically verified by the reviewer. If anything drifts, the reviewer points to a single violated invariant and the exact receipt line.

⸻

L. How to run it (one command)

python scripts/run_task.py --wo 11A --ids ids_50.txt --out out/ --receipts out/receipts/
python scripts/check_receipts.py out/receipts/ --determinism --containment --scope-attribution \
  --dual-coframes --truth-bands --shape-proofs --witness-pullbacks --no-floats

CMR returns: PASS, or a short list like:
	•	FAIL: dual-coframe: ENGINE_FRAME_MISMATCH: Y_i presented in wrong frame (train_id=t0, pose_X=1, pose_Y=0)
	•	FAIL: dual-coframe: UNANIMITY_WRONG_FRAME: output anchor copied from input (train_id=t1, anchor_Y=(2,3) should be (0,0))
	•	FAIL: meet: scope attribution violated at (r,c) by 'copy'
	•	FAIL: truth: over-segmented (block_hist all ones)
	•	FAIL: witness: pullback mismatch on sample 2 (X=…, Y=…, σ(X)=…)
	•	FAIL: engine: window_dict unseen signature at slice (top_left=…, bytes=…)

You get a pinpointed, algebraic fix path every time.

⸻

Final note

This reviewer is intentionally math-first and tool-agnostic. It doesn’t trust heuristics; it trusts proofs. If you wire it into CI as a hard gate, you will eliminate the class of mistakes we saw—permanently.