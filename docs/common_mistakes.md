Below is a field-tested, engineer-oriented catalogue of the mistakes that keep biting people on ARC—even when they think they’re using exact math. For each class of mistake I give:
	•	Symptom: what you’ll see in outputs,
	•	Root cause: why it’s wrong,
	•	Do this instead: the exact fix,
	•	Receipt to log: what to record so reviewers (and future you) can prove it’s right.

Treat this as your pre-flight checklist. If you follow it, your solver behaves like a mathematical operator: same inputs → same receipts → same outputs. No drift.

⸻

0) Absolute fundamentals (always true)
	•	Never derive anything from test outputs. Only from trainings.
	•	No floats for logic. If you use FFT/NTT for overlaps, re-verify by pixel equality; identity Δ excluded.
	•	Frozen tag set only. No dynamic features “just for this task”.
	•	Fail-closed. If a proof cannot be established, return an explicit contradiction receipt; do not guess.

⸻

1) Candidate-set mistakes

A1. Dropping a valid color (or treating it as “background-only”)
	•	Symptom: Picking the wrong winner for argmax/majority/vote; missing entries in small summaries.
	•	Root cause: You silently excluded 7 (or 1) when the family law allows it to win.
	•	Do this instead: Candidate sets are part of the law.
	•	Occupancy-argmax: all non-zero colors unless the family explicitly restricts (e.g., “8-only family”).
	•	Majority summaries: foreground set \mathcal F and background set \mathcal B are frozen per family (e.g., \mathcal F=\{4,6,7,8\}, \mathcal B=\{0,1\}).
	•	Receipt to log: foreground_colors, background_colors, per-color counts for the decision window/block.

A2. Ignoring an empty candidate (empty cluster/slot)
	•	Symptom: You put a non-border color in a coarse grid slot that had no marker.
	•	Root cause: Filling from “most frequent in test” rather than the family rule.
	•	Do this instead: Empty slot ⇒ border color (frozen).
	•	Receipt: for every coarse cell, either (marker_color, centroid) or {"empty": true, "fill": border}.

⸻

2) Aggregation-scope mistakes

B1. Stopping at the wrong level (too local)
	•	Symptom: You wrote 16 per-block votes into a 4×4 when the training law uses quadrant pooling to a different 4×4.
	•	Root cause: The family has a second-stage pooling (e.g., 2×2 blocks → quadrant vote), but you emitted after stage 1.
	•	Do this instead: Follow the family-specific scope: e.g., blocks → pooled quadrants; or motif-replication across all band argmax windows.
	•	Receipt: both the stage-1 votes and stage-2 pooled results.

B2. Missing replication across argmax windows
	•	Symptom: You output one 3×3 motif of 8; gold shows that motif repeated horizontally across all canonical argmax windows.
	•	Root cause: You found the motif but ignored other argmax windows on the frozen band row.
	•	Do this instead: Enumerate all canonical argmax windows on the chosen band row (non-overlapping, stride≥3), and concatenate the motif for each.
	•	Receipt: list of argmax windows (top-left coords), M8 value, and the replicated motif count.

⸻

3) Tie-break mistakes

C1. Unfrozen placement (x-offsets/argmax choice)
	•	Symptom: Same bars/motif, different horizontal position than gold.
	•	Root cause: You used “left-pack” or ad-hoc selection.
	•	Do this instead: Freeze tie in this order (or as your addendum states):
	•	Nearest to center (L1),
	•	then topmost,
	•	then leftmost,
	•	for skylines: first Π-appearance/max-rect anchor, then non-overlap left→right.
	•	Receipt: tie chain, candidate list with distances/keys, chosen index.

C2. Mode vs. strict-majority
	•	Symptom: Blocks that are ties (e.g., {4:2, 0:2}) were written as 4; gold has 7 (fallback).
	•	Root cause: You used mode. Family requires strict majority among \mathcal F, fallback 7.
	•	Do this instead: Strict majority in \mathcal F; if none, write fallback background; among \mathcal F ties, pick the smallest color.
	•	Receipt: per-block counts and decision rule string ("strict_majority_foreground_fallback_7").

⸻

4) Truth/feature mistakes

D1. Dynamic tags or fuzzy clustering
	•	Symptom: Good on some, drifts on others; two teams disagree.
	•	Root cause: You added a “nearby row/col” threshold or a bespoke tag.
	•	Do this instead: Use only frozen tags; cluster by sorted distinct centroid rows/cols (exact), not epsilon-based.
	•	Receipt: tag_set_version hash; the exact row_clusters, col_clusters arrays.

D2. 4-connected vs 8-connected mismatch
	•	Symptom: Marker/component count differs run to run or between teams.
	•	Root cause: One used 8-connectivity; law and receipts expect 4-connectivity.
	•	Do this instead: Components: 4-adjacency per color; record connectivity choice in receipts.
	•	Receipt: connectivity="4", component invariant tuples.

⸻

5) Shape/content proof mistakes

E1. Changing output shape without an S proof
	•	Symptom: Output dims don’t match gold or trainings’ size law.
	•	Root cause: You emitted a different shape (cropped/expanded) without S satisfying all training equalities.
	•	Do this instead: Learn S (AFFINE/PERIOD/FRAME/COUNT) and verify on every training; record branch and params.
	•	Receipt: S: {branch, params_bytes_hex, verified_train_ids}.

E2. Moving content without a φ equality witness
	•	Symptom: You “moved” objects (translate/rotate/mirror) without proof.
	•	Root cause: Missing φ proof of equality on per-component bbox.
	•	Do this instead: φ must be a partial function with exact equality; majority is disallowed.
	•	Receipt: φ encodings, per-component bbox equality checks; singleton S(p) mask.

⸻

6) Palette/Π mistakes

F1. Canonizing outputs or mixing test in palette semantics
	•	Symptom: Palette differences between teams; color ids shift.
	•	Root cause: Palette canon included training outputs or was recomputed per grid.
	•	Do this instead: Palette canon built on inputs only (all train inputs + test input), freq↓ then value↑; unseen output colors use identity fallback.
	•	Receipt: palette_freqs[(color,freq)...], palette_map.

F2. Non-idempotent Π
	•	Symptom: Π²≠Π; U⁻¹∘Π≠id.
	•	Do this instead: Always verify Π round-trip in receipts; fail-closed if it doesn’t hold.

⸻

7) Unanimity & copy mistakes

G1. Unanimity on empty pullback
	•	Symptom: You filled a block by unanimity when no training defined the pullback.
	•	Do this instead: Unanimity applies only if some trainings define the pullback coords and all defined agree.
	•	Receipt: for each block: (block_id, color, defined_train_ids); empty → unanimity off.

G2. Majority copy (illegal)
	•	Symptom: Combined different φ_i^* outputs to “copy” a value.
	•	Do this instead: S(p) = ⋂ {φ_i^*(p)}; if any φ undefined or values differ ⇒ no copy.
	•	Receipt: singleton_count, singleton_mask_hash (bitset).

⸻

8) Meet & repaint mistakes

H1. Wrong priority or multiple passes
	•	Symptom: Latent flicker: repaint changes the output.
	•	Root cause: You re-entered the write order or swapped priority.
	•	Do this instead: One pass with fixed priority: copy ▷ law ▷ unanimity ▷ bottom. Immediately repaint and assert idempotence.
	•	Receipt: {copy,law,unanimity,bottom} counts; repaint_hash.

H2. Bottom not frozen to 0
	•	Symptom: Empty becomes “min palette” instead of 0.
	•	Do this instead: Bottom=0. Always.
	•	Receipt: state bottom=0 once; enforce in tests.

⸻

9) FFT/NTT overlap mistakes (when used)

I1. Float-rounding drift
	•	Symptom: Δ differs between runs/teams.
	•	Do this instead: If using FFT, treat it as a candidate generator only; verify by exact pixel equality; identity Δ excluded.
	•	Receipt: method (fft_int/ntt), candidate Δ list, accepted Δ with equality check.

⸻

10) Determinism & environment

J1. Hidden nondeterminism
	•	Symptom: Same inputs, different outputs on another machine.
	•	Root cause: Unfrozen tie or environment-dependent iteration order; or missing env fingerprints.
	•	Do this instead:
	•	Freeze all ties and orders;
	•	Record env_fingerprint and double-run determinism (compare all receipt hashes).
	•	Receipt: env_fingerprint, Π/Truth/φ/S/tie/meet/final hashes; double-run compare.

⸻

11) Quick “before ship” checklist
	•	Π receipts present; Π²=Π; U⁻¹∘Π=id.
	•	S learned, verified on every training.
	•	Witness per training logged; φ equality proven (or φ empty for summary).
	•	Truth tag_set_version matches frozen string; partition hash logged.
	•	Candidate colors/windows/blocks recorded; none excluded silently.
	•	Aggregation scope matches family (replication/pooling as per law) and is receipted.
	•	Tie-breaks applied and receipted (distance/lex etc.).
	•	Unanimity used only on non-empty pullbacks; defined_train_ids recorded.
	•	Meet counts logged; repaint idempotence hash equals.
	•	Determinism: double-run; all receipt hashes equal; env_fingerprint recorded.

⸻

12) Minimal receipts to catch 99% of bugs
	•	Π: palette_freqs,map, pose_id, anchor, roundtrip_hash.
	•	S: branch, params_bytes_hex, verified_train_ids.
	•	Truth: tag_set_version, partition_hash, and the tables that decide (e.g., per-color counts for argmax windows / per-block counts / marker centroid lists).
	•	Witness: per-training φ/σ (or φ=∅), intersection result or tie table.
	•	Tie: candidate list, cost tuples or tie keys, chosen_idx.
	•	Meet: counts + repaint_hash.
	•	Final: output_hash, env_fingerprint.

⸻

One sentence summary

Most mistakes are not math errors; they’re spec gaps — omitted candidates, wrong aggregation level, unfrozen ties, or unproven shape/content moves. Freeze them, receipt them, and the solver is exact mathematics.

# Which WO takes care of which common mistake
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