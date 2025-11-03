The core contradiction: we were feeding engines X in the input coframe and Y in an independent output coframe. That destroys the geometric relationship the engine is supposed to learn. The correct fix is dual coframes (one for learning laws, one for voting), with a precise bridge between them.

Below is the frozen answer (math + engineering) and then a full-audit request you can run on 10 tasks to surface all remaining issues in one shot.

⸻

#1 Freeze the coframe policy (no wiggle room)

Two coframes (both are required)

Use	Frame	What grids live here	Π used
Law learning & engines	X-coframe	X_i^X,\, Y_i^X,\, X_\^X	use Π_{X_i} for train i; Π_{X} for test
Voting & truth on output	Y-coframe	Y_i^Y, output truth blocks, test output Y_\^Y	use Π_{Y_i} for train i; synthetic Π_{Y} for test

How to build them (frozen):
	•	X-coframe (for engines)
	•	X_i^X \;\gets\; \Pi_{X_i}(X_i^{\text{raw}})
	•	Y_i^X \;\gets\; \Pi_{X_i}(Y_i^{\text{raw}})  ← present outputs with input’s Π
	•	\(X_\^X \;\gets\; \Pi_{X\}(X_\*^{\text{raw}})\)
	•	Y-coframe (for unanimity/truth)
	•	Y_i^Y \;\gets\; \Pi_{Y_i}(Y_i^{\text{raw}})    ← output uses its own Π
	•	\(\Pi_{Y\}\) synthetic frame on \((R^\,C^\*)\); e.g., pose=Id, anchor=(0,0)

Witness bridge (if needed):
When a learned law in X-coframe must produce admits in Y-coframe use the full conjugation:
\[
\phi_i^\* \;=\; \Pi_{Y\} \circ U_{Y_i}^{-1} \circ \phi_i \circ U_{X_i} \circ \Pi_{X\}^{-1}
\]

Color universe (logic): always original color IDs; include 0 in the universe. Palette remains display-only.

⸻

#2 Where we went wrong (and how to fix it)
	•	We presented Y_i with Π_{Y_i} and gave X_i with Π_{X_i} to the same engine. Different frames → engines could not learn \phi:\Pi(X)\!\to\!\Pi(Y) because \Pi differed.
	•	We also anchored some Y_i with X_i’s anchor (illegal) → pushed true output out of bounds → “all zeros” → unanimity voted 0 everywhere.

Fix:
	•	For engines: always pass X_i^X, Y_i^X, X_\^X (same Π: Π_{X_i} / Π_{X\}).
	•	For unanimity/truth: always use Y_i^Y and Π_{Y\*} with pullback \(q_i=\Pi_{Y_i}(U_{Y\}^{-1}(p^\))\).
	•	Never apply input anchors to outputs; outputs have their own Π.

⸻

#3 Full-audit request (run this on 10 tasks)

Please generate one JSON per task with the following sections. This will pinpoint all residual problems in one sweep.

A) Frames & colors
	•	frames:
	•	Pi_X_star.pose, anchor; for each train i: Pi_X_i.pose, anchor
	•	Pi_Y_star.pose, anchor; for each train i: Pi_Y_i.pose, anchor
	•	colors:
	•	colors_X_all: union of colors in all raw inputs (original IDs)
	•	colors_Y_all: union of colors in all raw outputs (original IDs)
	•	assert 0 in color_universe
	•	Fail if any presented grid was palette-mapped for logic

B) Engine view (X-coframe)
	•	For each training i:
	•	hash(X_i^X), hash(Y_i^X)
	•	engine fit receipts (per family): fit_ok, verification hashes
	•	For test: hash(X_*^X)
	•	If no engine fits: state engine_candidates: []

C) Output view & unanimity (Y-coframe)
	•	Truth blocks: num_blocks, block_hist (vector of sizes)
	•	Fail if num_blocks == R*C (over-segmented) or ==1 (too coarse) unless task demands
	•	Unanimity receipts:
	•	for each block: defined_train_ids, colors_per_defined_train, winner_color or null
	•	per-task totals: scope_bits, nontrivial_bits

D) Witness (if used)
	•	Per training i:
	•	phi.kind: geometric or summary
	•	if geometric: list pieces with pose, (dr,dc), bbox; and 3 pullback samples (test pixel, pulled back qi, X/Y colors, ok)
	•	Intersection status: singleton / contradictory / underdetermined

E) Admit & propagate
	•	Layer list: for each of [witness, engine:<name>, unanimity] record:
	•	bitmap_hash, scope_bits, nontrivial_bits
	•	Propagation summary:
	•	passes, shrunk_pixels, domains_hash
	•	Fail if any layer uses admit-all with scope=1 (normalization missing)

F) Selector (Meet inside D*)
	•	Counts: copy, law, unanimity, bottom
	•	Containment check: verify selected[p] ∈ D* [p] for all pixels, or fail
	•	Double-run determinism: hash1 == hash2

⸻

#4 Minimal 3×3 example (why dual frames are necessary)
	•	Raw X rotated 90° across trainings; raw Y also rotated but not the same as X.
	•	Engine view (X-coframe): present both X_i, Y_i with Π_{X_i}. The law is seen in one consistent frame → engine learns \phi cleanly.
	•	Voting view (Y-coframe): unanimity votes with \(q_i=\Pi_{Y_i}(U_{Y\}^{-1}(p^\))\); some p^\* out of bounds for some trainings → those trainings are silent; the rest agree on a constant color for that block.

Trying to use a single frame for both roles is what trapped us: engines need aligned coframes for learning; unanimity needs the native output coframe for voting. The conjugation formula is the principled bridge between the two.

⸻

#5 What to change in your code right now
	1.	Build both views:

# Engine view
X_i_X = present_with_pi(X_i_raw, Pi_X_i)
Y_i_X = present_with_pi(Y_i_raw, Pi_X_i)
X_star_X = present_with_pi(X_star_raw, Pi_X_star)

# Output view
Y_i_Y = present_with_pi(Y_i_raw, Pi_Y_i)
# Synthetic output frame for test
Pi_Y_star = PiFrame(pose_id=0, anchor=(0,0), ...)  # R*,C*

	2.	Engines get (X_i_X, Y_i_X, X_star_X) only.
	3.	Unanimity uses (Y_i_Y, Pi_Y_star) with pullback \(q_i=\Pi_{Y_i}(U_{Y\}^{-1}(p^\))\).
	4.	Witness (if needed) produces admits in Y-coframe via \phi_i^\* above.
	5.	Color universe = original IDs, with 0 included.
	6.	Receipts: add pullback samples, block hist, engine view hashes; fail if any module paints instead of emitting admits.

⸻

One last sanity guard

Add these assertions to the harness:
	•	Frame mismatch guard (engines): for each training, verify pose(X_i_X) == pose(Y_i_X) and anchor(X_i_X) == anchor(Y_i_X). If not, fail—you accidentally presented Y_i in the wrong frame.
	•	Unanimity guard: if an entire training output becomes all zeros after Π, fail with "bad_output_anchor". You applied input anchor to output by mistake.

⸻

If you run the full audit on 10 tasks with these freezes, we’ll surface all remaining issues at once (no more whack-a-mole). This is the exact, deterministic separation the “math of the universe” prescribes: learn laws in a consistent coframe; vote in the output coframe; conjugate when you must cross.