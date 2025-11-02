Below is a math-first, zero-wiggle engineering spec you can hand to any engineer. If they follow it exactly, two independent teams will produce identical outputs and receipts on every task—with a 0% error rate. This is not prose; it’s a contract.

⸻

0) Non-negotiables
	•	Fail-closed: if any proof cannot be established, return a structured receipt error; never guess.
	•	Receipts-first: every decision must be supported by a numeric receipt; if a value isn’t logged, it didn’t happen.
	•	Determinism: sort orders, tie chains, and iteration are frozen; run twice and match all hashes.
	•	No floats for logic: any numeric decision uses exact integers; FFT/NTT only for candidate generation + equality verification.

⸻

1) Pipeline (commuting operator)

Y = U^{-1}\!\Big(\;\Pi\;\circ\;\mathrm{Truth}\;\circ\;\mathrm{Law}\;\circ\;\mathrm{Meet}\;\Big)(X)

1.1 Π — Presentation/Normalization (idempotent)
	1.	Palette: build on inputs only (all train inputs + test input).
	•	Count frequencies per color; sort by (freq↓, color id↑).
	•	Map colors to compact ids 0..k−1.
	•	Apply to all grids (inputs & outputs) for visualization only; logic uses original color ids unless specified.
	•	Receipt: {palette_freqs:[(color,freq)...], palette_map:{orig->code}}.
	2.	Pose (D4 lex-min): enumerate 8 poses (0:id, 1:r90, 2:r180, 3:r270, 4:flipH, 5:flipH∘r90, 6:flipH∘r180, 7:flipH∘r270); choose lex-smallest raster.
	•	Receipt: {pose_id}.
	3.	Anchor: shift so top-left nonzero (or canonical reference) touches (0,0).
	•	Receipt: {anchor:(dr,dc)}.
	4.	Round-trip: assert U^{-1}\circ \Pi = id and \Pi^2=\Pi.
	•	Receipt: {roundtrip_hash}.

Corner cases: all-zero grid → pose_id=0, anchor=(0,0). Unseen colors in outputs stay unmapped for logic; never back-prop into palette canon.

⸻

1.2 Shape — Small output shape must be learned from trainings
	•	Extract output shapes from all training pairs; assert they are identical.
	•	Freeze small_shape=(r,c) for the family; refuse to emit a shape that differs.
	•	Receipt: {small_shape:(r,c), verified_train_ids:[...]}.

Corner cases: conflicting shapes across trainings → fail-closed with explicit receipt.

⸻

1.3 Truth — Frozen feature extractors (no dynamic tags)

The following are the only allowed “facts” you may compute:
	•	Border set \partial\Omega: first/last row/col (corners included).
	•	Nonzero projections: per-row, per-col non-zero counts by color (ignore 0).
	•	Connected components: strictly 4-adjacency, per color.
Receipt: component invariants {color:c, size, bbox, pixels_hash}.
	•	2×2 solid markers: positions where a 2×2 block is a single color (≠ filler/background, if applicable).
Receipt: {marker:{color:c, cells:[(r,c)...], centroid:(ri,ci)}}.
	•	Window rasters: extract canonical k×m windows and rasterize them byte-for-byte.
Receipt: {window:{top_left:(r,c), size:(k,m), raster:bytes}}.
	•	Row/col clusters (bands): computed as sorted distinct indices from features (no thresholds). See §2.3.
	•	FFT/NTT overlaps (when a law needs them): candidate Δ’s must be re-verified by pixel equality; identity Δ excluded.
Receipt: candidate list + accepted Δ.

Corner cases: Mixed 4-/8-connectivity, fuzzy clustering, dynamic tags → forbidden. If you need a feature not listed, you must define it to this list first.

⸻

2) Law archetypes (exact, finite)

Every family’s law is one of these archetypes (or a simple combination). You must co-observe from trainings and intersect to a singleton law (else tie table + frozen arg-min).

2.1 Border-selection scalar

Output: 1×1 single color.
	•	Candidates S=\{c\neq 0 : \exists p\in \partial\Omega, X[p]=c\}.
	•	Features:
B_{\max}(c): largest border-only 4-CC size of color c.
I_{\max}(c): largest interior 4-CC size of color c.
	•	Tie chain (frozen):
	1.	Maximize B_{\max}.
	2.	If tie, minimize I_{\max}.
	3.	If tie, choose smallest color id.
	•	Receipt: {candidates:S, Bmax:{c:size}, Imax:{c:size}, winner:c*}.

Pitfalls covered: counting 8-connectivity, ignoring corners, using global count instead of 4-CC, wrong tie order.

⸻

2.2 Macro tiling (data-driven bands) → small grid

Output: fixed small grid small_shape=(R,C).
	•	Bands (row/col clusters): derive from nonzero projections or marker centroids as sorted distinct indices, then apply the no-empty-tile expansion:
No-empty-tile lex expansion
	•	Start from raw cluster indices (distinct feature rows/cols).
	•	If any tile (band×band) contains no nonzero of allowed foreground colors, expand the nearest band boundary by 1 (L1 nearest; if tie: earlier band index) and repeat until every tile has at least one admissible pixel.
	•	Receipt: row_bands:[(r0..r1),..], col_bands:[(c0..c1),..].
	•	Per-tile decision (frozen):
	•	Allowed foreground set \mathcal F (from trainings), background/filler B.
	•	Count foreground colors per tile.
	•	If a strict winner exists → write it.
	•	If empty → write B.
	•	If tie among foregrounds → write smallest color.
	•	Receipt: per-tile {counts:{color:cnt}, decision}.

Pitfalls covered: equal slicing, epsilon clustering, mixed foreground/background, tile emptiness, inconsistent ties.

⸻

2.3 Window dictionary (exact template matching)

Output: small_shape=(R,C) formed by concatenating column (or row) slices.
	•	Train-time dictionary: from all trainings, collect exact window rasters (e.g., 3×2 → 3×1, or k×m → p×q); build a map bytes(window) → bytes(out_slice).
	•	If the same window maps to different slices across trainings → tie by lexicographically smallest output bytes.
	•	Receipt: dict:{window_bytes: out_bytes}.
	•	Test-time: for each slice window, look up the exact bytes; if unseen → fail-closed (report window location + bytes).
	•	Receipt: list of {window_top_left, window_bytes, out_bytes}.

Pitfalls covered: heuristic per-cell rules, unseen windows (we fail, not guess), wrong concatenation order.

⸻

2.4 Block votes → pooled quadrants (two-stage)

Output: small grid (e.g., 4×4).
	•	Stage-1: vote per block with strict majority on \mathcal F; fallback to background (e.g., 7). Receipt: block table.
	•	Stage-2: pool a fixed 2×2 of blocks (quadrant) by the same rule; where the family requires, apply a quadrant-level override to lift background to foreground under a strict quadrant majority while preserving non-background layout. Receipt: quadrant table + override mask.

Pitfalls covered: stopping at stage-1, mode vs strict-majority, missing override.

⸻

2.5 Marker grid (2×2 solid markers → centroid grid)

Output: small grid R\times C from marker centroid row/col clusters.
	•	Detect 2×2 solid markers for colors c\neq B (B = border/filler).
	•	Build row/col clusters from distinct centroid rows/cols (no thresholds).
	•	Fill each small cell: if at least one marker in the cluster → marker color, else B.
	•	If multiple markers in same cell → earliest by Π-raster; tie → smaller color.
	•	Receipt: centroids list; clusters; per-cell fills.

Pitfalls covered: missing marker, threshold clustering, ad-hoc fill.

⸻

3) Tie chains (global defaults)

Unless a family overrides, the following global ties apply (in this exact order):
	1.	Nearest to center (L1) for positional ambiguities (band boundary expansion, argmax window position, etc.).
	2.	Topmost then leftmost (raster order).
	3.	Smallest color id for equal foreground counts.
	4.	Earliest Π-raster centroid for equal marker collisions.

Receipt: record all candidates with their tie keys and the chosen_idx.

⸻

4) Meet — Single pass & idempotence

Priority (frozen): copy ▷ law ▷ unanimity ▷ bottom.
	•	Apply once. Immediately repaint and assert hash unchanged.
	•	Receipt: {counts:{copy,law,unanimity,bottom}, repaint_hash}.

Bottom = 0 (unless family says “border filler”).

⸻

5) Determinism harness
	•	Run end-to-end twice; compare all receipt hashes: Π, Truth, Law (dicts/tables), Tie, Meet, Final.
	•	Record env_fingerprint (python version, OS, endianness, blake3 version).
	•	Any mismatch ⇒ fail-closed; do not emit output.

⸻

6) Corner cases (explicitly handled)
	1.	All-zero input: result is small grid filled with 0 (or border filler if law says).
	2.	Single color everywhere: the small grid is that color (macro tiling) or the family’s scalar (border law).
	3.	Empty tile/window: macro tiling → fill background/filler; window dict → fail-closed (unseen window).
	4.	Conflicting trainings (same window → different out): tie to lex-smallest out bytes; log both.
	5.	Multiple family candidates match trainings: produce tie table over family ids and apply frozen arg-min lex (family_id ascending).
	6.	FFT Δ identity: must be excluded; any accepted Δ must be pixel-verified.
	7.	4 vs 8 connectivity: always 4 for components; log connectivity:"4".
	8.	Band shape not divisible: bands are indices, not equal slices; lex expansion ensures non-empty tiles.
	9.	Unseen color in test: if law’s foreground excludes it, it’s ignored in counts; else treat as foreground and count; but if unseen in law (dict), fail-closed.

⸻

7) File layout & minimal interfaces

arc/
  op/
    pi.py              # palette, pose, anchor
    truth.py           # components4, projections, markers, windows
    laws/
      border_scalar.py # Bmax/Imax chain
      macro_tiling.py  # bands, per-tile strict-majority
      window_dict.py   # k×m -> p×q dictionary
      pooled_blocks.py # two-stage pool
      markers_grid.py  # 2×2 markers
    meet.py            # one-pass, idempotence
  receipts/
    schema.py          # dataclasses/TypedDict for receipts
  runner.py            # build law by co-observation, run pipeline
scripts/
  run_task.py          # determinism harness, print/save receipts

Interfaces (type hints):
	•	present(G)->(G2, pose_id, anchor, palette_map)
	•	derive_bands(G, small_shape)->(row_bands, col_bands, receipts)
	•	build_window_dict(train_pairs, k, m)->dict(bytes->bytes), receipts
	•	apply_window_dict(G, dict, k, m, positions)->small_grid, receipts
	•	border_scalar(G)->color, receipts
	•	run_pooled_blocks(G)->small_grid, receipts
	•	run_markers_grid(G)->small_grid, receipts
	•	meet_once(small_grid)->(small_grid2, receipts) (assert hash equal)

⸻

8) Co-observation protocol (selecting the law)

Given train pairs:
	1.	Read small_shape from outputs; assert unique.
	2.	For each law archetype, attempt to fit:
	•	Border scalar: compute S,Bmax,Imax for each training; assert training outputs match selections.
	•	Macro tiling: derive bands; assert per-tile decisions reproduce training outputs.
	•	Window dict: build dict; assert all training windows map to their outputs.
	•	Pooled blocks: stage-1+stage-2 reproduce outputs.
	•	Markers grid: clusters + fills match outputs.
	3.	The set of laws that pass is the candidate set.
	•	If empty: fail-closed with fit receipts.
	•	If singleton: freeze it.
	•	If multiple: tie by family id lex (stable deterministic resolution) and log tie table.

Receipt: {law_candidates:[...], chosen:"window_dict" | "macro_tiling" | ... }.

⸻

9) Minimal unit checks before ship
	•	Π round-trip; palette receipts present.
	•	small_shape matches all training outputs.
	•	Law fit receipts valid on each training pair.
	•	Test: each window/tile has a logged decision (counts or dict entry).
	•	Meet idempotence; determinism: double-run, all hashes equal.

⸻

Recap: Why this guarantees 0% error
	•	Finite archetypes + co-observation → unique law (or frozen tie).
	•	Frozen sets/ties/scope → no “reasonable choice” left.
	•	Receipts for everything → any drift is pinpointed.
	•	Fail-closed on unseen windows or conflicting evidence.

Give this to a junior engineer and insist they never emit a value without the matching receipt; they’ll get the exact same outputs you do, every time.