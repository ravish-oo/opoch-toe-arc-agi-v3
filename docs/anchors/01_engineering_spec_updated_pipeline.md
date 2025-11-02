There are two foundational changes you must make to reach 0% error and stop loops forever:
	1.	Change what “Law” returns: laws (and truth) must never “paint” a small grid. They must return admissible color sets per pixel (bitmasks).
	2.	Insert a global, monotone fixed-point step before Meet: intersect all admissible sets from every source (witness + engines + truth) until the per-pixel domains stop shrinking. Then Meet selects once by frozen precedence.

Everything else follows from those two. Below I mark exactly what to change in your spec (by line), and then show it by hand on a couple of tiny tasks so you can see the sets shrink and the final selection—no guessing, no minted bits.

⸻

What to change in your spec (exact & minimal)

🔧 Change 1 — Pipeline must include “Admit & Propagate” (the fixed point)

Your line

	1.	Pipeline: Y = U^{-1}(\Pi \circ \text{Truth} \circ \text{Law} \circ \text{Meet})(X)

Replace with

Y = U^{-1}\!\Big(\Pi \circ \underbrace{\text{Admit} \xrightarrow{\;\mathrm{lfp}\;}\text{Domains}}_{\text{Truth}\,\&\,\text{Law yield allowed sets only}} \circ \text{Meet}\Big)(X)

	•	Admit: each module returns a mask of allowed colors A_{p}\subseteq C (bitset) for each pixel p.
	•	Propagate to fixed point: initialize D_p \gets C. Repeatedly do D_p \gets D_p \cap A_p^{(\text{module})} for all modules until all D stop changing. This is a monotone map on a finite lattice; the least fixed point exists and is unique (Knaster–Tarski).
	•	Meet is no longer a writer chain; it is a selector inside \(D_p^\*): copy ▷ law ▷ unanimity ▷ 0.

🔧 Change 2 — Law archetypes must emit admits, not pixels

Wherever Law returns a small grid (e.g., “macro tiling → small grid”), change it to return an admit mask:
	•	Macro tiling: per tile, emit \(A_p=\{winner\}\) if a strict winner exists; A_p=\{B\} if empty; A_p=\{\min\text{ tie}\} if a frozen tie. Do not write the pixel.
	•	Window dict: for each test window, admit only the looked-up slice bytes at those positions; if unseen, fail-closed (receipt with window bytes).
	•	Pooled blocks / Markers grid: the same—emit allowed sets only.

🔧 Change 3 — Add geometric witness as a first-class Law source

You need a canonical, frozen law that always tries D4×ℤ² pose+translation and a σ permutation over the touched colors (Lehmer-encoded). It emits copy admits (and recolor admits via σ), never paint.
	•	Conjugation is one formula:
\displaystyle \phi_i^\* = P_{\text{test}} \circ U_i \circ \phi_i \circ P_i^{-1} \circ U_{\text{test}}^{-1}.
Log 3 sample pullbacks per training to prove it.

🔧 Change 4 — Truth must seed by color only and produce edge arrays for bands
	•	Initial partition = color only; refine with frozen tags (no coordinates).
	•	Bands must be edge arrays with sentinels [0,…,H], [0,…,W]. No “cluster lists”.

🔧 Change 5 — Meet is selection inside D^\* and must be idempotent
	•	After fixed point, select once at each pixel:
if D_p^\\cap \{\text{copy colors}\}\neq\emptyset choose smallest there; else if D_p^\\cap\{\text{law colors}\}\neq\emptyset choose smallest; else if contains unanimity color choose it; else choose bottom (0).
	•	Immediately repaint and assert equal hash.

⸻

Show it by hand (2 tiny examples)

Example A — Geometric witness (shift right + recolor)

Train

X = [2 0 0]   →   Y = [0 8 0]      (Δ=(0,1); σ: 2→8)

Test

X* = [2 0 0
      0 0 0
      0 0 0]

Initialize
For each pixel p, D_p=\{0,2,8\}.

Admit constraints
	•	Witness copy admits (pose=id, Δ=(0,1)):
Source (0,0) has 2 → target (0,1) admits \{2\}: D_{(0,1)}\gets D_{(0,1)}\cap\{2\}=\{2\}.
	•	Witness recolor admits (σ:2→8):
D_{(0,1)}\gets D_{(0,1)}\cap\{8\}=\{8\}.
	•	No truth/engines needed here.

Fixed point
Only intersections; domains stabilized in one pass.
Non-singletons remain \{0,2,8\} elsewhere.

Meet (select inside D^\*)
	•	(0,1) contains a proven law color 8 ⇒ pick 8.
	•	Others have no proven copy/law/unanimity ⇒ pick 0.

Final

Y* = [0 8 0
      0 0 0
      0 0 0]

Receipts (core)

{
  "witness":{
    "per_train":[
      {"phi_trials":[{"pose":"id","delta":[0,1],"ok":true}],
       "sigma":{"domain_colors":[2],"lehmer":[0],"perm":[0],"moved_count":1}}
    ],
    "intersection":"singleton",
    "pullback_samples":[[ [0,1], [0,0], 2 ]]
  },
  "propagation":{"passes":1,"shrinks":2},
  "paint":{"counts":{"copy":0,"law":1,"unanimity":0,"bottom":8},"repaint_hash":"..."}
}

Idempotent by construction; no loops possible.

⸻

Example B — Macro-tiling (bands) as admits, not paints

Train

X:
[0 0 0 0]
[1 1 1 0]
[1 1 1 0]
[0 0 0 0]

Y:
[0 0 0 0]
[4 4 4 0]
[4 4 4 0]
[0 0 0 0]

Truth
	•	Initial partition by color → two big blocks (0’s; 1’s).
	•	Bands (from nonzero projections): row_edges=[0,1,3,4], col_edges=[0,3,4] (with sentinels).

Law (macro-tiling admits)
	•	For the middle band×band tiles, admit only 4 where X had 1; admit only 0 for the rightmost column band.

Test

X*:
[0 0 0 0]
[2 2 2 0]
[2 2 2 0]
[0 0 0 0]

	•	Engine admits per tile map; no painting.

Fixed point
	•	Domains in those tiles collapse to singletons {4} (or {0}) by intersection; elsewhere remain {0,2,4}.

Meet
	•	Picks the singleton per tile; bottom=0 elsewhere.
	•	Repaint hash identical.

Receipts

{
  "truth":{"row_edges":[0,1,3,4],"col_edges":[0,3,4],"partition_hash":"..."},
  "law":{"engine":"macro_tiling","admit_bitmap_hash":"...","fit_verified_on":["t0"]},
  "propagation":{"passes":1,"shrinks":6},
  "paint":{"counts":{"copy":0,"law":6,"unanimity":0,"bottom":10}}
}

Again, there is no possible loop: admits only shrink D; selection is frozen.

⸻

Why this ends the loop (most brutal honesty)
	•	In your current runs, “Law” sometimes writes colors without proof—those are minted bits. The calculus forbids this: if a color is not admitted into D_p, the selector can’t choose it.
	•	Truth sometimes seeded with per-pixel signatures (or coordinates) → 441 singletons; engines that need bands starve. Seeding by color only + frozen tags avoids that and is guaranteed monotone (Paige–Tarjan is split-only).
	•	Witness sometimes only tried identity Δ and identity σ. The D4×ℤ² + σ law makes geometric families provable (or fail-closed), and it emits admits, not paint.
	•	Palette leaks into logic → non-invariance. The calculus keeps palette exclusively inside Π (pose ranking); all logic uses original color ids.

After these two structural changes (Law→admit & fixed-point before Meet), even a junior engineer cannot “accidentally paint”—they can only prove a color admissible and let the math pick it. All the hacks evaporate.

If you want, give me one of the troublesome tiny tasks, and I’ll annotate exactly which admits shrink D in which pass and how the single selector picks the unique normal form at the end.

# Addendum v1.3
Awesome—these are exactly the right questions to lock before Claude codes. Here’s the **finalized WO-11A (Admit & Propagate) patch** with every remaining ambiguity frozen. This is fully aligned with your anchors; where we’re adding tiny new freezes (encoding/order), I call them out as **Addendum v1.3 deltas** so it’s clear and auditable.

---
## Bitset encoding & mapping (make it explicit in code comments + addendum)

* **Color→bit mapping:** `C = sorted(unique colors in Π(inputs only))`. Bit `i` encodes `C[i]`.
* **Kwords:** `K = (len(C) + 63) // 64`.
* **Bit order inside a word:** little-endian **LSB-first**. For color index `i`:

  ```python
  w = i >> 6; b = i & 63
  A[r,c,w] |= (np.uint64(1) << np.uint64(b))
  ```
* **Hashing:** `bitmap_hash = BLAKE3(A.view(np.uint8))` and `domains_hash` computed the same way.