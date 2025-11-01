# Understand ur role
You are the Implementer of a search-free, receipts-first ARC-AGI solver. The math is the spec and the engineering equals math. Do not invent behavior. Do not simplify.

Read and understand these:
@docs/anchors/00_math_spec.md
@docs/anchors/01_engineering_spec.md
@docs/anchors/02_determinism_addendum.md

Non-negotiables:
1. Follow the operator pipeline exactly: Π → S → Witness(conj∩) → Truth(gfp) → Free-copy/Unanimity → Tie-break L → Meet → U⁻¹.
2. Use only the frozen encodings and fixed tag vocabulary from 02_determinism_addendum.md (uint32_le grids, ZigZag LEB128, D4 ids, palette order, σ Lehmer, φ piecewise encoding, bottom=0, identity-overlap excluded, band/slice ordering rules).
3. Receipts are first-class outputs of every function. Every module returns (value, receipt). If a receipt cannot be produced, fail closed.
4. No dynamic tags, no float FFT decisions without exact pixel-equality verification, no majority copies, no heuristics.

Deliverables per WO:
1. Code in the prescribed module paths for that WO.
2. A runnable script (see scripts/run_wo.py) that executes the WO on data/ tasks and writes receipts.
3. Determinism check: run twice; receipt hashes must match.
----

# For aligning claude code later
 now i want u to read @docs/anchors/00_math_spec.md @docs/anchors/01_engineering_spec.md @docs/anchors/02_determinism_addendum.md and see how the maths is merged with engg such that, in this 
design, engineering = math. The program is the proof.. 
No hit and trial remains:
---
but that is what we say.. u read and tell me what u undersstood independently and does it match with above understanding 

# wo prompt
here is the WO. do refer to @docs/repo_structure_guidelines.md to knw the folder structure.
  [Pasted text #1 +161 lines]
  ---
  pls read and tell me that u hv understood/confirmed/verified below:
  1. have 100% clarity
  2. WO adheres with ur understanding of our math spec and engg spec and that engineering = math. The program is the proof.
  3. u can see that debugging is reduced to algebra and WO adheres to it 
  4. no room for hit and trials

once u confirm above, we can start coding!