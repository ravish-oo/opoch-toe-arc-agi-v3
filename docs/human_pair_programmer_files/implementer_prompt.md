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

# Larger understanding prompt:
like i said we are just getting started and only hv docs setup, now i want u to read @docs/anchors/01_engineering_spec.md and
  @docs/anchors/02_determinism_addendum.md and assess for urself if this aligns with @docs/anchors/00_math_spec.md and ur understanding.
  looking at this can u establish:
  1. with these specs that engineering = math. The program is the proof.
  2. u can see that debugging is reduced to algebra
  3. no room for hit and trials

# wo prompt
here is the WO. do refer to @docs/anchors/03_repo_structure.md to knw the folder structure.
  [Pasted text #1 +161 lines]
  ---
  pls read and tell me that u hv understood/confirmed/verified below:
  1. have 100% clarity
  2. WO adheres with ur understanding of our math spec and engg spec and that engineering = math. The program is the proof.
  3. u can see that debugging is reduced to algebra and WO adheres to it 
  4. no room for hit and trials

once u confirm above, we can start coding!


# wo notes
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