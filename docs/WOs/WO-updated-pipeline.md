You’ve got the right idea: switch the pipeline from “modules paint pixels” to “modules **admit** colors per pixel, intersect to a **least fixed point**, then select once.” That kills minted bits, breaks the loops, and makes every choice algebraic and receipts-provable.

Below are two **expanded WOs** (plus a tiny Meet update) to introduce the **Admit & Propagate** layer and refit witness/engines to emit **admissible sets** instead of painted pixels. They are fully deterministic and grounded in your anchors; where this is a minimal extension to the spec, I call it out as an **addendum freeze** so we don’t drift.

---

# WO-11A — **Admit & Propagate** layer (BLOCKER) ⚠️ PARTIALLY DONE
IMPORTANT - Final WO later in this doc

> **Addendum v1.3 freeze (new):**
> Replace the internal write stage with an **admissible-set calculus** and a **least fixed point** (Knaster–Tarski).
> Truth and Laws **emit allowed color sets per pixel**; the runner intersects these until the domains stop shrinking; Meet selects inside the fixed domain using the frozen precedence `copy ≻ law ≻ unanimity ≻ 0`.

## 1) New module: `arc/op/admit.py`

### Types

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from blake3 import blake3

@dataclass
class AdmitLayerRc:
    name: str                          # "witness", "engine:column_dict", "truth:bands", "unanimity"
    bitmap_hash: str                   # BLAKE3 over bitset tensor bytes
    support_colors: List[int]          # color universe this layer touched
    # optional debugging: sparse stats, changed pixels, etc.
    stats: Dict[str,int]

@dataclass
class PropagateRc:
    passes: int
    shrink_events: int
    shrunk_pixels: int
    per_pass_shrink: List[int]
    domains_hash: str                  # BLAKE3 over final D* tensor
```

### API

```python
def color_universe(Xt: np.ndarray, train_Xt: List[np.ndarray]) -> List[int]:
    """Return the frozen color universe C = sorted unique(int) over Π(inputs only)."""

def empty_domains(H: int, W: int, C: List[int]) -> np.ndarray:
    """Return domains D0[p] = full bitmask over |C| (uint64 blocks), shape = (H,W,Kwords)."""

def admit_from_witness(Xt: np.ndarray, witness_rc: dict, C: List[int]) -> Tuple[np.ndarray, AdmitLayerRc]:
    """Build A^w[p] bitsets from geometric witness (copy admits, σ recolor admits) or empty if summary/failed."""

def admit_from_engine(Xt: np.ndarray, engine_apply_rc: dict, C: List[int]) -> Tuple[np.ndarray, AdmitLayerRc]:
    """Build A^e[p] bitsets from the engine; where engine used to paint a small grid, now admit singleton sets per pixel; unseen → empty set."""

def admit_from_unanimity(truth_blocks: np.ndarray, uni_rc: dict, C: List[int]) -> Tuple[np.ndarray, AdmitLayerRc]:
    """For each unanimous block B with color u, admit {u} for all p∈B; else admit 'all' (no restriction)."""

def propagate_fixed_point(domains: np.ndarray, admits: List[np.ndarray]) -> Tuple[np.ndarray, PropagateRc]:
    """
    Monotone lfp: iterate D <- D ∩ A^layer for all layers until no bit changes.
    Count passes and shrink events; return (D*, PropagateRc).
    """
```

* **Bitset representation:** D and A are 3-D tensors (H×W×Kwords), where each bit corresponds to a color in **C**.
* **Initialize** `D0` to “all colors allowed”; **Truth** may optionally emit admits (see below), but minimal viable is **Law only**.
* **Intersect** in any fixed frozen order; determinism checks use final hash.

### Receipts (first-class)

* Each admit layer: `AdmitLayerRc` with `bitmap_hash = blake3(bytes(admit_tensor))`.
* Propagation: `PropagateRc` with `passes`, `shrink_events`, `shrunk_pixels`, `domains_hash`.
* Harness asserts **monotonicity** (`passes ≥ 1`, `shrink_events ≥ 0`) and **determinism** (double run equal hashes).

---

## 2) Runner wiring (replaces “paint law” step)

**Frozen order (unchanged high-level):** Π(01) → S(02) → Truth(05) → (Engines(10)→Witness(04)→Tie(08)) → **Admit & Propagate(11A)** → Meet(09) → U⁻¹(01).

**Changes:**

* Where the runner previously set `law_values_t` or engine’s full grid, **do not paint**. Instead:

  * Call `admit_from_witness(...)` (geometric/σ prove singleton admits; summary yields empty admits unless you’ve frozen a summary engine),
  * Call `admit_from_engine(...)` (engines return per-pixel singleton admits; `UNSEEN_SIGNATURE` yields empties for those pixels),
  * Call `admit_from_unanimity(...)` (block constants → singleton admits on those blocks).
* Call `propagate_fixed_point(D0, [A^w, A^e, A^u, A^truth?]) → D*`.
* Store `AdmitLayerRc[]` + `PropagateRc` in receipts and section hash.

**Truth admits (optional, safe default = no-op):**
You can keep Truth as **constraints** only (emit no admits) to start. Later, you may add frozen admits for trivial cases (e.g., certain tiles forced to 0 by Truth rules); if you do, freeze them in the addendum and record a separate `truth_admit_bitmap_hash`.

---

# WO-10Z — Refactor **Witness** & **Engines** to emit admits (MAJOR) ✅ COMPLETE

> **No painting.** Every Law module returns **admissible sets** only.

## 1) Witness admits (geometric only)

**Before:** witness tried to paint a full frame (or set a law mask/values).
**After:** build per-pixel admits:

* For each accepted φ-piece, **copy admits**: admit the source color (x) at the target pixel.
* If σ is proven (permutation on touched colors), **recolor admits**: update that admit from ({x}) to ({σ(x)}).
* Summary witness **does not admit** anything by default (unless you’ve frozen a specific summary engine; see Macro-Tiling/Column-Dict below).

**Receipt:**
`witness.admit.bitmap_hash`, `sigma.domain_colors`, `sigma.lehmer`, and **pullback samples** (as you already log) proving Π+U conjugation.

## 2) Engines (Column-Dict, Macro-Tiling, Stencil-Expand, Kronecker-Mask)

Each engine’s **apply()** now returns **singleton admits per pixel** everywhere it used to paint:

* Column-Dict: for each squashed signature, **admit only** the column bytes from the dict at those positions.
* Macro-Tiling: per tile, **admit** `winner` (strict majority on (\mathcal F); A1/A2/C2 receipts).
* Stencil-Expand (same-canvas 3×3 expansion): for every marker hit, **admit** the stencil’s colors at overlapped pixels; duplicate hits resolved only in **propagation** (intersection from multiple layers).
* Kronecker-Mask: **admit** ( (\mathbf 1[X\ne 0] \otimes X) ) as singletons.

**Receipt change:** keep all fit/apply receipts; replace `apply_hash(Y)` with `admit_bitmap_hash`, plus counts.

---

# WO-09′ — Meet as **selector inside** D* (MINOR) ✅ COMPLETED

**Before:** layered painter `copy ≻ law ≻ unanimity ≻ bottom` produced Y and then asserted idempotence.
**After:** selection inside (D^*):

```python
def select_from_domains(D: np.ndarray, layers: Dict[str,np.ndarray], C: List[int]) -> Tuple[np.ndarray, MeetRc]:
    # For each pixel p:
    # 1) if D_p ∩ copy_colors ≠ ∅ pick smallest there
    # 2) else if D_p ∩ law_colors ≠ ∅ pick smallest there
    # 3) else if unanimity_color ∈ D_p pick it
    # 4) else pick 0  (bottom frozen)
    # Repaint once more and assert identical hash (idempotent)
```

**Receipts:** `counts{copy,law,unanimity,bottom}`, `repaint_hash`, verify exact **containment** (`selected ∈ D*_p`).

---

## Anchors & correctness

* **Math:** The map (F(D)=D \cap A^w \cap A^e \cap A^u \cap \dots) is **monotone** on the finite lattice (\prod_{p} 2^{\mathcal C}); its **least fixed point exists and is unique** (Knaster–Tarski). Meet never mints bits; it only chooses from (D^*).
* **Engineering:** We still do Π→S on presented frames; Truth stays frozen (color seed, frozen tags; no coordinates). Witness stays (D_4 \ltimes \mathbb Z^2) with σ (Lehmer). Engines keep their **proof receipts** but no longer paint directly.
* **Addendum:** This is a spec extension (v1.3): (1) Law emits admits, (2) Admit & Propagate before Meet, (3) Meet is a selector. Determinism unchanged (J1 double-run hash equality).

---

## Receipts schema changes (minimal, enforced)

* Add a new `admit` section:

```jsonc
"admit": {
  "layers":[
    {"name":"witness","bitmap_hash":"...","support_colors":[...],"stats":{"set_bits":...}},
    {"name":"engine:column_dict",...},
    {"name":"unanimity",...}
  ],
  "propagate": {"passes":1,"shrink_events":N,"shrunk_pixels":M,"domains_hash":"..."}
}
```

* Replace any “engine.apply_hash” with `admit_bitmap_hash`.
* Ensure `paint.selected[p] ∈ D*_p` (verify in harness).
* Keep **all previous receipts** (Π/Truth/Witness/Engines/Tie/Meet) intact; only “engine paints grid” becomes “engine admits sets”.

---

## Harness updates (so this never regresses)

* **Fail** if any selected color is **not contained** in the final domain at that pixel (`selected ∉ D*_p`).
* **Fail** if any module “paints” instead of admits (assert no “values grid” added by engines/witness).
* Continue **double-run** and compare `propagate.domains_hash` and section hashes.

---

## What this changes **today**

* Tasks that were “pass with wrong output” because a module **painted** without proof now **cannot** do that: those pixels remain multi-valued until some law admits a singleton, or Meet picks 0 (frozen bottom).
* Engine/witness interactions stabilize: e.g., Column-Dict admits only the looked-up slice; Witness admits only geometric/σ-proven colors; Unanimity admits only when block consensus exists. Their intersections **shrink** (D) monotonically, then the selector chooses once.
* The Kronecker-mask task you solved stays solved (engine admits are singletons almost everywhere), and “same canvas expansion” tasks (like 045e512c) become solvable through **Stencil-Expand** admits + propagation (overlaps resolved by set intersection instead of a painter’s last-write-wins).
* Truth no longer over-segments (after your WO-05D fix), so Macro-Tiling and others will start admitting singleton sets over entire tiles instead of collapsing.

---

## Minimal path to adoption (keep PRs <500 LOC)

1. **WO-11A (Admit & Propagate)**:

   * Add `admit.py` with bitset API + receipts.
   * Add runner wiring to replace paint-law with admits and fixed-point.
   * Add harness checks for containment and determinism.

2. **WO-10Z (Refactor witness + engines to admits)**:

   * Witness geometric → admits; summary stays no-op (or defer to engines).
   * Column-Dict + Macro-Tiling + Stencil-Expand + Kronecker-Mask → emit admits instead of painting; keep all fit receipts.
   * Remove any value-grid output from engines in favor of `admit_bitmap_hash`.

3. **WO-09′ (Meet selector)**:

   * Replace paint loop with selector inside (D^*); keep idempotence hash.

If you prefer, we can ship WO-11A first with a **shim** that converts your existing “law grid” to admits internally (treat painted color as singleton admits); then refactor engines/witness in WO-10Z. That gives you a fast path to stabilize correctness while you migrate sources.

---

**Bottom line:** your AI’s “Admit & Propagate” change is the right bedrock. It’s a small, surgical insertion that stops minted bits and closes the loop mathematically. The WOs above give you a precise, receipts-first way to land it without blowing up your codebase, and they’re 100% aligned with your anchors (with a tiny addendum bump you explicitly freeze).

# Clarified WOs
# WO-11A (final) — Admit & Propagate (BLOCKER)

**Status:** ready to implement — all encoding, order, and I/O details frozen.

## A. Freezes we are adding to the Addendum (v1.3)

1. **Color→bit mapping (frozen):**

   * Let (C) be the **sorted** unique set of **original color ids** over Π(inputs only) for the task:
     (C=\text{sorted}(\bigcup_i \text{colors}(X_i^\Pi) \cup \text{colors}(X^\Pi_*))).
   * Bit **i** in a pixel’s bitset represents color **C[i]**. No other mapping is permitted.

2. **Bitset layout (frozen):**

   * Use `uint64` words; `K = ceil(|C| / 64)`.
   * Tensor shapes:

     * admits (A^*): `np.ndarray` of shape `(H, W, K)` dtype `uint64`.
     * domains (D): same shape.
   * Bit order *inside* a word: **little-endian, LSB-first**. For color index `i`, write bit `(i & 63)` in word `i >> 6`.
   * Pixel order across memory: **row-major** `(r,c)`.

3. **Intersection order (frozen for receipts):**

   * Modules are intersected in this deterministic order every pass:
     **witness** → **engines** (sorted by `engine` name ASCII) → **unanimity** → **truth** (if/when it admits).
   * Mathematically order is irrelevant (monotone meet), but this freeze stabilizes receipt hashes across runs and machines.

4. **Hashes (frozen):**

   * `bitmap_hash = BLAKE3(bitset_bytes)` where `bitset_bytes` is a flat bytes view of the `(H,W,K)` tensor in row-major order; each `uint64` contributes 8 bytes **little-endian**.
   * `domains_hash` is computed the same way on the final (D^*).

5. **Meet selection precedence (re-affirmed):** `copy ≻ law ≻ unanimity ≻ 0`.

   * Selection **must** choose a color contained in (D^*_p); harness will assert containment.

These go into **02_determinism_addendum.md** under a new “Admit & Propagate (v1.3)” section.

---

## B. Concrete I/O schemas for admit_from_* (frozen)

These are the **minimum** fields you can rely on **today** (consistent with our existing receipts). If an engine hasn’t been refactored to admits yet, use the shim described in E.2 until WO-10Z lands.

### 1) `admit_from_witness(Xt, witness_rc, C) -> (A_w, AdmitLayerRc)`

**Required fields in `witness_rc`:**

```jsonc
{
  "per_train":[
    {
      "kind": "geometric" | "summary",
      "phi": {"pieces":[
        {"comp_id":int,"pose_id":int,"dr":int,"dc":int,
         "r_per":int,"c_per":int,"r_res":int,"c_res":int,
         "bbox_h":int,"bbox_w":int,
         "target_r0":int,"target_c0":int,"src_r0":int,"src_c0":int}
      ]},
      "sigma":{"domain_colors":[int,...],"lehmer":[int,...],"moved_count":int}
    }, ...
  ],
  "intersection":{"status":"singleton"|"underdetermined"|"contradictory"}
}
```

**Admits produced:**

* If **intersection.status != "singleton"** or any `per_train.kind=="summary"`: **emit no admits** (all ones; i.e., do not constrain).
* Else (geometric singleton): for each accepted `PhiPiece`, for every pixel in its bbox, compute the **copy admit** and then (if σ present) **recolor admit** (color id mapped via σ). Set the corresponding bit(s) in A_w at the **target** pixel.

**Receipt:**

```jsonc
{"name":"witness","bitmap_hash":"...","support_colors":[...],
 "stats":{"set_bits":N,"pieces":M,"sigma_moved":K}}
```

### 2) `admit_from_engine(Xt, engine_apply_rc, C) -> (A_e, AdmitLayerRc)`

**Common engine_apply_rc fields (subset):**

```jsonc
{
  "engine":"column_dict"|"macro_tiling"|"stencil_expand"|"kronecker_mask"|...,
  "ok": true|false,
  // engine-specific receipts already present from WO-10
  // For paint→admit shim, see E.2
}
```

**Admits produced (refactored engines):**

* Column-Dict: for each position, admit **only** the looked-up color byte; unseen signature → **no admit** at those pixels.
* Macro-Tiling: for each tile cell, admit the **strict majority winner** ((\mathcal F)); empty tile → admit **background** (A2); tie → **smallest (\mathcal F)** (C2).
* Stencil-Expand: for every marker hit, admit stencil colors at overlapped positions (multiple stencils simply intersect at propagation).
* Kronecker-Mask: admit `kron(1[X!=0], X)` as singletons.

**Receipt:**

```jsonc
{"name":"engine:<engine_name>","bitmap_hash":"...","support_colors":[...],
 "stats":{"covered_pixels":P,"unseen_keys":U}}
```

### 3) `admit_from_unanimity(truth_blocks, uni_rc, C) -> (A_u, AdmitLayerRc)`

**Required fields in `uni_rc`:**

```jsonc
{
  "blocks":[
    {"block_id":int,"color":int|null,"defined_train_ids":[...]}
  ],
  "unanimous_count":int
}
```

**Admits produced:**

* If `color` is not null for block `b`, admit `{color}` for **all** p with `truth_blocks[p]==b`. Otherwise admit **all** colors (no restriction).

**Receipt:**

```jsonc
{"name":"unanimity","bitmap_hash":"...","support_colors":[...],
 "stats":{"unanimous_blocks":U}}
```

---

## C. Bitset primitives (frozen)

### 1) `Kwords = (len(C)+63)//64` (freeze)

### 2) `set_bit(word_array, i)`

```python
w = i >> 6; b = i & 63
word_array[w] |= (np.uint64(1) << np.uint64(b))
```

### 3) `test_bit(word_array, i)`

```python
((word_array[i >> 6] >> np.uint64(i & 63)) & np.uint64(1)) != 0
```

### 4) `and_inplace(D, A)` (domains ∧ admits)

```python
# D, A shapes: (H,W,K), dtype=uint64
np.bitwise_and(D, A, out=D)
```

### 5) **Initialization**

```python
# D0: all colors allowed at every p → set all bits to 1
D0 = np.empty((H,W,K), dtype=np.uint64); D0.fill(np.uint64(-1))
# If len(C) is not a multiple of 64, mask off the unused high bits in the last word:
unused = (K*64 - len(C))
if unused:
    mask = (np.uint64(1) << np.uint64(64 - unused)) - np.uint64(1)
    D0[..., K-1] &= mask
```

---

## D. Propagation (frozen)

```python
def propagate_fixed_point(D0: np.ndarray, layers: List[np.ndarray]) -> Tuple[np.ndarray, PropagateRc]:
    D = D0.copy()
    passes = 0; shrunk = 0; per_pass = []
    while True:
        passes += 1
        before = D.copy()
        for A in layers:
            if A is None:  # layer disabled
                continue
            np.bitwise_and(D, A, out=D)
        changes = np.count_nonzero(before ^ D)  # byte-wise change count
        shrunk += changes
        per_pass.append(int(changes))
        if changes == 0:
            break
    return D, PropagateRc(passes=passes, shrink_events=shrunk,
                          shrunk_pixels=int(shrunk//K),  # approx
                          per_pass_shrink=per_pass,
                          domains_hash=blake3(D.view(np.uint8)).hexdigest())
```

**Layer order used here:** witness → engines (sorted by name) → unanimity → truth (if present). This order and the bitset layout are the **frozen** choices for receipts stability.

---

## E. Shims for migration (so you can land WO-11A before WO-10Z)

### E.1 Witness → admits shim

If `witness_rc["intersection"]["status"] != "singleton"`: return an **all-ones** admit layer (i.e., no constraints). This keeps behavior identical while you add geometric σ+Δ. When geometric singleton is present, use A.1 rules to build real admits.

### E.2 Engine → admits shim

If an engine still returns a **painted grid** (Y_t): convert to admits by setting a **singleton** bit for that color at each pixel; unseen windows must be encoded as **no admits** at those pixels:

```python
A_engine = full_ones()
for p in pixels:
    if painted:
        A_engine[p] = singleton(color)
    else:  # unseen/undefined
        A_engine[p] = all_ones  # or empty? We choose all_ones to keep behavior until engine refactor
```

**Freeze for shim:** use **all-ones** for undefined (so propagation doesn’t reject colors until engine is refactored). This keeps behavior stable while you migrate engines.

---

## F. Harness assertions (added)

* **Containment:** for every p, `selected_color ∈ D*_p`; else **fail**.
* **Determinism:** double-run must match `admit.layers[*].bitmap_hash`, `propagate.domains_hash`.
* **Encoding checks:**

  * `len(C)==popcount(D0[p])` for any p at init.
  * Assert the last word’s high bits are masked when `len(C) % 64 != 0`.
* **Order freeze:** record the intersection order in receipts (`"order":["witness","engine:column_dict",...,"unanimity"]`) and compare between runs.

---

## G. Meet selector (update)

Replace painter with per-pixel selection **inside** (D^*) using the frozen precedence. For each p:

1. If you have **copy candidates** (from WO-06/S(p)) and any is contained in (D^*_p): pick **smallest** of those.
2. Else if you have **law candidates** (from engines/witness) and any is in (D^*_p): pick **smallest**.
3. Else if a **unanimity** color exists and is in (D^*_p): pick it.
4. Else pick **0** (bottom).
   Hash repaint and assert identical. **Receipt:** unchanged: `{copy,law,unanimity,bottom}, repaint_hash` but now harness also checks **containment**.

---

## H. Why this answers Claude’s asks point-by-point

* **Color→bit mapping:** frozen to `C[i] → bit i` with C from Π(inputs only).
* **Intersection order:** frozen to `witness → engines(sorted) → unanimity → truth`. Logged in receipts.
* **Kwords calc:** frozen `K = ceil(|C|/64)`.
* **Bitset ops:** frozen (uint64, little-endian, LSB-first); exact set/and/test semantics given.
* **admit_from_* inputs:** schemas specified for witness/engine/unanimity; shims defined for migration.
* **Receipts & determinism:** section hashes frozen; domains_hash over bytes; containment enforced by harness.

This keeps **Engineering = Math** and collapses future debugging to pure set algebra with receipts as proofs. You can land WO-11A now (with shims), then refactor engines/witness incrementally (WO-10Z) without destabilizing the pipeline.

---
Great—let’s freeze **exactly** how to implement witness conjugation so Claude can code it without guesswork. Below is a **complete, anchor-grounded spec** for `conjugate_to_test`, including the Π structure, D4 primitives, the algebra for composition, a **functional algorithm** you can implement immediately (no hit-and-trial), how residues transform, and the precise receipts to log.

---

# WO-04 Conjugation — Final, Frozen Spec ✅ COMPLETED

## 0) What we’re doing (math)

We want the witness on training (i), expressed in the Π-frame of the training input, to be transported into the Π-frame of the **test** input:
[
\boxed{;\phi_i^* ;=; \Pi_* ;\circ; U_i ;\circ; \phi_i ;\circ; \Pi_i^{-1} ;\circ; U_*^{-1};}\qquad
\boxed{;\sigma_i^* ;=; \sigma_i;}\tag{A}
]
Here (\Pi) is the **presentation** on positions (D4 pose + anchor), (U=\Pi^{-1}). Palette mapping **does not affect coordinates**; σ acts on colors only and is **frame-invariant**.

---

## 1) Π structure (frozen) and access

### Dataclass

```python
@dataclass
class PiFrame:
    pose_id: int              # D4 id ∈ {0..7}
    anchor: Tuple[int,int]    # (ar,ac) top-left offset subtracted by Π
    R: np.ndarray             # 2×2 integer D4 matrix (row-major coords)
    R_inv: np.ndarray         # inverse D4 matrix (R^-1)
```

**Where it comes from:** your WO-01 `present_all(...)` must already record, per grid, `pose_id`, `anchor`, and the D4 pose matrix; build `R`/`R_inv` table once (see §2).

* **Π on positions (frozen):**
  (p_\pi = \Pi(p_\text{raw}) = R,p_\text{raw} - a) where (a=(ar,ac)).
* **U on positions (frozen):**
  (p_\text{raw} = U(p_\pi) = R^{-1}(p_\pi + a)).

Palette maps are irrelevant to position conjugation and must **not** be used here.

---

## 2) D4 primitives (frozen)

Coordinates are column vectors (p=(r,c)^T) with row increasing downward.

| pose_id | op                   |                                 R matrix |
| ------: | -------------------- | ---------------------------------------: |
|       0 | I                    |   (\begin{bmatrix} 1&0\0&1\end{bmatrix}) |
|       1 | R90 (clockwise)      |  (\begin{bmatrix} 0&1\-1&0\end{bmatrix}) |
|       2 | R180                 | (\begin{bmatrix} -1&0\0&-1\end{bmatrix}) |
|       3 | R270                 |  (\begin{bmatrix} 0&-1\1&0\end{bmatrix}) |
|       4 | FH (flip left-right) |  (\begin{bmatrix} 1&0\0&-1\end{bmatrix}) |
|       5 | FH∘R90               |                                 (R_1R_4) |
|       6 | FH∘R180              |                                 (R_2R_4) |
|       7 | FH∘R270              |                                 (R_3R_4) |

**Frozen helpers:**

```python
D4_R = {pid: R_matrix}              # as above
D4_R_INV = {pid: np.linalg.inv(R)}  # integer matrices; equals transpose for rotations/flips
def inv_pose(pid): ...              # inverse id via table
def compose_pose(p1,p2): ...        # pid for R[p1]·R[p2]
def rotate_offset(dr,dc,pid):       # returns R·(dr,dc)
```

---

## 3) φ piece representation (frozen)

You already use:

```python
@dataclass
class PhiPiece:
    comp_id: int
    pose_id: int                # D4 on source
    dr: int; dc: int            # translation in TRAIN Π frame (target_tl - src_tl)
    r_per: int; c_per: int      # residue basis
    r_res: int; c_res: int      # residue class
    bbox_h: int; bbox_w: int
    target_r0: int; target_c0: int   # TRAIN Π coords
    src_r0: int;    src_c0: int      # TRAIN Π coords
```

We’ll return **the same schema** in **TEST Π** coords for `phi_star`.

**Interpretation:** in the training Π frame, (\phi_i) maps a source position (p) to
[
q = R_{\text{piece}} ;p ;+; t_{\text{piece}}\quad\text{with};;
R_{\text{piece}}=D4_\text{pose_matrix}(pose_id),; t=(dr,dc)^T;.
]
The recorded `target_r0,c0` and `src_r0,c0` are the top-left anchors of the two bboxes; equality was verified on that rectangle.

---

## 4) Conjugation algorithm — **two frozen implementations**

You have **two** deterministic ways to implement (A); you may pick either. The **functional** method is simplest and avoids algebraic mistakes; the **closed-form** derives explicit ((R^*,t^*)) for receipts.

### 4.1 Functional (preferred; easy to code, zero ambiguity)

For **each φ piece**, build a new piece in TEST Π by **transporting coordinates pointwise**:

* Given a **target** pixel (q_* \in) TEST Π **inside** the conjugated bbox, compute the **source** pixel in TEST Π:
  [
  \boxed{ ;p_* ;=; \Pi_* !\big( ; U_i\big( ; \phi_i \big( \Pi_i^{-1}!\big( U_*^{-1}(q_*) \big) \big) ;\big);\big) ;}\tag{B}
  ]
  All maps are defined in §1.

* To **materialize** a `PhiPiece*`, compute the **conjugated bboxes** first: transport the **four corners** of the training target bbox by ((B)) (with (\phi_i) set to identity) to get the target bbox in TEST Π; similarly map the source bbox corners through the inner pieces to get source bbox in TEST Π; take the min row/col of the images as `target_r0*, target_c0*` (and `src_r0*, src_c0*`), height/width from extents.

* Populate `pose_id*` and `(dr*,dc*)` by fitting:

  * Set (R^*=R_* R_i^{-1} R_{\text{piece}} R_i R_*^{-1}) (see 4.2) and lookup its pose id via table; or, if you don’t implement 4.2, set `pose_id* = pose_id` and **verify** that `R^\*` equals that D4 (common cases are rotations/flips).
  * Compute `t*` from two anchor points: pick src top-left (p) and its image (q) and solve (t^* = q - R^* p).

* **Residues transform**:
  If (R^*) swaps axes (90/270), swap `(r_per,c_per)` and `(r_res,c_res)`. Otherwise keep them; keep residue values modulo the basis.

**This method is exact** and only uses frozen primitives. You can also compute admits **without** constructing `PhiPiece*` explicitly (WO-10Z): evaluate (B) to fill **witness admits**.

### 4.2 Closed-form (optional; for receipts & speed)

Represent every transform as an **affine pair** ((R,t)) with composition:
[
(R_a,t_a)\circ(R_b,t_b)=(R_aR_b,;R_a t_b + t_a),\qquad (R,t)^{-1}=(R^{-1},-R^{-1}t)
]

* **Piece** (T=(R_p,t_p)) where (R_p) is D4(pose_id) and (t_p=(dr,dc)^T).
* **P/U (positions)**:
  ( \Pi=(R, -a),\quad U=\Pi^{-1}=(R^{-1}, R^{-1} a) ).

Compute:
[
\begin{aligned}
M_1 &= \Pi_i^{-1} = (R_i^{-1},; R_i^{-1} a_i) \
M_2 &= T\circ M_1 = (R_p R_i^{-1},; R_p (R_i^{-1} a_i) + t_p) \
M_3 &= U_i\circ M_2 = (R_i^{-1} R_p R_i^{-1},; R_i^{-1}(R_p (R_i^{-1} a_i) + t_p) + R_i^{-1} a_i) \
M_4 &= \Pi_* \circ M_3 = (R_* R_i^{-1} R_p R_i^{-1},; R_* \text{(M3.t)} - a_*) \
M_5 &= M_4 \circ U_*^{-1},\quad U_*^{-1}=(R_*, -a_*) \
T^* &= \boxed{(R_* R_i^{-1} R_p R_i^{-1} R_*,; R_*,R_i^{-1},t_p;+; \underbrace{R_*(R_i^{-1}(R_p R_i^{-1} a_i + a_i)) - R_* a_* - a_*}_{\text{anchor-offset correction}} )}
\end{aligned}
]
You can then read off:

* `pose_id*` from the D4 matrix (R^*) via lookup,
* `dr*,dc*` from (t^*) (integer vector),
* swap residues on 90/270 as above.

> If you use 4.1 to construct bboxes & admits, keep 4.2 for **receipts** only (pose/Δ after conjugation).

---

## 5) σ and palette edge cases (frozen rules)

* **σ does not change** under conjugation: `sigma* = sigma`. You already encode σ via **Lehmer** on **touched colors**; that remains valid.
* **Palette**: Π’s palette canon (values) **never** participates in conjugation; all equality proofs and φ operate on the **original** integer colors (as per Addendum “logic_color_space: original”). If a module accidentally uses palette codes, harness must fail with a receipt hint.

---

## 6) Receipts (must log)

For each training i:

```jsonc
{
  "witness": {
    "per_train": [
      {
        "kind": "geometric" | "summary",
        "phi": {
          "pieces":[ { ... as PhiPiece* in TEST Π ... } ],
          "trials": [ {"comp_id":k, "trials":[{"pose":p,"dr":dr,"dc":dc,"ok":true|false,"reason":"..."}]} ],
          "domain_pixels": N
        },
        "sigma": { "domain_colors":[...], "lehmer":[...], "moved_count": K },
        "conjugation": {
          "train_id": "ti",
          "pose_train": pose_i, "anchor_train": [ar_i,ac_i],
          "pose_test":  pose_*, "anchor_test":  [ar_*,ac_*],
          "pullback_samples": [
            { "q_test":[rt,ct], "p_test":[rs,cs], "X[q_src]": x, "Y[q_tgt]": y, "ok": x==y_or_sigma(x) }
          ]
        }
      }
    ],
    "intersection": { "status": "singleton"|"underdetermined"|"contradictory" }
  }
}
```

* **Pullback samples:** choose 3 test-frame target pixels in the piece bbox; compute `p_test` via (B); log `(q_test, p_test, X[p_test], Y[q_test], ok)`.
* **Determinism:** double-run equality on the JSON receipts hash.

---

## 7) Residue handling (frozen)

* If the composed pose (R^*) swaps axes (pose id in {1,3,5,7}), **swap** `(r_per,c_per)` and `(r_res,c_res)`; else keep them.
* Always keep residues **modulo** their basis.
* For the functional approach (4.1): residue doesn’t affect admits; fill it in receipts from 4.2’s (R^*)/axis swap rule.

---

## 8) Edge cases (frozen)

* **Component missing in test** (e.g., class map mismatch): do **not** synthesize a φ* piece; mark this training as **summary** or fail the geometric witness for that training (the intersection will handle it).
* **Different palettes**: ignored for conjugation; σ operates on **original** colors; Π palette is display-only.
* **Row coframe**: row-coframe retry (optional WO-04.1) occurs **before** conjugation; if geometric row-coframe is accepted, conjugation applies as above on the row piece(s).

---

## 9) Minimal pseudo-code (functional)

```python
def conjugate_to_test(phi_train: PhiRc, sigma_train: SigmaRc,
                      pi_train: PiFrame, pi_test: PiFrame,
                      Xt: np.ndarray, Yt: np.ndarray):
    pieces_star = []
    trials_star = []
    for pc in phi_train.pieces:
        # derive test-frame bboxes by mapping corners
        tgt_train = rect_corners(pc.target_r0, pc.target_c0, pc.bbox_h, pc.bbox_w)
        src_train = rect_corners(pc.src_r0,   pc.src_c0,   pc.bbox_h, pc.bbox_w)

        tgt_test_pts = [ P_test(U_train(pt)) for pt in tgt_train ]   # φ=Id for bbox transport
        src_test_pts = [ P_test(U_train( T_piece(pt) )) for pt in src_train ]  # with piece

        tgt_r0,tgt_c0, h,w = bbox_from_points(tgt_test_pts)
        src_r0,src_c0, _,_ = bbox_from_points(src_test_pts)

        # (optional) derive pose*, dr*,dc* using closed-form (4.2) for receipts
        pose_star, (dr_star,dc_star) = derive_pose_and_delta(pi_train, pi_test, pc)

        pieces_star.append(PhiPiece(
            comp_id=pc.comp_id, pose_id=pose_star, dr=dr_star, dc=dc_star,
            r_per=pc.r_per, c_per=pc.c_per, r_res=pc.r_res, c_res=pc.c_res,  # swap if pose* swaps axes
            bbox_h=h, bbox_w=w, target_r0=tgt_r0, target_c0=tgt_c0, src_r0=src_r0, src_c0=src_c0
        ))
    sigma_star = sigma_train  # unchanged
    # log 3 pullback samples (B) for receipts
    return PhiRc(pieces=pieces_star, bbox_equal=[True]*len(pieces_star),
                 domain_pixels=sum(p.bbox_h*p.bbox_w for p in pieces_star),
                 trials=trials_star), sigma_star
```

---

## 10) Why this is now “Engineering = Math” & “No hit-and-trial”

* **All primitives are frozen** (Π structure, D4 tables, affine composition).
* **Two fully specified algorithms** (functional + closed-form) give you equivalent φ*; you can implement the functional now and add closed-form receipts if desired.
* **Receipts make it algebraic**: pullback samples prove the formula; pose/Δ after conjugation are logged; residues are transformed by a frozen rule; σ is unchanged and Lehmers are visible.
* **No degrees of freedom left**: no palette leaks, no thresholds, no unstated ordering, no “eyeballing.”

With this, Claude can implement `conjugate_to_test()` and you can proceed to **WO-10Z** (witness → admits) knowing conjugation is precise, deterministic, and receipts-proven.

---

