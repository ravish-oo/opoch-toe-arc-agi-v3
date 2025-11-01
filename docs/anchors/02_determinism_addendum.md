# ARC‑AGI Operator — Determinism Freezes & Receipts Addendum

**Purpose.** This addendum freezes every implicit “dial” so the solver is a mathematical operator in practice, not a heuristic program. It supplements the Math Spec and the Engineering Spec with canonical encodings, fixed tag vocabulary, tie‑break costs, and receipts.

**Scope.** Applies to all tasks; no runtime switches. Any deviation is a bug. When this doc conflicts with code, this doc wins.

---

## 0) Global invariants

* **No dynamic tags.** Truth uses the frozen tag set (#2) only.
* **Integer arithmetic.** Any frequency/FFT/period calculation must be integer‑safe; floating results must be verified by exact pixel equality before being trusted.
* **Canonical encodings** in this doc are the only valid ones. All receipts must include the *encoded* form, not re‑derived prose.
* **Fail‑closed.** Contradiction → route to §7/§9; never silently choose.
* **Cell encoding for hashes.** When serializing any grid for hashing/receipts, encode each color as **uint32 little‑endian (uint32_le)** in row‑major order.
* **Signed integers encoding.** All signed integers (e.g., dr, dc, residues) are **ZigZag‑encoded LEB128 varints**. Unsigned integers use plain LEB128 varints. Any overflow is an error and must abort.

---

## 1) Canonical encodings (freezes)

### 1.1 Branch tags for Shape S

Use the following byte codes (single ASCII):

* `A` = AFFINE (R=aH+b, C=cW+d)
* `P` = PERIOD_MULTIPLE
* `C` = COUNT_BASED
* `F` = FRAME_OR_MULTIPLE

**Param serialization:** big‑endian varint per integer (unsigned via LEB128; signed values first ZigZag, then LEB128), sequence framed as:

```
<count:varint><p1:varint>...<pk:varint>
```

Examples:

* AFFINE → `<4><a><b><c><d>`
* PERIOD_MULTIPLE → `<2><k_r><k_c>` with implicit p_r^min, p_c^min recorded in receipt as separate framed tuples `<row_periods_lcm><col_periods_lcm>`
* COUNT_BASED → `<4><alpha1><beta1><alpha2><beta2>`, plus a *named* qual id (ASCII) hashed and recorded alongside

**S candidate ordering key:**

```
(branch_byte, params_bytes, R, C)
```

Lexicographic on bytes; integers compared as bytes (not numeric) to forbid platform variance.

**Axis tie for PERIOD_MULTIPLE:** If admissible candidates exist for both orientations with identical `(branch_byte, params_bytes)`, choose the orientation whose band run count is larger; if equal, choose **rows over columns** (axis_code rows=0, cols=1) and record `axis_code` in receipts.

### 1.2 D4 pose ids and anchor offsets

* D4 ids (0..7): 0=identity, 1=rot90, 2=rot180, 3=rot270, 4=flipH, 5=flipH∘rot90, 6=flipH∘rot180, 7=flipH∘rot270.
* Anchor offsets: **signed ZigZag‑encoded LEB128 varints** `(dr, dc)` in row‑major coords.

### 1.3 Palette canon

* Build over `⋃ inputs ∪ {test}` **inputs only**.
* Sort colors by *descending frequency*; ties by *ascending color value*.
* Outputs are *displayed* via this map; unseen output colors use identity fallback.
* Receipt must include `(color, freq)` list and the final map as `(orig→code)` pairs.

### 1.4 σ permutation encoding

* Encode σ as **Lehmer code** over the *active palette in the law’s domain*. Record `(domain_colors:list, lehmer:list)`.
* `recolor_bits(σ)` = number of *moved* colors in domain (k). This is the frozen measure used in tie‑break.

### 1.5 φ encoding (piecewise)

For each component id `cid` in X’s canon order:

```
<cid:varint><pose_id:byte><dr:varint><dc:varint><residue_basis:(r_per:varint,c_per:varint)><residue:(r:varint,c:varint)>
```

If not periodic, set periods=1 and residue=(0,0).

---

## 2) Truth tag vocabulary (frozen)

Only these tags are allowed. Any extra tag at runtime is an error.

**Local (radius=1,2):**

* `color` (exact value)
* `n4_adj` (N,E,S,W same‑color booleans)
* `n8_adj` (8‑neighborhood same‑color boolean)
* `samecomp_r2` (same connected component within r=2 window)
* `parity` (r%2, c%2)
* `row_period_2`, `row_period_3` (when every cell in the window’s row obeys p‑periodicity)
* `col_period_2`, `col_period_3`

**Global:**

* `per_color_overlap` via integer convolution/NTT or int‑FFT; every candidate Δ must be *verified by exact equality* on the implied overlap; record accepted Δ. **Identity Δ is excluded** (see §7).

  * **Transform freeze:** Implementations may use integer FFT or NTT. If using **NTT**, use a 64‑bit friendly prime `p = 2^{64} − 2^{32} + 1` (or an equivalently documented prime) with a **fixed primitive root**; record `(method="ntt", modulus, root)` in receipts. If using **integer FFT** with rounding, record `(method="fft_int", verified=true)` and the final accepted Δ set.
* `per_line_min_period` via KMP (exact), both rows and columns; record LCMs.
* `exact_tile` flags where a bbox is tiled by a motif.
* `bbox_mirror` and `bbox_rotate` flags (bitwise exact) within components’ bboxes.

**Receipts:**

* `tag_set_version` = BLAKE3("color|n4_adj|n8_adj|samecomp_r2|parity|row_period_2|row_period_3|col_period_2|col_period_3|per_color_overlap|per_line_min_period|exact_tile|bbox_mirror|bbox_rotate")
* `refinement_steps`: int; `block_hist`: list[int]; `partition_hash` = BLAKE3(block_id raster).
* **Line mode ties (if used):** mode = most frequent color; ties → smallest color value. Record `(line_id, tie_colors:list, chosen_color)`.

---

## 3) Component matching (frozen)

* Connectivity: 4‑adjacency per color.
* Invariants tuple (lex key): `(area, bbox_h, bbox_w, perim4, outline_hash, anchor_rc)` where

  * `outline_hash` = BLAKE3 of the **D4‑min raster of the component’s mask** (independent of pose/color).
  * `anchor_rc` = **top‑left coordinate of the bbox in Π frame**.
* Stable match: sort X and Y component lists by the tuple; pair left‑to‑right. If counts differ or any equality check fails, flag contradiction.
* Verification: after applying candidate (pose,Δ,residue), compare pixelwise across bbox; must be equal.

---

## 4) Partial φ semantics (free copy)

* Each conjugated φ_i^* is a **partial function**.
* Define `S(p) = ⋂_i dom_images(φ_i^*, p)` where `dom_images` is {φ_i^*(p)} if defined, else ∅.
* If *any* i is undefined at p, intersection is ∅. **Never** copy by majority or union.
* **Singleton mask encoding:** serialize as a bitset (row‑major, LSB‑first per byte) and record `singleton_mask_hash`.

---

## 5) Unanimity pullback (frozen)

For a truth block B in test:

* Map positions via `p_i = Π_i ∘ Π_*^{-1}(p)`.
* If shape sizes differ, pull back by S to the maximal common sub‑domain; positions outside a training’s Y are **undefined** for unanimity.
* Unanimity holds iff **for all trainings where p_i is defined**, `Y_i(p_i)` is the same color, for all p∈B.
* If the pullback is **empty for every training**, unanimity **does not apply** to B and may not be used to write it.
* Record `(block_id, color, defined_train_ids)`.

---

## 6) Tie‑break L (fixed)

**Cost tuple:**

```
C(φ,σ) = (
  L1_disp_anchors(φ),     # sum of |Δr|+|Δc| over component anchors only
  param_len_bytes(φ),     # length of φ encoding in bytes
  recolor_bits(σ),        # moved-colors count in σ domain
  object_breaks(φ),       # Δ(#components) in φ’s domain
  tie_code(φ)             # REF=0, ROT=1, TRANS=2
)
```

* Compare lex on the above tuple.
* `tie_code`: pose class priority (any reflection < any rotation < pure translation).
* **object_breaks(φ)** = max(0, components(X∘φ) − components(X)) counting all color components; merges do **not** count as breaks. Record pre/post counts in receipts.
* Residue preference: for equal classes, prefer *smaller* residues `(r mod per_r, c mod per_c)` by lex.
* σ domain = colors touched by law. If two σ with same recolor_bits remain equal, choose lex on Lehmer code.

---

## 7) Identity‑overlap guard (Truth)

* When ranking overlaps/mirrors/rotations inside Truth, **exclude identity** from the candidate set. Identity equivalences are already captured by the base color tag.

---

## 8) Ordering rules for symbolic writes

When the law writes symbolic slices/bands/rows not tied to φ copying:

* **Color selection:** from witness intersection or unanimity only (never guessed from test).
* **Ordering:** freeze to **first Π‑raster appearance** order of the selected colors in the *test* input (ties by ascending color value). If a task family’s trainings demonstrate a different canonical order (e.g., left→mid→right), record a *family code* and enforce it consistently; receipt must include the order list.
* **Band‑compression outputs:** Horizontal bands ⇒ emit a **column vector** (top→bottom). Vertical bands ⇒ emit a **row vector** (left→right). If both bandings exist and are admissible, use the orientation chosen by §1.1’s PERIOD_MULTIPLE axis rule; record `(axis, band_edges:list, band_colors:list)`.

---

## 9) Meet write & idempotence

* Priority: `copy ▷ law ▷ unanimity ▷ bottom` (strict). One pass.
* **Bottom color is canonical 0.** Writing bottom means writing color 0 in the presented palette.
* After write, immediately recompute a BLAKE3 hash of the output; apply the same write pass again and assert the hash is unchanged (idempotence receipt).

---

## 10) Receipts schema (minimum fields)

* Π: `palette_hash`, `palette_freqs[(color,freq)...]`, `pose_id`, `anchor_drdc`, `roundtrip_hash`.
* S: `branch_byte`, `params_bytes_hex`, `R`, `C`, `verified_train_ids`.
* Truth: `tag_set_version`, `refinement_steps`, `block_hist`, `partition_hash`.
* Witness (per training): matched component pairs with invariant tuples; per‑component `(pose, dr, dc, residues)`; σ `(domain_colors, lehmer)`; after conjugation: encoded `(φ*, σ*)`.
* Intersection: either `(φ̂,σ̂)` encoded, or flags `{contradictory|underdetermined}`.
* Free copy: `singleton_count` and `singleton_mask_hash` (bitset bytes).
* Unanimity: list of `(block_id, color, defined_train_ids)`.
* Tie‑break: candidates with cost tuples; `chosen_idx`.
* Meet: counts and `repaint_hash`.
* Final: `output_hash`.

---

## 11) Determinism harness

* Run the full operator twice with the same inputs. Compare **all** receipt hashes (Π, S, Truth partition, φ/σ encodings, tie‑break table, repaint hash, final hash). If any differs, raise `NONDETERMINISTIC_EXECUTION` with the first differing receipt key.
* Receipts must include `env_fingerprint` with `{platform, endian, blake3_version, compiler_version, build_flags_hash}`; mismatch across runs triggers `NONDETERMINISTIC_ENV`.

---

## 12) Golden pitfall tests (must pass)

1. **Band grid derivation**: cases where visual guessing fails; assert equality of partition hash and band counts.
2. **Summary law**: force column/bit codebook derivation; ensure no invented bits; compare against receipts.
3. **Partial φ**: ensure S(p)=∅ if any φ_i^* undefined; forbid majority copy.
4. **FFT overlap**: float vs int; require equality verification; identity‑overlap excluded.
5. **Tie‑break**: synthetic underdetermined laws; assert argmin with recorded costs.
6. **Slice ordering**: enforce Π‑raster first‑appearance; regression when order drifts.
7. **Meet idempotence**: second pass hash identical.

---

## 13) Glossary (codes & IDs)

* D4 ids: 0..7 as in §1.2.
* Branch bytes: `A,P,C,F`.
* tie_code: REF=0, ROT=1, TRANS=2.
* tag_set_version: BLAKE3 of `"color|n4_adj|n8_adj|samecomp_r2|parity|row_period_2|row_period_3|col_period_2|col_period_3|per_color_overlap|per_line_min_period|exact_tile|bbox_mirror|bbox_rotate"` (exact string and order).

---

## 14) Compliance checklist (for reviewers)

**Severity legend:** BLOCKER = must pass; MAJOR = should pass (fail requires explicit waiver); MINOR = should pass (non-breaking).

* [ ] **BLOCKER** Π receipts present and invertible (palette_hash, pose_id, anchor, roundtrip_hash)
* [ ] **BLOCKER** S candidate list shown; winner matches `(branch_byte, params_bytes, R, C)` ordering; axis tie rule applied if P-branch
* [ ] **BLOCKER** Component invariants + stable matching receipts; pixelwise verification per component
* [ ] **BLOCKER** φ verified per component; σ unique (Lehmer); conjugations recorded; intersection result present
* [ ] **BLOCKER** Truth tags = frozen set; tag_set_version matches canonical; partition hash recorded
* [ ] **BLOCKER** Free copy singleton_count and singleton_mask_hash recorded
* [ ] **MAJOR** Unanimity blocks listed (if any) with defined_train_ids; empty-pullback blocks omitted
* [ ] **BLOCKER** Tie-break candidate table with cost tuples; chosen_idx matches lex rule
* [ ] **BLOCKER** Meet counts + repaint idempotence hash recorded; bottom=0 enforced
* [ ] **BLOCKER** Determinism harness receipts included; env_fingerprint present
* [ ] **BLOCKER** Final output_hash recorded

### 14.1 Machine-readable checklist (YAML)

```yaml
checks:
  - key: pi_receipts
    severity: BLOCKER
    must_include: [palette_hash, pose_id, anchor_drdc, roundtrip_hash]
  - key: shape_selection
    severity: BLOCKER
    fields: [branch_byte, params_bytes_hex, R, C]
    ordering_key: [branch_byte, params_bytes, R, C]
    period_axis_rule: rows_over_cols_when_equal: true
  - key: components_matching
    severity: BLOCKER
    invariants: [area, bbox_h, bbox_w, perim4, outline_hash, anchor_rc]
    verification: pixelwise_equal: true
  - key: witnesses
    severity: BLOCKER
    fields: [phi_encoded, sigma_domain_colors, sigma_lehmer, conjugated_phi_sigma, intersection]
  - key: truth_partition
    severity: BLOCKER
    tag_set_version: "color|n4_adj|n8_adj|samecomp_r2|parity|row_period_2|row_period_3|col_period_2|col_period_3|per_color_overlap|per_line_min_period|exact_tile|bbox_mirror|bbox_rotate"
    partition_hash: required
  - key: free_copy
    severity: BLOCKER
    fields: [singleton_count, singleton_mask_hash]
  - key: unanimity
    severity: MAJOR
    fields: [blocks, color, defined_train_ids]
    empty_pullback_disallowed: true
  - key: tie_break
    severity: BLOCKER
    cost_tuple_order: [L1_disp_anchors, param_len_bytes, recolor_bits, object_breaks, tie_code]
    residue_preference: smaller_is_better
  - key: meet
    severity: BLOCKER
    priority: [copy, law, unanimity, bottom]
    bottom_value: 0
    repaint_idempotence_required: true
  - key: determinism
    severity: BLOCKER
    env_fingerprint: [platform, endian, blake3_version, compiler_version, build_flags_hash]
  - key: final
    severity: BLOCKER
    fields: [output_hash]
```

---

**End of addendum.** This document plus the Math and Engineering specs are sufficient for a third party to implement a deterministic, receipts‑first ARC solver with no heuristics or hidden choices.
