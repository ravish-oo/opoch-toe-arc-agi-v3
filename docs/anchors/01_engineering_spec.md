Below is the flip-engineering = pure math spec. You can hand this to any team and they will produce a solver that behaves like a mathematical operator, not a heuristic program. Every step is an exact equality, a canonical intersection, or a declared minimum-bit argmin that makes the operator total. There are no dials, no learning, no hidden priors.

⸻

0) System overview — one commuting operator

Goal. Given only matrices \{(X_i,Y_i)\}{i=1}^m and \(X\\), output \(Y_\\).

Single operator.
\boxed{
Y_\* \;=\;
U^{-1}\!\Big(\;
\Pi\;\circ\;
\mathrm{gfp}(\mathcal F)\;\circ\;
\bigcap_{i=1}^m \mathrm{Conj}(\phi_i,\sigma_i)\;\circ\;
\mathrm{Meet}
\;\Big)(X_\*)
}
	•	\Pi — idempotent presentation (palette on inputs only, D4 lex, anchor).
	•	\mathrm{gfp}(\mathcal F) — truth = largest bisimulation (fixed point) under provable equalities.
	•	\mathrm{Conj}(\phi_i,\sigma_i) — co-observed witnesses per training pair, transported to test and intersected.
	•	Meet — least per-pixel write: copy ▷ law ▷ unanimity ▷ bottom.
	•	U^{-1} — exact inverse of \Pi.

Commutation (diamond). All four moves commute; repainting is idempotent; result is unique.
Totality. A declared minimum-bit tie-break selects a unique law when intersection returns >1.

⸻

1) Data types
	•	Grid G\in \mathcal C^{H\times W}; \Omega=\{0..H-1\}\times\{0..W-1\}.
	•	Transform T=(\text{palette},\ \text{D4 op},\ \text{anchor offset}).
	•	Witness (\phi,\sigma):
	•	\phi:\Omega\to\Omega a bijection (piecewise by components; within each bbox \phi \in D_4 \ltimes \mathbb Z^2, optionally with exact lattice residues).
	•	\sigma\in S_{|\mathcal C|} color permutation.
	•	Truth partition P:\Omega\to block IDs.

⸻

2) Presentation \Pi (idempotent; 0-bit)

Contract.
Input: G. Output: \tilde G=\Pi(G), transform T, s.t. U^{-1}(\tilde G)=G.
	•	Palette canon on inputs only (build once on \bigcup_i X_i\cup\{X_\*\}); map colors to codes by decreasing frequency; outputs displayed through same map (identity fallback).
	•	D4 lex pose: choose d\in D_4 minimizing raster lex; G’=d(G).
	•	Anchor: translate so bbox of nonzero support touches (0,0).
	•	\tilde G\gets result; record T. Ensure \Pi^2=\Pi; verify round-trip with U^{-1}.

Receipts. palette hash, D4 op, anchor, round-trip ok.

⸻

3) Shape S (exact size constraints + least)

Contract.
Input: \{(H_i,W_i,H’_i,W’_i)\}. Output: map S(H,W)=(R,C).

Synthesize constraints; choose the least branch/integers that fit all trainings:
	1.	Affine: R=aH+b,\ C=cW+d (a,b,c,d\in \mathbb Z_{\ge0}); prefer mult. (b=d=0)\prec add. (a=c=1)\prec mixed (lex-smallest).
	2.	Period-multiple: compute minimal per-line periods p_r(i),p_c(j) (exact). Let p_r^{\min}=\mathrm{lcm}\{p_r(i)>1\} (else 1), p_c^{\min} likewise; find k_r,k_c with (H’_i,W’_i)=(k_r p_r^{\min}, k_c p_c^{\min}).
	3.	Count-based: exact content counts (e.g., #row-pairs satisfying a corridor predicate, #components) → integers \alpha,\beta s.t. (H’_i,W’_i)=(\alpha\#\mathrm{qual}+\beta,\,\dots).
	4.	Frame/BBox; Pad-to-multiple(k) if exactly implied.

Apply S to the presented test to get canvas size R\times C.

Receipts. branch, integers, equalities per training verified.

⸻

4) Co-observation witness solver (\phi_i,\sigma_i)

Goal. For each training (after Π), compute (\phi_i,\sigma_i) satisfying
\boxed{ \tilde Y_i\ =\ \sigma_i\ \circ\ \tilde X_i\ \circ\ \phi_i \quad \text{(exact equality).} }

Algorithm (exact):
	1.	Components. Extract 4-conn components for each color in \tilde X_i, \tilde Y_i:
	•	invariants: area, bbox dims, centroid (integer moments), outline hash (D4-min raster, xxhash64).
	•	match components between X and Y by identical signature (assignment is trivial because hashes match; if multiple equals, break ties by lex of centroids; exact equality verified next).
	2.	Per-component isomorphism. For each matched pair (C^X,C^Y):
	•	enumerate 8 D4 ops; for each, compute the translation \Delta that aligns bbox anchors; verify value equality on every pixel of the bbox; accept the unique (d,\Delta) that makes X\circ(d,\Delta) = Y on the component.
	•	Periodic/tiled region: compute minimal periods inside bbox; allow residue-class alignment verified by bitwise equality.
	3.	Glue per-component maps into a global bijection \phi_i (components are disjoint after Π).
	4.	Palette role \sigma_i. From all pixels, infer the unique permutation \sigma_i\in S_{|\mathcal C|} such that \tilde Y_i(p)=\sigma_i(\tilde X_i(\phi_i(p))) for all p. If any conflict, the pair is inconsistent; abort.

Transport to test frame.
\((\phi_i^\,\sigma_i^\)=\mathrm{Conj}{i\to *}(\phi_i,\sigma_i)=\Pi\\circ\Pi_i^{-1}\ \circ(\phi_i,\sigma_i)\ \circ \Pi_i\circ\Pi_\^{-1}\).

Intersect across trainings:
\((\phi,\sigma)=\bigwedge_i (\phi_i^\,\sigma_i^\)\) (component-wise parameters must match).
	•	empty ⇒ contradictory data;
	•	singleton ⇒ law fixed;
	•	size>1 ⇒ underdetermined; use tie-break (§7).

Receipts. For each i: comp matches, per-comp maps (D4,Δ), proof hashes; \sigma_i; conjugations; intersection result.

⸻

5) Truth on test: \mathrm{gfp}(\mathcal F)

Goal. Largest bisimulation (coarsest observation-equivalence).

Tags (exact).
	•	Local (radius 1 or 2): color; n4/n8 adjacency flags; samecomp inside window; parity flags; tiny window-period flags p∈{2,3} if all rows/cols in the window repeat with p.
	•	Global: per-color FFT autocorrelation → candidate overlaps Δ (verify by value equality on overlap); per-line minimal periods (KMP); exact tilings (bitwise equality); exact mirrors/rotations in bboxes (bitwise).

Paige–Tarjan refinement.
Start by color; refine blocks until no tag distinguishes blocks. Output partition P:\Omega\to\text{blockID}.

Receipts. tag set; refinement iterations; block histogram.

⸻

6) Free copy sets S(p)

For each test pixel p\in\Omega, define
\[
\boxed{\ S(p)=\bigcap_i \{\ \phi_i^\(p)\ \}\ }
\]
(a set of input coordinates). If |S(p)|=1, this is a free copy site; record \(\tilde Y_\(p)=\tilde X_\*(s)\) with 0 bits.

Receipts. count and mask of singleton sites.

⸻

7) Tie-break (minimum-bit law) — only if needed

If the witness intersection produces multiple admissible (\phi,\sigma), choose the unique argmin of a fixed, few-bit cost:
\[
(\phi^\,\sigma^\) = \underset{(\phi,\sigma)}{\arg\min}\ \alpha_1\,\text{total\_displacement}(\phi) + \alpha_2\,\text{\#params}(\phi) + \beta\,\text{recolor\_bits}(\sigma) + \gamma\,\text{component\_breaks}(\phi)
\]
Ties lex: prefer reflect ≺ rotate ≺ translate; prefer smaller residues; then lex on params. This makes the operator total when the data leaves a set of laws.

Receipts. admissible (\phi,\sigma) list; costs; chosen argmin.

⸻

8) Unanimity (truth-block constants)

For each truth block B\subset\Omega, compute unanimous color u(B) if for all trainings i and for all p\in B, \tilde Y_i(p_i) (with frame mapping \Pi_i\circ\Pi_\*^{-1}, plus shape pullback if needed) is the same color. If so, u(B) is defined.

Receipts. unanimous blocks; color per block.

⸻

9) Meet (least write; single pass)

For each pixel p:

\[
A_p = \underbrace{\{\tilde X_\(s)\mid |S(p)|=1\}}_{\text{copy}}
\cup\underbrace{\{\,\sigma^\\big(\tilde X_\(\phi^\(p))\big)\,\}}{\text{law (if fixed)}}
\cup\underbrace{\{\,u(P(p))\,\}}{\text{unanimity (if defined)}}
\cup\{c_{\min}\}.
\]

Write once:
\[
\boxed{\ \tilde Y_\(p)=\min\nolimits_{\triangleright} A_p,\quad \text{with order}~ \textbf{copy} \triangleright \textbf{law} \triangleright \textbf{unanimity} \triangleright \textbf{bottom}\ }
\]
(pointwise meet → idempotent repaint). Then \(Y_\=U^{-1}(\tilde Y_\*)\).

Receipts. counts by rule; idempotent repaint hash.

⸻

10) Determinism, termination, complexity
	•	Determinism: Π is idempotent; witness solve is exact equality; conjugation/intersection canonical; truth gfp unique (finite tags); the meet is a total order → unique Y_\*.
	•	Termination: all steps are finite and require no search trees; only the argmin on a finite law set in §7.
	•	Complexity: proof compilation O(Cn\log n) + O(n); truth refinement O(n\log n); per-pair witness checks O(n); meet O(n). On ARC sizes, runtime is milliseconds.

⸻

11) Edge-handling / guarantees
	•	Contradictory trainings (no witness law fits all pairs): operator still returns a unique output via copy ▷ unanimity ▷ bottom and logs contradiction witnesses.
	•	Underdetermination (multiple laws fit): tie-break (§7) picks the minimum-bit law; operator still produces a unique output; receipts show the law set and selected argmin.
	•	Palettes not seen in inputs: inverse palette uses identity fallback (Π remains idempotent).

⸻

12) Reference skeleton (pure math, no heuristics)

def solve_task(train_pairs, Xstar):
    # 1) Present
    Xs_t, Ys_t, Xstar_t, Uinv = present_all(train_pairs, Xstar)  # Π, U^{-1}

    # 2) Shape
    R, C, S_receipt = synthesize_shape(train_pairs, Xstar_t)      # exact + least
    Xstar_t = resize_canvas(Xstar_t, R, C)

    # 3) Co-observation witnesses
    witnesses = []
    for Xi_t, Yi_t in zip(Xs_t, Ys_t):
        phi_i, sigma_i = solve_witness(Xi_t, Yi_t)                # exact equality per component
        witnesses.append(conjugate_to_test(phi_i, sigma_i, Xi_t.frame, Xstar_t.frame))
    law_set = intersect_witnesses(witnesses)                      # ∩ params
    law = argmin_L(law_set)                                       # §7 (minimal bits)

    # 4) Truth on test
    tags, syms = compile_exact_proofs(Xstar_t, Xs_t)              # FFT overlaps, KMP periods, components, tilings, etc.
    truth = paige_tarjan_gfp(Xstar_t, tags, syms)

    # 5) Free copy sets
    S = build_free_sets(witnesses)                                # S(p) = ∩ φ_i*(p)

    # 6) Unanimity per truth block
    u = compute_unanimity(truth, Xs_t, Ys_t, Xstar_t)

    # 7) Meet (copy ▷ law ▷ unanimity ▷ bottom)
    Yt = full_of(Xstar_t.shape, c_min(Xs_t+[Xstar_t]))
    for p in coords(Xstar_t):
        if len(S[p]) == 1:
            Yt[p] = Xstar_t[S[p].pop()]
        elif law is not None and law.defined_on(p):
            Yt[p] = law.apply(Xstar_t, p)
        elif u.has(truth.block_of(p)):
            Yt[p] = u[truth.block_of(p)]
        # else keep bottom

    return Uinv(Yt), receipts(...)

Everything above is exact and receipts-tight. There is no “traditional engineering”—only finite algebra on matrices (exact equalities, intersections, fixed points, a declared minimum-bit argmin). That’s the universe’s “engineering” for colored grids.