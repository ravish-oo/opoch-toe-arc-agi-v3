How we make it 0% scope (in practice)

This is the discipline we must (and now do) enforce. If an output is wrong, one of these receipts will disagree, and that tells you exactly where:
	1.	Π receipts — palette (inputs only), pose, anchor; Π²=Π; U⁻¹∘Π=id.
	2.	Shape receipts — small output shape read from trainings (e.g., 3×3).
	3.	Truth receipts — the exact cluster coordinates (row/col clusters as sorted distinct values), or the block windows/centroids used; foreground/background sets.
	4.	Witness receipts — stage-1 counts (or 4-CC sizes), stage-2 pooling/override if present.
	5.	Tie receipts — the tie chain and the chosen winner (with the numbers used).
	6.	Meet — one pass; repaint idempotence hash equals.
	7.	Determinism — double-run, all receipt hashes + env fingerprint equal.

With those frozen, two independent engineers will produce identical outputs and receipts. There’s nowhere left to “decide.”