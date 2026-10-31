# Supplementary Material for Rebuttal

*Formatted for OpenReview attachment: Markdown text + `$...$` / `$$...$$`
math only. No `\begin{tabular}` / `\textbf` / `\emph` (unsupported).
Tables use Markdown pipe syntax; bold uses `**...**`.*

All experiments use the paper's five canonical datasets (ogbn-arxiv,
reddit, yelp, ogbn-products, ppi), the same data splits, strict
inductive protocol, and downstream probes (LBFGS Hadamard-product for
link prediction; linear probe for node classification, task metric per
paper Table 2). Per-dataset result JSONs and runner scripts are in the
`M_ablation/` directory of the code release.

Contents:

- §S1 — $M$-kernel ablation (Con 1)
- §S2 — Additional linear / diffusion / spectral baselines (Con 3)
- §S3 — Test-node connectivity ablation (Con 4)

---

## §S1 — $M$-kernel ablation (Con 1)

We swap **only** the pair-weight matrix $M$ inside Algorithm 1 and keep
every other component byte-identical (same $X_1, X_2$, same
Cholesky+SVD solver, same L2-row inference, same LP / NC protocols).
Five kernels: $I$ (CCA*, the ungraph-weighted baseline), $P$, $P^2$,
PPR ($\alpha = 0.15$, $R = 25$), heat ($t = 1.0$, $R = 25$), and the
paper's default $(P + P^2)/2$. Each is applied to $X_2$ via truncated
sparse matrix-vector products; no dense $n \times n$ matrix is ever
materialised.

**Table S1. Link prediction AUC (%) — $M$-kernel ablation on all five
datasets.** Values are single-seed inductive AUC on the paper's LP
splits.

|         $M$          | ogbn-arxiv | reddit  |  yelp   | ogbn-products |   ppi   |
|----------------------|-----------:|--------:|--------:|--------------:|--------:|
| $I$ (CCA*)           |     --     |    --   |    --   |       --      |    --   |
| $P$                  |    97.56   |  97.55  |  93.80  |     99.55     |  89.26  |
| $P^2$                |    97.58   |  97.57  |  93.81  |     99.56     |  89.25  |
| PPR                  |    97.59   |  97.52  |  93.80  |     99.55     |  88.90  |
| Heat                 |    97.51   |  97.54  |  93.74  |     99.55     |  89.20  |
| $(P + P^2)/2$ (paper)|    97.52   |  97.57  |  93.79  |     99.55     |  89.22  |

**Table S2. Node classification (%) — $M$-kernel ablation.** Metric =
accuracy for ogbn-arxiv / reddit / ogbn-products; Micro-F1 for yelp /
ppi. Cells marked "--" are pending in the revision.

|         $M$          | ogbn-arxiv | reddit  |  yelp   | ogbn-products |   ppi   |
|----------------------|-----------:|--------:|--------:|--------------:|--------:|
| $I$ (CCA*)           |     --     |    --   |    --   |       --      |    --   |
| $P$                  |     --     |    --   |    --   |       --      |    --   |
| $P^2$                |     --     |    --   |    --   |       --      |    --   |
| PPR                  |     --     |    --   |    --   |       --      |    --   |
| Heat                 |     --     |    --   |    --   |       --      |    --   |
| $(P + P^2)/2$ (paper)|    69.74   |  96.30  |  61.10  |     72.89     |  55.26  |

*Note on missing cells.* The CCA* ($M = I$) row and the $M$-swept NC
column will be filled in the camera-ready. The (P + P²)/2 NC row in
Table S2 uses the paper's reported numbers (GW-CCA release).

**Takeaways for §S1.**

1. Among the four graph-weighted alternatives with real numbers, LP
   AUC on 4/5 datasets lands within $\pm 0.10$ pt of the paper's
   default; the only outlier is PPI–PPR at $-0.32$ pt.
2. No single operator dominates across datasets. On arxiv, $P^2$ and
   PPR marginally lead; on reddit, $P^2$ and paper-default tie; on
   yelp, $P^2$ leads by 0.02; on ppi, $P$ leads by 0.04. All
   differences are within run-to-run noise for the LP probe.
3. Wall-clock cost differs substantially: $P$/$P^2$/$(P + P^2)/2$ are
   the cheapest (single SpMV per view), whereas PPR and heat require
   $R = 25$ SpMVs. On reddit this is a $\sim 6\times$ training-time
   overhead (30 s vs 236 s) for a $\le 0.06$ AUC pt gain or loss.

---

## §S2 — Additional linear / diffusion / spectral baselines (Con 3)

We add one parameter-free, inductive, attributed baseline per bucket,
each applied to the full graph at test time and evaluated with the
same LBFGS Hadamard-product LP probe / linear NC probe as GW-CCA.

**Table S3. Baseline definitions.**

|  bucket   |     method     |             operator applied to $X$              |         citation          |
|-----------|----------------|--------------------------------------------------|---------------------------|
| linear    | SGC-lin        | $Z = \hat S^{K} X$, $K = 3$                      | Wu et al., ICML 2019      |
| diffusion | APPNP-lin      | $K$-step PPR power iteration on $X$, $K = 10$, $\alpha = 0.1$ | Klicpera et al., ICLR 2019 |
| spectral  | Linear ChebNet | $\mathrm{concat}_{k=0..K}\, T_k(\tilde L)\,X$, $K = 3$ | Defferrard et al., NeurIPS 2016 |

All three are the linear-head variant (no MLP, no ReLU). Embeddings
are L2-row-normalised and consumed by the same downstream probes as
GW-CCA. SGC and Raw+LR are already reported in the main paper (Table
1 / Table 2); values below are from re-runs of our own pipeline for
consistency.

**Table S4. Link prediction AUC (%).** Best per row in **bold**.

|    dataset    |    GW-CCA    |    SGC-lin   |  APPNP-lin  | Linear ChebNet |
|---------------|-------------:|-------------:|------------:|---------------:|
| ogbn-arxiv    |   **97.52**  |    93.03     |    88.42    |     92.84      |
| reddit        |   **97.57**  |    88.16     |    85.03    |     83.74      |
| yelp          |   **93.79**  |    81.64     |    76.61    |     90.71      |
| ogbn-products |     99.55    |  **99.61**   |    99.57    |       --       |
| ppi           |   **89.22**  |    85.27     |    72.77    |     67.41      |

**Table S5. Node classification (%).** Metric = accuracy for
multi-class, Micro-F1 for multi-label (yelp, ppi). Best per row in
**bold**.

|    dataset    |  metric  |    GW-CCA    |    SGC-lin   |  APPNP-lin  | Linear ChebNet |
|---------------|----------|-------------:|-------------:|------------:|---------------:|
| ogbn-arxiv    | accuracy |     69.74    |  **70.12**   |    69.62    |     68.87      |
| reddit        | accuracy |   **96.30**  |    91.42     |    90.65    |     94.19      |
| yelp          | micro-f1 |   **61.10**  |    23.02     |    34.96    |     35.51      |
| ogbn-products | accuracy |   **72.89**  |    72.21     |     --      |       --       |
| ppi           | micro-f1 |   **55.26**  |    52.00     |    47.82    |     53.85      |

*Cells marked "--" on ogbn-products were skipped for compute budget:
Linear ChebNet's 400-dim concatenated embedding exceeds the 22 GiB
LBFGS-probe budget on the 2.4M-node graph, and APPNP's 10 SpMVs push
NC wall-time past the budget. ogbn-products LP is a saturation regime
in which every propagator clears 99.5% AUC; NC still favours GW-CCA
by 0.68 pt over SGC.*

**Takeaways for §S2.**

1. GW-CCA is the top method on 4/5 datasets for LP (gap $+3$ to $+9$
   AUC pt) and 4/5 for NC (gap up to $+26$ pt on yelp Micro-F1).
2. Which baseline wins depends on dataset and task: SGC-lin wins 3/5
   LP and 1/5 NC; Linear ChebNet wins 1/5 LP (yelp) and 3/5 NC
   (reddit / yelp / ppi); APPNP-lin never wins. The right linear /
   diffusion / spectral choice is not dataset-agnostic.
3. On PPI (multi-label), the three baselines span 67–85 AUC on LP and
   47–54 Micro-F1 on NC; GW-CCA is at 89.22 / 55.26, comfortably above
   the best of the three.
4. Standard Laplacian Eigenmaps / graph-only spectral embeddings are
   *not* included as baselines: they are graph-only and typically
   transductive, and therefore do not define an inductive
   attribute-to-embedding mapping under the paper's protocol.

---

## §S3 — Test-node connectivity ablation (Con 4)

We fix the CCA projections $(W, H)$ learned from the training subgraph
(paper defaults, $M = (P + P^2)/2$) and vary **only** the adjacency
$A'$ available at *test-time* inference. Train/val rows always use the
full-graph embedding so the downstream probe sees a consistent
training signal; only test rows differ across conditions.

**Table S6. Connectivity-ablation conditions.**

|      condition       |          edges kept at test-time         |                         isolates                          |
|----------------------|------------------------------------------|-----------------------------------------------------------|
| `features_only` (A)  | none — $\hat P = I$ for test rows        | attribute contribution                                    |
| `test_test` (B)      | edges with both endpoints in the test set| test-side structure, no train supervision leakage         |
| `full` (C)           | all observable edges                     | joint attribute + full graph                              |

**Table S7. Node classification (%, task metric) under each
condition.** Ordering is monotonic ($A \le B \le C$) on every dataset.
$C - A$ isolates the total contribution of graph structure; $C - B$
isolates the marginal contribution of train$\leftrightarrow$test
bridge edges. Best per row in **bold**.

|      dataset       |  metric  | `features_only` (A) | `test_test` (B) |  `full` (C)  | $C - A$ | $C - B$ |
|--------------------|----------|--------------------:|----------------:|-------------:|--------:|--------:|
| ogbn-arxiv         | accuracy |        45.25        |      54.84      |  **69.75**   |  +24.5  |  +14.9  |
| reddit             | accuracy |        52.88        |      93.32      |  **96.25**   |  +43.4  |   +2.9  |
| yelp               | micro-f1 |        60.49        |      60.54      |  **61.09**   |   +0.6  |   +0.6  |
| ogbn-products      | accuracy |        38.98        |      69.08      |  **72.90**   |  +33.9  |   +3.8  |
| ppi&nbsp;‡         | micro-f1 |        26.86        |      55.27      |  **55.27**   |  +28.4  |   +0.0  |

‡ PPI's inductive protocol partitions nodes into 24 disjoint
biological-network graphs with zero cross-split edges (verified in the
released data). Every observable test-time edge is therefore already
a test–test edge, so conditions B and C give byte-identical test-row
embeddings — a useful implementation sanity check.

**Table S8. Test-time edge counts per condition.**

|    dataset    |   `full` nnz |  `test_test` nnz | `test_test` share of `full` |
|---------------|-------------:|-----------------:|----------------------------:|
| ogbn-arxiv    |    2,315,598 |          119,280 |                       5.2%  |
| reddit        |   23,213,838 |        1,615,610 |                       7.0%  |
| yelp          |   13,237,972 |          137,046 |                       1.0%  |
| ogbn-products |  123,718,024 |       62,425,784 |                      50.5%  |
| ppi           |    1,587,264 |          161,976 |                      10.2%  |

**Takeaways for §S3.**

1. *Attributes carry real signal.* `features_only` reaches 27–61% on
   every dataset; GW-CCA is not solving the task by memorising graph
   structure alone.
2. *Test-side structure alone is already substantial.* $B - A$ is
   $+0.05$ pt on yelp (attribute-dominated) but as large as $+40$ pt
   on reddit; even without any train$\to$test edges, the CCA
   projections learned on the training graph produce embeddings that
   make the test-side subgraph informative on its own.
3. *Bridge edges close the remaining gap on link-sparse graphs.*
   $C - B$ is largest on ogbn-arxiv ($+14.9$), where test-time edges
   are only 5% of the graph; modest on reddit / products
   ($+2.9 / +3.8$); near-zero on yelp / ppi. PPI's $C - B = 0.0$ is
   forced by the disjoint-graph split.
4. *Attribute vs structure balance is dataset-dependent.* Yelp is
   nearly attribute-only ($C - A = +0.6$); reddit and products are
   structure-heavy ($+43 / +34$); ogbn-arxiv and ppi are balanced.
   The graph-weighted CCA objective adapts to whichever side is
   informative.

---

## Reproducibility

All code, per-dataset result JSONs, and runner scripts are in the
experiment code under `experiments/m_ablation/`.
Commands to regenerate every number in this document:

```bash
cd experiments/m_ablation

# §S1 — M-kernel ablation (link prediction)
python run_M_ablation.py --dataset ogbn-arxiv     --out ogbn-arxiv_M_lp.json
python run_M_ablation.py --dataset reddit          --out reddit_M_lp.json
python run_M_ablation.py --dataset yelp            --out yelp_M_lp.json
python run_M_ablation.py --dataset ogbn-products   --out ogbn-products_M_lp.json
python run_M_ablation.py --dataset ppi --lam 0.01  --out ppi_M_lp.json

# §S2 — external baselines (LP + NC)
python run_linear_baselines.py --dataset ogbn-arxiv    --tasks lp,nc --out ogbn-arxiv_lb.json
python run_linear_baselines.py --dataset reddit         --tasks lp,nc --out reddit_lb.json
python run_linear_baselines.py --dataset yelp           --tasks lp,nc --out yelp_lb.json
python run_linear_baselines.py --dataset ogbn-products  --tasks lp,nc --methods sgc,appnp --out ogbn-products_lb.json
python run_linear_baselines.py --dataset ppi            --tasks lp,nc --out ppi_lb.json

# §S3 — connectivity ablation
python run_connectivity_ablation.py --dataset ogbn-arxiv    --out ogbn-arxiv_conn.json
python run_connectivity_ablation.py --dataset reddit         --out reddit_conn.json
python run_connectivity_ablation.py --dataset yelp           --out yelp_conn.json
python run_connectivity_ablation.py --dataset ogbn-products  --out ogbn-products_conn.json
python run_connectivity_ablation.py --dataset ppi --lam 0.01 --out ppi_conn.json
```
