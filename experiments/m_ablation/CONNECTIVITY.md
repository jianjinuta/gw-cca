# Con 4 — Connectivity Ablation

Reviewer asks: "clarify the relative contributions of attributes and
graph structure." We fix the CCA projections $(W, H)$ learned from the
training subgraph (paper defaults, $M = (P + P^2)/2$) and vary **only**
the adjacency $A'$ used at *test-time* inference. Train/val rows are
always embedded on the full graph so the downstream LR probe sees a
consistent training signal; only test rows differ across conditions.
Node classification, all 5 datasets.

## Conditions

| # | slug | edges kept at test-time | isolates |
|---|------|-----------|----------|
| A | `features_only` | none — $P_\text{hat} = I$ for test rows, propagation is a no-op | attribute contribution |
| B | `test_test` | edges with both endpoints in test set | test-side structure with zero train supervision leakage |
| C | `full` | all observable edges (paper's inductive protocol) | joint attribute + full graph |

Implementation: embed once on $A_\text{full}$ to get $Z_\text{full}$;
embed a second time on the masked $A'$; splice $Z_\text{test}$ from the
masked run into $Z_\text{full}$; run the LR probe.
Runner: `run_connectivity_ablation.py`. $W, H$ are learned once per dataset.

## Node classification (%)

Metric = accuracy for multi-class, Micro-F1 for multi-label (yelp, ppi).

| dataset | metric | features_only (A) | test_test (B) | full (C) | C − A | C − B |
|---|---|---:|---:|---:|---:|---:|
| ogbn-arxiv | accuracy | 45.25 | 54.84 | **69.75** | +24.50 | +14.91 |
| reddit | accuracy | 52.88 | 93.32 | **96.25** | +43.37 | +2.93 |
| yelp | micro-f1 | 60.49 | 60.54 | **61.09** | +0.60 | +0.55 |
| ogbn-products | accuracy | 38.98 | 69.08 | **72.90** | +33.92 | +3.82 |
| ppi ‡ | micro-f1 | 26.86 | 55.27 | **55.27** | +28.41 | +0.00 |

Ordering is monotonic ($A \le B \le C$) on every dataset, as it should
be under a strictly nested edge budget.

‡ **PPI: `test_test == full` by construction.** PPI's inductive
protocol partitions nodes into 24 disjoint biological-network graphs;
train/val/test splits contain zero cross-split edges (verified: 0
edges between any two splits in `A_full`). Since every observable
test-time edge is already a test–test edge, condition B and condition
C give byte-identical test-row embeddings — a useful sanity check on
the implementation.

## Edges kept at test-time

| dataset | full nnz | test_test nnz | test-test share of full |
|---|---:|---:|---:|
| ogbn-arxiv | 2,315,598 | 119,280 | 5.2% |
| reddit | 23,213,838 | 1,615,610 | 7.0% |
| yelp | 13,237,972 | 137,046 | 1.0% |
| ogbn-products | 123,718,024 | 62,425,784 | 50.5% |
| ppi | 1,587,264 | 161,976 | 10.2% (of these, 100% are test–test — see disjoint note) |

## Takeaways for the rebuttal

1. **Attributes carry real signal.** `features_only` reaches 27–60% on
   every dataset — GW-CCA is not solving the task by memorising graph
   structure alone. This directly refutes the "structure-only
   embedding" reading of the method.
2. **Test-side structure alone is already substantial.** `test_test`
   ranges from a tie with `features_only` (yelp) to +44 pt over it
   (reddit). Even without any train→test edges, the CCA projections
   learned on the training graph produce embeddings that make the
   test-side subgraph structure informative on its own.
3. **Train↔test edges close the remaining gap on structure-heavy
   graphs.** `full − test_test` is largest on ogbn-arxiv (+14.9),
   modest on reddit / products (+2.9 / +3.8), and near-zero on yelp /
   ppi. On the datasets where test-side structure is already rich
   (reddit, products), the bridge edges add marginal signal; on
   ogbn-arxiv, where test-time edges are sparse (5% of the graph), the
   bridge dominates.
4. **Attribute vs structure balance is dataset-dependent.** Reddit and
   products lean on structure (`C − A` = +43 / +34 pt); yelp is
   essentially attribute-only (`C − A` = +0.6); ogbn-arxiv sits in
   between; ppi is structure-dependent but with disjoint splits, so
   the "structure" it uses is entirely intra-test. GW-CCA's joint CCA
   objective adapts to whichever side is informative — this is
   exactly the "graph-weighted" aspect of the design.

## Reproducing

```bash
cd experiments/m_ablation
python run_connectivity_ablation.py --dataset ogbn-arxiv     --out ogbn-arxiv_conn.json
python run_connectivity_ablation.py --dataset reddit          --out reddit_conn.json
python run_connectivity_ablation.py --dataset yelp            --out yelp_conn.json
python run_connectivity_ablation.py --dataset ogbn-products   --out ogbn-products_conn.json
python run_connectivity_ablation.py --dataset ppi --lam 0.01  --out ppi_conn.json
```

PPI needs `--lam 0.01` (matches historical setting; default
$\lambda = 0$ produces a non-PD covariance).
