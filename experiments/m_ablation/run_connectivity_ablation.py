"""Con 4 — connectivity ablation for GW-CCA.

Disentangle attribute vs graph-structure contribution by fixing the learned
projections (W, H) and varying only the adjacency used at inference:

    features_only   A' = 0  ->  P_hat = I  ->  X_2 = X
                    (attribute-only baseline; propagation is a no-op)
    test_test       A' = A_full restricted to edges (u, v) with u, v in test set
                    (structure only, no train supervision leakage)
    full            A' = A_full
                    (paper's inductive protocol)

W, H come from `gw_cca_train` on the training subgraph (paper defaults, p1p2 M).
Reports NC accuracy / Micro-F1 (task metric) per (condition, dataset) cell.

Example:
    python run_connectivity_ablation.py --dataset ogbn-arxiv --out ogbn-arxiv_conn.json
    python run_connectivity_ablation.py --dataset ppi --lam 0.01 --out ppi_conn.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_RELEASE = Path(__file__).resolve().parents[2]  # repo root: gw_cca.py, data.py, probe.py
sys.path.insert(0, str(_RELEASE))

from data import DATASETS  # noqa: E402
from gw_cca import DEFAULT_D, DEFAULT_LAM, gw_cca_embed, gw_cca_train  # noqa: E402
from probe import eval_node_cls  # noqa: E402


CONDITIONS = ["features_only", "test_test", "full"]


def mask_adjacency(A: sp.spmatrix, condition: str, node_set: np.ndarray) -> sp.spmatrix:
    """Return A' derived from A under the given ablation condition."""
    n = A.shape[0]
    if condition == "full":
        return A
    if condition == "features_only":
        # Empty adjacency; P_hat = (D + I)^{-1}(A + I) = I on this graph, so
        # propagation is a no-op and X_2 = X.
        return sp.csr_matrix((n, n), dtype=A.dtype)
    if condition == "test_test":
        keep_node = np.zeros(n, dtype=bool)
        keep_node[node_set] = True
        Ac = A.tocoo()
        edge_mask = keep_node[Ac.row] & keep_node[Ac.col]
        return sp.coo_matrix(
            (Ac.data[edge_mask], (Ac.row[edge_mask], Ac.col[edge_mask])),
            shape=A.shape,
        ).tocsr()
    raise ValueError(f"Unknown condition {condition!r}")


def run_one(ds, W, H, Z_full: np.ndarray, condition: str, K: int) -> dict:
    """Only test-node rows are re-embedded on the masked graph.

    Reviewer's Con 4 concerns *test-time* graph information: how much of the
    observable graph is available at inference for test nodes. Train/val
    embeddings are always taken from the full-graph run (Z_full) so the
    downstream LR probe sees a consistent, high-quality training signal;
    only test rows differ across conditions.
    """
    A_full = ds["A_full"]
    A_masked = mask_adjacency(A_full, condition, ds["test"])
    t0 = time.time()
    Z_masked = gw_cca_embed(ds["X_full"], A_masked, W, H, K=K, l2_normalize=True)
    if condition == "features_only":
        # Consistent-geometry probe: train and test rows both use the
        # graph-free embedding, so the number is directly comparable to
        # Raw+LR. The prior splice fit LR on Z_full[train] and evaluated
        # on Z_masked[test], introducing a train/test embedding-distribution
        # shift that inflated the C-A gap.
        Z = Z_masked
    else:
        # Splice: train/val from full-graph embedding, test from masked-graph embedding.
        Z = Z_full.copy()
        Z[ds["test"]] = Z_masked[ds["test"]]
    embed_s = time.time() - t0
    nc = eval_node_cls(
        Z,
        ds["y"],
        ds["train"],
        ds["val"],
        ds["test"],
        metric=ds.get("metric", "accuracy"),
    )
    return {
        "condition": condition,
        "nnz_masked": int(A_masked.nnz),
        "embed_s": float(embed_s),
        "nc_val": nc["val_acc"],
        "nc_test": nc["test_acc"],
        "nc_metric": nc["metric"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"Unknown condition: {c!r} (available: {CONDITIONS})")

    d = args.d if args.d is not None else DEFAULT_D[args.dataset]
    lam = args.lam if args.lam is not None else DEFAULT_LAM[args.dataset]

    print(f"Loading {args.dataset}...", flush=True)
    ds = DATASETS[args.dataset]()
    d = min(d, ds["X_train"].shape[1])
    print(
        f"  n_full={ds['n']:,}  n_train={len(ds['train']):,}  "
        f"n_test={len(ds['test']):,}  A_full nnz={ds['A_full'].nnz:,}  "
        f"d={d}  K={args.K}  lam={lam}"
    )

    print("\nTraining GW-CCA (paper defaults, p1p2 kernel)...", flush=True)
    t0 = time.time()
    W, H = gw_cca_train(ds["X_train"], ds["A_train"], d=d, K=args.K, lam=lam)
    print(f"  train {time.time() - t0:.2f} s   W {W.shape}   H {H.shape}", flush=True)

    print("Embedding full graph once (train/val rows use this)...", flush=True)
    t0 = time.time()
    Z_full = gw_cca_embed(ds["X_full"], ds["A_full"], W, H, K=args.K, l2_normalize=True)
    print(f"  embed_full {time.time() - t0:.2f} s   Z_full {Z_full.shape}", flush=True)

    results = []
    for cond in conditions:
        print(f"\n>>> {cond}", flush=True)
        r = run_one(ds, W, H, Z_full, cond, K=args.K)
        r["dataset"] = args.dataset
        results.append(r)
        print(
            f"  {cond:>14s}  nnz {r['nnz_masked']:>12,}   NC {r['nc_metric']:>8s} "
            f"{r['nc_test'] * 100:6.2f}   embed {r['embed_s']:6.2f} s",
            flush=True,
        )

    print("\n=== summary ===")
    for r in results:
        print(
            f"  {r['condition']:>14s}   NC {r['nc_metric']:>8s} {r['nc_test'] * 100:6.2f}   "
            f"edges kept {r['nnz_masked']:>12,}"
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
