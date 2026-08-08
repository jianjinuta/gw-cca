"""Run GW-CCA on a single dataset and report node-classification Micro-F1
and link-prediction AUC/AP.

Example:
    python run.py --dataset ogbn-arxiv
    python run.py --dataset yelp --d 300 --K 3 --lam 0.01
    python run.py --dataset reddit --swap
    python run.py --dataset yelp --center
"""
from __future__ import annotations

import argparse
import time

from data import DATASETS
from gcca import DEFAULT_D, DEFAULT_LAM, gcca_embed, gcca_train
from probe import eval_link_pred, eval_node_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--swap", action="store_true", help="Use the GW-CCA(swap) variant (§4.3)")
    ap.add_argument("--d", type=int, default=None, help="Embedding dim (default from paper §5.3)")
    ap.add_argument("--K", type=int, default=3, help="Propagation depth")
    ap.add_argument("--lam", type=float, default=None, help="Ridge (default from paper §5.3)")
    ap.add_argument("--center", action="store_true",
                    help="Column-centre X and X_2 using training-subgraph means (§ ablation)")
    args = ap.parse_args()

    d = args.d if args.d is not None else DEFAULT_D[args.dataset]
    lam = args.lam if args.lam is not None else DEFAULT_LAM[args.dataset]
    swap = args.swap
    center = args.center
    method_name = "gw-cca-centered" if center else ("gw-cca-swap" if swap else "gw-cca")

    print(f"Loading {args.dataset}...", flush=True)
    ds = DATASETS[args.dataset]()
    d = min(d, ds["X_train"].shape[1])
    print(f"  n_full={ds['n']:,}  n_train={len(ds['train']):,}  p={ds['X_train'].shape[1]}  d={d}  K={args.K}  lam={lam}")

    print(f"Training {method_name} (Algorithm 1 training phase)...", flush=True)
    t0 = time.time()
    W, H, mu1, mu2 = gcca_train(ds["X_train"], ds["A_train"], d=d, K=args.K,
                                lam=lam, swap=swap, center=center)
    train_s = time.time() - t0
    print(f"  train_s = {train_s:.2f}")

    print("Inductive inference: full-graph embedding + link-prediction-graph embedding...", flush=True)
    Z_nc = gcca_embed(ds["X_full"], ds["A_full"], W, H,
                      K=args.K, swap=swap, mu1=mu1, mu2=mu2)
    Z_lp = gcca_embed(ds["X_full"], ds["A_lp"], W, H,
                      K=args.K, swap=swap, mu1=mu1, mu2=mu2)

    print("Evaluating...", flush=True)
    nc = eval_node_cls(Z_nc, ds["y"], ds["train"], ds["val"], ds["test"], ds["num_classes"])
    lp = eval_link_pred(Z_lp, ds)

    print()
    print(f"=== {method_name} on {args.dataset} ===")
    print(f"  Node classification ({nc['metric']}): val {nc['val_acc']*100:.2f}   test {nc['test_acc']*100:.2f}")
    print(f"  Link prediction:                  AUC {lp['auc']*100:.2f}   AP {lp['ap']*100:.2f}")
    print(f"  Encoder wall-clock:               {train_s:.2f} s (CPU)")


if __name__ == "__main__":
    main()
