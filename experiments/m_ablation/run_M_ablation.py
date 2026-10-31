"""M-kernel ablation for GW-CCA on link prediction (rebuttal Con 1).

Fork of run.py (repo root) that swaps the pair-weight matrix M while
keeping every other piece of Algorithm 1 identical (same X_1 = X,
X_2 = P_hat^K X, same Cholesky+SVD solver, same L2-row inference, same
LP protocol). Reports LP AUC/AP per (M, dataset) cell.

M variants:
    p1p2       (P + P^2) / 2                       -- paper default (baseline)
    p1         P                                    -- first-order only
    p2         P^2                                  -- second-order only
    ppr        (1 - alpha) sum_r alpha^r P^r        -- Personalized PageRank (alpha=0.15)
    heat       exp(-t (I - P)) = e^{-t} exp(t P)    -- heat kernel (t=1.0)

Both PPR and Heat are applied to X_2 via truncated power series (R=25) so
we never materialise a dense n x n matrix -- same SpMV cost model as the
default kernel. See docstrings of `apply_M_*` below for the recurrences.

Example:
    python run_M_ablation.py --dataset ppi
    python run_M_ablation.py --dataset ogbn-arxiv --kernels p1p2,p1,p2,ppr,heat

Uses gw_cca.py / data.py / probe.py from the repo root; the root is prepended
to sys.path automatically so no install is needed.
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

# Make the release code importable
_RELEASE = Path(__file__).resolve().parents[2]  # repo root: gw_cca.py, data.py, probe.py
sys.path.insert(0, str(_RELEASE))

from data import DATASETS  # noqa: E402
from gw_cca import (  # noqa: E402
    DEFAULT_D,
    DEFAULT_LAM,
    _P_transition,
    _propagate,
    _l2_rows,
    gw_cca_embed,
)
from probe import eval_link_pred as eval_link_pred_sklearn  # noqa: E402


# ----------------------------------------------------------------------
# GPU LBFGS link-prediction probe -- torch analogue of probe.eval_link_pred
# ----------------------------------------------------------------------


def eval_link_pred_gpu(Z_lp, ds, max_iter: int = 100, C: float = 1.0, seed: int = 42):
    """LP probe on GPU. Fits a torch.nn.Linear via LBFGS on Hadamard products
    of endpoint embeddings; deterministic given seed. Faster than sklearn on
    the larger datasets where the training-edge count is in the millions.
    """
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def had(edges):
        return Z_lp[edges[:, 0]] * Z_lp[edges[:, 1]]

    X_tr = np.concatenate([had(ds["lp_train_pos"]), had(ds["lp_train_neg"])], axis=0).astype(np.float32)
    y_tr = np.concatenate([np.ones(len(ds["lp_train_pos"])), np.zeros(len(ds["lp_train_neg"]))]).astype(np.float32)
    X_te = np.concatenate([had(ds["lp_test_pos"]), had(ds["lp_test_neg"])], axis=0).astype(np.float32)
    y_te = np.concatenate([np.ones(len(ds["lp_test_pos"])), np.zeros(len(ds["lp_test_neg"]))]).astype(np.float32)

    # Standardise using training statistics (no centering, same as sklearn probe).
    sd = X_tr.std(0) + 1e-6
    X_tr = X_tr / sd
    X_te = X_te / sd

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    Xe = torch.from_numpy(X_te).to(device)

    torch.manual_seed(seed)
    n_in = X_tr.shape[1]
    model = torch.nn.Linear(n_in, 1).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        history_size=20,
        tolerance_grad=1e-4,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        logits = model(Xt).squeeze(-1)
        loss = loss_fn(logits, yt)
        for p in model.parameters():
            loss = loss + (0.5 / C) * (p**2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    model.eval()
    with torch.no_grad():
        scores = model(Xe).squeeze(-1).cpu().numpy()
    return {
        "auc": float(roc_auc_score(y_te, scores)),
        "ap": float(average_precision_score(y_te, scores)),
    }


def eval_link_pred(Z_lp, ds, use_gpu: bool = True):
    if use_gpu:
        import torch

        if torch.cuda.is_available():
            return eval_link_pred_gpu(Z_lp, ds)
    return eval_link_pred_sklearn(Z_lp, ds)


# ----------------------------------------------------------------------
# M-kernel applications (M applied to X_2 via SpMV, never materialised)
# ----------------------------------------------------------------------


def apply_M_p1p2(P: sp.spmatrix, X: np.ndarray) -> np.ndarray:
    """(P + P^2)/2 -- paper default."""
    PX = P @ X
    return 0.5 * (PX + P @ PX)


def apply_M_p1(P: sp.spmatrix, X: np.ndarray) -> np.ndarray:
    """P -- first-order random-walk proximity."""
    return P @ X


def apply_M_p2(P: sp.spmatrix, X: np.ndarray) -> np.ndarray:
    """P^2 -- second-order (shared-neighbour) proximity."""
    return P @ (P @ X)


def apply_M_ppr(P: sp.spmatrix, X: np.ndarray, alpha: float = 0.15, R: int = 25) -> np.ndarray:
    """Personalized PageRank: (1-alpha) sum_{r=0}^{R} alpha^r P^r X.

    Truncation R=25 with alpha=0.15 -> residual weight alpha^{R+1} ~ 1e-22.
    """
    y = X.astype(np.float64, copy=True)
    total = X.astype(np.float64, copy=True)
    for _ in range(R):
        y = alpha * (P @ y)
        total = total + y
    return (1.0 - alpha) * total


def apply_M_heat(P: sp.spmatrix, X: np.ndarray, t: float = 1.0, R: int = 25) -> np.ndarray:
    """Heat kernel exp(-t L_rw) X = e^{-t} sum_{k=0}^{R} (t^k / k!) P^k X.

    L_rw = I - P is the random-walk Laplacian. Truncation R=25 with t=1
    gives series tail ~t^{R+1}/(R+1)! ~ 10^{-25}.
    """
    y = X.astype(np.float64, copy=True)
    total = X.astype(np.float64, copy=True)
    for k in range(1, R + 1):
        y = (t / k) * (P @ y)
        total = total + y
    return float(np.exp(-t)) * total


APPLIERS = {
    "p1p2": apply_M_p1p2,
    "p1": apply_M_p1,
    "p2": apply_M_p2,
    "ppr": apply_M_ppr,
    "heat": apply_M_heat,
}


# ----------------------------------------------------------------------
# Training with a pluggable M applier (rest of Algorithm 1 unchanged)
# ----------------------------------------------------------------------


def gcca_train_M(
    X_train: np.ndarray,
    A_train: sp.spmatrix,
    d: int,
    K: int,
    lam: float,
    apply_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1 training phase with a swappable M-applier.

    Mirrors gcca.gw_cca_train exactly except at the M X_2 step.
    """
    from scipy.linalg import cholesky, solve_triangular

    X1 = X_train.astype(np.float64)
    X2 = _propagate(A_train, X1, K)

    P = _P_transition(A_train)
    M_X2 = apply_M(P, X2)

    p1, p2 = X1.shape[1], X2.shape[1]
    C11 = X1.T @ X1 + lam * np.eye(p1)
    C22 = X2.T @ X2 + lam * np.eye(p2)
    T = X1.T @ M_X2

    L1 = cholesky(C11, lower=True)
    L2 = cholesky(C22, lower=True)

    A_tmp = solve_triangular(L1, T, lower=True)
    B = solve_triangular(L2, A_tmp.T, lower=True).T
    U, sigma, Vt = np.linalg.svd(B, full_matrices=False)
    d_eff = min(d, len(sigma))
    U = U[:, :d_eff]
    V = Vt[:d_eff, :].T

    W = solve_triangular(L1.T, U, lower=False)
    H = solve_triangular(L2.T, V, lower=False)
    return W, H


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_one(ds, kernel: str, d: int, K: int, lam: float) -> dict:
    apply_M = APPLIERS[kernel]

    t0 = time.time()
    W, H = gcca_train_M(ds["X_train"], ds["A_train"], d=d, K=K, lam=lam, apply_M=apply_M)
    train_s = time.time() - t0

    Z_lp = gw_cca_embed(ds["X_full"], ds["A_lp"], W, H, K=K, l2_normalize=True)
    lp = eval_link_pred(Z_lp, ds)

    return {
        "kernel": kernel,
        "train_s": float(train_s),
        "auc": lp["auc"],
        "ap": lp["ap"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument(
        "--kernels",
        default="p1p2,p1,p2,ppr,heat",
        help="Comma-separated list of M-kernel names to run",
    )
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--out", type=str, default=None, help="JSON output file")
    args = ap.parse_args()

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    for k in kernels:
        if k not in APPLIERS:
            raise SystemExit(f"Unknown kernel: {k!r} (available: {list(APPLIERS)})")

    d = args.d if args.d is not None else DEFAULT_D[args.dataset]
    lam = args.lam if args.lam is not None else DEFAULT_LAM[args.dataset]

    print(f"Loading {args.dataset}...", flush=True)
    ds = DATASETS[args.dataset]()
    d = min(d, ds["X_train"].shape[1])
    print(
        f"  n_full={ds['n']:,}  n_train={len(ds['train']):,}  p={ds['X_train'].shape[1]}  d={d}  K={args.K}  lam={lam}"
    )
    if ds.get("skip_lp", False):
        raise SystemExit(f"Dataset {args.dataset} has skip_lp=True; no link prediction to run.")

    results = []
    for kernel in kernels:
        print(f"\n>>> M = {kernel}", flush=True)
        r = run_one(ds, kernel=kernel, d=d, K=args.K, lam=lam)
        r["dataset"] = args.dataset
        r["d"] = d
        r["K"] = args.K
        r["lam"] = lam
        results.append(r)
        print(
            f"  {kernel:>6s}  AUC {r['auc'] * 100:6.2f}   AP {r['ap'] * 100:6.2f}   train {r['train_s']:6.2f} s",
            flush=True,
        )

    print("\n=== summary ===")
    print(f"{'kernel':>6s}  {'AUC (%)':>8s}  {'AP (%)':>8s}  {'train_s':>8s}")
    for r in results:
        print(f"{r['kernel']:>6s}  {r['auc'] * 100:8.2f}  {r['ap'] * 100:8.2f}  {r['train_s']:8.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
