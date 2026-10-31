"""Rebuttal Con 3 — external linear/diffusion/spectral baselines.

Three parameter-free feature-propagation baselines, one per reviewer bucket:

    SGC-lin       Z = S_hat^K X                             (linear)
    APPNP-lin     Z = alpha (I - (1-alpha) A_hat)^{-1} X    (diffusion / PPR)
    ChebNet-lin   Z = concat_{k=0..K} T_k(L_tilde) X        (spectral)

Notation:
    S_hat = (D+I)^{-1/2} (A+I) (D+I)^{-1/2}     (Kipf sym-normalised)
    A_hat = D^{-1/2} A D^{-1/2}                 (sym-normalised, no self-loops
                                                for L = I - A_hat convention)
    L_tilde = 2 L / lambda_max - I,  with lambda_max = 2, so L_tilde = -A_hat.
    T_0(x) = I,  T_1(x) = x,  T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x).

Every method:
    * is applied to the FULL graph at inference time (`ds["A_lp"]`, `ds["X_full"]`);
    * needs no learned parameters -> the strict-inductive protocol is trivially
      satisfied (no peeking at LP edges during "training", because there is no
      training);
    * outputs an L2-row-normalised embedding matrix, matching GW-CCA's `_l2_rows`;
    * is evaluated by the same LBFGS-on-Hadamard-products LP probe as
      `run_M_ablation.py`.

Example:
    python run_linear_baselines.py --dataset ogbn-arxiv --out ogbn-arxiv_lb.json
    python run_linear_baselines.py --dataset ppi --lam 0.01 --out ppi_lb.json  # lam unused
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import DATASETS  # noqa: E402
from run_M_ablation import eval_link_pred  # noqa: E402
from probe import eval_link_pred as eval_link_pred_sklearn, eval_node_cls  # noqa: E402


# ----------------------------------------------------------------------
# Graph operators
# ----------------------------------------------------------------------


def _sym_norm_with_self_loops(A: sp.spmatrix) -> sp.spmatrix:
    """S_hat = (D+I)^{-1/2} (A+I) (D+I)^{-1/2}."""
    n = A.shape[0]
    A_sl = A + sp.eye(n)
    d = np.asarray(A_sl.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = sp.diags(d_inv_sqrt)
    return D @ A_sl @ D


def _sym_norm_no_self_loops(A: sp.spmatrix) -> sp.spmatrix:
    """A_hat = D^{-1/2} A D^{-1/2} (for L = I - A_hat)."""
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = sp.diags(d_inv_sqrt)
    return D @ A @ D


def _l2_rows(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return A / n


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------


def embed_sgc(X: np.ndarray, A: sp.spmatrix, K: int = 3) -> np.ndarray:
    """SGC-lin: Z = S_hat^K X (parameter-free — the W in SGC is a supervised head)."""
    S = _sym_norm_with_self_loops(A)
    Z = X.astype(np.float64)
    for _ in range(K):
        Z = S @ Z
    return _l2_rows(Z.astype(np.float32))


def embed_appnp(X: np.ndarray, A: sp.spmatrix, K: int = 10, alpha: float = 0.1) -> np.ndarray:
    """APPNP-lin: K-step personalised PageRank power iteration on X.

    H^{(0)} = X
    H^{(k+1)} = (1 - alpha) A_hat H^{(k)} + alpha H^{(0)}

    Uses A_hat = (D+I)^{-1/2} (A+I) (D+I)^{-1/2} (Klicpera's default).
    """
    A_hat = _sym_norm_with_self_loops(A)
    H0 = X.astype(np.float64)
    H = H0.copy()
    for _ in range(K):
        H = (1.0 - alpha) * (A_hat @ H) + alpha * H0
    return _l2_rows(H.astype(np.float32))


def embed_chebnet(X: np.ndarray, A: sp.spmatrix, K: int = 3) -> np.ndarray:
    """Linear-ChebNet: concat of T_k(L_tilde) X for k = 0..K.

    With lambda_max = 2 for the normalised Laplacian, L_tilde = L - I = -A_hat.
    So T_1 X = -A_hat X, T_{k+1} X = -2 A_hat T_k X - T_{k-1} X.
    """
    A_hat = _sym_norm_no_self_loops(A)
    X64 = X.astype(np.float64)
    T_prev = X64
    T_curr = -(A_hat @ X64)
    views = [T_prev, T_curr]
    for _ in range(K - 1):
        T_next = -2.0 * (A_hat @ T_curr) - T_prev
        views.append(T_next)
        T_prev, T_curr = T_curr, T_next
    Z = np.concatenate(views, axis=1)
    return _l2_rows(Z.astype(np.float32))


BASELINES = {
    "sgc": lambda X, A: embed_sgc(X, A, K=3),
    "appnp": lambda X, A: embed_appnp(X, A, K=10, alpha=0.1),
    "chebnet": lambda X, A: embed_chebnet(X, A, K=3),
}


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def run_one(ds, name: str, cpu_eval: bool = False, do_lp: bool = True, do_nc: bool = True) -> dict:
    """Embed via `name`, then optionally run LP and NC probes."""
    embed = BASELINES[name]
    t0 = time.time()
    # NC uses the full inductive graph's adjacency (ds["A"]); LP has its own
    # supervision-scrubbed adjacency (ds["A_lp"]) — they can differ.
    A_nc = ds.get("A", ds["A_lp"])
    Z_nc = embed(ds["X_full"], A_nc)
    if do_lp:
        # LP embedding uses A_lp (edges removed to form pos/neg pairs)
        if id(A_nc) == id(ds.get("A_lp", A_nc)) or ds.get("A_lp") is None:
            Z_lp = Z_nc
        else:
            Z_lp = embed(ds["X_full"], ds["A_lp"])
    embed_s = time.time() - t0

    out = {"method": name, "embed_s": float(embed_s)}

    if do_lp and not ds.get("skip_lp", False):
        if cpu_eval:
            lp = eval_link_pred_sklearn(Z_lp, ds)
        else:
            try:
                lp = eval_link_pred(Z_lp, ds)
            except Exception as e:
                print(f"  GPU LP probe failed ({type(e).__name__}); falling back to sklearn", flush=True)
                import torch

                torch.cuda.empty_cache()
                lp = eval_link_pred_sklearn(Z_lp, ds)
        out["lp_auc"] = lp["auc"]
        out["lp_ap"] = lp["ap"]

    if do_nc:
        nc = eval_node_cls(
            Z_nc,
            ds["y"],
            ds["train"],
            ds["val"],
            ds["test"],
            metric=ds.get("metric", "accuracy"),
        )
        out["nc_val"] = nc["val_acc"]
        out["nc_test"] = nc["test_acc"]
        out["nc_metric"] = nc["metric"]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--methods", default="sgc,appnp,chebnet")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--cpu-eval",
        action="store_true",
        help="Force sklearn LP probe (CPU) — needed for wide embeddings on ogbn-products",
    )
    ap.add_argument("--tasks", default="lp,nc", help="Comma-separated: lp, nc")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in BASELINES:
            raise SystemExit(f"Unknown baseline: {m!r} (available: {list(BASELINES)})")

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    do_lp = "lp" in tasks
    do_nc = "nc" in tasks

    print(f"Loading {args.dataset}...", flush=True)
    ds = DATASETS[args.dataset]()
    print(f"  n_full={ds['n']:,}  p={ds['X_full'].shape[1]}", flush=True)

    results = []
    for name in methods:
        print(f"\n>>> {name}", flush=True)
        r = run_one(ds, name, cpu_eval=args.cpu_eval, do_lp=do_lp, do_nc=do_nc)
        r["dataset"] = args.dataset
        results.append(r)
        parts = [f"embed {r['embed_s']:6.2f} s"]
        if "lp_auc" in r:
            parts.append(f"LP AUC {r['lp_auc'] * 100:6.2f}  AP {r['lp_ap'] * 100:6.2f}")
        if "nc_test" in r:
            parts.append(f"NC {r['nc_metric']:>7s} {r['nc_test'] * 100:6.2f}")
        print("  " + f"{name:>8s}  " + "   ".join(parts), flush=True)

    print("\n=== summary ===")
    for r in results:
        parts = [f"{r['method']:>8s}"]
        if "lp_auc" in r:
            parts.append(f"LP {r['lp_auc'] * 100:6.2f}/{r['lp_ap'] * 100:6.2f}")
        if "nc_test" in r:
            parts.append(f"NC {r['nc_metric']:>7s} {r['nc_test'] * 100:6.2f}")
        parts.append(f"({r['embed_s']:.1f}s)")
        print("  " + "   ".join(parts))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
