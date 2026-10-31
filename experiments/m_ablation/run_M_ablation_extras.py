"""M-kernel ablation on additional benchmarks (Platonov heterophilic +
LINKX small). The loaders below build each dataset with `data._finalize_ds`,
which always creates the link-prediction split (`_make_lp_split`).

Adds these datasets to the sweep:
    Platonov (Luan 2023): roman-empire, amazon-ratings, minesweeper, tolokers, questions
    LINKX-small:          penn94, genius

Uses the same M variants as run_M_ablation.py: p1p2 (paper baseline), p1,
p2, ppr, heat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RELEASE = _HERE.parents[1]  # repo root: gw_cca.py, data.py, probe.py
sys.path.insert(0, str(_RELEASE))
sys.path.insert(0, str(_HERE))

import data as data_module  # noqa: E402
from gw_cca import DEFAULT_D, DEFAULT_LAM  # noqa: E402
from run_M_ablation import run_one, APPLIERS  # noqa: E402


# ----------------------------------------------------------------------
# Re-expose Platonov / LINKX loaders with LP enabled
# ----------------------------------------------------------------------


def _platonov_with_lp(pyg_name: str, ds_name: str):
    def loader():
        from torch_geometric.datasets import HeterophilousGraphDataset
        from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected
        import numpy as np

        data_module._ogb_safe_globals()
        ds = HeterophilousGraphDataset(name=pyg_name, root="/tmp/platonov")
        d = ds[0]
        n = d.num_nodes
        edge_index_full = to_undirected(d.edge_index, num_nodes=n)
        A_full = data_module._symmetrize(to_scipy_sparse_matrix(edge_index_full, num_nodes=n))
        X_full = d.x.numpy().astype(np.float32)
        y = d.y.numpy().astype(np.int64)
        train_idx = d.train_mask[:, 0].numpy().nonzero()[0]
        val_idx = d.val_mask[:, 0].numpy().nonzero()[0]
        test_idx = d.test_mask[:, 0].numpy().nonzero()[0]
        return data_module._finalize_ds(
            ds_name,
            n,
            X_full,
            edge_index_full,
            A_full,
            y,
            train_idx,
            val_idx,
            test_idx,
        )

    return loader


def _linkx_pyg_with_lp(pyg_name: str, ds_name: str):
    def loader():
        from torch_geometric.datasets import LINKXDataset
        from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected
        import numpy as np

        data_module._ogb_safe_globals()
        ds = LINKXDataset(root="/tmp/linkx-pyg", name=pyg_name)
        d = ds[0]
        n = d.num_nodes
        edge_index_full = to_undirected(d.edge_index, num_nodes=n)
        A_full = data_module._symmetrize(to_scipy_sparse_matrix(edge_index_full, num_nodes=n))
        X_full = d.x.numpy().astype(np.float32)
        y = d.y.view(-1).numpy().astype(np.int64)
        if hasattr(d, "train_mask") and d.train_mask is not None:
            tm, vm, sm = d.train_mask, d.val_mask, d.test_mask
            if tm.dim() == 2:
                tm, vm, sm = tm[:, 0], vm[:, 0], sm[:, 0]
            train_idx = tm.nonzero(as_tuple=False).view(-1).numpy()
            val_idx = vm.nonzero(as_tuple=False).view(-1).numpy()
            test_idx = sm.nonzero(as_tuple=False).view(-1).numpy()
        else:
            train_idx, val_idx, test_idx = data_module._linkx_random_split(n, seed=42)
        return data_module._finalize_ds(
            ds_name,
            n,
            X_full,
            edge_index_full,
            A_full,
            y,
            train_idx,
            val_idx,
            test_idx,
        )

    return loader


EXTRAS = {
    "roman-empire": _platonov_with_lp("Roman-empire", "roman-empire"),
    "amazon-ratings": _platonov_with_lp("Amazon-ratings", "amazon-ratings"),
    "minesweeper": _platonov_with_lp("Minesweeper", "minesweeper"),
    "tolokers": _platonov_with_lp("Tolokers", "tolokers"),
    "questions": _platonov_with_lp("Questions", "questions"),
    "penn94": _linkx_pyg_with_lp("penn94", "penn94"),
    "genius": _linkx_pyg_with_lp("genius", "genius"),
}

# Sensible defaults for datasets not in DEFAULT_D
_EXTRA_LAM = {
    "roman-empire": 1e-2,
    "amazon-ratings": 1e-2,
    "minesweeper": 1e-2,
    "tolokers": 1e-2,
    "questions": 1e-2,
    "penn94": 1e-2,
    "genius": 1e-2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(EXTRAS.keys()))
    ap.add_argument("--kernels", default="p1p2,p1,p2,ppr,heat")
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    for k in kernels:
        if k not in APPLIERS:
            raise SystemExit(f"Unknown kernel: {k!r}")

    d = args.d if args.d is not None else DEFAULT_D.get(args.dataset, 128)
    lam = args.lam if args.lam is not None else _EXTRA_LAM.get(args.dataset, 1e-2)

    print(f"Loading {args.dataset}...", flush=True)
    ds = EXTRAS[args.dataset]()
    d = min(d, ds["X_train"].shape[1])
    print(
        f"  n_full={ds['n']:,}  n_train={len(ds['train']):,}  p={ds['X_train'].shape[1]}  d={d}  K={args.K}  lam={lam}"
    )

    results = []
    for kernel in kernels:
        print(f"\n>>> M = {kernel}", flush=True)
        try:
            r = run_one(ds, kernel=kernel, d=d, K=args.K, lam=lam)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            r = {"kernel": kernel, "auc": None, "ap": None, "train_s": None, "error": str(e)}
        r["dataset"] = args.dataset
        r["d"] = d
        r["K"] = args.K
        r["lam"] = lam
        results.append(r)
        if r.get("auc") is not None:
            print(
                f"  {kernel:>6s}  AUC {r['auc'] * 100:6.2f}   AP {r['ap'] * 100:6.2f}   train {r['train_s']:6.2f} s",
                flush=True,
            )

    print("\n=== summary ===")
    print(f"{'kernel':>6s}  {'AUC (%)':>8s}  {'AP (%)':>8s}  {'train_s':>8s}")
    for r in results:
        if r.get("auc") is not None:
            print(f"{r['kernel']:>6s}  {r['auc'] * 100:8.2f}  {r['ap'] * 100:8.2f}  {r['train_s']:8.2f}")
        else:
            print(f"{r['kernel']:>6s}  {'FAIL':>8s}  {'FAIL':>8s}  {'-':>8s}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
