"""Dataset loaders and strict-inductive splits for the five benchmarks.

Each loader returns a dict containing both the full-graph views (for inductive
inference) and the training-subgraph views (for training the encoder), plus
train/val/test node indices and the link-prediction split.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _symmetrize(A: sp.spmatrix) -> sp.spmatrix:
    """Undirected + binarize + no self-loops."""
    A = (A + A.T).tocsr()
    A.data[:] = 1
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _sample_neg_edges(existing, node_pool, n_samples, rng, max_attempts_mult=100):
    """Sample n_samples unordered node pairs (u < v) that are NOT in `existing`."""
    negs = []
    for _ in range(max_attempts_mult * n_samples):
        if len(negs) >= n_samples:
            break
        u, v = int(rng.choice(node_pool)), int(rng.choice(node_pool))
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) not in existing:
            negs.append((u, v))
            existing.add((u, v))
    return np.array(negs, dtype=np.int64)


def _make_lp_split(edge_index_full, train_idx, test_idx, n_nodes, ratio=0.2, seed=42):
    """Strict-inductive link-prediction split: hold out 20% of test-test edges."""
    from torch_geometric.utils import to_scipy_sparse_matrix

    rng = np.random.default_rng(seed)
    ei_np = edge_index_full.numpy()
    src, dst = ei_np[0], ei_np[1]
    test_mask = np.zeros(n_nodes, dtype=bool); test_mask[test_idx] = True
    train_mask = np.zeros(n_nodes, dtype=bool); train_mask[train_idx] = True
    order = src < dst

    both_test = test_mask[src] & test_mask[dst]
    test_edges = np.stack([src[both_test & order], dst[both_test & order]], axis=1)
    n_hold = max(1, int(len(test_edges) * ratio))
    perm = rng.permutation(len(test_edges))
    holdout_pos = test_edges[perm[:n_hold]]
    holdout_set = set(map(tuple, holdout_pos.tolist()))

    keep = np.ones(len(src), dtype=bool)
    for i in range(len(src)):
        u, v = (src[i], dst[i]) if src[i] < dst[i] else (dst[i], src[i])
        if (u, v) in holdout_set:
            keep[i] = False
    ei_lp = torch.from_numpy(np.stack([src[keep], dst[keep]]))
    A_lp = _symmetrize(to_scipy_sparse_matrix(ei_lp, num_nodes=n_nodes))

    existing_test_set = set(map(tuple, test_edges.tolist()))
    holdout_neg = _sample_neg_edges(existing_test_set, test_idx, n_hold, rng)

    both_train = train_mask[src] & train_mask[dst]
    train_edges_all = np.stack([src[both_train & order], dst[both_train & order]], axis=1)
    if len(train_edges_all) > n_hold:
        train_edges = train_edges_all[rng.choice(len(train_edges_all), n_hold, replace=False)]
    else:
        train_edges = train_edges_all
    train_edge_set = set(map(tuple, train_edges.tolist()))
    train_neg = _sample_neg_edges(train_edge_set, train_idx, len(train_edges), rng)

    return dict(edge_index_lp=ei_lp, A_lp=A_lp,
                lp_train_pos=train_edges, lp_train_neg=train_neg,
                lp_test_pos=holdout_pos, lp_test_neg=holdout_neg)


def _finalize_ds(name, n, X_full, edge_index_full, A_full, y,
                 train_idx, val_idx, test_idx, multilabel=False, lp_seed=42):
    from torch_geometric.utils import subgraph, to_scipy_sparse_matrix

    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
    ei_train, _ = subgraph(train_mask, edge_index_full, relabel_nodes=True, num_nodes=n)
    n_train = int(train_mask.sum())
    A_train = _symmetrize(to_scipy_sparse_matrix(ei_train, num_nodes=n_train))
    X_train = X_full[np.sort(np.asarray(train_idx))]
    lp = _make_lp_split(edge_index_full, train_idx, test_idx, n, seed=lp_seed)
    num_classes = int(y.shape[1]) if multilabel else int(y.max() + 1)
    ds = dict(
        name=name, n=n,
        X_full=X_full, A_full=A_full, edge_index_full=edge_index_full,
        X_train=X_train, A_train=A_train, edge_index_train=ei_train,
        train=np.asarray(train_idx), val=np.asarray(val_idx), test=np.asarray(test_idx),
        y=y, num_classes=num_classes, multilabel=multilabel,
    )
    ds.update(lp)
    return ds


def _ogb_safe_globals():
    """Register OGB tensor classes for torch.load + auto-accept dataset downloads."""
    try:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
    except Exception:
        pass
    try:
        import ogb.utils.url as _u
        _u.decide_download = lambda url: True
    except Exception:
        pass


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def load_ogbn_arxiv(root="/tmp/ogb"):
    _ogb_safe_globals()
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected

    ds = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    d = ds[0]; n = d.num_nodes
    edge_index_full = to_undirected(d.edge_index, num_nodes=n)
    A_full = _symmetrize(to_scipy_sparse_matrix(edge_index_full, num_nodes=n))
    X_full = d.x.numpy().astype(np.float32)
    y = d.y.view(-1).numpy().astype(np.int64)
    split = ds.get_idx_split()
    return _finalize_ds("ogbn-arxiv", n, X_full, edge_index_full, A_full, y,
                        split["train"].numpy(), split["valid"].numpy(), split["test"].numpy())


def load_ogbn_products(root="/tmp/ogb"):
    _ogb_safe_globals()
    from ogb.nodeproppred import PygNodePropPredDataset
    from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected

    ds = PygNodePropPredDataset(name="ogbn-products", root=root)
    d = ds[0]; n = d.num_nodes
    edge_index_full = to_undirected(d.edge_index, num_nodes=n)
    A_full = _symmetrize(to_scipy_sparse_matrix(edge_index_full, num_nodes=n))
    X_full = d.x.numpy().astype(np.float32)
    y = d.y.view(-1).numpy().astype(np.int64)
    split = ds.get_idx_split()
    return _finalize_ds("ogbn-products", n, X_full, edge_index_full, A_full, y,
                        split["train"].numpy(), split["valid"].numpy(), split["test"].numpy())


def load_reddit(root="/tmp/reddit"):
    _ogb_safe_globals()
    from torch_geometric.datasets import Reddit2
    from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected

    ds = Reddit2(root=root)
    d = ds[0]; n = d.num_nodes
    edge_index_full = to_undirected(d.edge_index, num_nodes=n)
    A_full = _symmetrize(to_scipy_sparse_matrix(edge_index_full, num_nodes=n))
    X_full = d.x.numpy().astype(np.float32)
    y = d.y.numpy().astype(np.int64)
    train_idx = d.train_mask.numpy().nonzero()[0]
    val_idx = d.val_mask.numpy().nonzero()[0]
    test_idx = d.test_mask.numpy().nonzero()[0]
    return _finalize_ds("reddit", n, X_full, edge_index_full, A_full, y,
                        train_idx, val_idx, test_idx)


def load_yelp(root="/tmp/yelp"):
    """Yelp (GraphSAINT split) — 717K users, 100-label multi-label."""
    _ogb_safe_globals()
    from torch_geometric.datasets import Yelp
    from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected

    ds = Yelp(root=root)
    d = ds[0]; n = d.num_nodes
    edge_index_full = to_undirected(d.edge_index, num_nodes=n)
    A_full = _symmetrize(to_scipy_sparse_matrix(edge_index_full, num_nodes=n))
    X_full = d.x.numpy().astype(np.float32)
    y = d.y.numpy().astype(np.int64)
    train_idx = d.train_mask.numpy().nonzero()[0]
    val_idx = d.val_mask.numpy().nonzero()[0]
    test_idx = d.test_mask.numpy().nonzero()[0]
    return _finalize_ds("yelp", n, X_full, edge_index_full, A_full, y,
                        train_idx, val_idx, test_idx, multilabel=True)


def load_ppi(root="/tmp/ppi"):
    """PPI — 24 protein-protein interaction graphs, 121-label multi-label.

    Genuinely graph-inductive: train on 20 graphs, val on 2, test on 2. We
    stitch them into a disjoint-union graph so the node-level protocol handles
    it transparently — no cross-graph edges means inductivity is exact.
    """
    _ogb_safe_globals()
    from torch_geometric.datasets import PPI
    from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected

    Xs, ys, edge_index_parts, split_masks = [], [], [], {"train": [], "val": [], "test": []}
    offset = 0
    for split_name in ("train", "val", "test"):
        for d in PPI(root=root, split=split_name):
            n_i = d.num_nodes
            Xs.append(d.x.numpy().astype(np.float32))
            ys.append(d.y.numpy().astype(np.int64))
            edge_index_parts.append(d.edge_index + offset)
            split_masks[split_name].append(np.arange(offset, offset + n_i))
            offset += n_i
    X_full = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    edge_index_full = to_undirected(torch.cat(edge_index_parts, dim=1), num_nodes=offset)
    A_full = _symmetrize(to_scipy_sparse_matrix(edge_index_full, num_nodes=offset))
    train_idx = np.concatenate(split_masks["train"])
    val_idx = np.concatenate(split_masks["val"])
    test_idx = np.concatenate(split_masks["test"])
    return _finalize_ds("ppi", offset, X_full, edge_index_full, A_full, y,
                        train_idx, val_idx, test_idx, multilabel=True)


DATASETS = {
    "ogbn-arxiv": load_ogbn_arxiv,
    "reddit": load_reddit,
    "yelp": load_yelp,
    "ogbn-products": load_ogbn_products,
    "ppi": load_ppi,
}
