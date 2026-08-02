"""Downstream evaluation: node-classification LR probe + link-prediction probe.

The node-classification probe is a torch LBFGS linear classifier on frozen
embeddings — algorithmically equivalent to sklearn's
`LogisticRegression(solver="lbfgs", C=1.0)` but faster on GPU and
bit-deterministic. Link prediction uses a scikit-learn LR on Hadamard products
of endpoint embeddings.
"""
from __future__ import annotations

import numpy as np
import torch


def _is_multilabel(y: np.ndarray) -> bool:
    return isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] > 1


def _torch_lr_probe(
    Z: np.ndarray,
    y: np.ndarray,
    tr_idx, val_idx, te_idx,
    *, multilabel: bool, num_classes: int,
    max_iter: int = 100, C: float = 1.0, seed: int = 42,
) -> tuple[float, float]:
    """LBFGS-trained linear probe on frozen embeddings. Deterministic."""
    from sklearn.metrics import f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = Z[tr_idx].mean(0, keepdims=True)
    sd = Z[tr_idx].std(0, keepdims=True) + 1e-6
    Zs = ((Z - mu) / sd).astype(np.float32)
    Xt = torch.from_numpy(Zs).to(device)

    n_in = Zs.shape[1]
    if multilabel:
        yt = torch.from_numpy(y.astype(np.float32)).to(device)
        n_out = y.shape[1]
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    else:
        yt = torch.from_numpy(y.astype(np.int64)).to(device)
        n_out = int(num_classes)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    tr_t = torch.from_numpy(np.asarray(tr_idx, dtype=np.int64)).to(device)
    val_t = torch.from_numpy(np.asarray(val_idx, dtype=np.int64)).to(device)
    te_t = torch.from_numpy(np.asarray(te_idx, dtype=np.int64)).to(device)

    torch.manual_seed(seed)
    model = torch.nn.Linear(n_in, n_out).to(device)
    Xtr = Xt[tr_t]; ytr = yt[tr_t]

    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=max_iter,
        history_size=20, tolerance_grad=1e-4, line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        for p in model.parameters():
            loss = loss + (0.5 / C) * (p ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)

    model.eval()
    with torch.no_grad():
        if multilabel:
            pv = (model(Xt[val_t]) > 0).cpu().numpy()
            pt = (model(Xt[te_t]) > 0).cpu().numpy()
            fv = f1_score(y[val_idx], pv, average="micro", zero_division=0)
            ft = f1_score(y[te_idx], pt, average="micro", zero_division=0)
        else:
            pv = model(Xt[val_t]).argmax(dim=1).cpu().numpy()
            pt = model(Xt[te_t]).argmax(dim=1).cpu().numpy()
            fv = float((pv == y[val_idx]).mean())
            ft = float((pt == y[te_idx]).mean())
    return fv, ft


def eval_node_cls(Z, y, train_idx, val_idx, test_idx, num_classes=None):
    """Node-classification LR probe. Returns dict with val/test Micro-F1."""
    multilabel = _is_multilabel(y)
    if num_classes is None:
        num_classes = y.shape[1] if multilabel else int(y.max() + 1)
    v, t = _torch_lr_probe(Z, y, train_idx, val_idx, test_idx,
                           multilabel=multilabel, num_classes=num_classes)
    return {"val_acc": v, "test_acc": t,
            "metric": "micro-f1" if multilabel else "accuracy"}


def eval_link_pred(Z_lp, ds, max_iter: int = 100):
    """Link-prediction probe (paper §5.5): LR on Hadamard products of endpoint
    embeddings, trained on train-node edges, evaluated on held-out test edges."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler

    def had(edges):
        return Z_lp[edges[:, 0]] * Z_lp[edges[:, 1]]

    X_train = np.concatenate([had(ds["lp_train_pos"]), had(ds["lp_train_neg"])], axis=0)
    y_train = np.concatenate([np.ones(len(ds["lp_train_pos"])),
                              np.zeros(len(ds["lp_train_neg"]))])
    X_test = np.concatenate([had(ds["lp_test_pos"]), had(ds["lp_test_neg"])], axis=0)
    y_test = np.concatenate([np.ones(len(ds["lp_test_pos"])),
                             np.zeros(len(ds["lp_test_neg"]))])

    sc = StandardScaler(with_mean=False).fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    clf = LogisticRegression(max_iter=max_iter, solver="lbfgs")
    clf.fit(X_train, y_train)
    scores = clf.decision_function(X_test)
    return {"auc": float(roc_auc_score(y_test, scores)),
            "ap": float(average_precision_score(y_test, scores))}
