"""GW-CCA: Graph-Weighted Canonical Correlation Analysis.

Closed-form, training-free inductive attributed graph embedding.
Implements Algorithm 1 of the paper (Cholesky + truncated SVD form).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# ----------------------------------------------------------------------
# Adjacency normalisations
# ----------------------------------------------------------------------

def row_normalize_with_self_loops(A: sp.spmatrix) -> sp.spmatrix:
    """P_hat = (D + I)^{-1} (A + I) — row-stochastic self-looped adjacency."""
    n = A.shape[0]
    A_sl = A + sp.eye(n)
    d = np.asarray(A_sl.sum(axis=1)).ravel()
    d_inv = np.where(d > 0, 1.0 / d, 0.0)
    return sp.diags(d_inv) @ A_sl


def _P_transition(A: sp.spmatrix) -> sp.spmatrix:
    """P = D^{-1} A — row-normalised transition matrix, no self-loops."""
    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv = np.where(d > 0, 1.0 / d, 0.0)
    return sp.diags(d_inv) @ A


def _propagate(A: sp.spmatrix, X: np.ndarray, K: int) -> np.ndarray:
    """Return P_hat^K X via K sparse matrix-vector products (Algorithm 1, line 2)."""
    P_hat = row_normalize_with_self_loops(A)
    Z = X
    for _ in range(K):
        Z = P_hat @ Z
    return np.asarray(Z)


def _l2_rows(A: np.ndarray) -> np.ndarray:
    """L2-normalise each row (post-projection, outside Algorithm 1)."""
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return A / n


# ----------------------------------------------------------------------
# Algorithm 1 — Training (lines 1–7)
# ----------------------------------------------------------------------

def gw_cca_train(
    X_train: np.ndarray,
    A_train: sp.spmatrix,
    d: int,
    K: int = 3,
    lam: float = 0.0,
    swap: bool = False,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Learn projections (W_tilde, H_tilde) using only the training subgraph.

    Follows Algorithm 1 of the paper line by line:

        Line 1: P = D^{-1} A;  P_hat = (D + I)^{-1} (A + I)
        Line 2: X_2 = P_hat^K X               (K sparse SpMVs)
        Line 3: M = (P + P^2) / 2             (sparse; never materialised, applied to X_2)
        Line 4: C_11 = X^T X + lambda I;  C_22 = X_2^T X_2 + lambda I;  T = X^T M X_2
        Line 5: Cholesky factorise C_kk = L_k L_k^T
        Line 6: U, Sigma, V = TruncatedSVD(C_11^{-1/2} T C_22^{-1/2}, rank=d)
        Line 7: W_tilde = C_11^{-1/2} U;  H_tilde = C_22^{-1/2} V

    If ``center=True``, column-centre X and X_2 using training-subgraph means
    (mu_1 = mean(X_train), mu_2 = mean(P_hat^K X_train)) before Line 4. The
    means are returned so that inductive inference can subtract the *training*
    statistics from the full-graph features (no test leakage). Returns
    (W_tilde, H_tilde, mu_1, mu_2); mu_1/mu_2 are ``None`` when centre=False.
    """
    from scipy.linalg import cholesky, solve_triangular

    # Line 2
    X1 = X_train.astype(np.float64)
    X2 = _propagate(A_train, X1, K)

    if center:
        mu1 = X1.mean(axis=0, keepdims=True)
        mu2 = X2.mean(axis=0, keepdims=True)
        X1 = X1 - mu1
        X2 = X2 - mu2
    else:
        mu1 = None
        mu2 = None

    if swap:
        X1, X2 = X2, X1

    # Line 3: apply M = (P + P^2)/2 to X_2 without materialising M
    P = _P_transition(A_train)
    M_X2 = 0.5 * (P @ X2 + P @ (P @ X2))

    # Line 4
    p1, p2 = X1.shape[1], X2.shape[1]
    C11 = X1.T @ X1 + lam * np.eye(p1)
    C22 = X2.T @ X2 + lam * np.eye(p2)
    T = X1.T @ M_X2

    # Line 5: Cholesky C_kk = L_k L_k^T (lower triangular)
    L1 = cholesky(C11, lower=True)
    L2 = cholesky(C22, lower=True)

    # Line 6: TruncatedSVD of C_11^{-1/2} T C_22^{-1/2} = L_1^{-1} T L_2^{-T}
    # (two triangular solves; never materialise C_kk^{-1/2} explicitly)
    A_tmp = solve_triangular(L1, T, lower=True)              # L_1^{-1} T
    B = solve_triangular(L2, A_tmp.T, lower=True).T          # (L_1^{-1} T) L_2^{-T}
    U, sigma, Vt = np.linalg.svd(B, full_matrices=False)
    d_eff = min(d, len(sigma))
    U = U[:, :d_eff]
    V = Vt[:d_eff, :].T

    # Line 7: W_tilde = L_1^{-T} U;  H_tilde = L_2^{-T} V
    W = solve_triangular(L1.T, U, lower=False)
    H = solve_triangular(L2.T, V, lower=False)
    return W, H, mu1, mu2


# ----------------------------------------------------------------------
# Algorithm 1 — Inductive Inference (lines 8–10)
# ----------------------------------------------------------------------

def gw_cca_embed(
    X_full: np.ndarray,
    A: sp.spmatrix,
    W: np.ndarray,
    H: np.ndarray,
    K: int = 3,
    swap: bool = False,
    l2_normalize: bool = True,
    mu1: np.ndarray | None = None,
    mu2: np.ndarray | None = None,
) -> np.ndarray:
    """Apply learned projections to any graph (Algorithm 1 inductive-inference phase).

        Line 8:  P_hat' = (D' + I)^{-1} (A' + I)
        Line 9:  X_2' = P_hat'^K X'
        Line 10: Z' = [X' W_tilde | X_2' H_tilde]

    l2_normalize (default True) L2-normalises rows of each view before
    concatenation — an implementation choice not part of Algorithm 1.

    mu1/mu2: optional training-subgraph column means (as returned by
    gw_cca_train(..., center=True)). When provided, they are subtracted from
    X' and X_2' respectively before applying the projections, so inductive
    inference uses *training* statistics — no test-node features enter mu.
    """
    X1 = X_full.astype(np.float64)
    X2 = _propagate(A, X1, K)
    if mu1 is not None:
        X1 = X1 - mu1
    if mu2 is not None:
        X2 = X2 - mu2
    if swap:
        X1, X2 = X2, X1
    Z1 = (X1 @ W).astype(np.float32)
    Z2 = (X2 @ H).astype(np.float32)
    if l2_normalize:
        Z1 = _l2_rows(Z1)
        Z2 = _l2_rows(Z2)
    return np.concatenate([Z1, Z2], axis=1)


# ----------------------------------------------------------------------
# Default hyperparameters (paper §5.3)
# ----------------------------------------------------------------------

DEFAULT_D = {
    "ogbn-arxiv": 120,
    "reddit": 120,
    "yelp": 300,
    "ogbn-products": 100,
    "ppi": 50,
}

# Only Yelp needs a positive ridge (its covariances are singular);
# other datasets use lambda = 0.
DEFAULT_LAM = {
    "ogbn-arxiv": 0.0,
    "reddit": 0.0,
    "yelp": 1e-2,
    "ogbn-products": 0.0,
    "ppi": 0.0,
}
