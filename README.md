# GW-CCA: Graph-Weighted Canonical Correlation Analysis

Closed-form, training-free inductive attributed graph embedding.

This repo contains the exact code used in the paper — an implementation of
**Algorithm 1** (training + inductive inference). Every GW-CCA result in the
paper's tables can be reproduced with a single `python run.py` call.

Code released with an anonymous submission; identifying information has been
removed.

## Contents

```
gcca.py            Algorithm 1: gcca_train (Cholesky + truncated SVD)
                                gcca_embed (inductive inference with L2 rows)
data.py            Strict-inductive dataset loaders + train/test edge splits
probe.py           Downstream evaluation (torch LBFGS LR probe + link-pred LR)
run.py             CLI entry: python run.py --dataset <name>
requirements.txt   Minimal dependencies (numpy, scipy, torch, torch-geometric, ogb, sklearn)
```

## Setup

```bash
pip install -r requirements.txt
```

Datasets are downloaded on first use via `torch-geometric` / `ogb` into
`/tmp/{ogb,reddit,yelp,ppi}` — set a different root by editing the loader calls
in `data.py` if you prefer.

## Reproduce every Table-2 / Table-3 GW-CCA cell

Each command below reproduces one column of the paper's node-classification and
link-prediction tables. All use `K = 3`; `d` and `λ` default to the paper's
§5.3 values (see `DEFAULT_D` / `DEFAULT_LAM` in `gcca.py`).

```bash
python run.py --dataset ogbn-arxiv       # d=120, λ=0
python run.py --dataset reddit           # d=120, λ=0
python run.py --dataset yelp             # d=300, λ=0.01
python run.py --dataset ogbn-products    # d=100, λ=0
python run.py --dataset ppi              # d=50,  λ=0
```

Add `--swap` to get the GW-CCA(swap) variant (§4.3). GW-CCA is **deterministic**:
the same command yields bit-identical output every run.

Expected output on `ogbn-arxiv`:

```
=== gw-cca on ogbn-arxiv ===
  Node classification (accuracy): val 71.5   test 69.7
  Link prediction:                AUC 97.5   AP 97.7
  Encoder wall-clock:             ~1.8 s (CPU)
```

## Baseline implementations

Every deep baseline in the paper uses the **authors' official code**; we do not
reimplement any encoder. Point sources:

| Method    | Reference                          | Official code |
|-----------|------------------------------------|---------------|
| SGC       | Wu et al., ICML 2019               | 5 lines of NumPy (paper Eq. 11: $\hat{P}^K X$) |
| GraphSAGE | Hamilton et al., NeurIPS 2017      | PyG canonical `SAGEConv` |
| DGI       | Veličković et al., ICLR 2019      | PyG canonical `DeepGraphInfomax` |
| GRACE     | Zhu et al., ICML SSL workshop 2020 | https://github.com/CRIPAC-DIG/GRACE |
| CCA-SSG   | Zhang et al., NeurIPS 2021         | https://github.com/hengruizhang98/CCA-SSG |
| BGRL      | Thakoor et al., ICLR 2022          | https://github.com/nerdslab/bgrl |
| GGD       | Zheng et al., NeurIPS 2022         | https://github.com/zyzisastudyreallyhardguy/Graph-Group-Discrimination |
| GraphACL  | Xiao et al., NeurIPS 2023          | https://github.com/tengxiao1/GraphACL |
| GraphMAE2 | Hou et al., WWW 2023               | https://github.com/THUDM/GraphMAE2 |

All baselines are trained on the training-node-induced subgraph (`A_train`) only
— the **strict-inductive protocol** described in the paper's §5.2. At inference,
the frozen encoder is applied to the full graph `A_full`. Baselines use their
official default hyperparameters, mini-batched with the neighbourhood-subsampling
scheme of Hamilton et al. 2017 when the full adjacency exceeds GPU memory.
Multi-seed variance (4 seeds per method × dataset) is reported as mean ± std in
the paper's tables.
