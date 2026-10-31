# Experiments

Supplementary experiments for GW-CCA, run with the method code in the repo root
(`gw_cca.py`, `data.py`, `probe.py`). Each script prepends the repo root to
`sys.path`, so nothing needs to be installed.

## `m_ablation/`

| Script | What it runs |
|---|---|
| `run_M_ablation.py` | Link prediction with the pair-weight matrix M swapped: `p1p2` (paper default), `p1`, `p2`, `ppr`, `heat`. Datasets from `data.DATASETS`. |
| `run_M_ablation_extras.py` | The same ablation on Platonov heterophilic graphs (roman-empire, amazon-ratings, minesweeper, tolokers, questions) and LINKX-small. |
| `run_connectivity_ablation.py` | Effect of graph connectivity on node classification. |
| `run_linear_baselines.py` | Parameter-free linear baselines (SGC, APPNP, ChebNet) under the same protocol. |
| `_build_notebooks.py` | Regenerates the three notebooks from the saved results. |

```bash
python experiments/m_ablation/run_M_ablation.py --dataset ppi
python experiments/m_ablation/run_M_ablation_extras.py --dataset minesweeper --out out.json
```

Saved results are the `*_M_lp.json`, `*_conn*.json` and `*_lb.json` /
`*_nc_lb.json` files; write-ups are `RESULTS.md`, `CONNECTIVITY.md`,
`BASELINES.md` and `SUPPLEMENTARY.md`.

## Known gaps

- `BASELINES.ipynb` reads the paper's per-dataset GW-CCA results from
  `results/_res_GCCA__<dataset>.json`. Those files are not included, so its
  GW-CCA column cannot be regenerated until they are produced again.
- Datasets are downloaded on first use (PyG / OGB) into `/tmp`; `ogb` must be
  installed for the OGB loaders.
