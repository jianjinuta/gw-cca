"""Build BASELINES.ipynb and CONNECTIVITY.ipynb, exec-in-place."""

from __future__ import annotations

import subprocess
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build_baselines() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# External linear / diffusion / spectral baselines (rebuttal to Con 3)\n"
            "\n"
            "Reviewer asks for more comparisons with existing linear, diffusion-based, "
            "and spectral graph embedding methods. We add one **parameter-free**, "
            "**inductive**, **attributed** baseline per bucket and evaluate under the "
            "same strict-inductive protocol as GW-CCA.\n"
            "\n"
            "| bucket | method | operator applied to $X$ | citation |\n"
            "|---|---|---|---|\n"
            "| linear | **SGC-lin** | $Z = \\hat S^{K} X$, $K{=}3$ | Wu et al., ICML 2019 |\n"
            "| diffusion | **APPNP-lin** | $K$-step PPR power iteration on $X$, $K{=}10, \\alpha{=}0.1$ | Klicpera et al., ICLR 2019 |\n"
            "| spectral | **Linear ChebNet** | $\\text{concat}_{k=0..K}\\, T_k(\\tilde L)\\,X$, $K{=}3$ | Defferrard et al., NeurIPS 2016 |\n"
            "\n"
            "All three are the linear-head variants (no MLP / no ReLU). Embeddings are "
            "L2-row-normalised and fed to the same LBFGS Hadamard-product LP probe / "
            "linear NC probe as GW-CCA."
        ),
        md(
            "## How to reproduce\n"
            "\n"
            "```bash\n"
            "cd experiments/m_ablation\n"
            "# link prediction\n"
            "python run_linear_baselines.py --dataset ogbn-arxiv     --tasks lp --out ogbn-arxiv_lb.json\n"
            "python run_linear_baselines.py --dataset reddit          --tasks lp --out reddit_lb.json\n"
            "python run_linear_baselines.py --dataset yelp            --tasks lp --out yelp_lb.json\n"
            "python run_linear_baselines.py --dataset ogbn-products   --tasks lp --methods sgc,appnp --out ogbn-products_lb.json\n"
            "python run_linear_baselines.py --dataset ppi             --tasks lp --out ppi_lb.json\n"
            "# node classification\n"
            "python run_linear_baselines.py --dataset ogbn-arxiv     --tasks nc --out ogbn-arxiv_nc_lb.json\n"
            "python run_linear_baselines.py --dataset reddit          --tasks nc --out reddit_nc_lb.json\n"
            "python run_linear_baselines.py --dataset yelp            --tasks nc --out yelp_nc_lb.json\n"
            "python run_linear_baselines.py --dataset ogbn-products   --tasks nc --methods sgc --out ogbn-products_nc_lb.json\n"
            "python run_linear_baselines.py --dataset ppi             --tasks nc --out ppi_nc_lb.json\n"
            "```"
        ),
        code(
            "import json\n"
            "\n"
            "DATASETS = ['ogbn-arxiv', 'reddit', 'yelp', 'ogbn-products', 'ppi']\n"
            "METHODS = ['sgc', 'appnp', 'chebnet']\n"
            "\n"
            "def _load_lp(ds):\n"
            "    with open(f'{ds}_lb.json') as f:\n"
            "        return {r['method']: r for r in json.load(f)}\n"
            "\n"
            "def _load_nc(ds):\n"
            "    try:\n"
            "        with open(f'{ds}_nc_lb.json') as f:\n"
            "            return {r['method']: r for r in json.load(f)}\n"
            "    except FileNotFoundError:\n"
            "        return {}\n"
            "\n"
            "def _load_gcca(ds):\n"
            "    with open(f'../../results/_res_GCCA__{ds}.json') as f:\n"
            "        return json.load(f)\n"
            "\n"
            "LP = {ds: _load_lp(ds) for ds in DATASETS}\n"
            "NC = {ds: _load_nc(ds) for ds in DATASETS}\n"
            "GCCA = {ds: _load_gcca(ds) for ds in DATASETS}\n"
            "print('loaded', len(DATASETS), 'datasets')"
        ),
        md("## LP AUC (%)"),
        code(
            "import pandas as pd\n"
            "\n"
            "def _lp_auc(rec):\n"
            "    return rec.get('auc', rec.get('lp_auc', float('nan')))\n"
            "\n"
            "rows = {\n"
            "    'GW-CCA (p1p2)': [GCCA[ds]['auc'] * 100 for ds in DATASETS],\n"
            "    'SGC': [_lp_auc(LP[ds].get('sgc', {})) * 100 if LP[ds].get('sgc') else None for ds in DATASETS],\n"
            "    'APPNP-lin': [_lp_auc(LP[ds].get('appnp', {})) * 100 if LP[ds].get('appnp') else None for ds in DATASETS],\n"
            "    'Linear ChebNet': [_lp_auc(LP[ds].get('chebnet', {})) * 100 if LP[ds].get('chebnet') else None for ds in DATASETS],\n"
            "}\n"
            "auc = pd.DataFrame(rows, index=DATASETS)\n"
            "auc.index.name = 'dataset'\n"
            "auc.round(2)"
        ),
        md("## LP AP (%)"),
        code(
            "def _lp_ap(rec):\n"
            "    return rec.get('ap', rec.get('lp_ap', float('nan')))\n"
            "\n"
            "rows = {\n"
            "    'GW-CCA (p1p2)': [GCCA[ds]['ap'] * 100 for ds in DATASETS],\n"
            "    'SGC': [_lp_ap(LP[ds].get('sgc', {})) * 100 if LP[ds].get('sgc') else None for ds in DATASETS],\n"
            "    'APPNP-lin': [_lp_ap(LP[ds].get('appnp', {})) * 100 if LP[ds].get('appnp') else None for ds in DATASETS],\n"
            "    'Linear ChebNet': [_lp_ap(LP[ds].get('chebnet', {})) * 100 if LP[ds].get('chebnet') else None for ds in DATASETS],\n"
            "}\n"
            "ap = pd.DataFrame(rows, index=DATASETS)\n"
            "ap.index.name = 'dataset'\n"
            "ap.round(2)"
        ),
        md("## LP AUC gap: GW-CCA − best linear baseline (points)\n\nPositive = GW-CCA is ahead."),
        code(
            "best_baseline = auc[['SGC', 'APPNP-lin', 'Linear ChebNet']].max(axis=1)\n"
            "gap = (auc['GW-CCA (p1p2)'] - best_baseline).rename('AUC gap (pt)')\n"
            "gap.round(2).to_frame()"
        ),
        md(
            "## Node classification (%)\n"
            "\n"
            "Metric per dataset: accuracy for multi-class, Micro-F1 for multi-label "
            "(yelp, ppi). GW-CCA numbers pulled from the paper's `_res_GCCA__*.json` "
            "files (main-paper GW-CCA run)."
        ),
        code(
            "def _nc(rec):\n"
            "    return rec.get('nc_test', float('nan'))\n"
            "\n"
            "rows = {\n"
            "    'metric': [GCCA[ds].get('metric', 'accuracy') for ds in DATASETS],\n"
            "    'GW-CCA (p1p2)': [GCCA[ds]['test_acc'] * 100 for ds in DATASETS],\n"
            "    'SGC': [_nc(NC[ds].get('sgc', {})) * 100 if NC[ds].get('sgc') else None for ds in DATASETS],\n"
            "    'APPNP-lin': [_nc(NC[ds].get('appnp', {})) * 100 if NC[ds].get('appnp') else None for ds in DATASETS],\n"
            "    'Linear ChebNet': [_nc(NC[ds].get('chebnet', {})) * 100 if NC[ds].get('chebnet') else None for ds in DATASETS],\n"
            "}\n"
            "nc = pd.DataFrame(rows, index=DATASETS)\n"
            "nc.index.name = 'dataset'\n"
            "nc.round(2)"
        ),
        md("## NC gap: GW-CCA − best linear baseline (points)"),
        code(
            "num_cols = ['SGC', 'APPNP-lin', 'Linear ChebNet']\n"
            "best_nc = nc[num_cols].max(axis=1)\n"
            "nc_gap = (nc['GW-CCA (p1p2)'] - best_nc).rename('NC gap (pt)')\n"
            "pd.concat([nc[['metric']], nc_gap.to_frame()], axis=1).round(2)"
        ),
        md(
            "## Takeaways for the rebuttal\n"
            "\n"
            "1. **GW-CCA beats all three linear/diffusion/spectral baselines on 4/5 "
            "datasets on LP** (gap +3 to +9 AUC pt) and on 4/5 on NC (gap up to +26 pt "
            "on yelp Micro-F1).\n"
            "2. **ogbn-products LP is a saturation regime.** Every propagator clears "
            "99.5% AUC; GW-CCA is within 0.06 pt of the best. NC still favours GW-CCA.\n"
            "3. **The three buckets don't collapse to one story.** For LP: SGC best on "
            "3/5, ChebNet on 1/5 (yelp), APPNP-lin never wins. For NC: ChebNet best on "
            "reddit / yelp / ppi, SGC on ogbn-arxiv. GW-CCA works uniformly — the "
            "CCA-style joint objective, not the propagation operator, is doing the work.\n"
            "4. **Compare with the M-ablation.** Sweeping $M$ inside GW-CCA moves "
            "accuracy by $\\pm 0.10$ AUC pt on 4/5 datasets; replacing the *whole* "
            "method with a shallow linear/diffusion/spectral propagator drops accuracy "
            "by $3$–$9$ pt for LP and up to $26$ pt for NC. The pair-weight matrix is "
            "a robust design knob; the CCA objective is load-bearing."
        ),
    ]
    path = HERE / "BASELINES.ipynb"
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {path}")
    subprocess.run(
        ["../../../.venv/bin/jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(path)],
        check=True,
        cwd=str(HERE),
    )


def build_connectivity() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# Connectivity ablation (rebuttal to Con 4)\n"
            "\n"
            "Reviewer asks: clarify the relative contributions of attributes and graph "
            "structure by evaluating under progressively restricted test-time graph "
            "information. We fix the CCA projections $(W, H)$ learned from the training "
            "subgraph (paper defaults, $M = (P + P^2)/2$) and vary **only** the "
            "adjacency $A'$ used at *test-time* inference. Train/val rows are always "
            "embedded on the full graph so the downstream LR probe sees a consistent "
            "training signal; only test rows differ across conditions.\n"
            "\n"
            "| condition | edges kept at test-time | isolates |\n"
            "|---|---|---|\n"
            "| `features_only` (A) | none — $P_\\text{hat} = I$, propagation is a no-op | attribute contribution |\n"
            "| `test_test` (B) | edges with both endpoints in the test set | test-side structure with no train supervision leakage |\n"
            "| `full` (C) | all observable edges (paper's inductive protocol) | joint attribute + full graph |\n"
            "\n"
            "Runner: `run_connectivity_ablation.py`."
        ),
        md(
            "## How to reproduce\n"
            "\n"
            "```bash\n"
            "cd experiments/m_ablation\n"
            "python run_connectivity_ablation.py --dataset ogbn-arxiv     --out ogbn-arxiv_conn.json\n"
            "python run_connectivity_ablation.py --dataset reddit          --out reddit_conn.json\n"
            "python run_connectivity_ablation.py --dataset yelp            --out yelp_conn.json\n"
            "python run_connectivity_ablation.py --dataset ogbn-products   --out ogbn-products_conn.json\n"
            "python run_connectivity_ablation.py --dataset ppi --lam 0.01  --out ppi_conn.json\n"
            "```\n"
            "\n"
            "PPI needs `--lam 0.01` (default $\\lambda=0$ produces a non-PD covariance)."
        ),
        code(
            "import json\n"
            "\n"
            "DATASETS = ['ogbn-arxiv', 'reddit', 'yelp', 'ogbn-products', 'ppi']\n"
            "CONDITIONS = ['features_only', 'test_test', 'full']\n"
            "\n"
            "def _load(ds):\n"
            "    with open(f'{ds}_conn.json') as f:\n"
            "        return {r['condition']: r for r in json.load(f)}\n"
            "\n"
            "RESULTS = {ds: _load(ds) for ds in DATASETS}\n"
            "print('loaded', len(RESULTS), 'datasets ×', len(CONDITIONS), 'conditions')"
        ),
        md("## Node classification (%)"),
        code(
            "import pandas as pd\n"
            "\n"
            "def _row(ds):\n"
            "    r = RESULTS[ds]\n"
            "    metric = r['full'].get('nc_metric', 'accuracy')\n"
            "    return [metric] + [r[c]['nc_test'] * 100 for c in CONDITIONS]\n"
            "\n"
            "nc = pd.DataFrame(\n"
            "    [_row(ds) for ds in DATASETS],\n"
            "    index=DATASETS,\n"
            "    columns=['metric'] + CONDITIONS,\n"
            ")\n"
            "nc.index.name = 'dataset'\n"
            "nc[CONDITIONS] = nc[CONDITIONS].astype(float).round(2)\n"
            "nc"
        ),
        md(
            "## Gap columns\n"
            "\n"
            "$C - A$ is the total contribution of graph structure; $C - B$ is the "
            "marginal contribution of train↔test edges beyond test-side structure."
        ),
        code(
            "gaps = pd.DataFrame({\n"
            "    'metric': nc['metric'],\n"
            "    'C - A (total structure)': (nc['full'] - nc['features_only']).round(2),\n"
            "    'C - B (train-test bridge)': (nc['full'] - nc['test_test']).round(2),\n"
            "}, index=DATASETS)\n"
            "gaps.index.name = 'dataset'\n"
            "gaps"
        ),
        md("## Edges kept at test-time"),
        code(
            "edges = pd.DataFrame({\n"
            "    'full nnz':      [RESULTS[ds]['full']['nnz_masked']       for ds in DATASETS],\n"
            "    'test_test nnz': [RESULTS[ds]['test_test']['nnz_masked']  for ds in DATASETS],\n"
            "}, index=DATASETS)\n"
            "edges['test_test share (%)'] = (100 * edges['test_test nnz'] / edges['full nnz']).round(2)\n"
            "edges.index.name = 'dataset'\n"
            "edges"
        ),
        md(
            "## Takeaways for the rebuttal\n"
            "\n"
            "1. **Attributes carry real signal.** `features_only` reaches $27$–$61$% on "
            "every dataset — GW-CCA is not solving the task by memorising graph "
            "structure alone.\n"
            "2. **Test-side structure alone is already substantial.** The "
            "`test_test − features_only` gap is $+0.6$ to $+43.4$ pt across datasets. "
            "Even without any train→test edges, the CCA projections learned on the "
            "training graph produce embeddings that make the test-side subgraph "
            "informative on its own.\n"
            "3. **Bridge edges close the remaining gap on link-sparse graphs.** "
            "`full − test_test` is largest on ogbn-arxiv ($+14.9$), where test-time "
            "edges are only 5% of the graph; modest on reddit / products "
            "($+2.9 / +3.8$); near-zero on yelp / ppi.\n"
            "4. **Attribute vs structure balance is dataset-dependent.** Yelp is nearly "
            "attribute-only ($C-A = +0.6$); reddit and products are structure-heavy "
            "($+43 / +34$); ogbn-arxiv and ppi are balanced. GW-CCA's joint CCA "
            "objective adapts to whichever side is informative.\n"
            "5. **PPI: `test_test == full` by construction.** PPI's inductive "
            "protocol partitions nodes into 24 disjoint biological-network graphs "
            "with zero cross-split edges. Every observable test-time edge is already "
            "a test-test edge, so B and C give byte-identical test-row embeddings — "
            "a useful sanity check on the masking implementation."
        ),
    ]
    path = HERE / "CONNECTIVITY.ipynb"
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {path}")
    subprocess.run(
        ["../../../.venv/bin/jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(path)],
        check=True,
        cwd=str(HERE),
    )


if __name__ == "__main__":
    build_baselines()
    build_connectivity()
