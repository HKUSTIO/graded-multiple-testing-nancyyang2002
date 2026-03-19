[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/i-dPW--H)
# Multiple Testing with FWER and FDR Exercise

This assignment asks you to implement multiple-testing analysis for family-wise and false-discovery error control in one reproducible pipeline.

## Learning goals

You will implement Bonferroni, Holm, Benjamini-Hochberg, and Benjamini-Yekutieli corrections, compute FWER and FDR, and compare error control against power in repeated simulations.

## Project structure

```text
multiple_testing_fwer_fdr/
|-- .github/workflows/classroom.yml
|-- config/assignment.json
|-- input/
|-- cleaned/
|   |-- null_pvalues.csv
|   `-- mixed_pvalues.csv
|-- output/
|   `-- results.json
|-- report/solution.qmd
|-- scripts/
|   |-- run_cleaning.py
|   |-- run_analysis.py
|   |-- run_pipeline.py
|   `-- run_assignment.py
|-- src/
|   |-- __init__.py
|   `-- multiple_testing.py
|-- tests/
|   |-- test_core.py
|   |-- test_report.py
|   `-- test_workflow_modifiable.py
|-- pyproject.toml
`-- README.md
```

## Role of scripts

The `scripts` directory defines the full assignment pipeline. `scripts/run_cleaning.py` generates and saves simulation p-values using pre-implemented data-generation functions in `src/multiple_testing.py`. `scripts/run_analysis.py` loads cleaned p-values and calls your analysis implementation to produce summary metrics in `output/results.json`. `scripts/run_pipeline.py` runs cleaning first and analysis second. `scripts/run_assignment.py` is a thin entrypoint that calls `run_pipeline.py`.

## Your tasks

All parameters and hyperparameters are defined in `config/assignment.json`. Use this file as the single source of truth and do not hard-code constants in `src/` or `scripts/`.

Data generation is already provided in `src/multiple_testing.py` through:

1. `simulate_null_pvalues(config)`
2. `simulate_mixed_pvalues(config)`

You only implement the analysis functions listed below.

### Mapping from symbols to config keys

| Symbol or quantity | `config/assignment.json` key | Notes |
|---|---|---|
| $N$ | `N` | sample size per simulation |
| $M$ | `M` | number of hypotheses per simulation |
| $M_0$ | `M0` | number of true null hypotheses in mixed design |
| $L$ | `L` | number of simulations |
| $p_t$ | `p_treat` | treatment-assignment probability in DGP |
| $\tau$ | `tau_alternative` | treatment effect for false null hypotheses |
| $\alpha$ | `alpha` | significance level |

### Mathematical targets

Bonferroni rejection rule:
$$
p_m \le \frac{\alpha}{M}.
$$

Holm step-down rule on sorted p-values $p^{(1)} \le \cdots \le p^{(M)}$:
$$
p^{(k)} \le \frac{\alpha}{M-k+1}.
$$

Benjamini-Hochberg rule:
$$
p^{(k)} \le \frac{k}{M}\alpha.
$$

Benjamini-Yekutieli rule:
$$
p^{(k)} \le \frac{k}{M}\frac{\alpha}{\sum_{j=1}^{M} 1/j}.
$$

For one simulation:
$$
FDR = \frac{\#\{\text{rejected true nulls}\}}{\#\{\text{all rejections}\}},
$$
with the convention $FDR=0$ when there are no rejections.

$$
Power = \frac{\#\{\text{rejected false nulls}\}}{\#\{\text{all false nulls}\}}.
$$

Across complete-null simulations:
$$
FWER = \Pr(\text{at least one rejection}).
$$

### Required function signatures

Implement these functions in `src/multiple_testing.py` with the exact signatures:

```python
def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray: ...
def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray: ...
def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray: ...
def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray: ...
def compute_fwer(rejections_null: np.ndarray) -> float: ...
def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float: ...
def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float: ...
def summarize_multiple_testing(null_pvalues: pd.DataFrame, mixed_pvalues: pd.DataFrame, alpha: float) -> dict[str, float]: ...
```

### Function-level requirements

`bonferroni_rejections(p_values, alpha)`

1. Input is a one-dimensional array of p-values for one simulation.
2. Return a boolean array with the same shape and index order as input.
3. Element `m` is `True` if hypothesis `m` is rejected by Bonferroni.

`holm_rejections(p_values, alpha)`

1. Use Holm step-down over sorted p-values.
2. If rank `k` fails, all larger ranks must be non-rejections.
3. Return decisions in the original hypothesis order.

`benjamini_hochberg_rejections(p_values, alpha)`

1. Find the largest rank satisfying the BH threshold.
2. Reject all hypotheses up to that rank.
3. Return decisions in the original hypothesis order.

`benjamini_yekutieli_rejections(p_values, alpha)`

1. Use the harmonic correction term $\sum_{j=1}^{M}1/j$.
2. Apply the same largest-rank logic as BH with BY thresholds.
3. Return decisions in the original hypothesis order.

`compute_fwer(rejections_null)`

1. Input is a boolean matrix of shape `[L, M]` under the complete null.
2. Compute the fraction of rows with at least one `True`.
3. Return a scalar float.

`compute_fdr(rejections, is_true_null)`

1. Inputs are one-dimensional boolean arrays for one simulation.
2. Compute false discoveries among all discoveries.
3. If no discovery occurs, return `0.0`.

`compute_power(rejections, is_true_null)`

1. Inputs are one-dimensional boolean arrays for one simulation.
2. Compute true rejections among false null hypotheses.
3. Return a scalar float.

`summarize_multiple_testing(null_pvalues, mixed_pvalues, alpha)`

1. `null_pvalues` has columns `sim_id`, `hypothesis_id`, `p_value`.
2. `mixed_pvalues` has columns `sim_id`, `hypothesis_id`, `p_value`, `is_true_null`.
3. Use complete-null simulations to compute:
   - `fwer_uncorrected`
   - `fwer_bonferroni`
   - `fwer_holm`
4. Use mixed simulations to compute simulation-level FDR and power, then average across simulations for:
   - `fdr_uncorrected`, `fdr_bh`, `fdr_by`
   - `power_uncorrected`, `power_bh`, `power_by`
5. Return exactly this dictionary:

```python
{
  "fwer_uncorrected": ...,
  "fwer_bonferroni": ...,
  "fwer_holm": ...,
  "fdr_uncorrected": ...,
  "fdr_bh": ...,
  "fdr_by": ...,
  "power_uncorrected": ...,
  "power_bh": ...,
  "power_by": ...,
}
```

After implementation, run `scripts/run_pipeline.py` and fill `report/solution.qmd` with values from `output/results.json`.

## Workflow

Step 1 is to accept the assignment link on GitHub Classroom, which creates your private repository.

Step 2 is to clone your repository locally.

```bash
git clone https://github.com/HKUSTIO/<your-repo-name>.git
cd <your-repo-name>
```

Step 3 is to install dependencies.

```bash
uv sync
```

Step 4 is to implement all required analysis functions in `src/multiple_testing.py`.

Step 5 is to run the pipeline and render the report.

```bash
uv run python scripts/run_pipeline.py
uv run quarto render report/solution.qmd
```

Step 6 is to commit and push your submission.

```bash
git add -A
git commit -m "your message"
git push
```

Step 7 is to check scores. Each push to `main` triggers `Autograding Tests` on GitHub Actions.

You may push multiple times before the deadline. The latest score is recorded.

## Grading policy

Grading has two components.

1. Core implementation correctness (70 points).
   This checks rejection-rule logic, metric formulas, ordering and shape consistency, and required dictionary outputs in `src/multiple_testing.py`.

2. Report and output completeness (30 points).
   This checks pipeline outputs in `cleaned/` and `output/`, required keys in `output/results.json`, and successful rendering/content checks for `report/solution.html`.

Commit-message style and other immutable repository-history details are not grading targets.