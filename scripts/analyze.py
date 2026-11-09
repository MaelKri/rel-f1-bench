"""Turn results/runs.jsonl and the saved test predictions into the README tables and figures.

Usage: uv run python scripts/analyze.py [--ref lgbm_fe] [--n_boot 1000]

Prints every table and writes the same text to results/summary.txt.
Figures go to results/figures/. Needs results/raw/ (written by scripts/run.py).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rel_f1_bench.data import TASKS, load  # noqa: E402  (first: sets the RelBench cache dir)
from rel_f1_bench.evaluation import RESULTS, load_runs, test_target  # noqa: E402
from rel_f1_bench.features import build_features  # noqa: E402
from rel_f1_bench.stats import cold_mask, paired_bootstrap, seed_mean  # noqa: E402

ORDER = ["lgbm_fe", "lgbm_fe_staleness", "tabpfn_fe", "gnn_ref", "gnn_e20", "gnn_e20_last"]
MAIN = {"lgbm_fe": "LightGBM + FE", "tabpfn_fe": "TabPFN + FE", "gnn_ref": "GNN (RelBench ref.)"}

parser = argparse.ArgumentParser()
parser.add_argument("--ref", default="lgbm_fe")
parser.add_argument("--n_boot", type=int, default=1000)
args = parser.parse_args()
pd.set_option("display.width", 250)

runs = load_runs()
methods = [m for m in ORDER if m in set(runs.method)] + sorted(set(runs.method) - set(ORDER))
tasks = [t for t in TASKS if t in set(runs.task)]
runs["method"] = pd.Categorical(runs.method, methods, ordered=True)
runs["task"] = pd.Categorical(runs.task, tasks, ordered=True)
runs["tune_time_s"] = runs.extra.str["tune_time_s"].fillna(0.0)
runs["n_params"] = runs.extra.str["n_params"]
out = []


def show(title, df, note=None):
    out.extend([title, df.to_string(index=False)] + ([note] if note else []) + [""])


def pm(x):
    return f"{x.mean():.4f} ± {x.std():.4f}"


# ---------------------------------------------------------------- 1. scores and cost
g = runs.groupby(["task", "method"], observed=True)
show("Scores (mean ± std over seeds; val = official val split, test = official leaderboard eval)",
     g.agg(metric=("metric", "first"), seeds=("seed", "nunique"),
           val=("val", pm), test=("test", pm)).reset_index())
cost = g.agg(train_s=("train_time_s", "mean"), tuning_s=("tune_time_s", "first"),
             infer_s=("infer_time_s", "mean"), peak_gpu_mb=("peak_gpu_mem_mb", "max"),
             params=("n_params", "first")).round(3).reset_index()
cost["peak_gpu_mb"] = cost.peak_gpu_mb.map(lambda v: "CPU" if pd.isna(v) else f"{v:.0f}")
cost["params"] = cost.params.map(lambda v: "-" if pd.isna(v) else f"{v / 1e6:.1f}M")
show("Cost (train/infer: mean per seed; infer = val + test; tuning: once per task)", cost)

# ---------------------------------------------------------------- 2. per task: test set, warm/cold, bootstrap
composition, staleness, subsets_scores, boots = [], [], [], []
cold_pos_drivers = {}  # binary tasks: drivers with a positive among cold rows
for task_name in tasks:
    task, db, tables = load(task_name)
    metric = task.metrics[0]
    higher = metric.__name__ == "roc_auc"
    y = test_target(task, task_name, tables["test"])
    ent = tables["test"].df[task.entity_col].to_numpy()
    cold = cold_mask(db, task, tables["test"])
    task_methods = [m for m in methods if ((runs.task == task_name) & (runs.method == m)).any()]
    preds = {}
    for m in task_methods:
        seeds = sorted(runs[(runs.task == task_name) & (runs.method == m)].seed)
        preds[m] = np.stack([np.load(RESULTS / "raw" / m / task_name / f"seed{s}_test.npy")
                             for s in seeds])

    every = np.arange(len(y))
    subsets = {"all": every, "warm": every[~cold], "cold": every[cold]}
    pos = y == 1
    if higher:
        cold_pos_drivers[task_name] = len(np.unique(ent[pos & cold]))
    composition.append({
        "task": task_name, "rows": len(y), "drivers": len(np.unique(ent)),
        "cold drivers": len(np.unique(ent[cold])), "cold rows": f"{cold.mean():.0%}",
        "positives": int(pos.sum()) if higher else "-",
        "drivers with a positive (warm/cold)":
            f"{len(np.unique(ent[pos]))} ({len(np.unique(ent[pos & ~cold]))}/"
            f"{len(np.unique(ent[pos & cold]))})" if higher else "-",
    })
    # how old is the latest known race at prediction time? (feature not used by the main models)
    staleness.append({"task": task_name, **{
        split: build_features(db, t, task.time_col, task.entity_col, staleness=True)
        ["days_since_last_race"].median() for split, t in tables.items()}})
    for m in task_methods:
        subsets_scores.append({"task": task_name, "method": m, **{
            name: round(seed_mean(metric, y, preds[m], idx), 4) for name, idx in subsets.items()}})
    for name in ["all", "warm"]:
        table, n_ok = paired_bootstrap(metric, y, preds, ent, subsets[name], args.ref,
                                       n_boot=args.n_boot)
        boots.append(table.assign(task=task_name, rows=name, resamples=n_ok))

show("Test set composition (warm = driver raced before the test cutoff, cold = debut after it)",
     pd.DataFrame(composition))
show("Median days since the driver's last known race, per split", pd.DataFrame(staleness))
show("Test score by subset (mean over seeds)", pd.DataFrame(subsets_scores),
     "Cold AUROC on driver-top3 rests on very few positive drivers: not interpretable.")
boot = pd.concat(boots)
show(f"Paired driver-block bootstrap on test: method - {args.ref} on the same resampled drivers",
     boot[["task", "rows", "method"] + [c for c in boot.columns
                                        if c not in ("task", "rows", "method")]],
     "driver-position: lower is better, so a positive diff means worse than the reference.")

text = "\n".join(out)
(RESULTS / "summary.txt").write_text(text)
print(text)

# ---------------------------------------------------------------- 3. figures
fig_dir = RESULTS / "figures"
fig_dir.mkdir(exist_ok=True)
main = [m for m in MAIN if m in methods]
colors = dict(zip(MAIN, ["tab:blue", "tab:orange", "tab:green"]))
arrow = {"roc_auc": "AUROC ↑", "nmae": "NMAE ↓"}

fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4))
for ax, task_name in zip(np.atleast_1d(axes), tasks):
    for m in main:
        r = runs[(runs.task == task_name) & (runs.method == m)]
        ax.errorbar(r.infer_time_s.mean(), r.test.mean(), yerr=r.test.std(), fmt="o",
                    capsize=4, color=colors[m], label=MAIN[m])
    ax.set_xscale("log")
    ax.set_xlabel("inference time, val + test (s, log scale)")
    ax.set_ylabel(f"test {arrow.get(r.metric.iloc[0], r.metric.iloc[0])}")
    ax.set_title(task_name)
    ax.grid(alpha=0.3)
np.atleast_1d(axes)[0].legend()
fig.tight_layout()
fig.savefig(fig_dir / "cost_vs_performance.png", dpi=150)

sub = pd.DataFrame(subsets_scores)
fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4))
for ax, task_name in zip(np.atleast_1d(axes), tasks):
    s = sub[sub.task == task_name].set_index("method")
    x = np.arange(2)
    for i, m in enumerate(main):
        ax.bar(x + (i - 1) * 0.27, s.loc[m, ["warm", "cold"]], 0.27, color=colors[m], label=MAIN[m])
    metric_name = runs[runs.task == task_name].metric.iloc[0]
    ax.set_xticks(x, ["warm", "cold"])
    ax.set_ylabel(f"test {arrow.get(metric_name, metric_name)}")
    if metric_name == "roc_auc":
        ax.set_ylim(0, 1)
        ax.axhline(0.5, ls="--", lw=1, color="grey")
        if cold_pos_drivers.get(task_name, 99) < 5:
            ax.text(1, 0.97, f"not interpretable:\nonly {cold_pos_drivers[task_name]} positive drivers",
                    ha="center", va="top", fontsize=8)
    ax.set_title(task_name)
np.atleast_1d(axes)[0].legend()
fig.tight_layout()
fig.savefig(fig_dir / "warm_vs_cold.png", dpi=150)
print(f"figures written to {fig_dir}")
