"""Shared evaluation for every method: val metric, official test metric, CIs, cost, results/runs.jsonl.

This is the only module that reads test labels.
"""
import json
import subprocess
import time
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from rel_f1_bench.data import DATASET, ROOT  # first: sets the RelBench cache dir
from relbench.load import RelBenchDataset
from relbench.submit import _fetch_eval_files, evaluate_task, write_prediction_table

from rel_f1_bench.stats import block_bootstrap

RESULTS = ROOT / "results"


def start():
    """Reset the GPU peak-memory counter and return a start time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def stop(t0):
    """Return (elapsed seconds, peak GPU memory in MB or None)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**20
    return time.perf_counter() - t0, None


def git_commit():
    """Short commit hash, with -dirty if the code (not results/) has uncommitted changes."""
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
                           capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return sha + ("-dirty" if dirty else "")


@lru_cache
def _test_table(task_name):
    # private relbench helper, same path as relbench.submit.evaluate_task (pinned to 3.1.x)
    local = _fetch_eval_files(DATASET, [task_name])
    return RelBenchDataset(local).load_task(task_name).get_table("test", mask_input_cols=False)


def test_target(task, task_name, masked_table):
    """Test targets aligned row by row with the masked test table."""
    gt = _test_table(task_name).df
    keys = [task.time_col, task.entity_col]
    merged = masked_table.df[keys].merge(gt[keys + [task.target_col]], on=keys, how="left")
    return merged[task.target_col].to_numpy()


def log_run(method, task_name, task, tables, seed, val_pred, test_pred,
            train_time, infer_time, peak_mem_mb=None, config=None, extra=None):
    """Score one run, save its predictions and append it to results/runs.jsonl."""
    metric = task.metrics[0]
    name = metric.__name__
    run_dir = RESULTS / "raw" / method / task_name
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / f"seed{seed}_val.npy", val_pred)
    np.save(run_dir / f"seed{seed}_test.npy", test_pred)

    val = task.evaluate(val_pred, tables["val"])[name]
    csv = write_prediction_table(task, test_pred, run_dir / f"seed{seed}_test.csv")
    test = evaluate_task(f"{DATASET}/{task_name}", csv)[name]

    val_df, test_df = tables["val"].df, tables["test"].df
    val_ci = block_bootstrap(metric, val_df[task.target_col].to_numpy(), val_pred,
                             val_df[task.entity_col].to_numpy())
    test_ci = block_bootstrap(metric, test_target(task, task_name, tables["test"]), test_pred,
                              test_df[task.entity_col].to_numpy())

    record = dict(
        method=method, task=task_name, seed=seed, metric=name,
        val=float(val), test=float(test), val_ci=val_ci, test_ci=test_ci,
        train_time_s=round(train_time, 2), infer_time_s=round(infer_time, 3),
        peak_gpu_mem_mb=None if peak_mem_mb is None else round(peak_mem_mb, 1),
        config=config or {}, extra=extra or {},
        commit=git_commit(), timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    with open(RESULTS / "runs.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[{method} | {task_name} | seed {seed}] val {name}={val:.4f} | "
          f"test {name}={test:.4f} | train {train_time:.1f}s")
    return record


def load_runs():
    """All runs, keeping the last one for each (method, task, seed)."""
    runs = pd.read_json(RESULTS / "runs.jsonl", lines=True)
    return runs.drop_duplicates(["method", "task", "seed"], keep="last").reset_index(drop=True)
