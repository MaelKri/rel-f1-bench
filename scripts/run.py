"""Run one experiment: every task and seed of a config, logged to results/runs.jsonl.

Usage: uv run python scripts/run.py configs/<exp>.yaml
       (TabPFN: uv run --env-file .env python scripts/run.py configs/tabpfn.yaml)
"""
import importlib
import sys

import yaml

from rel_f1_bench.data import load  # first: sets the RelBench cache dir
from rel_f1_bench.evaluation import log_run

cfg = yaml.safe_load(open(sys.argv[1]))
model = importlib.import_module(f"rel_f1_bench.models.{cfg['model']}")

for task_name in cfg["tasks"]:
    task, db, tables = load(task_name)
    for result in model.run(cfg, task, db, tables):
        log_run(cfg["method"], task_name, task, tables, **result)
