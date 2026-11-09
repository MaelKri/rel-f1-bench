"""Dataset and task loading. The RelBench cache lives in <repo>/data."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("RELBENCH_CACHE_DIR", str(ROOT / "data"))

import relbench  # noqa: E402  (after the cache dir is set)

DATASET = "rel-f1"
TASKS = ["driver-dnf", "driver-top3", "driver-position"]


def load(task_name):
    """Return (task, db, tables). db stops at the test cutoff and tables["test"] has its target masked."""
    dataset = relbench.load_dataset(DATASET)
    task = dataset.load_task(task_name)
    db = dataset.get_db()
    # get_table() is not cached in relbench v3: build each split once
    tables = {split: task.get_table(split) for split in ["train", "val", "test"]}
    return task, db, tables
