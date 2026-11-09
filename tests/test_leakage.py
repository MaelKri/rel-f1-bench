"""Leakage test on the real data: no feature may read rows at or after the row's own timestamp."""
from functools import cache

import pytest

from rel_f1_bench.data import TASKS, load
from rel_f1_bench.features import leakage_test

load = cache(load)


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("task_name", TASKS)
def test_no_leakage(task_name, split):
    task, db, tables = load(task_name)
    # staleness=True so that every feature is checked
    leakage_test(db, tables[split], task.time_col, task.entity_col, windows=(5, 10), staleness=True)
