"""Driver-level bootstraps and warm/cold split of the test set."""
import numpy as np
import pandas as pd

from rel_f1_bench.features import build_features


def block_bootstrap(metric, y, pred, entities, n_boot=1000, seed=0):
    """95% CI of the metric, resampling drivers: rows of one driver are strongly correlated."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(entities)
    idx_by_ent = {e: np.flatnonzero(entities == e) for e in uniq}
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_ent[e] for e in rng.choice(uniq, size=len(uniq))])
        if metric.__name__ == "roc_auc" and len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], pred[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def cold_mask(db, task, table):
    """True for rows of drivers with no race in the db (debut after the cutoff)."""
    return build_features(db, table, task.time_col, task.entity_col)["n_races"].isna().to_numpy()


def seed_mean(metric, y, preds, idx):
    """Metric on rows idx, averaged over seeds (preds has shape n_seeds x n_rows)."""
    return np.mean([metric(y[idx], p[idx]) for p in preds])


def paired_bootstrap(metric, y, preds, entities, idx_pool, ref, n_boot=1000, seed=0):
    """CI of (method - ref) for every method in `preds`, all scored on the same resampled drivers.

    Returns the table and the number of valid resamples.
    """
    higher = metric.__name__ == "roc_auc"
    rng = np.random.default_rng(seed)
    rows_of = {e: idx_pool[entities[idx_pool] == e] for e in np.unique(entities[idx_pool])}
    drivers = list(rows_of)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[e] for e in rng.choice(drivers, size=len(drivers))])
        if higher and len(np.unique(y[idx])) < 2:
            continue
        boots.append({m: seed_mean(metric, y, p, idx) for m, p in preds.items()})
    B = pd.DataFrame(boots)
    rows = []
    for m in preds:
        if m == ref:
            continue
        d = B[m] - B[ref]
        better = (d > 0).mean() if higher else (d < 0).mean()
        rows.append({"method": m, f"diff vs {ref}": round(d.mean(), 4),
                     "95% CI": f"[{d.quantile(.025):.4f}, {d.quantile(.975):.4f}]",
                     "P(better than ref)": round(better, 3)})
    return pd.DataFrame(rows), len(B)
