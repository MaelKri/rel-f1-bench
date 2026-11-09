"""TabPFN on the same features as LightGBM: train table as in-context set, no tuning, val never used."""
import tabpfn
from relbench.base import TaskType
from tabpfn import TabPFNClassifier, TabPFNRegressor

from rel_f1_bench.evaluation import start, stop
from rel_f1_bench.features import build_split_features


def run(cfg, task, db, tables):
    """Yield one result per seed (predictions on val and test, cost, config)."""
    X = build_split_features(db, task, tables, **cfg["features"])
    y_train = tables["train"].df[task.target_col].to_numpy()
    binary = task.task_type == TaskType.BINARY_CLASSIFICATION
    Model = TabPFNClassifier if binary else TabPFNRegressor

    for seed in cfg["seeds"]:
        model = Model(device="cuda", random_state=seed, **cfg["tabpfn"])
        t0 = start()
        model.fit(X["train"], y_train)
        train_time, train_mem = stop(t0)

        t0 = start()
        if binary:
            val_pred = model.predict_proba(X["val"])[:, 1]
            test_pred = model.predict_proba(X["test"])[:, 1]
        else:  # the median of the predictive distribution minimises the MAE
            val_pred = model.predict(X["val"], output_type="median")
            test_pred = model.predict(X["test"], output_type="median")
        infer_time, infer_mem = stop(t0)

        yield dict(seed=seed, val_pred=val_pred, test_pred=test_pred,
                   train_time=train_time, infer_time=infer_time,
                   peak_mem_mb=max(train_mem, infer_mem),
                   config={**cfg["tabpfn"], **cfg["features"]},
                   extra=dict(tabpfn_version=tabpfn.__version__, n_context=len(y_train),
                              n_features=X["train"].shape[1]))
