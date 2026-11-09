"""LightGBM on the hand-crafted features: Optuna on val (seed 0), then one refit per seed."""
import lightgbm as lgb
import optuna
from relbench.base import TaskType

from rel_f1_bench.evaluation import start, stop
from rel_f1_bench.features import build_split_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


def search_space(trial):
    # the order of the suggest calls matters for the TPE sampler: keep it fixed
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def run(cfg, task, db, tables):
    """Yield one result per seed (predictions on val and test, cost, config)."""
    X = build_split_features(db, task, tables, **cfg["features"])
    y = {s: tables[s].df[task.target_col].to_numpy() for s in ["train", "val"]}
    binary = task.task_type == TaskType.BINARY_CLASSIFICATION
    metric = task.metrics[0]
    base = dict(cfg["lgbm"], objective="binary" if binary else "l1",
                metric="auc" if binary else "l1", verbose=-1)
    Model = lgb.LGBMClassifier if binary else lgb.LGBMRegressor

    def fit(params, seed):
        model = Model(**params, random_state=seed)
        model.fit(X["train"], y["train"], eval_X=X["val"], eval_y=y["val"],
                  callbacks=[lgb.early_stopping(cfg["early_stopping"], verbose=False)])
        return model

    def predict(model, split):
        return model.predict_proba(X[split])[:, 1] if binary else model.predict(X[split])

    def objective(trial):
        return metric(y["val"], predict(fit(dict(base, **search_space(trial)), 0), "val"))

    study = optuna.create_study(direction="maximize" if binary else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    t0 = start()
    study.optimize(objective, n_trials=cfg["n_trials"])
    tune_time, _ = stop(t0)
    best = dict(base, **study.best_params)

    for seed in cfg["seeds"]:
        t0 = start()
        model = fit(best, seed)
        train_time, _ = stop(t0)
        t0 = start()
        val_pred, test_pred = predict(model, "val"), predict(model, "test")
        infer_time, _ = stop(t0)
        yield dict(seed=seed, val_pred=val_pred, test_pred=test_pred,
                   train_time=train_time, infer_time=infer_time,
                   config={**best, **cfg["features"]},
                   extra=dict(tune_time_s=round(tune_time, 1), n_trials=cfg["n_trials"],
                              best_iter=model.best_iteration_, n_features=X["train"].shape[1]))
