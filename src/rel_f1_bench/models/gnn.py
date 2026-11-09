"""Reference RelBench GNN: heterogeneous GraphSAGE on the pkey-fkey graph, PyTorch Frame encoders.

Graph, model and training loop follow the RelBench example (relbench@07ec0ce);
only the evaluation goes through the shared protocol.
"""
import copy
import json

import numpy as np
import torch
from relbench.base import TaskType
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything

from rel_f1_bench.data import DATASET, ROOT
from rel_f1_bench.evaluation import start, stop
from rel_f1_bench.models.relbench_nn import GloveTextEmbedding, Model

CACHE = ROOT / "data" / "relbench_examples" / DATASET


def get_stypes(db):
    """Column types proposed by RelBench, cached so that the materialized graph stays the same."""
    path = CACHE / "stypes.json"
    if path.exists():
        raw = json.load(open(path))
        return {t: {c: stype(s) for c, s in cols.items()} for t, cols in raw.items()}
    col_to_stype = get_stype_proposal(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(col_to_stype, open(path, "w"), indent=2, default=str)
    return col_to_stype


def build_graph(db, task):
    """Heterogeneous graph of the db; primary and foreign keys are removed from the node features."""
    return make_pkey_fkey_graph(
        db,
        col_to_stype_dict=get_stypes(db),
        text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device="cpu"),
                                             batch_size=256),
        cache_dir=str(CACHE / "materialized"),
        remove_columns=task.hidden_columns(),
    )


def run(cfg, task, db, tables):
    """Yield one result per seed (predictions on val and test, cost, config)."""
    hp = cfg["gnn"]
    device = torch.device("cuda")
    torch.set_num_threads(1)
    data, col_stats_dict = build_graph(db, task)
    binary = task.task_type == TaskType.BINARY_CLASSIFICATION
    loss_fn = BCEWithLogitsLoss() if binary else L1Loss()
    clamp = None if binary else np.percentile(tables["train"].df[task.target_col], [2, 98])
    y_val = tables["val"].df[task.target_col].to_numpy()
    metric = task.metrics[0]

    def make_loader(split):
        ti = get_node_train_table_input(table=tables[split], task=task)
        return NeighborLoader(
            data,
            num_neighbors=[int(hp["num_neighbors"] / 2**i) for i in range(hp["num_layers"])],
            time_attr="time",
            input_nodes=ti.nodes,
            input_time=ti.time,
            transform=ti.transform,
            batch_size=hp["batch_size"],
            temporal_strategy=hp["temporal_strategy"],
            shuffle=split == "train",
        )

    @torch.no_grad()
    def predict(model, loader, seed):
        model.eval()
        torch.manual_seed(seed)  # neighbour sampling is random at inference too
        out = []
        for batch in loader:
            pred = model(batch.to(device), task.entity_table).view(-1)
            out.append((torch.sigmoid(pred) if binary else pred.clamp(*clamp)).cpu())
        return torch.cat(out).numpy()

    for seed in cfg["seeds"]:
        # the order of the calls below fixes the random streams: keep it
        seed_everything(seed)
        loaders = {s: make_loader(s) for s in ["train", "val", "test"]}
        model = Model(data=data, col_stats_dict=col_stats_dict, num_layers=hp["num_layers"],
                      channels=hp["channels"], out_channels=1, aggr=hp["aggr"], norm="batch_norm",
                      gnn=hp["gnn"], gat_heads=hp["gat_heads"],
                      gat_dropout=hp["gat_dropout"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

        best = best_state = best_epoch = None
        t0 = start()
        for epoch in range(1, hp["epochs"] + 1):
            model.train()
            for step, batch in enumerate(loaders["train"]):
                if step >= hp["max_steps_per_epoch"]:
                    break
                batch = batch.to(device)
                opt.zero_grad()
                pred = model(batch, task.entity_table).view(-1)
                loss = loss_fn(pred.float(), batch[task.entity_table].y.float())
                loss.backward()
                opt.step()
            val = metric(y_val, predict(model, loaders["val"], seed))
            if best is None or (val >= best if binary else val <= best):  # best epoch on val
                best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
        train_time, train_mem = stop(t0)

        model.load_state_dict(best_state)
        t0 = start()
        val_pred = predict(model, loaders["val"], seed)
        test_pred = predict(model, loaders["test"], seed)
        infer_time, infer_mem = stop(t0)
        # how much does inference-time neighbour sampling move the predictions?
        jitter = float(np.abs(predict(model, loaders["val"], seed + 1000) - val_pred).max())

        yield dict(seed=seed, val_pred=val_pred, test_pred=test_pred,
                   train_time=train_time, infer_time=infer_time,
                   peak_mem_mb=max(train_mem, infer_mem), config=hp,
                   extra=dict(best_epoch=best_epoch, val_eval_jitter=jitter,
                              n_params=sum(p.numel() for p in model.parameters())))
