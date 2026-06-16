from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common.seed import set_random_seed
from dynamics_model import config as defaults
from dynamics_model.model import CrossAttnFusionModel
from dynamics_model.train_fn.ce_train_fn import ce_train_fn
from dynamics_model.train_fn.focal_train_fn import focal_train_fn


def _str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the dynamics response prediction model.")
    parser.add_argument("--seq-path", type=Path, default=Path(defaults.SEQ_DATASET_PATH))
    parser.add_argument("--track-path", type=Path, default=Path(defaults.TRACK_DATASET_PATH))
    parser.add_argument("--split-json", type=Path, default=Path(defaults.TEST_TRAIN_SPLIT_ANNOTATION_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path(defaults.RESULTS_DIR))
    parser.add_argument("--training-method", choices=["focal", "ce"], default="focal")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=defaults.BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=defaults.MAX_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--early-stop-patience", type=int, default=defaults.EARLY_STOP_PATIENCE)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--fusion-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=defaults.DROPOUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--use-class-weight",
        type=_str_to_bool,
        default=True,
        help="Use class weights in CE/focal loss. Set false for tiny demo data that does not contain every class.",
    )
    parser.add_argument("--use-wandb", type=_str_to_bool, default=False)
    parser.add_argument("--use-wandb-best", type=_str_to_bool, default=False)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="organoid-analyzer")
    parser.add_argument("--sweep-name", type=str, default="grid_val_v1")
    parser.add_argument("--metric-name", type=str, default="val/best_auc")
    return parser.parse_args(argv)


def _get_wandb_best(entity=None, project="organoid-analyzer", sweep_name="grid_val_v1", metric_name="val/best_auc"):
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    sweep_id = None
    for sweep in api.project(name=project, entity=entity).sweeps():
        if getattr(sweep, "name", None) == sweep_name:
            sweep_id = getattr(sweep, "id", None) or getattr(sweep, "sweep_id", None)
            break
    runs = api.sweep(f"{path}/{sweep_id}").runs if sweep_id else api.runs(path)
    best_run = None
    best_metric = -1e18
    for run in runs:
        value = (run.summary or {}).get(metric_name)
        if value is not None and value > best_metric:
            best_metric = value
            best_run = run
    if best_run is None:
        raise RuntimeError(f"No W&B run found with metric {metric_name!r}.")
    cfg = best_run.config or {}
    return {
        "learning_rate": cfg.get("learning_rate", 1e-3),
        "batch_size": cfg.get("batch_size", defaults.BATCH_SIZE),
        "dropout": cfg.get("dropout", defaults.DROPOUT),
        "focal_gamma": cfg.get("focal_gamma", 2.0),
        "training_method": cfg.get("training_method", "focal"),
        "hidden_size": cfg.get("hidden_size", 16),
        "fusion_size": cfg.get("fusion_size", 32),
        "run_id": best_run.id,
        "run_name": best_run.name,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    set_random_seed(args.seed)

    if args.use_wandb_best:
        best = _get_wandb_best(
            entity=args.wandb_entity,
            project=args.wandb_project,
            sweep_name=args.sweep_name,
            metric_name=args.metric_name,
        )
        args.learning_rate = best["learning_rate"]
        args.batch_size = best["batch_size"]
        args.dropout = best["dropout"]
        args.focal_gamma = best["focal_gamma"]
        args.training_method = best["training_method"]
        args.hidden_size = best["hidden_size"]
        args.fusion_size = best["fusion_size"]
        run_name = best["run_id"] or best["run_name"]
    else:
        run_name = "manual"

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    out_dir = args.output_dir / f"run_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = CrossAttnFusionModel(
        seq_input_size=defaults.FEATURE_LEN,
        track_input_size=defaults.TRACK_LEN,
        hidden_size=args.hidden_size,
        fusion_size=args.fusion_size,
        dropout=args.dropout,
    )

    train_kwargs = {
        "seq_path": args.seq_path,
        "track_path": args.track_path,
        "result_path": out_dir,
        "test_train_split_annotation_path": args.split_json,
        "model": model,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "weight_decay": args.weight_decay,
        "early_stop_patience": args.early_stop_patience,
        "use_class_weight": args.use_class_weight,
        "wandb_run": wandb_run,
    }
    if args.training_method == "focal":
        metrics = focal_train_fn(focal_gamma=args.focal_gamma, **train_kwargs)
    else:
        metrics = ce_train_fn(**train_kwargs)

    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "metrics_keys": sorted(metrics.keys())}, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()
    print(f"Saved dynamics run to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
