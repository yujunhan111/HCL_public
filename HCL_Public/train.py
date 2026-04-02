import os
import argparse
import glob
from typing import List, Dict
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))

from dataset.dataset import EHRDataset
from dataset.collate_func import ehr_collate_fn
from model.EHR_model import EHRModel
from model.fusion import build_fusion

from utils import (
    move_batch_to_device, compute_metrics, save_predictions,
    append_csv_row, init_logging, set_seed,
    load_code_embedding_matrix, get_code_vocab_size,
    build_mappings_from_patients, run_mlp_branch,
)
from baseline_runners import (
    extract_all_representations, run_slide_on_full_data,
    project_with_V, run_hnn_on_full_data,
    project_hnn_with_loadings, make_repr_loader,
    run_mmfl_on_full_data, project_mmfl,
    run_jive_on_full_data, project_jive,
    run_sjive_on_full_data, project_sjive,
)


# ---------------------------------------------------------------------------
# Argument parser (extracted so run_experiments.py can reuse it)
# ---------------------------------------------------------------------------

def build_full_parser():
    """
    Build the complete argument parser for a single experiment.
    Extracted as a standalone function so that run_experiments.py
    can import and reuse it for argument merging.
    """
    parser = argparse.ArgumentParser(description="Train EHR model")

    # Data
    parser.add_argument("--data_dir",         type=str, default="")
    parser.add_argument("--cxr_path",         type=str, default="")
    parser.add_argument("--output_dir",       type=str, default="outputs")
    parser.add_argument("--task",             type=str, default="readmission",
                        choices=["readmission", "mortality"])
    parser.add_argument("--missing_mode",     type=str, default=None)
    parser.add_argument("--chunk_size",       type=int, default=2,
                        help="Number of pkl files to load per training chunk")
    parser.add_argument("--label_file",       type=str, default=None)
    parser.add_argument("--label_task",       type=str, default=None)
    parser.add_argument("--label_task_type",  type=str, default=None,
                        choices=["classification", "regression"])
    parser.add_argument("--label_visit_policy", type=str, default=None,
                        choices=["all_visits", "history_before_last"])
    parser.add_argument("--label_pid_col",    type=str, default="patient_id")
    parser.add_argument("--exclude_death_codes_from_code_branch", action="store_true")

    # Fusion
    parser.add_argument("--fusion_type",      type=str, default="hcl",
                        help="Fusion method: hcl, slide, hnn, convirt, mmfl, misa, dlf, tsd")

    # Training
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--hcl_lam",          type=float, default=1.0)
    parser.add_argument("--patience",         type=int,   default=10)
    parser.add_argument("--num_workers",      type=int,   default=4)

    # Model architecture
    parser.add_argument("--r",                type=int,   default=512)
    parser.add_argument("--hidden_size",      type=int,   default=512)
    parser.add_argument("--code_emb_dim",     type=int,   default=1024)
    parser.add_argument("--note_input_dim",   type=int,   default=768)
    parser.add_argument("--lab_proj_dim",     type=int,   default=512)
    parser.add_argument("--rnn_layers",       type=int,   default=1)
    parser.add_argument("--dropout",          type=float, default=0.0)
    parser.add_argument("--rnn_type",         type=str,   default="GRU",
                        choices=["GRU", "LSTM", "RNN"])
    parser.add_argument("--hcl_hidden_dims",  type=int,   nargs="+", default=[256, 128])
    parser.add_argument("--patient_csv_path", type=str,   default="")
    parser.add_argument("--code_emb_path",    type=str,   default=None)

    # HNN-specific
    parser.add_argument("--hnn_tau",     type=float, default=1.0)
    parser.add_argument("--hnn_kappa",   type=float, default=1.0)
    parser.add_argument("--hnn_lam_all", type=float, default=0.1)

    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--pretrain_weight", type=float, default=1.0)

    # Optuna / r_list
    parser.add_argument("--r_candidates",  type=int, nargs="+", default=None)
    parser.add_argument("--r_list",        type=int, nargs="+", default=None)
    parser.add_argument("--optuna_trials", type=int, default=50)

    # Training mode
    parser.add_argument("--training_mode",   type=str, default="joint",
                        choices=["joint", "pretrain_finetune"])
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--finetune_strategy", type=str, default="freeze",
                        choices=["freeze", "partial", "full"])
    parser.add_argument("--finetune_backbone_lr_ratio", type=float, default=0.1)
    parser.add_argument("--pretrain_patience", type=int, default=5)

    # MMFL-specific
    parser.add_argument("--mmfl_lam",   type=float, default=1.0)
    parser.add_argument("--mmfl_gamma", type=float, default=1.0)
    parser.add_argument("--mmfl_mu",    type=float, default=1.0)

    # MISA-specific
    parser.add_argument("--misa_alpha", type=float, default=1.0)
    parser.add_argument("--misa_beta",  type=float, default=1.0)
    parser.add_argument("--misa_gamma", type=float, default=1.0)
    parser.add_argument("--misa_cmd_K", type=int,   default=5)
    parser.add_argument("--n_heads",    type=int,   default=2)
    # sJIVE-specific
    parser.add_argument("--sjive_eta", type=float, default=0.5,
                        help="sJIVE weight: 1.0=unsupervised JIVE, 0.0=pure prediction")
    parser.add_argument("--summary_csv", type=str, default=None)

    return parser


def resolve_task_settings(args):
    use_external = bool(getattr(args, "label_file", None) and getattr(args, "label_task", None))
    if use_external:
        missing = [
            name for name in ["label_file", "label_task", "label_task_type", "label_visit_policy"]
            if getattr(args, name, None) is None
        ]
        if missing:
            raise ValueError(f"Missing external label args: {missing}")
        args.task_type = args.label_task_type.lower()
        args.task_name = args.label_task
    else:
        if any(getattr(args, name, None) is not None for name in ["label_file", "label_task", "label_task_type", "label_visit_policy"]):
            raise ValueError(
                "label_file, label_task, label_task_type, and label_visit_policy must be provided together"
            )
        args.task_type = "classification"
        args.task_name = args.task
    return args


def build_dataset_kwargs(args):
    return {
        "label_file": getattr(args, "label_file", None),
        "label_task": getattr(args, "label_task", None),
        "label_task_type": getattr(args, "label_task_type", None),
        "label_visit_policy": getattr(args, "label_visit_policy", None),
        "label_pid_col": getattr(args, "label_pid_col", "patient_id"),
        "exclude_death_codes_from_code_branch": getattr(
            args, "exclude_death_codes_from_code_branch", False
        ),
    }


def get_loss_name(task_type: str) -> str:
    return "MSE" if task_type == "regression" else "CE"


def get_monitor_name(task_type: str) -> str:
    return "MSE" if task_type == "regression" else "AUROC"


def get_monitor_value(metrics: dict, task_type: str) -> float:
    return metrics["mse"] if task_type == "regression" else metrics["auroc"]


def is_better_metric(current: float, best: float, task_type: str) -> bool:
    if task_type == "regression":
        return current < best
    return current > best


def make_metric_row(epoch, split, stage, train_total, train_task_loss, train_pretrain, metrics, task_type, skipped_total):
    return {
        "epoch"          : epoch,
        "split"          : split,
        "stage"          : stage,
        "train_total"    : train_total,
        "train_task_loss": train_task_loss,
        "train_pretrain" : train_pretrain,
        "auroc"          : round(metrics["auroc"], 4) if task_type == "classification" else "",
        "auprc"          : round(metrics["auprc"], 4) if task_type == "classification" else "",
        "f1"             : round(metrics["f1"], 4) if task_type == "classification" else "",
        "acc"            : round(metrics["acc"], 4) if task_type == "classification" else "",
        "precision"      : round(metrics["precision"], 4) if task_type == "classification" else "",
        "recall"         : round(metrics["recall"], 4) if task_type == "classification" else "",
        "threshold"      : round(metrics["threshold"], 4) if task_type == "classification" else "",
        "mse"            : round(metrics["mse"], 4) if task_type == "regression" else "",
        "mae"            : round(metrics["mae"], 4) if task_type == "regression" else "",
        "medae"          : round(metrics["medae"], 4) if task_type == "regression" else "",
        "mape"           : round(metrics["mape"], 4) if task_type == "regression" else "",
        "smape"          : round(metrics["smape"], 4) if task_type == "regression" else "",
        "huber"          : round(metrics["huber"], 4) if task_type == "regression" else "",
        "r2"             : round(metrics["r2"], 4) if task_type == "regression" else "",
        "num_samples"    : metrics.get("num_samples", ""),
        "skipped_total"  : skipped_total,
    }


def log_metrics(logger, prefix: str, metrics: dict, task_type: str, skipped_total: int):
    if task_type == "classification":
        logger.info(
            f"{prefix} AUROC={metrics['auroc']:.4f} | "
            f"AUPRC={metrics['auprc']:.4f} | "
            f"F1={metrics['f1']:.4f} | ACC={metrics['acc']:.4f} | "
            f"Used={metrics.get('num_samples', 'NA')} | Skipped={skipped_total}"
        )
    else:
        logger.info(
            f"{prefix} MSE={metrics['mse']:.4f} | "
            f"MAE={metrics['mae']:.4f} | "
            f"R2={metrics['r2']:.4f} | "
            f"Used={metrics.get('num_samples', 'NA')} | Skipped={skipped_total}"
        )


# ---------------------------------------------------------------------------
# Optuna search (HCL only)
# ---------------------------------------------------------------------------

def run_optuna_search(
    args,
    r_candidates: List[int],
    n_trials: int,
    device: torch.device,
    logger: logging.Logger,
):
    """
    Use Optuna to search the best per-structure r_list for HCL.

    Parameters
    ----------
    args          : parsed CLI args
    r_candidates  : list of candidate dims, e.g. [10, 50, 100]
    n_trials      : number of Optuna trials
    device        : torch device
    logger        : logger instance

    Returns
    -------
    best_r_list : list of 7 ints, the best per-structure dimensions
    """
    import optuna
    resolve_task_settings(args)

    # Pre-build mappings and datasets (shared across all trials)
    mappings        = build_mappings_from_patients(args.patient_csv_path)
    code_vocab_size = get_code_vocab_size(args.code_emb_path)

    pretrained_code_emb = None
    if args.code_emb_path is not None:
        pretrained_code_emb = load_code_embedding_matrix(
            parquet_path=args.code_emb_path,
            vocab_size=code_vocab_size,
            embed_dim=args.code_emb_dim,
        )

    def make_dataset(split_name):
        pkl_paths = sorted(
            glob.glob(os.path.join(args.data_dir, split_name, "*.pkl"))
        )
        ds = EHRDataset(
            pkl_paths=pkl_paths,
            cxr_embeddings_path=args.cxr_path,
            mappings=mappings,
            index_set=set(),
            task=args.task,
            missing_mode=args.missing_mode,
            **build_dataset_kwargs(args),
        )
        ds.replace_data(ds.load_chunk(pkl_paths))
        return ds

    train_ds = make_dataset("train")
    val_ds   = make_dataset("val")

    collate = partial(ehr_collate_fn, make_onehot_label=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    demo_dim = train_ds[0]["demographic"].shape[0]

    def objective(trial):
        # Sample r_list from candidates
        r_list = [
            trial.suggest_categorical(f"r_{k}", r_candidates)
            for k in range(7)
        ]
        logger.info(f"Trial {trial.number}: r_list = {r_list}")

        # Build fusion and model
        fusion = build_fusion(
            "hcl",
            input_dims=[args.hidden_size] * 3,
            hidden_dims=args.hcl_hidden_dims,
            r_list=r_list,
        )

        model = EHRModel(
            fusion_module=fusion,
            code_vocab_size=code_vocab_size,
            lab_vocab_size=12000,
            demo_dim=demo_dim,
            task_type=args.task_type,
            hidden_size=args.hidden_size,
            num_classes=2,
            code_emb_dim=args.code_emb_dim,
            note_input_dim=args.note_input_dim,
            lab_proj_dim=args.lab_proj_dim,
            rnn_layers=args.rnn_layers,
            dropout=args.dropout,
            rnn_type=args.rnn_type,
            pretrained_code_emb=pretrained_code_emb,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_metric = float("inf") if args.task_type == "regression" else 0.0
        patience_counter = 0

        for epoch in range(args.epochs):
            train_metrics = run_epoch_joint(
                model, train_loader, device, optimizer,
                pretrain_weight=args.pretrain_weight,
                hcl_lam=args.hcl_lam,
            )

            val_labels, val_predictions, _ = run_inference(model, val_loader, device)
            val_metrics = compute_metrics(val_labels, val_predictions, task_type=args.task_type)
            monitor = get_monitor_value(val_metrics, args.task_type)

            # Report intermediate value for pruning
            report_value = -monitor if args.task_type == "regression" else monitor
            trial.report(report_value, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch + 1}")
                raise optuna.exceptions.TrialPruned()

            if is_better_metric(monitor, best_metric, args.task_type):
                best_metric = monitor
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break

        logger.info(
            f"Trial {trial.number} finished: best_{get_monitor_name(args.task_type).lower()} = {best_metric:.4f}"
        )
        return -best_metric if args.task_type == "regression" else best_metric

    # Create study with pruning
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials)

    # Extract best r_list
    best_r_list = [study.best_params[f"r_{k}"] for k in range(7)]
    logger.info(f"Optuna search done. Best r_list = {best_r_list}")
    if args.task_type == "regression":
        logger.info(f"Best MSE = {-study.best_value:.4f}")
    else:
        logger.info(f"Best AUROC = {study.best_value:.4f}")

    return best_r_list


# ---------------------------------------------------------------------------
# Training epoch functions
# ---------------------------------------------------------------------------

def run_epoch_joint(model, loader, device, optimizer, pretrain_weight=1.0, **pretrain_kwargs):
    """
    Run one epoch of joint training (fusion loss + CE loss, all params updated).
    """
    model.train()
    total_loss = 0
    total_task = 0
    total_pt   = 0
    n_batches  = 0

    from tqdm import tqdm

    for batch in tqdm(loader, desc="Joint Train", leave=False):
        batch     = move_batch_to_device(batch, device)
        loss_dict = model.compute_joint_loss(batch, pretrain_weight=pretrain_weight, **pretrain_kwargs)
        loss      = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_task += loss_dict["task_loss"]
        total_pt   += loss_dict["pretrain"]
        n_batches  += 1

    return {
        "total"    : total_loss / n_batches,
        "task_loss": total_task / n_batches,
        "pretrain" : total_pt   / n_batches,
    }


def run_epoch_pretrain(model, loader, device, optimizer, **pretrain_kwargs):
    """
    Run one epoch of pretrain-only training (contrastive loss only, no CE).
    Only encoder + fusion parameters are updated.
    """
    model.train()
    total_loss = 0
    n_batches = 0

    from tqdm import tqdm

    for batch in tqdm(loader, desc="Pretrain", leave=False):
        batch = move_batch_to_device(batch, device)
        x_list = model._encode_modalities(batch)
        loss_dict = model.fusion.compute_pretrain_loss(x_list, **pretrain_kwargs)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"total": total_loss / n_batches}


@torch.no_grad()
def run_val_pretrain(model, loader, device, **pretrain_kwargs):
    """
    Compute pretrain (contrastive) loss on validation set for early stopping.
    """
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        x_list = model._encode_modalities(batch)
        loss_dict = model.fusion.compute_pretrain_loss(x_list, **pretrain_kwargs)
        total_loss += loss_dict["total"].item()
        n_batches += 1

    return total_loss / n_batches


def run_epoch_finetune(model, loader, device, optimizer):
    """
    Run one epoch of finetune training (CE loss only, no contrastive loss).
    Which parameters are updated depends on the optimizer's param groups.
    """
    model.train()
    total_loss = 0
    n_batches = 0

    from tqdm import tqdm

    for batch in tqdm(loader, desc="Finetune", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        labels = batch["label"]
        if model.task_type == "classification":
            loss = nn.functional.cross_entropy(outputs, labels.long())
        else:
            loss = nn.functional.mse_loss(outputs, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"total": total_loss / n_batches, "task_loss": total_loss / n_batches}


# ---------------------------------------------------------------------------
# Freeze / finetune helpers
# ---------------------------------------------------------------------------

def freeze_fusion(model):
    """
    Freeze all parameters except the classifier head,
    and switch frozen modules to eval mode.
    """
    for enc in [model.code_enc, model.note_enc, model.lab_enc]:
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()

    for p in model.fusion.parameters():
        p.requires_grad = False
    model.fusion.eval()


def build_finetune_optimizer(model, strategy, lr, backbone_lr_ratio=0.1):
    """
    Build optimizer for the finetune stage based on the chosen strategy.

    Parameters
    ----------
    model              : EHRModel
    strategy           : 'freeze' | 'partial' | 'full'
    lr                 : base learning rate for classifier
    backbone_lr_ratio  : lr multiplier for encoder+fusion in 'partial' mode

    Returns
    -------
    optimizer : torch.optim.Adam
    """
    if strategy == "freeze":
        for enc in [model.code_enc, model.note_enc, model.lab_enc]:
            for p in enc.parameters():
                p.requires_grad = False
        for p in model.fusion.parameters():
            p.requires_grad = False
        return torch.optim.Adam(model.classifier.parameters(), lr=lr)

    elif strategy == "partial":
        backbone_params = []
        for enc in [model.code_enc, model.note_enc, model.lab_enc]:
            backbone_params.extend(enc.parameters())
        backbone_params.extend(model.fusion.parameters())

        return torch.optim.Adam([
            {"params": backbone_params, "lr": lr * backbone_lr_ratio},
            {"params": model.classifier.parameters(), "lr": lr},
        ])

    elif strategy == "full":
        return torch.optim.Adam(model.parameters(), lr=lr)

    else:
        raise ValueError(f"Unknown finetune_strategy: {strategy}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    """
    Run model in eval mode over loader.
    Returns (all_labels, all_predictions, all_patient_ids).
    """
    model.eval()
    all_labels, all_predictions, all_pids = [], [], []

    for batch in loader:
        batch   = move_batch_to_device(batch, device)
        outputs = model(batch)
        if model.task_type == "classification":
            preds = torch.softmax(outputs, dim=-1)[:, 1]
        else:
            preds = outputs
        all_predictions.extend(preds.cpu().numpy().tolist())
        all_labels.extend(batch["label"].cpu().numpy().tolist())
        all_pids.extend(batch["patient_id"])

    return (
        np.array(all_labels),
        np.array(all_predictions),
        all_pids,
    )


# ---------------------------------------------------------------------------
# Summary CSV helper
# ---------------------------------------------------------------------------

def _append_summary(args, test_metrics, logger, mode_override=None, skipped_total=0):
    """
    Append final test metrics to the global summary CSV if --summary_csv is set.
    Cleans up checkpoint files after recording results.
    """
    if args.summary_csv is None:
        return
    from datetime import datetime
    summary_row = {
        "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fusion_type"  : args.fusion_type,
        "training_mode": mode_override if mode_override is not None else args.training_mode,
        "task"         : getattr(args, "task_name", args.task),
        "task_type"    : args.task_type,
        "seed"         : args.seed,
        "missing_mode" : args.missing_mode or "none",
        "r"            : args.r,
        "auroc"        : round(test_metrics["auroc"],      4) if args.task_type == "classification" else "",
        "auprc"        : round(test_metrics["auprc"],      4) if args.task_type == "classification" else "",
        "f1"           : round(test_metrics["f1"],         4) if args.task_type == "classification" else "",
        "acc"          : round(test_metrics["acc"],        4) if args.task_type == "classification" else "",
        "precision"    : round(test_metrics["precision"],  4) if args.task_type == "classification" else "",
        "recall"       : round(test_metrics["recall"],     4) if args.task_type == "classification" else "",
        "threshold"    : round(test_metrics["threshold"],  4) if args.task_type == "classification" else "",
        "mse"          : round(test_metrics["mse"],        4) if args.task_type == "regression" else "",
        "mae"          : round(test_metrics["mae"],        4) if args.task_type == "regression" else "",
        "medae"        : round(test_metrics["medae"],      4) if args.task_type == "regression" else "",
        "mape"         : round(test_metrics["mape"],       4) if args.task_type == "regression" else "",
        "smape"        : round(test_metrics["smape"],      4) if args.task_type == "regression" else "",
        "huber"        : round(test_metrics["huber"],      4) if args.task_type == "regression" else "",
        "r2"           : round(test_metrics["r2"],         4) if args.task_type == "regression" else "",
        "num_samples"  : test_metrics.get("num_samples", ""),
        "skipped_total": skipped_total,
    }
    append_csv_row(args.summary_csv, summary_row)
    logger.info(f"Summary appended to {args.summary_csv}")

    # Clean up checkpoint files to save disk space
    for fname in ["best_model.pt", "pretrain_best.pt"]:
        fpath = os.path.join(args.output_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            logger.info(f"Removed checkpoint: {fpath}")


# ---------------------------------------------------------------------------
# Core experiment logic (shared by standalone and batch runner)
# ---------------------------------------------------------------------------

def run_single_experiment(args, shared_data, device):
    """
    Run a single experiment using pre-loaded shared data.

    Parameters
    ----------
    args        : argparse.Namespace with all experiment-specific settings
    shared_data : dict containing loaders, vocab size, embeddings, demo_dim
    device      : torch.device
    """
    resolve_task_settings(args)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger   = init_logging(args.output_dir)
    csv_path = os.path.join(args.output_dir, "training_record.csv")

    logger.info(f"Using device: {device}")
    logger.info(f"Args: {vars(args)}")

    # Unpack shared data
    train_loader      = shared_data["train_loader"]
    val_loader        = shared_data["val_loader"]
    test_loader       = shared_data["test_loader"]
    code_vocab_size   = shared_data["code_vocab_size"]
    pretrained_code_emb = shared_data["pretrained_code_emb"]
    demo_dim          = shared_data["demo_dim"]
    split_stats       = shared_data.get("split_stats", {})
    train_skipped_total = sum(split_stats.get("train", {}).get(k, 0) for k in ["skipped_short_history", "skipped_missing_label", "skipped_nan_label"])
    val_skipped_total = sum(split_stats.get("val", {}).get(k, 0) for k in ["skipped_short_history", "skipped_missing_label", "skipped_nan_label"])
    test_skipped_total = sum(split_stats.get("test", {}).get(k, 0) for k in ["skipped_short_history", "skipped_missing_label", "skipped_nan_label"])
    args.train_skipped_total = train_skipped_total
    args.val_skipped_total = val_skipped_total
    args.test_skipped_total = test_skipped_total

    for split_name, stats in split_stats.items():
        skipped_total = (
            stats.get("skipped_short_history", 0)
            + stats.get("skipped_missing_label", 0)
            + stats.get("skipped_nan_label", 0)
        )
        logger.info(
            f"{split_name}: used={stats.get('kept', 0)} | "
            f"skipped_total={skipped_total} | "
            f"short_history={stats.get('skipped_short_history', 0)} | "
            f"missing_label={stats.get('skipped_missing_label', 0)} | "
            f"nan_label={stats.get('skipped_nan_label', 0)}"
        )

    # ---------------------------------------------------------------
    # Optuna search for per-structure r_list (HCL only)
    # ---------------------------------------------------------------
    if args.fusion_type.lower() == "hcl" and args.r_candidates is not None:
        logger.info(
            f"Running Optuna search with candidates={args.r_candidates}, "
            f"n_trials={args.optuna_trials}"
        )
        best_r_list = run_optuna_search(
            args=args,
            r_candidates=args.r_candidates,
            n_trials=args.optuna_trials,
            device=device,
            logger=logger,
        )
        args.r_list = best_r_list
        logger.info(f"Using searched r_list: {args.r_list}")
    elif args.fusion_type.lower() == "hcl" and args.r_list is not None:
        logger.info(f"Using provided r_list: {args.r_list}")
    else:
        if not hasattr(args, 'r_list') or args.r_list is None:
            args.r_list = None

    # Build fusion module
    fusion_kwargs = dict(
        input_dims=[args.hidden_size, args.hidden_size, args.hidden_size],
        hidden_dims=args.hcl_hidden_dims,
        r=args.r,
        tau=args.hnn_tau,
        kappa=args.hnn_kappa,
        lam_all=args.hnn_lam_all,
        lam=args.mmfl_lam,
        mu=args.mmfl_mu,
        cmd_K=args.misa_cmd_K,
        n_heads=args.n_heads,
        alpha=args.misa_alpha,
        beta=args.misa_beta,
    )
    # Resolve gamma conflict: MMFL uses 'gamma', MISA uses 'misa_gamma'
    if args.fusion_type.lower() == "misa":
        fusion_kwargs["gamma"] = args.misa_gamma
    else:
        fusion_kwargs["gamma"] = args.mmfl_gamma

    if args.fusion_type.lower() == "hcl" and args.r_list is not None:
        fusion_kwargs["r_list"] = args.r_list
        fusion_kwargs.pop("r", None)
    if args.fusion_type.lower() == "sjive":
            fusion_kwargs["eta"] = args.sjive_eta
    fusion = build_fusion(args.fusion_type, **fusion_kwargs)

    model = EHRModel(
        fusion_module=fusion,
        code_vocab_size=code_vocab_size,
        lab_vocab_size=12000,
        demo_dim=demo_dim,
        task_type=args.task_type,
        hidden_size=args.hidden_size,
        num_classes=2,
        code_emb_dim=args.code_emb_dim,
        note_input_dim=args.note_input_dim,
        lab_proj_dim=args.lab_proj_dim,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        pretrained_code_emb=pretrained_code_emb,
    ).to(device)

    # ---------------------------------------------------------------
    # SLIDE branch
    # ---------------------------------------------------------------
    if args.fusion_type.lower() == "slide":
        logger.info("SLIDE mode: extracting full representations from frozen encoders...")
        freeze_fusion(model)

        logger.info("Extracting train representations...")
        tr_xc, tr_xn, tr_xl, tr_labels, tr_demos, tr_pids = \
            extract_all_representations(model, train_loader, device)

        logger.info("Extracting val representations...")
        val_xc, val_xn, val_xl, val_labels, val_demos, val_pids = \
            extract_all_representations(model, val_loader, device)

        logger.info("Extracting test representations...")
        te_xc, te_xn, te_xl, te_labels, te_demos, te_pids = \
            extract_all_representations(model, test_loader, device)

        logger.info("Running SLIDE on full training data...")
        tr_U, V_train = run_slide_on_full_data(model, tr_xc, tr_xn, tr_xl, device)

        tr_X      = torch.cat([tr_xc, tr_xn, tr_xl], dim=1)
        col_means = tr_X.mean(dim=0)

        logger.info("Projecting val data onto training SLIDE space...")
        val_U = project_with_V(val_xc, val_xn, val_xl, V_train, col_means, device)

        logger.info("Projecting test data onto training SLIDE space...")
        te_U = project_with_V(te_xc, te_xn, te_xl, V_train, col_means, device)

        test_metrics = run_mlp_branch(
            model, tr_U, val_U, te_U, tr_labels, val_labels, te_labels,
            tr_demos, val_demos, te_demos, te_pids, "SLIDE",
            device, args, logger, csv_path,
        )
        _append_summary(args, test_metrics, logger, mode_override="decomposition", skipped_total=test_skipped_total)
        return

    # ---------------------------------------------------------------
    # HNN branch
    # ---------------------------------------------------------------
    if args.fusion_type.lower() == "hnn":
        logger.info("HNN mode: extracting full representations from frozen encoders...")
        freeze_fusion(model)

        logger.info("Extracting train representations...")
        tr_xc, tr_xn, tr_xl, tr_labels, tr_demos, tr_pids = \
            extract_all_representations(model, train_loader, device)

        logger.info("Extracting val representations...")
        val_xc, val_xn, val_xl, val_labels, val_demos, val_pids = \
            extract_all_representations(model, val_loader, device)

        logger.info("Extracting test representations...")
        te_xc, te_xn, te_xl, te_labels, te_demos, te_pids = \
            extract_all_representations(model, test_loader, device)

        logger.info("Running HNN on full training data...")
        tr_U = run_hnn_on_full_data(model, tr_xc, tr_xn, tr_xl, device)

        logger.info("Projecting val data onto training HNN space...")
        val_U = project_hnn_with_loadings(val_xc, val_xn, val_xl, model.fusion, device)

        logger.info("Projecting test data onto training HNN space...")
        te_U = project_hnn_with_loadings(te_xc, te_xn, te_xl, model.fusion, device)

        test_metrics = run_mlp_branch(
            model, tr_U, val_U, te_U, tr_labels, val_labels, te_labels,
            tr_demos, val_demos, te_demos, te_pids, "HNN",
            device, args, logger, csv_path,
        )
        _append_summary(args, test_metrics, logger, mode_override="decomposition", skipped_total=test_skipped_total)
        return

    # ---------------------------------------------------------------
    # MMFL branch
    # ---------------------------------------------------------------
    if args.fusion_type.lower() == "mmfl":
        logger.info("MMFL mode: extracting full representations from frozen encoders...")
        freeze_fusion(model)

        logger.info("Extracting train representations...")
        tr_xc, tr_xn, tr_xl, tr_labels, tr_demos, tr_pids = \
            extract_all_representations(model, train_loader, device)

        logger.info("Extracting val representations...")
        val_xc, val_xn, val_xl, val_labels, val_demos, val_pids = \
            extract_all_representations(model, val_loader, device)

        logger.info("Extracting test representations...")
        te_xc, te_xn, te_xl, te_labels, te_demos, te_pids = \
            extract_all_representations(model, test_loader, device)

        logger.info("Running MMFL fit on full training data...")
        tr_U_np = run_mmfl_on_full_data(
            model.fusion, tr_xc, tr_xn, tr_xl, tr_labels,
        )

        logger.info("Projecting val data onto training MMFL space...")
        val_U_np = project_mmfl(model.fusion, val_xc, val_xn, val_xl)

        logger.info("Projecting test data onto training MMFL space...")
        te_U_np = project_mmfl(model.fusion, te_xc, te_xn, te_xl)

        tr_U  = torch.from_numpy(tr_U_np).float()
        val_U = torch.from_numpy(val_U_np).float()
        te_U  = torch.from_numpy(te_U_np).float()

        test_metrics = run_mlp_branch(
            model, tr_U, val_U, te_U,
            tr_labels, val_labels, te_labels,
            tr_demos, val_demos, te_demos,
            te_pids, "MMFL",
            device, args, logger, csv_path,
        )
        _append_summary(args, test_metrics, logger, mode_override="decomposition", skipped_total=test_skipped_total)
        return
    # ---------------------------------------------------------------
    # JIVE branch
    # ---------------------------------------------------------------
    if args.fusion_type.lower() == "jive":
        logger.info("JIVE mode: extracting full representations from frozen encoders...")
        freeze_fusion(model)

        logger.info("Extracting train representations...")
        tr_xc, tr_xn, tr_xl, tr_labels, tr_demos, tr_pids = \
            extract_all_representations(model, train_loader, device)

        logger.info("Extracting val representations...")
        val_xc, val_xn, val_xl, val_labels, val_demos, val_pids = \
            extract_all_representations(model, val_loader, device)

        logger.info("Extracting test representations...")
        te_xc, te_xn, te_xl, te_labels, te_demos, te_pids = \
            extract_all_representations(model, test_loader, device)

        logger.info("Running JIVE fit on full training data...")
        tr_U_np = run_jive_on_full_data(
            model.fusion, tr_xc, tr_xn, tr_xl,
        )

        logger.info("Projecting val data onto training JIVE space...")
        val_U_np = project_jive(model.fusion, val_xc, val_xn, val_xl)

        logger.info("Projecting test data onto training JIVE space...")
        te_U_np = project_jive(model.fusion, te_xc, te_xn, te_xl)

        tr_U  = torch.from_numpy(tr_U_np).float()
        val_U = torch.from_numpy(val_U_np).float()
        te_U  = torch.from_numpy(te_U_np).float()

        test_metrics = run_mlp_branch(
            model, tr_U, val_U, te_U,
            tr_labels, val_labels, te_labels,
            tr_demos, val_demos, te_demos,
            te_pids, "JIVE",
            device, args, logger, csv_path,
        )
        _append_summary(args, test_metrics, logger, mode_override="decomposition", skipped_total=test_skipped_total)
        return
    # ---------------------------------------------------------------
    # sJIVE branch
    # ---------------------------------------------------------------
    if args.fusion_type.lower() == "sjive":
        logger.info("sJIVE mode: extracting full representations from frozen encoders...")
        freeze_fusion(model)

        logger.info("Extracting train representations...")
        tr_xc, tr_xn, tr_xl, tr_labels, tr_demos, tr_pids = \
            extract_all_representations(model, train_loader, device)

        logger.info("Extracting val representations...")
        val_xc, val_xn, val_xl, val_labels, val_demos, val_pids = \
            extract_all_representations(model, val_loader, device)

        logger.info("Extracting test representations...")
        te_xc, te_xn, te_xl, te_labels, te_demos, te_pids = \
            extract_all_representations(model, test_loader, device)

        logger.info("Running sJIVE fit on full training data...")
        tr_U_np = run_sjive_on_full_data(
            model.fusion, tr_xc, tr_xn, tr_xl, tr_labels,
        )

        logger.info("Projecting val data onto training sJIVE space...")
        val_U_np = project_sjive(model.fusion, val_xc, val_xn, val_xl)

        logger.info("Projecting test data onto training sJIVE space...")
        te_U_np = project_sjive(model.fusion, te_xc, te_xn, te_xl)

        tr_U  = torch.from_numpy(tr_U_np).float()
        val_U = torch.from_numpy(val_U_np).float()
        te_U  = torch.from_numpy(te_U_np).float()

        test_metrics = run_mlp_branch(
            model, tr_U, val_U, te_U,
            tr_labels, val_labels, te_labels,
            tr_demos, val_demos, te_demos,
            te_pids, "sJIVE",
            device, args, logger, csv_path,
        )
        _append_summary(args, test_metrics, logger, mode_override="decomposition", skipped_total=test_skipped_total)
        return
    # ---------------------------------------------------------------
    # Joint training branch (HCL / ConVIRT / DLF / MISA / TSD)
    # ---------------------------------------------------------------
    best_model_ckpt = os.path.join(args.output_dir, "best_model.pt")

    if args.training_mode == "joint":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_metric  = float("inf") if args.task_type == "regression" else 0.0
        patience_counter = 0

        logger.info("Starting joint training...\n")

        for epoch in range(args.epochs):
            logger.info(f"{'=' * 40}")
            logger.info(f"Epoch {epoch + 1} / {args.epochs}")
            logger.info(f"{'=' * 40}")

            train_metrics = run_epoch_joint(
                model, train_loader, device, optimizer,
                pretrain_weight=args.pretrain_weight,
                hcl_lam=args.hcl_lam,
            )
            logger.info(
                f"Train total={train_metrics['total']:.4f} | "
                f"{get_loss_name(args.task_type)}={train_metrics['task_loss']:.4f} | "
                f"pretrain={train_metrics['pretrain']:.4f}"
            )

            # --- Validation ---
            val_labels, val_predictions, _ = run_inference(model, val_loader, device)
            val_metrics = compute_metrics(val_labels, val_predictions, task_type=args.task_type)
            val_monitor = get_monitor_value(val_metrics, args.task_type)
            log_metrics(logger, "Val", val_metrics, args.task_type, val_skipped_total)
            append_csv_row(
                csv_path,
                make_metric_row(
                    epoch=epoch + 1,
                    split="val",
                    stage="joint",
                    train_total=round(train_metrics["total"], 6),
                    train_task_loss=round(train_metrics["task_loss"], 6),
                    train_pretrain=round(train_metrics["pretrain"], 6),
                    metrics=val_metrics,
                    task_type=args.task_type,
                    skipped_total=val_skipped_total,
                ),
            )

            # --- Test (monitoring only, not used for early stopping) ---
            test_labels_ep, test_predictions_ep, _ = run_inference(model, test_loader, device)
            test_metrics_ep = compute_metrics(test_labels_ep, test_predictions_ep, task_type=args.task_type)
            log_metrics(logger, "Test", test_metrics_ep, args.task_type, test_skipped_total)
            append_csv_row(
                csv_path,
                make_metric_row(
                    epoch=epoch + 1,
                    split="test",
                    stage="joint",
                    train_total=round(train_metrics["total"], 6),
                    train_task_loss=round(train_metrics["task_loss"], 6),
                    train_pretrain=round(train_metrics["pretrain"], 6),
                    metrics=test_metrics_ep,
                    task_type=args.task_type,
                    skipped_total=test_skipped_total,
                ),
            )

            # --- Early stopping on validation only ---
            if is_better_metric(val_monitor, best_val_metric, args.task_type):
                logger.info(
                    f"New best ({get_monitor_name(args.task_type)} {best_val_metric:.4f} -> {val_monitor:.4f}), saving checkpoint."
                )
                best_val_metric  = val_monitor
                patience_counter = 0
                torch.save(model.state_dict(), best_model_ckpt)
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience = {patience_counter} / {args.patience}")
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered.")
                    break

    # ---------------------------------------------------------------
    # Pretrain + Finetune branch (HCL / ConVIRT / DLF / MISA / TSD)
    # ---------------------------------------------------------------
    elif args.training_mode == "pretrain_finetune":

        # === Stage 1: Pretrain (contrastive loss only) ===
        logger.info("=" * 60)
        logger.info("Stage 1: Pretrain (contrastive loss only)")
        logger.info("=" * 60)

        pretrain_params = []
        for enc in [model.code_enc, model.note_enc, model.lab_enc]:
            pretrain_params.extend(enc.parameters())
        pretrain_params.extend(model.fusion.parameters())
        pretrain_optimizer = torch.optim.Adam(pretrain_params, lr=args.lr)

        best_val_pretrain_loss    = float("inf")
        pretrain_patience_counter = 0
        pretrain_ckpt             = os.path.join(args.output_dir, "pretrain_best.pt")

        for epoch in range(args.pretrain_epochs):
            logger.info(f"{'=' * 40}")
            logger.info(f"Pretrain Epoch {epoch + 1} / {args.pretrain_epochs}")
            logger.info(f"{'=' * 40}")

            train_metrics = run_epoch_pretrain(
                model, train_loader, device, pretrain_optimizer,
                hcl_lam=args.hcl_lam,
            )
            logger.info(f"Pretrain train loss={train_metrics['total']:.4f}")

            val_pretrain_loss = run_val_pretrain(
                model, val_loader, device,
                hcl_lam=args.hcl_lam,
            )
            logger.info(f"Pretrain val loss={val_pretrain_loss:.4f}")

            csv_row = {
                "epoch"          : epoch + 1,
                "split"          : "val",
                "stage"          : "pretrain",
                "train_total"    : round(train_metrics["total"], 6),
                "train_task_loss": "",
                "train_pretrain" : round(train_metrics["total"], 6),
                "auroc"          : "",
                "auprc"          : "",
                "f1"             : "",
                "acc"            : "",
                "precision"      : "",
                "recall"         : "",
                "threshold"      : "",
                "mse"            : "",
                "mae"            : "",
                "medae"          : "",
                "mape"           : "",
                "smape"          : "",
                "huber"          : "",
                "r2"             : "",
                "num_samples"    : "",
                "skipped_total"  : "",
            }
            append_csv_row(csv_path, csv_row)

            if val_pretrain_loss < best_val_pretrain_loss:
                logger.info(
                    f"New best pretrain loss ({best_val_pretrain_loss:.4f} -> "
                    f"{val_pretrain_loss:.4f}), saving pretrain checkpoint."
                )
                best_val_pretrain_loss    = val_pretrain_loss
                pretrain_patience_counter = 0
                torch.save(model.state_dict(), pretrain_ckpt)
            else:
                pretrain_patience_counter += 1
                logger.info(
                    f"No improvement. Pretrain patience = "
                    f"{pretrain_patience_counter} / {args.pretrain_patience}"
                )
                if pretrain_patience_counter >= args.pretrain_patience:
                    logger.info("Pretrain early stopping triggered.")
                    break

        logger.info("Loading best pretrain checkpoint for finetune stage...")
        model.load_state_dict(torch.load(pretrain_ckpt, map_location=device))

        # === Stage 2: Finetune (supervised loss only) ===
        logger.info("=" * 60)
        logger.info(
            f"Stage 2: Finetune ({get_loss_name(args.task_type)} only, strategy={args.finetune_strategy})"
        )
        logger.info("=" * 60)

        finetune_optimizer = build_finetune_optimizer(
            model,
            strategy=args.finetune_strategy,
            lr=args.lr,
            backbone_lr_ratio=args.finetune_backbone_lr_ratio,
        )

        best_val_metric  = float("inf") if args.task_type == "regression" else 0.0
        patience_counter = 0

        for epoch in range(args.epochs):
            logger.info(f"{'=' * 40}")
            logger.info(f"Finetune Epoch {epoch + 1} / {args.epochs}")
            logger.info(f"{'=' * 40}")

            train_metrics = run_epoch_finetune(
                model, train_loader, device, finetune_optimizer,
            )
            logger.info(f"Finetune train {get_loss_name(args.task_type)}={train_metrics['task_loss']:.4f}")

            val_labels, val_predictions, _ = run_inference(model, val_loader, device)
            val_metrics = compute_metrics(val_labels, val_predictions, task_type=args.task_type)
            val_monitor = get_monitor_value(val_metrics, args.task_type)
            log_metrics(logger, "Val", val_metrics, args.task_type, val_skipped_total)
            append_csv_row(
                csv_path,
                make_metric_row(
                    epoch=epoch + 1,
                    split="val",
                    stage="finetune",
                    train_total=round(train_metrics["total"], 6),
                    train_task_loss=round(train_metrics["task_loss"], 6),
                    train_pretrain="",
                    metrics=val_metrics,
                    task_type=args.task_type,
                    skipped_total=val_skipped_total,
                ),
            )

            test_labels_ep, test_predictions_ep, _ = run_inference(model, test_loader, device)
            test_metrics_ep = compute_metrics(test_labels_ep, test_predictions_ep, task_type=args.task_type)
            log_metrics(logger, "Test", test_metrics_ep, args.task_type, test_skipped_total)
            append_csv_row(
                csv_path,
                make_metric_row(
                    epoch=epoch + 1,
                    split="test",
                    stage="finetune",
                    train_total=round(train_metrics["total"], 6),
                    train_task_loss=round(train_metrics["task_loss"], 6),
                    train_pretrain="",
                    metrics=test_metrics_ep,
                    task_type=args.task_type,
                    skipped_total=test_skipped_total,
                ),
            )

            if is_better_metric(val_monitor, best_val_metric, args.task_type):
                logger.info(
                    f"New best ({get_monitor_name(args.task_type)} {best_val_metric:.4f} -> {val_monitor:.4f}), saving checkpoint."
                )
                best_val_metric  = val_monitor
                patience_counter = 0
                torch.save(model.state_dict(), best_model_ckpt)
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience = {patience_counter} / {args.patience}")
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered.")
                    break

    # ---------------------------------------------------------------
    # Final test evaluation (shared by joint and pretrain_finetune)
    # ---------------------------------------------------------------
    logger.info("\nLoading best model checkpoint for test evaluation...")
    model.load_state_dict(torch.load(best_model_ckpt, map_location=device))

    test_labels, test_predictions, test_pids = run_inference(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_predictions, task_type=args.task_type)
    log_metrics(logger, "Final Test", test_metrics, args.task_type, test_skipped_total)
    append_csv_row(
        csv_path,
        make_metric_row(
            epoch="",
            split="test",
            stage="final",
            train_total="",
            train_task_loss="",
            train_pretrain="",
            metrics=test_metrics,
            task_type=args.task_type,
            skipped_total=test_skipped_total,
        ),
    )

    save_predictions(
        patient_ids=test_pids,
        labels=test_labels,
        save_path=os.path.join(args.output_dir, "test_predictions.csv"),
        predictions=test_predictions,
        task_type=args.task_type,
        threshold=test_metrics.get("threshold"),
    )

    _append_summary(args, test_metrics, logger, skipped_total=test_skipped_total)

    logger.info(f"Training record saved to {csv_path}")
    logger.info("Done.")


# ---------------------------------------------------------------------------
# Standalone entry point (python train.py --...)
# ---------------------------------------------------------------------------

def main(args):
    """
    Standalone entry: loads data itself, then calls run_single_experiment.
    Used when running `python train.py --...` directly for a single experiment.
    """
    resolve_task_settings(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build mappings and vocab
    mappings        = build_mappings_from_patients(args.patient_csv_path)
    code_vocab_size = get_code_vocab_size(args.code_emb_path)

    pretrained_code_emb = None
    if args.code_emb_path is not None:
        pretrained_code_emb = load_code_embedding_matrix(
            parquet_path=args.code_emb_path,
            vocab_size=code_vocab_size,
            embed_dim=args.code_emb_dim,
        )

    def make_dataset(split_name):
        pkl_paths = sorted(
            glob.glob(os.path.join(args.data_dir, split_name, "*.pkl"))
        )
        ds = EHRDataset(
            pkl_paths=pkl_paths,
            cxr_embeddings_path=args.cxr_path,
            mappings=mappings,
            index_set=set(),
            task=args.task,
            missing_mode=args.missing_mode,
            **build_dataset_kwargs(args),
        )
        ds.replace_data(ds.load_chunk(pkl_paths))
        return ds

    train_ds = make_dataset("train")
    val_ds   = make_dataset("val")
    test_ds  = make_dataset("test")

    collate = partial(ehr_collate_fn, make_onehot_label=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    demo_dim = train_ds[0]["demographic"].shape[0]

    shared_data = {
        "train_loader":      train_loader,
        "val_loader":        val_loader,
        "test_loader":       test_loader,
        "code_vocab_size":   code_vocab_size,
        "pretrained_code_emb": pretrained_code_emb,
        "demo_dim":          demo_dim,
        "split_stats": {
            "train": dict(train_ds.stats),
            "val": dict(val_ds.stats),
            "test": dict(test_ds.stats),
        },
    }

    run_single_experiment(args, shared_data, device)


if __name__ == "__main__":
    parser = build_full_parser()
    args = parser.parse_args()
    main(args)
