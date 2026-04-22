import os
import csv
import logging
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, f1_score,
    accuracy_score, mean_squared_error,
    mean_absolute_error, median_absolute_error, r2_score,
)
from dataset.mapping import create_all_mappings
from baseline_runners import make_repr_loader


HUBER_DELTA = 1.0


def _compute_mape(labels: np.ndarray, predictions: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(labels), eps)
    return float(np.mean(np.abs((labels - predictions) / denom)) * 100.0)


def _compute_smape(labels: np.ndarray, predictions: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(labels) + np.abs(predictions), eps)
    return float(np.mean(2.0 * np.abs(predictions - labels) / denom) * 100.0)


def _compute_huber(labels: np.ndarray, predictions: np.ndarray, delta: float = HUBER_DELTA) -> float:
    err = predictions - labels
    abs_err = np.abs(err)
    quadratic = np.minimum(abs_err, delta)
    linear = abs_err - quadratic
    return float(np.mean(0.5 * quadratic ** 2 + delta * linear))


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Recursively move all tensors in batch dict to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    task_type: str = "classification",
    threshold: float = None,
):
    """
    Compute metrics for either classification or regression.
    """
    task_type = task_type.lower()
    if task_type == "regression":
        labels = np.asarray(labels, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        return {
            "mse": mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions),
            "medae": median_absolute_error(labels, predictions),
            "mape": _compute_mape(labels, predictions),
            "smape": _compute_smape(labels, predictions),
            "huber": _compute_huber(labels, predictions),
            "r2": r2_score(labels, predictions),
            "threshold": None,
            "num_samples": int(len(labels)),
        }

    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)

    if threshold is None:
        thresholds = np.linspace(0, 1, 201)
        f1s = [
            f1_score(labels, (predictions >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        threshold = float(thresholds[np.argmax(f1s)])

    preds = (predictions >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "auroc"    : auroc,
        "auprc"    : auprc,
        "f1"       : f1,
        "acc"      : acc,
        "precision": prec,
        "recall"   : rec,
        "threshold": threshold,
        "num_samples": int(len(labels)),
    }


def save_predictions(
    patient_ids: list,
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: str,
    task_type: str = "classification",
    threshold: float = None,
):
    """Save per-patient predictions to csv."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        if task_type == "regression":
            fieldnames = ["patient_id", "label", "prediction"]
        else:
            fieldnames = ["patient_id", "label", "prob", "pred"]
            preds = (predictions >= threshold).astype(int)

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if task_type == "regression":
            for pid, lbl, pred in zip(patient_ids, labels, predictions):
                writer.writerow({
                    "patient_id": pid,
                    "label"     : round(float(lbl), 6),
                    "prediction": round(float(pred), 6),
                })
        else:
            for pid, lbl, prob, pred in zip(patient_ids, labels, predictions, preds):
                writer.writerow({
                    "patient_id": pid,
                    "label"     : int(lbl),
                    "prob"      : round(float(prob), 6),
                    "pred"      : int(pred),
                })
    print(f"[INFO] Predictions saved to {save_path}")


def append_csv_row(csv_path: str, row: dict):
    """
    Append one row to the training record CSV.
    If file exists with a different header, expand the header and
    rewrite the existing rows before appending the new one.
    """
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    new_fields = list(row.keys())
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    if existing_fields == new_fields:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fields)
            writer.writerow(row)
        return

    merged_fields = list(existing_fields)
    for field in new_fields:
        if field not in merged_fields:
            merged_fields.append(field)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        for existing_row in existing_rows:
            normalized = {field: existing_row.get(field, "") for field in merged_fields}
            writer.writerow(normalized)
        writer.writerow({field: row.get(field, "") for field in merged_fields})


def init_logging(output_dir: str) -> logging.Logger:
    """
    Initialize a logger that writes to both stdout and a log file.
    Log file: <output_dir>/train.log
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def set_seed(seed: int):
    """Fix all random seeds for reproducibility across runs."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_code_embedding_matrix(
    parquet_path: str,
    vocab_size: int,
    embed_dim: int = 1024,
) -> torch.Tensor:
    """
    Load BGE code embeddings from parquet and build an embedding matrix.

    The parquet file must have columns:
        - 'index'         : int, code_index (used as row index in the matrix)
        - 'bge_embedding' : list/array of floats, length embed_dim

    Row 0 is reserved for padding and kept as zeros.
    Any code_index that exceeds vocab_size is silently skipped.

    Returns
    -------
    emb_matrix : torch.Tensor, shape [vocab_size + 1, embed_dim]
    """
    df = pd.read_parquet(parquet_path)
    emb_matrix = torch.zeros(vocab_size + 1, embed_dim, dtype=torch.float32)
    for row in df.itertuples(index=False):
        code_idx = int(row.index)
        if code_idx > vocab_size:
            continue
        vec = torch.tensor(row.bge_embedding, dtype=torch.float32)
        emb_matrix[code_idx] = vec
    return emb_matrix


def get_code_vocab_size(code_emb_path: str) -> int:
    """
    Read max code index directly from the parquet embedding file,
    avoiding the need to scan all patient pkl files.
    """
    df = pd.read_parquet(code_emb_path, columns=["index"])
    return int(df["index"].max())


def build_mappings_from_patients(patient_csv_path: str) -> Dict:
    """
    Build demographic mappings directly from the patients_dict csv file.
    Expects columns: gender, ethnicity (mapped to 'race'), marital_status, language.
    """
    df = pd.read_csv(patient_csv_path)
    if "ethnicity" in df.columns and "race" not in df.columns:
        df = df.rename(columns={"ethnicity": "race"})
    return create_all_mappings(df)


def run_mlp_branch(
    model,
    tr_U, val_U, te_U,
    tr_labels, val_labels, te_labels,
    tr_demos, val_demos, te_demos,
    te_pids,
    method_name,
    device,
    args,
    logger,
    csv_path,
):
    """
    Train the shared MLP head on pre-computed score matrices U
    (from SLIDE / HNN / MMFL), then evaluate on val and test.
    """
    import torch.nn as nn
    from baseline_runners import make_repr_loader

    # Re-initialize classifier weights so it starts fresh
    for layer in model.classifier:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    # Build DataLoaders from pre-computed representations
    train_loader = make_repr_loader(
        tr_U, torch.zeros_like(tr_U[:, :0]),
        torch.zeros_like(tr_U[:, :0]),
        tr_labels, tr_demos,
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = make_repr_loader(
        val_U, torch.zeros_like(val_U[:, :0]),
        torch.zeros_like(val_U[:, :0]),
        val_labels, val_demos,
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader = make_repr_loader(
        te_U, torch.zeros_like(te_U[:, :0]),
        torch.zeros_like(te_U[:, :0]),
        te_labels, te_demos,
        batch_size=args.batch_size, shuffle=False,
    )

    task_type = getattr(args, "task_type", "classification").lower()
    val_skipped_total = getattr(args, "val_skipped_total", 0)
    test_skipped_total = getattr(args, "test_skipped_total", 0)

    # Only train the classifier head
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    model.classifier.to(device)

    best_monitor = float("inf") if task_type == "regression" else float("-inf")
    patience_counter = 0
    best_state = None

    logger.info(f"\n{method_name}: Training MLP head on pre-computed scores...")

    for epoch in range(args.epochs):
        # --- Train ---
        model.classifier.train()
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            U_batch = batch["x_code"].to(device)
            demo    = batch["demographic"].to(device)
            labels  = batch["label"].to(device)

            feat    = torch.cat([U_batch, demo], dim=1)
            outputs = model.classifier(feat)
            if task_type == "classification":
                loss = nn.functional.cross_entropy(outputs, labels.long())
            else:
                loss = nn.functional.mse_loss(outputs.squeeze(-1), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # --- Validate ---
        model.classifier.eval()
        val_predictions_list = []
        val_labels_list = []
        with torch.no_grad():
            for batch in val_loader:
                U_batch = batch["x_code"].to(device)
                demo    = batch["demographic"].to(device)
                labels  = batch["label"]

                feat    = torch.cat([U_batch, demo], dim=1)
                outputs = model.classifier(feat)
                if task_type == "classification":
                    preds = torch.softmax(outputs, dim=-1)[:, 1]
                else:
                    preds = outputs.squeeze(-1)

                val_predictions_list.extend(preds.cpu().numpy().tolist())
                val_labels_list.extend(labels.numpy().tolist())

        val_metrics = compute_metrics(
            np.array(val_labels_list),
            np.array(val_predictions_list),
            task_type=task_type,
        )
        monitor = val_metrics["mse"] if task_type == "regression" else val_metrics["auroc"]

        if task_type == "classification":
            logger.info(
                f"{method_name} Epoch {epoch+1}/{args.epochs} | "
                f"Train CE={avg_loss:.4f} | "
                f"Val AUROC={monitor:.4f} | AUPRC={val_metrics['auprc']:.4f} | "
                f"F1={val_metrics['f1']:.4f} | ACC={val_metrics['acc']:.4f} | "
                f"Used={val_metrics['num_samples']} | Skipped={val_skipped_total}"
            )
        else:
            logger.info(
                f"{method_name} Epoch {epoch+1}/{args.epochs} | "
                f"Train MSE={avg_loss:.4f} | "
                f"Val MSE={val_metrics['mse']:.4f} | "
                f"MAE={val_metrics['mae']:.4f} | R2={val_metrics['r2']:.4f} | "
                f"Used={val_metrics['num_samples']} | Skipped={val_skipped_total}"
            )

        csv_row = {
            "epoch"         : epoch + 1,
            "split"         : "val",
            "stage"         : method_name.lower(),
            "train_total"   : round(avg_loss, 6),
            "train_task_loss": round(avg_loss, 6),
            "train_pretrain": "",
            "auroc"         : round(val_metrics["auroc"],      4) if task_type == "classification" else "",
            "auprc"         : round(val_metrics["auprc"],      4) if task_type == "classification" else "",
            "f1"            : round(val_metrics["f1"],         4) if task_type == "classification" else "",
            "acc"           : round(val_metrics["acc"],        4) if task_type == "classification" else "",
            "precision"     : round(val_metrics["precision"],  4) if task_type == "classification" else "",
            "recall"        : round(val_metrics["recall"],     4) if task_type == "classification" else "",
            "threshold"     : round(val_metrics["threshold"],  4) if task_type == "classification" else "",
            "mse"           : round(val_metrics["mse"],        4) if task_type == "regression" else "",
            "mae"           : round(val_metrics["mae"],        4) if task_type == "regression" else "",
            "medae"         : round(val_metrics["medae"],      4) if task_type == "regression" else "",
            "mape"          : round(val_metrics["mape"],       4) if task_type == "regression" else "",
            "smape"         : round(val_metrics["smape"],      4) if task_type == "regression" else "",
            "huber"         : round(val_metrics["huber"],      4) if task_type == "regression" else "",
            "r2"            : round(val_metrics["r2"],         4) if task_type == "regression" else "",
            "num_samples"   : val_metrics["num_samples"],
            "skipped_total" : val_skipped_total,
        }
        append_csv_row(csv_path, csv_row)

        improved = monitor < best_monitor if task_type == "regression" else monitor > best_monitor
        if improved:
            if task_type == "classification":
                logger.info(f"New best (AUROC {best_monitor:.4f} -> {monitor:.4f})")
            else:
                logger.info(f"New best (MSE {best_monitor:.4f} -> {monitor:.4f})")
            best_monitor = monitor
            patience_counter = 0
            best_state = {
                k: v.clone() for k, v in model.classifier.state_dict().items()
            }
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience = {patience_counter} / {args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"{method_name}: Early stopping triggered.")
                break

    # --- Test with best classifier ---
    if best_state is not None:
        model.classifier.load_state_dict(best_state)

    model.classifier.eval()
    te_predictions_list = []
    te_labels_list = []
    with torch.no_grad():
        for batch in test_loader:
            U_batch = batch["x_code"].to(device)
            demo    = batch["demographic"].to(device)
            labels  = batch["label"]

            feat    = torch.cat([U_batch, demo], dim=1)
            outputs = model.classifier(feat)
            if task_type == "classification":
                preds = torch.softmax(outputs, dim=-1)[:, 1]
            else:
                preds = outputs.squeeze(-1)

            te_predictions_list.extend(preds.cpu().numpy().tolist())
            te_labels_list.extend(labels.numpy().tolist())

    te_labels_arr = np.array(te_labels_list)
    te_predictions_arr  = np.array(te_predictions_list)
    test_metrics  = compute_metrics(te_labels_arr, te_predictions_arr, task_type=task_type)

    if task_type == "classification":
        logger.info(
            f"{method_name} Test AUROC={test_metrics['auroc']:.4f} | "
            f"AUPRC={test_metrics['auprc']:.4f} | "
            f"F1={test_metrics['f1']:.4f} | "
            f"ACC={test_metrics['acc']:.4f} | "
            f"Precision={test_metrics['precision']:.4f} | "
            f"Recall={test_metrics['recall']:.4f} | "
            f"Threshold={test_metrics['threshold']:.4f} | "
            f"Used={test_metrics['num_samples']} | Skipped={test_skipped_total}"
        )
    else:
        logger.info(
            f"{method_name} Test MSE={test_metrics['mse']:.4f} | "
            f"MAE={test_metrics['mae']:.4f} | "
            f"R2={test_metrics['r2']:.4f} | "
            f"Used={test_metrics['num_samples']} | Skipped={test_skipped_total}"
        )

    test_csv_row = {
        "epoch"         : "",
        "split"         : "test",
        "stage"         : method_name.lower(),
        "train_total"   : "",
        "train_task_loss": "",
        "train_pretrain": "",
        "auroc"         : round(test_metrics["auroc"],      4) if task_type == "classification" else "",
        "auprc"         : round(test_metrics["auprc"],      4) if task_type == "classification" else "",
        "f1"            : round(test_metrics["f1"],         4) if task_type == "classification" else "",
        "acc"           : round(test_metrics["acc"],        4) if task_type == "classification" else "",
        "precision"     : round(test_metrics["precision"],  4) if task_type == "classification" else "",
        "recall"        : round(test_metrics["recall"],     4) if task_type == "classification" else "",
        "threshold"     : round(test_metrics["threshold"],  4) if task_type == "classification" else "",
        "mse"           : round(test_metrics["mse"],        4) if task_type == "regression" else "",
        "mae"           : round(test_metrics["mae"],        4) if task_type == "regression" else "",
        "medae"         : round(test_metrics["medae"],      4) if task_type == "regression" else "",
        "mape"          : round(test_metrics["mape"],       4) if task_type == "regression" else "",
        "smape"         : round(test_metrics["smape"],      4) if task_type == "regression" else "",
        "huber"         : round(test_metrics["huber"],      4) if task_type == "regression" else "",
        "r2"            : round(test_metrics["r2"],         4) if task_type == "regression" else "",
        "num_samples"   : test_metrics["num_samples"],
        "skipped_total" : test_skipped_total,
    }
    append_csv_row(csv_path, test_csv_row)

    save_predictions(
        patient_ids=te_pids,
        labels=te_labels_arr,
        save_path=os.path.join(args.output_dir, "test_predictions.csv"),
        predictions=te_predictions_arr,
        task_type=task_type,
        threshold=test_metrics.get("threshold"),
    )

    logger.info(f"{method_name} done.")

    return test_metrics
