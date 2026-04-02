"""
Batch experiment runner.
Loads data ONCE, then iterates over experiment configs from a text file.

Usage:
    python run_experiments.py \
      --data_dir ../data \
      --code_emb_path ../data/dict_note_code.parquet \
      --patient_csv_path ../data/patients_dict.csv \
      --cxr_path ../data/mimic_cxr_embeddings_final_with_visit_relative_time.pkl \
      --task readmission \
      --missing_mode all_exist \
      --batch_size 256 \
      --seeds 1 2 3 \
      --summary_csv Result/summary.csv \
      --experiment_file experiments.txt

Each line in experiments.txt is one experiment's per-run args, e.g.:
    --output_dir Result/TSD_joint --fusion_type tsd --training_mode joint --epochs 20
    --output_dir Result/SLIDE --fusion_type slide --epochs 50
"""

import os
import sys
import glob
import shlex
import argparse
import traceback
from functools import partial

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))

from dataset.dataset import EHRDataset
from dataset.collate_func import ehr_collate_fn
from utils import (
    load_code_embedding_matrix,
    get_code_vocab_size,
    build_mappings_from_patients,
)
from train import build_full_parser, run_single_experiment, resolve_task_settings, build_dataset_kwargs


# ---------------------------------------------------------------------------
# Outer-level argument parser (shared across all experiments)
# ---------------------------------------------------------------------------

def parse_runner_args():
    """Parse the shared (outer-level) arguments for run_experiments.py."""
    parser = argparse.ArgumentParser(description="Batch experiment runner")

    # Data paths (shared across all experiments)
    parser.add_argument("--data_dir",         type=str, required=True)
    parser.add_argument("--code_emb_path",    type=str, required=True)
    parser.add_argument("--patient_csv_path", type=str, required=True)
    parser.add_argument("--cxr_path",         type=str, required=True)

    # Shared defaults that can still be overridden per-experiment
    parser.add_argument("--task",          type=str,   default="readmission")
    parser.add_argument("--missing_mode",  type=str,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--seeds",         type=int,   nargs="+", default=None)
    parser.add_argument("--summary_csv",   type=str,   default=None)
    parser.add_argument("--label_file",    type=str,   default=None)
    parser.add_argument("--label_task",    type=str,   default=None)
    parser.add_argument("--label_task_type", type=str, default=None,
                        choices=["classification", "regression"])
    parser.add_argument("--label_visit_policy", type=str, default=None,
                        choices=["all_visits", "history_before_last"])
    parser.add_argument("--label_pid_col", type=str, default="patient_id")
    parser.add_argument("--exclude_death_codes_from_code_branch", action="store_true")
    parser.add_argument("--hidden_size",   type=int,   default=512)
    parser.add_argument("--code_emb_dim",  type=int,   default=1024)
    parser.add_argument("--note_input_dim",type=int,   default=768)
    parser.add_argument("--lab_proj_dim",  type=int,   default=512)
    parser.add_argument("--rnn_layers",    type=int,   default=1)
    parser.add_argument("--dropout",       type=float, default=0.0)
    parser.add_argument("--rnn_type",      type=str,   default="GRU")
    parser.add_argument("--lr",            type=float, default=1e-4)

    # Experiment file
    parser.add_argument("--experiment_file", type=str, required=True,
                        help="Text file where each non-empty, non-comment line "
                             "is one experiment's per-run args")

    return parser.parse_args()


def resolve_runner_seeds(runner_args):
    """
    Normalize the runner seed list.
    Falls back to --seed for backward compatibility.
    """
    if runner_args.seeds is not None:
        return runner_args.seeds
    return [runner_args.seed]


# ---------------------------------------------------------------------------
# Data loading (done once)
# ---------------------------------------------------------------------------

def load_shared_data(runner_args):
    """
    Load all data artifacts exactly once.
    Returns a dict containing everything needed by run_single_experiment.
    """
    print("=" * 60)
    print("Loading shared data (once for all experiments)...")
    print("=" * 60)
    resolve_task_settings(runner_args)

    mappings        = build_mappings_from_patients(runner_args.patient_csv_path)
    code_vocab_size = get_code_vocab_size(runner_args.code_emb_path)
    print(f"code_vocab_size = {code_vocab_size}")

    pretrained_code_emb = load_code_embedding_matrix(
        parquet_path=runner_args.code_emb_path,
        vocab_size=code_vocab_size,
        embed_dim=runner_args.code_emb_dim,
    )
    print("Code embeddings loaded.")

    def make_dataset(split_name):
        pkl_paths = sorted(
            glob.glob(os.path.join(runner_args.data_dir, split_name, "*.pkl"))
        )
        ds = EHRDataset(
            pkl_paths=pkl_paths,
            cxr_embeddings_path=runner_args.cxr_path,
            mappings=mappings,
            index_set=set(),
            task=runner_args.task,
            missing_mode=runner_args.missing_mode,
            **build_dataset_kwargs(runner_args),
        )
        ds.replace_data(ds.load_chunk(pkl_paths))
        return ds

    train_ds = make_dataset("train")
    val_ds   = make_dataset("val")
    test_ds  = make_dataset("test")

    collate = partial(ehr_collate_fn, make_onehot_label=False)

    train_loader = DataLoader(
        train_ds, batch_size=runner_args.batch_size, shuffle=True,
        num_workers=runner_args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=runner_args.batch_size, shuffle=False,
        num_workers=runner_args.num_workers, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=runner_args.batch_size, shuffle=False,
        num_workers=runner_args.num_workers, collate_fn=collate,
    )

    demo_dim = train_ds[0]["demographic"].shape[0]
    print(f"demo_dim = {demo_dim}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print("=" * 60)
    print("Data loading complete.\n")

    return {
        "train_loader":        train_loader,
        "val_loader":          val_loader,
        "test_loader":         test_loader,
        "code_vocab_size":     code_vocab_size,
        "pretrained_code_emb": pretrained_code_emb,
        "demo_dim":            demo_dim,
        "split_stats": {
            "train": dict(train_ds.stats),
            "val": dict(val_ds.stats),
            "test": dict(test_ds.stats),
        },
    }


# ---------------------------------------------------------------------------
# Experiment file parsing
# ---------------------------------------------------------------------------

def parse_experiment_lines(filepath):
    """
    Read experiment file.
    Each non-empty, non-comment (#) line is one experiment.
    Returns list of (line_number, arg_string) tuples.
    """
    experiments = []
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            experiments.append((line_num, line))
    return experiments


# ---------------------------------------------------------------------------
# Argument merging
# ---------------------------------------------------------------------------

def merge_args(runner_args, experiment_arg_str, seed_override=None):
    """
    Build a complete args namespace by:
    1. Starting from train.py's full parser defaults
    2. Overlaying shared runner_args
    3. Overlaying per-experiment args (highest priority)

    Auto-appends {task}/seed_{seed} to output_dir.
    """
    full_parser = build_full_parser()

    # Parse per-experiment tokens
    exp_tokens    = shlex.split(experiment_arg_str)
    exp_args, _   = full_parser.parse_known_args(exp_tokens)

    # Start from full parser defaults
    merged = full_parser.parse_args([])

    # Overlay shared runner_args onto defaults
    shared_fields = [
        "data_dir", "code_emb_path", "patient_csv_path", "cxr_path",
        "task", "missing_mode", "batch_size", "num_workers", "seed",
        "summary_csv", "hidden_size", "code_emb_dim", "note_input_dim",
        "lab_proj_dim", "rnn_layers", "dropout", "rnn_type", "lr",
        "label_file", "label_task", "label_task_type", "label_visit_policy", "label_pid_col",
        "exclude_death_codes_from_code_branch",
    ]
    for field in shared_fields:
        if hasattr(runner_args, field):
            setattr(merged, field, getattr(runner_args, field))

    # Overlay per-experiment args (only those explicitly provided,
    # i.e. different from the full parser defaults)
    exp_defaults = full_parser.parse_args([])
    for key, val in vars(exp_args).items():
        default_val = getattr(exp_defaults, key, None)
        if val != default_val:
            setattr(merged, key, val)

    if seed_override is not None:
        merged.seed = seed_override

    # Auto-append task/seed to output_dir
    merged.output_dir = os.path.join(
        merged.output_dir,
        merged.task,
        f"seed_{merged.seed}",
    )

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    runner_args = parse_runner_args()
    seed_list   = resolve_runner_seeds(runner_args)
    shared_data = load_shared_data(runner_args)
    experiments = parse_experiment_lines(runner_args.experiment_file)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total   = len(experiments) * len(seed_list)
    success = 0
    failed  = []

    print(
        f"Found {len(experiments)} experiment(s) in {runner_args.experiment_file} "
        f"across {len(seed_list)} seed(s): {seed_list}\n"
    )

    run_idx = 0
    for seed in seed_list:
        print(f"\n{'=' * 60}")
        print(f"Starting seed {seed}")
        print(f"{'=' * 60}\n")

        for exp_idx, (line_num, exp_str) in enumerate(experiments, 1):
            run_idx += 1
            print(f"\n{'#' * 60}")
            print(
                f"# Run {run_idx}/{total}  |  seed {seed}  |  "
                f"experiment {exp_idx}/{len(experiments)} (line {line_num})"
            )
            print(f"# {exp_str}")
            print(f"{'#' * 60}\n")

            try:
                args = merge_args(runner_args, exp_str, seed_override=seed)
                run_single_experiment(args, shared_data, device)
                success += 1
            except Exception as e:
                print(
                    f"\n[ERROR] Run {run_idx} FAILED "
                    f"(seed {seed}, experiment line {line_num}): {e}"
                )
                traceback.print_exc()
                failed.append((seed, exp_idx, line_num, str(e)))
                continue

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"All runs done: {success}/{total} succeeded.")
    if failed:
        print("Failed runs:")
        for seed, exp_idx, line_num, err in failed:
            print(f"  seed={seed}, experiment={exp_idx} (line {line_num}): {err}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()