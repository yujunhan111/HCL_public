# HCL: Hierarchical Contrastive Learning for Multimodal Data

This repository contains the code for **Hierarchical Contrastive Learning (HCL)**, a multimodal fusion framework. The codebase also includes implementations of 9 baseline fusion methods (ConVIRT, MISA, DLF, TSD, SLIDE, HNN, JIVE, sJIVE, MMFL) under the same pipeline for reproducibility.

## Architecture

```
Medical Codes  ──→  CodeEncoder (Embedding + GRU)  ──→  [B, hidden_size]  ─┐
Clinical Notes ──→  NoteEncoder (GRU)              ──→  [B, hidden_size]  ─┼──→  FusionModule  ──→  MLP Classifier  ──→  Prediction
Lab Results    ──→  LabEncoder  (Projection + GRU)  ──→  [B, hidden_size]  ─┘         ↑
                                                                            Demographics ─┘
```

Three modality-specific RNN encoders produce fixed-size patient representations, which are fused by a swappable fusion module and combined with demographics for downstream prediction.

## Project Structure

```
HCL_Public/
├── train.py                  # Single experiment entry point
├── run_experiments.py        # Batch runner: load data once, run many configs
├── baseline_runners.py       # Helpers for decomposition-based baselines
├── utils.py                  # Metrics, logging, seed, CSV utilities
├── dataset/
│   ├── dataset.py            # EHRDataset: patient preprocessing & data loading
│   ├── collate_func.py       # Padding & batching for variable-length sequences
│   └── mapping.py            # Demographic feature encoding
└── model/
    ├── EHR_model.py          # Main model: encoders + fusion + classifier
    ├── encoders.py           # Per-modality encoders
    ├── rnn.py                # Shared RNNEncoder with masking & packing
    ├── building_blocks.py    # StructureEncoder FFN
    └── fusion/
        ├── base.py           # FusionModule abstract base class
        ├── HCL.py            # Our method
        └── *.py              # Baseline implementations
```

## Data

```
data/
├── train/                        # Train split (pkl files)
├── val/                          # Validation split
├── test/                         # Test split
├── dict_note_code.parquet        # Pretrained code embeddings (columns: index, bge_embedding)
└── patients_dict.csv             # Patient demographics
```

Each `.pkl` file contains a list of patient dicts with `patient_id`, `demographics`, and a list of `visits`. Each visit contains medical code events, clinical note embeddings, lab events, and task labels. See `dataset/dataset.py` for the full schema.

## Usage

### Single Experiment

```bash
python train.py \
  --data_dir ./data \
  --code_emb_path ./data/dict_note_code.parquet \
  --patient_csv_path ./data/patients_dict.csv \
  --task readmission \
  --fusion_type hcl \
  --training_mode joint \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --output_dir Result/HCL_joint \
  --seed 42
```

### Batch Experiments

`run_experiments.py` loads data once and runs multiple experiment configs (defined in a text file) across multiple seeds:

```bash
python run_experiments.py \
  --data_dir ./data \
  --code_emb_path ./data/dict_note_code.parquet \
  --patient_csv_path ./data/patients_dict.csv \
  --task readmission \
  --seeds 1 2 3 \
  --summary_csv Result/summary.csv \
  --experiment_file experiments.txt
```

### Optuna Search for HCL r-list

HCL allows each of its 7 hierarchical structures to have a different latent dimension. Use `--r_candidates` to search via Optuna, or `--r_list` to specify directly:

```bash
# Search
python train.py --fusion_type hcl --r_candidates 10 20 50 100 --optuna_trials 100 ...

# Direct
python train.py --fusion_type hcl --r_list 100 100 50 100 100 20 20 ...
```

## Key Arguments

Run `python train.py --help` for the full list. Highlights:

| Argument | Default | Description |
|----------|---------|-------------|
| `--fusion_type` | `hcl` | Fusion method (`hcl`, `convirt`, `misa`, `dlf`, `tsd`, `slide`, `hnn`, `jive`, `sjive`, `mmfl`) |
| `--training_mode` | `joint` | `joint` or `pretrain_finetune` |
| `--task` | `readmission` | `readmission` or `mortality` |
| `--r` | `512` | Fusion latent dimension |
| `--hidden_size` | `512` | RNN hidden dimension |
| `--epochs` | `50` | Max training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--patience` | `10` | Early stopping patience |
| `--missing_mode` | `None` | Modality ablation (e.g., `code_only`, `note_only`, `all_exist`) |

Custom prediction tasks are supported via `--label_file`, `--label_task`, `--label_task_type`, and `--label_visit_policy`.

## Outputs

Each experiment writes to its `output_dir`:

- `training_record.csv` — per-epoch train/val/test metrics
- `test_predictions.csv` — per-patient test predictions
- `train.log` — full training log

## Requirements

- Python 3.8+
- PyTorch, NumPy, Pandas, scikit-learn, tqdm
- Optuna (optional, for r-list search)
