# HCL: Hierarchical Contrastive Learning for Multimodal EHR Fusion

A research framework for multimodal Electronic Health Record (EHR) fusion, implementing **Hierarchical Contrastive Learning (HCL)** and 9 baseline fusion methods under a unified pipeline.

## Overview

This framework takes multi-modal patient data — medical codes, clinical notes, and lab results — encodes each modality with RNN-based encoders, and fuses them via a swappable fusion module for downstream clinical prediction tasks (e.g., 30-day readmission, in-hospital mortality).

### Architecture

```
Medical Codes  ──→  CodeEncoder (Embedding + GRU)  ──→  [B, hidden_size]  ─┐
Clinical Notes ──→  NoteEncoder (GRU)              ──→  [B, hidden_size]  ─┼──→  FusionModule  ──→  MLP Classifier  ──→  Prediction
Lab Results    ──→  LabEncoder  (Projection + GRU)  ──→  [B, hidden_size]  ─┘         ↑
                                                                            Demographics ─┘
```

### Supported Fusion Methods

| Method | Type | Training Mode |
|--------|------|---------------|
| **HCL** (ours) | Hierarchical contrastive learning | Joint / Pretrain-finetune |
| ConVIRT | Pairwise contrastive learning | Joint / Pretrain-finetune |
| MISA | Modality-invariant/specific decomposition | Joint / Pretrain-finetune |
| DLF | Disentangled language-focused fusion | Joint / Pretrain-finetune |
| TSD | Tri-subspace disentanglement | Joint / Pretrain-finetune |
| SLIDE | Structured matrix factorization | Decomposition + MLP |
| HNN | Hierarchical nuclear norm | Decomposition + MLP |
| JIVE | Joint and Individual Variation Explained | Decomposition + MLP |
| sJIVE | Supervised JIVE | Decomposition + MLP |
| MMFL | Supervised multi-modal fission learning | Decomposition + MLP |

## Project Structure

```
HCL_Public/
├── train.py                  # Single experiment entry point
├── run_experiments.py        # Batch experiment runner (load data once, run many configs)
├── baseline_runners.py       # Helpers for decomposition-based methods (SLIDE, HNN, JIVE, etc.)
├── utils.py                  # Metrics, logging, seed, CSV utilities
├── dataset/
│   ├── dataset.py            # EHRDataset: patient preprocessing & multi-modal data loading
│   ├── collate_func.py       # Padding & batching for variable-length sequences
│   └── mapping.py            # Demographic feature encoding
└── model/
    ├── EHR_model.py          # Main model: encoders + fusion + classifier
    ├── encoders.py           # CodeModalityEncoder, NoteModalityEncoder, LabModalityEncoder
    ├── rnn.py                # Shared RNNEncoder (GRU/LSTM) with masking & packing
    ├── building_blocks.py    # StructureEncoder FFN (shared building block)
    └── fusion/
        ├── base.py           # FusionModule abstract base class
        ├── HCL.py            # Hierarchical Contrastive Learning (ours)
        ├── SLIDE.py          # SLIDE matrix factorization
        ├── HNN.py            # Hierarchical Nuclear Norm
        ├── JIVE.py           # JIVE decomposition
        ├── sJIVE.py          # Supervised JIVE
        ├── MMFL.py           # Multi-Modal Fission Learning
        ├── ConVIRT.py        # ConVIRT contrastive fusion
        ├── MISA.py           # MISA invariant/specific fusion
        ├── DLF.py            # Disentangled Language-Focused fusion
        ├── TSD.py            # Tri-Subspace Disentanglement
        └── __init__.py       # Fusion registry & build_fusion() factory
```

## Data Preparation

### Expected Directory Layout

```
data/
├── train/
│   ├── chunk_0.pkl
│   ├── chunk_1.pkl
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
├── dict_note_code.parquet        # Pretrained BGE code embeddings
├── patients_dict.csv             # Patient demographics (gender, race, marital_status, language)
└── mimic_cxr_embeddings_final_with_visit_relative_time.pkl  # CXR embeddings
```

### Patient PKL Format

Each `.pkl` file contains a list of patient dictionaries. Each patient has:

- `patient_id`: unique identifier
- `demographics`: dict with `age`, `gender`, `race`, `marital_status`, `language`
- `visits`: list of visit dicts, each containing:
  - `ccs_events`, `icd10_events`, `icd9_events`, `phecode_events` — diagnosis codes
  - `rxnorm_events` — medication codes
  - `drg_APR_events`, `drg_HCFA_events` — DRG codes
  - `dis_embeddings`, `rad_embeddings` — pre-computed note embeddings (768-dim)
  - `lab_events` — lab results with `code_index`, `relative_time`, `standardized_value`
  - `30_days_readmission`, `in_hospital_mortality` — binary labels

### Code Embedding Parquet

The `dict_note_code.parquet` file must have columns:
- `index`: int, the code_index
- `bge_embedding`: list of floats (1024-dim BGE embeddings)

## Usage

### Single Experiment

```bash
python train.py \
  --data_dir ./data \
  --code_emb_path ./data/dict_note_code.parquet \
  --patient_csv_path ./data/patients_dict.csv \
  --cxr_path ./data/mimic_cxr_embeddings_final_with_visit_relative_time.pkl \
  --task readmission \
  --fusion_type hcl \
  --training_mode joint \
  --r 512 \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-4 \
  --output_dir Result/HCL_joint \
  --seed 42
```

### Batch Experiments

Create an experiment file (e.g., `experiments.txt`), one experiment per line:

```
# experiments.txt
--output_dir Result/HCL_joint    --fusion_type hcl    --training_mode joint           --epochs 50
--output_dir Result/HCL_pf       --fusion_type hcl    --training_mode pretrain_finetune --pretrain_epochs 20 --epochs 30
--output_dir Result/ConVIRT      --fusion_type convirt --training_mode joint           --epochs 50
--output_dir Result/MISA         --fusion_type misa   --training_mode joint            --epochs 50
--output_dir Result/DLF          --fusion_type dlf    --training_mode joint            --epochs 50
--output_dir Result/TSD          --fusion_type tsd    --training_mode joint            --epochs 50
--output_dir Result/SLIDE        --fusion_type slide  --epochs 50
--output_dir Result/HNN          --fusion_type hnn    --epochs 50
--output_dir Result/JIVE         --fusion_type jive   --epochs 50
--output_dir Result/sJIVE        --fusion_type sjive  --epochs 50
--output_dir Result/MMFL         --fusion_type mmfl   --epochs 50
```

Then run:

```bash
python run_experiments.py \
  --data_dir ./data \
  --code_emb_path ./data/dict_note_code.parquet \
  --patient_csv_path ./data/patients_dict.csv \
  --cxr_path ./data/mimic_cxr_embeddings_final_with_visit_relative_time.pkl \
  --task readmission \
  --batch_size 256 \
  --seeds 1 2 3 \
  --summary_csv Result/summary.csv \
  --experiment_file experiments.txt
```

This loads the data **once** and runs all experiments across all seeds. Results are appended to `summary.csv`.

### Optuna Hyperparameter Search (HCL)

HCL supports per-structure latent dimension search via Optuna:

```bash
python train.py \
  --fusion_type hcl \
  --r_candidates 64 128 256 512 \
  --optuna_trials 50 \
  ...
```

This searches the optimal `r_list` (7 latent dimensions, one per hierarchical structure) on the validation set.

## Key Arguments

### Shared Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `readmission` | Prediction task: `readmission` or `mortality` |
| `--fusion_type` | `hcl` | Fusion method (see table above) |
| `--training_mode` | `joint` | `joint` or `pretrain_finetune` |
| `--hidden_size` | `512` | RNN hidden dimension |
| `--r` | `512` | Fusion latent dimension |
| `--epochs` | `50` | Max training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--batch_size` | `32` | Batch size |
| `--patience` | `10` | Early stopping patience |
| `--seed` | `42` | Random seed |
| `--missing_mode` | `None` | Modality missing simulation (e.g., `code_only`, `note_only`, `all_exist`) |

### Training Mode Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrain_epochs` | `20` | Epochs for the pretrain stage |
| `--pretrain_patience` | `5` | Early stopping patience for pretrain |
| `--finetune_strategy` | `freeze` | `freeze`, `partial`, or `full` |
| `--finetune_backbone_lr_ratio` | `0.1` | LR multiplier for backbone in `partial` mode |
| `--pretrain_weight` | `1.0` | Weight of fusion loss relative to task loss in joint training |

### External Label Support

For custom prediction tasks beyond readmission/mortality:

```bash
python train.py \
  --label_file ./labels.csv \
  --label_task length_of_stay \
  --label_task_type regression \
  --label_visit_policy all_visits \
  ...
```

| Argument | Description |
|----------|-------------|
| `--label_file` | Path to CSV/Parquet/Pickle with labels |
| `--label_task` | Column name for the target variable |
| `--label_task_type` | `classification` or `regression` |
| `--label_visit_policy` | `all_visits` or `history_before_last` |

## Outputs

Each experiment writes to its `output_dir`:

- `training_record.csv` — per-epoch metrics (train loss, val/test AUROC, AUPRC, F1, etc.)
- `test_predictions.csv` — per-patient predictions on the test set
- `train.log` — full training log

If `--summary_csv` is specified, a single row of final test metrics is appended for cross-experiment comparison.

## Training Modes Explained

### Neural Fusion Methods (HCL, ConVIRT, MISA, DLF, TSD)

These methods have learnable fusion parameters and support two training strategies:

- **Joint**: Fusion loss (contrastive/regularization) and task loss (CE/MSE) are optimized simultaneously end-to-end.
- **Pretrain-Finetune**: Stage 1 trains only the fusion loss (task-agnostic); Stage 2 freezes/partially unfreezes the backbone and trains the classifier.

### Decomposition Methods (SLIDE, HNN, JIVE, sJIVE, MMFL)

These methods run matrix factorization on the **full** training set representations:

1. Freeze RNN encoders, extract all patient representations
2. Run the decomposition algorithm on the training set
3. Project val/test data onto the learned loading matrices
4. Train a small MLP classifier on the resulting score matrices

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, scikit-learn
- tqdm
- Optuna (optional, for HCL hyperparameter search)
