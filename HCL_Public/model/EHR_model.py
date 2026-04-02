import torch
import torch.nn as nn
from fusion.base import FusionModule
from encoders import (
    CodeModalityEncoder,
    NoteModalityEncoder,
    LabModalityEncoder,
)


class EHRModel(nn.Module):
    """
    General EHR model pipeline:
        3 modality RNN encoders -> [B, hidden_size] each
        -> FusionModule (any method, e.g. HCL) -> [B, fusion.out_dim]
        -> concat demographic -> [B, fusion.out_dim + demo_dim]
        -> classifier head -> [B, num_classes]

    The fusion method is injected at construction time, so swapping
    methods only requires passing a different FusionModule instance.

    Parameters
    ----------
    fusion_module      : FusionModule instance (e.g. HCLFusion)
    code_vocab_size    : number of unique medical code indices
    lab_vocab_size     : number of unique lab type indices
    demo_dim           : dimension of demographic feature vector
    hidden_size        : RNN hidden size (output dim per modality encoder)
    num_classes        : 2 for binary classification
    code_emb_dim       : embedding dim for medical codes (BGE = 1024)
    note_input_dim     : pre-computed note embedding dim (BGE = 768)
    lab_proj_dim       : projection dim inside LabModalityEncoder
    rnn_layers         : number of RNN layers per modality
    dropout            : dropout rate
    rnn_type           : "GRU" | "LSTM" | "RNN"
    pretrained_code_emb: optional pretrained embedding matrix [vocab_size+1, code_emb_dim],
                         if provided the embedding layer is initialized with it and frozen
    """

    def __init__(
        self,
        fusion_module: FusionModule,
        code_vocab_size: int,
        lab_vocab_size: int,
        demo_dim: int,
        task_type: str = "classification",
        hidden_size: int = 512,
        num_classes: int = 2,
        code_emb_dim: int = 1024,
        note_input_dim: int = 768,
        lab_proj_dim: int = 512,
        rnn_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "GRU",
        pretrained_code_emb: torch.Tensor = None,
    ):
        super().__init__()
        self.task_type = task_type.lower()
        if self.task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")

        # --- Modality encoders ---
        self.code_enc = CodeModalityEncoder(
            vocab_size=code_vocab_size,
            emb_dim=code_emb_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            pretrained_emb=pretrained_code_emb,
        )
        self.note_enc = NoteModalityEncoder(
            input_dim=note_input_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )
        self.lab_enc = LabModalityEncoder(
            lab_vocab_size=lab_vocab_size,
            proj_dim=lab_proj_dim,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )

        # --- Fusion module (swappable) ---
        self.fusion = fusion_module

        # --- Classifier head ---
        clf_input_dim = fusion_module.out_dim + demo_dim
        output_dim = num_classes if self.task_type == "classification" else 1
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def _compute_supervised_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return nn.functional.cross_entropy(outputs, labels.long())
        preds = outputs.squeeze(-1)
        return nn.functional.mse_loss(preds, labels.float())

    def _encode_modalities(self, batch: dict):
        """
        Run all three modality encoders.
        Returns x_list: [x_code, x_note, x_lab], each [B, hidden_size].
        """
        # Code modality
        med    = batch["medical"]
        x_code = self.code_enc(med["codes"], med["mask"])

        # Note modality: empty dict when no notes in batch
        if batch["notes"]:
            x_note = self.note_enc(
                batch["notes"]["embeddings"], batch["notes"]["mask"]
            )
        else:
            x_note = torch.zeros_like(x_code)

        # Lab modality
        lab   = batch["labs"]
        x_lab = self.lab_enc(
            lab["values"], lab["times"], lab["types"].long(), lab["mask"]
        )

        return [x_code, x_note, x_lab]

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict returned by ehr_collate_fn

        Returns
        -------
        logits : [B, num_classes]
        """
        x_list = self._encode_modalities(batch)          # list of [B, hidden_size]
        h      = self.fusion(x_list)                     # [B, fusion.out_dim]
        demo   = batch["demographic"]                    # [B, demo_dim]
        feat   = torch.cat([h, demo], dim=1)
        outputs = self.classifier(feat)
        return outputs if self.task_type == "classification" else outputs.squeeze(-1)

    def compute_pretrain_loss(self, batch: dict, **kwargs) -> dict:
        """
        Pretraining step: delegates to fusion module.
        Only called when fusion.has_pretrain is True.

        Returns
        -------
        dict with at least key 'total' (scalar Tensor)
        """
        x_list = self._encode_modalities(batch)
        return self.fusion.compute_pretrain_loss(x_list, **kwargs)

    def compute_classifier_loss(self, batch: dict) -> dict:
        """
        Classifier training step.
        Fusion encoders are frozen externally before calling this.

        Returns
        -------
        dict:
            total : CE loss (scalar Tensor)
            ce    : CE loss value (float)
        """
        outputs  = self.forward(batch)
        labels   = batch["label"]
        task_loss = self._compute_supervised_loss(outputs, labels)

        return {
            "total"    : task_loss,
            "task_loss": task_loss.item(),
        }
    def compute_joint_loss(self, batch: dict, pretrain_weight: float = 1.0, **pretrain_kwargs) -> dict:
        """
        Joint training: fusion pretrain loss + downstream CE loss.
        All parameters (encoders, fusion, classifier) are updated together.

        Parameters
        ----------
        batch           : dict returned by ehr_collate_fn
        pretrain_weight : weight to balance fusion loss vs classification loss
        **pretrain_kwargs : extra kwargs passed to fusion.compute_pretrain_loss()

        Returns
        -------
        dict:
            total    : combined loss (scalar Tensor)
            ce       : CE loss value (float)
            pretrain : fusion pretrain loss value (float)
        """
        x_list = self._encode_modalities(batch)

        # Pass labels to fusion pretrain loss (needed by DLF triplet mining)
        pretrain_dict = self.fusion.compute_pretrain_loss(
            x_list, labels=batch["label"], **pretrain_kwargs
        )
        loss_pretrain = pretrain_dict["total"]

        # Downstream classification loss
        h      = self.fusion(x_list)
        demo   = batch["demographic"]
        feat   = torch.cat([h, demo], dim=1)
        outputs = self.classifier(feat)
        if self.task_type == "regression":
            outputs = outputs.squeeze(-1)
        labels = batch["label"]
        task_loss = self._compute_supervised_loss(outputs, labels)

        total = task_loss + pretrain_weight * loss_pretrain

        return {
            "total"    : total,
            "task_loss": task_loss.item(),
            "pretrain" : loss_pretrain.item(),
        }
