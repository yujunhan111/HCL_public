import torch
import torch.nn as nn
from rnn import RNNEncoder


class CodeModalityEncoder(nn.Module):
    """
    Encodes medical code sequences using pretrained BGE embeddings.

    Pipeline: codes [B, L] (int) -> Embedding (frozen, pretrained)
              -> [B, L, emb_dim] -> RNNEncoder -> [B, hidden_size]
    """
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 1024,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "GRU",
        pretrained_emb: torch.Tensor = None,  # shape [vocab_size+1, emb_dim]
    ):
        super().__init__()
        # +1 for padding index 0
        self.embedding = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)

        if pretrained_emb is not None:
            # Load pretrained weights and freeze
            self.embedding.weight = nn.Parameter(pretrained_emb, requires_grad=False)

        self.rnn = RNNEncoder(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        codes : [B, L]  long
        mask  : [B, L]  bool
        returns: [B, hidden_size]
        """
        x = self.embedding(codes)   # [B, L, emb_dim]
        return self.rnn(x, mask)
class NoteModalityEncoder(nn.Module):
    """
    Encodes pre-computed note embeddings (e.g. BGE-768).
    Pipeline: embs [B, L, 768] -> RNNEncoder -> [B, hidden_size]
    """
    def __init__(
        self,
        input_dim: int = 768,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.rnn = RNNEncoder(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )

    def forward(self, embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        embs : [B, L, input_dim]  float
        mask : [B, L]             bool
        returns: [B, hidden_size]
        """
        return self.rnn(embs, mask)


class CXRModalityEncoder(nn.Module):
    """
    Encodes pre-computed CXR embeddings (e.g. 2048-dim visual features).
    Pipeline: embs [B, L, 2048] -> RNNEncoder -> [B, hidden_size]
    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.rnn = RNNEncoder(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )

    def forward(self, embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        embs : [B, L, input_dim]  float
        mask : [B, L]             bool
        returns: [B, hidden_size]
        """
        return self.rnn(embs, mask)


class LabModalityEncoder(nn.Module):
    """
    Encodes lab event sequences.
    Each timestep: values [B,L,1] + times [B,L,1] + types [B,L] (int)
    Pipeline:
        values -> Linear(1 -> proj_dim) -> [B, L, proj_dim]
        times  -> Linear(1 -> proj_dim) -> [B, L, proj_dim]
        types  -> Embedding(lab_vocab -> proj_dim) -> [B, L, proj_dim]
        element-wise sum -> [B, L, proj_dim]
        -> RNNEncoder -> [B, hidden_size]
    """
    def __init__(
        self,
        lab_vocab_size: int,
        proj_dim: int = 512,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.value_proj = nn.Linear(1, proj_dim)
        self.time_proj  = nn.Linear(1, proj_dim)
        # lab type indices start from 100 in dataset; use padding_idx=0
        self.type_emb   = nn.Embedding(lab_vocab_size + 1, proj_dim, padding_idx=0)
        self.rnn = RNNEncoder(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )

    def forward(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        types: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        values : [B, L]  float
        times  : [B, L]  float
        types  : [B, L]  long
        mask   : [B, L]  bool
        returns: [B, hidden_size]
        """
        v = self.value_proj(values.unsqueeze(-1))   # [B, L, proj_dim]
        t = self.time_proj(times.unsqueeze(-1))     # [B, L, proj_dim]
        e = self.type_emb(types)                    # [B, L, proj_dim]
        x = v + t + e                               # element-wise sum [B, L, proj_dim]
        return self.rnn(x, mask)