import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from fusion.base import FusionModule
from building_blocks import StructureEncoder


# ---------------------------------------------------------------------------
# Modality pair indices for 3 modalities: (A,B), (A,C), (B,C)
# ---------------------------------------------------------------------------
_MODALITY_PAIRS = [(0, 1), (0, 2), (1, 2)]
_NUM_MODALITIES = 3


# ---------------------------------------------------------------------------
# Bidirectional InfoNCE loss for one modality pair
# ---------------------------------------------------------------------------

def _bidirectional_infonce(
    v: torch.Tensor,
    u: torch.Tensor,
    tau: float,
    lam: float,
) -> torch.Tensor:
    """
    Compute bidirectional InfoNCE (ConVIRT) loss for one modality pair.

    Both v and u must be L2-normalized before calling this function.
    Positive pairs are on the diagonal (same sample index).

    Parameters
    ----------
    v   : [B, d], L2-normalized projected features from modality 1
    u   : [B, d], L2-normalized projected features from modality 2
    tau : temperature parameter
    lam : weight for v->u direction; (1-lam) for u->v direction

    Returns
    -------
    loss : scalar Tensor
    """
    B      = v.size(0)
    labels = torch.arange(B, device=v.device)  # diagonal = positive pairs

    # Cosine similarity matrix [B, B]: sim[i,j] = <v_i, u_j>
    # v and u are already L2-normalized so dot product == cosine similarity
    sim = v @ u.T / tau                         # [B, B]

    # v -> u: for each v_i, find the matching u_i among all u_k
    loss_v2u = F.cross_entropy(sim, labels)

    # u -> v: for each u_i, find the matching v_i among all v_k
    loss_u2v = F.cross_entropy(sim.T, labels)

    return lam * loss_v2u + (1.0 - lam) * loss_u2v


# ---------------------------------------------------------------------------
# ConVIRTFusion: FusionModule implementation
# ---------------------------------------------------------------------------

class ConVIRTFusion(FusionModule):
    """
    ConVIRT-style pairwise contrastive pre-training fusion module for 3 modalities.

    Pre-training stage:
        Each modality encoder output (hidden_size) is passed through a
        StructureEncoder (FFN with ReLU) to project it to r dimensions,
        then L2-normalized. Pairwise bidirectional InfoNCE losses are
        computed for all 3 modality pairs:
            Total_Loss = Loss(A,B) + Loss(A,C) + Loss(B,C)

    Downstream stage:
        The same StructureEncoders are kept (not discarded).
        Outputs of all 3 StructureEncoders are concatenated: [B, 3*r].
        The downstream MLP classifier receives this 3*r-dimensional vector.

    Parameters
    ----------
    input_dims  : list of encoder output dims per modality, e.g. [512, 512, 512]
    hidden_dims : hidden layer sizes inside each StructureEncoder FFN
    r           : output dimension of each StructureEncoder (shared latent dim)
    tau         : InfoNCE temperature parameter (default 0.07)
    lam         : weight for v->u loss direction (default 0.75)
    """

    def __init__(
        self,
        input_dims: List[int],
        hidden_dims: List[int],
        r: int,
        tau: float = 0.07,
        lam: float = 0.75,
    ):
        # out_dim = 3 * r (concat of 3 modality structure encoder outputs)
        super().__init__(out_dim=_NUM_MODALITIES * r, has_pretrain=True)

        self.r   = r
        self.tau = tau
        self.lam = lam

        # One StructureEncoder per modality: hidden_size -> r
        # Reuses the same building block as HCLFusion
        self.structure_encoders = nn.ModuleList([
            StructureEncoder(
                input_dim=input_dims[m],
                hidden_dims=hidden_dims,
                output_dim=r,
            )
            for m in range(_NUM_MODALITIES)
        ])

    def _encode(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Pass each modality through its StructureEncoder.

        Parameters
        ----------
        x_list : list of [B, d_m] tensors, one per modality

        Returns
        -------
        list of [B, r] tensors, one per modality
        """
        return [
            self.structure_encoders[m](x_list[m])
            for m in range(_NUM_MODALITIES)
        ]

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Downstream forward pass.
        Pass each modality through its StructureEncoder, then concatenate.

        Parameters
        ----------
        x_list : list of [B, d_m] tensors, one per modality

        Returns
        -------
        [B, 3*r] tensor
        """
        encoded = self._encode(x_list)                 # list of [B, r]
        return torch.cat(encoded, dim=-1)              # [B, 3*r]

    def compute_pretrain_loss(
        self,
        x_list: List[torch.Tensor],
        **kwargs,
    ) -> dict:
        """
        Compute pairwise ConVIRT contrastive pre-training loss.

        For each modality pair (A,B), (A,C), (B,C):
            1. Pass through StructureEncoder -> [B, r]
            2. L2-normalize
            3. Compute bidirectional InfoNCE loss

        Total_Loss = Loss(A,B) + Loss(A,C) + Loss(B,C)

        Parameters
        ----------
        x_list : list of [B, d_m] tensors, one per modality

        Returns
        -------
        dict:
            total   : total pre-training loss (scalar Tensor)
            loss_AB : InfoNCE loss for pair (A, B) (float)
            loss_AC : InfoNCE loss for pair (A, C) (float)
            loss_BC : InfoNCE loss for pair (B, C) (float)
        """
        # Encode and L2-normalize each modality
        encoded = self._encode(x_list)                 # list of [B, r]
        normed  = [
            F.normalize(z, p=2, dim=-1)
            for z in encoded
        ]                                              # list of [B, r], unit vectors

        # Compute pairwise bidirectional InfoNCE losses
        pair_losses = [
            _bidirectional_infonce(
                v=normed[m1],
                u=normed[m2],
                tau=self.tau,
                lam=self.lam,
            )
            for m1, m2 in _MODALITY_PAIRS
        ]

        total = pair_losses[0] + pair_losses[1] + pair_losses[2]

        return {
            "total"  : total,
            "loss_AB": pair_losses[0].item(),
            "loss_AC": pair_losses[1].item(),
            "loss_BC": pair_losses[2].item(),
        }