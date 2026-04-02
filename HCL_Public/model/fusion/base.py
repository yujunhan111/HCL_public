from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn as nn


class FusionModule(ABC, nn.Module):
    """
    Abstract base class for all fusion methods.

    Every fusion method must implement:
        - forward(x_list)            : fuse modality encodings -> [B, out_dim]
        - compute_pretrain_loss(...) : return pretrain loss dict, or None if not applicable

    Attributes
    ----------
    out_dim    : int  - output dimension after fusion (used by EHR_model to build classifier)
    has_pretrain : bool - whether this method requires a pretraining stage
    """

    def __init__(self, out_dim: int, has_pretrain: bool):
        super().__init__()
        self.out_dim     = out_dim
        self.has_pretrain = has_pretrain

    @abstractmethod
    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse modality encodings into a single representation.

        Parameters
        ----------
        x_list : list of [B, d_m] tensors, one per modality

        Returns
        -------
        [B, out_dim] tensor
        """
        raise NotImplementedError

    @abstractmethod
    def compute_pretrain_loss(
        self,
        x_list: List[torch.Tensor],
        **kwargs,
    ) -> Optional[dict]:
        """
        Compute the pretraining loss for this fusion method.

        Parameters
        ----------
        x_list : list of [B, d_m] tensors, one per modality

        Returns
        -------
        dict with at least key 'total' (scalar Tensor), or None if has_pretrain=False
        """
        raise NotImplementedError
