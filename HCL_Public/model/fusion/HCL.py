import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from typing import List, Optional

from fusion.base import FusionModule
from building_blocks import StructureEncoder


# ---------------------------------------------------------------------------
# Structure table helpers (module-level, built once at import time)
# ---------------------------------------------------------------------------

_NUM_MODALITIES = 3
_NUM_STRUCTURES = 7  # 2^3 - 1


def _build_structure_tables(M: int):
    """
    Build lookup tables for M modalities.

    Returns
    -------
    structure_list : list of frozenset
        All 2^M - 1 non-empty subsets, ordered by descending size then
        ascending lex order.
    structure_mask : dict[int, dict[int, int]]
        structure_mask[m][k] = s means modality m participates in structure k
        at encoder level s.  Level s = M - subset_size.
        Key k is absent when modality m does not belong to structure k.
    """
    subsets = []
    for size in range(M, 0, -1):
        for combo in combinations(range(M), size):
            subsets.append(frozenset(combo))

    mask = {m: {} for m in range(M)}
    for k, subset in enumerate(subsets):
        level = M - len(subset)
        for m in subset:
            mask[m][k] = level

    return subsets, mask


_STRUCTURE_LIST, _STRUCTURE_MASK = _build_structure_tables(_NUM_MODALITIES)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_hierarchical_structures(h_raw: list) -> list:
    """
    Expand per-level encoder outputs to all 7 structures.

    Parameters
    ----------
    h_raw : list[list[Tensor]]
        h_raw[m][s] : [B, r],  m in 0..2,  s in 0..2

    Returns
    -------
    h_full : list[list[Tensor]]
        h_full[m][k] : [B, r],  k in 0..6
        Zero tensor when modality m does not belong to structure k.
    """
    B, r   = h_raw[0][0].shape
    device = h_raw[0][0].device
    dtype  = h_raw[0][0].dtype
    zero   = torch.zeros(B, r, device=device, dtype=dtype)

    h_full = []
    for m in range(_NUM_MODALITIES):
        hm = []
        for k in range(_NUM_STRUCTURES):
            if k in _STRUCTURE_MASK[m]:
                s = _STRUCTURE_MASK[m][k]
                hm.append(h_raw[m][s])
            else:
                hm.append(zero)
        h_full.append(hm)
    return h_full


def _hcl_loss(h_full: list, lam: float = 1.0) -> torch.Tensor:
    """
    Hierarchical Contrastive Learning loss.
    Representations are L2-normalised before computing phi.

    Supports per-structure latent dimensions (r_list[k] may differ across k).

    Parameters
    ----------
    h_full : list[list[Tensor]]  h_full[m][k] : [B, r_list[k]]
    lam    : regularisation weight

    Returns
    -------
    loss : scalar Tensor
    """
    B      = h_full[0][0].size(0)
    device = h_full[0][0].device

    # L2-normalise all representations
    h_norm = [
        [F.normalize(h_full[m][k], p=2, dim=-1) for k in range(_NUM_STRUCTURES)]
        for m in range(_NUM_MODALITIES)
    ]

    # Build phi matrix: sum of outer products over all (structure, modality-pair)
    # For each structure k, all modalities share the same r_list[k], so the
    # matrix multiply h_norm[m][k] @ h_norm[mp][k].T is always valid.
    phi = torch.zeros(B, B, device=device)
    for k in range(_NUM_STRUCTURES):
        for m in range(_NUM_MODALITIES):
            for mp in range(_NUM_MODALITIES):
                phi = phi + h_norm[m][k] @ h_norm[mp][k].T

    diag = torch.diag(phi)
    pos  = diag.sum() / (2 * B)
    neg  = (phi.sum() - diag.sum()) / (2 * B * (B - 1))

    # Representation regulariser
    # When r_list[k] != r_list[kp], we use the minimum dimension to compute
    # the overlap between representations of the same modality across
    # different structures.
    reg = torch.tensor(0.0, device=device)
    for m in range(_NUM_MODALITIES):
        for k in range(_NUM_STRUCTURES):
            for kp in range(_NUM_STRUCTURES):
                rk  = h_norm[m][k].size(1)
                rkp = h_norm[m][kp].size(1)
                r_min = min(rk, rkp)
                reg = reg + (h_norm[m][k][:, :r_min] * h_norm[m][kp][:, :r_min]).sum()

    return neg - pos + lam * reg

def _average_hierarchical_outputs(h_full: list) -> dict:
    """
    Average representations across modalities for each structure.

    Returns
    -------
    h_avg : dict[int, Tensor]   h_avg[k] : [B, r]
    """
    h_avg = {}
    for k, subset in enumerate(_STRUCTURE_LIST):
        participating = [h_full[m][k] for m in subset]
        h_avg[k] = torch.stack(participating, dim=0).mean(dim=0)
    return h_avg


# ---------------------------------------------------------------------------
# HCLFusion: FusionModule implementation
# ---------------------------------------------------------------------------
class HCLFusion(FusionModule):
    """
    Hierarchical Contrastive Learning fusion module.

    Now supports per-structure latent dimensions via r_list.
    r_list[k] specifies the latent dim for structure k (k = 0..6).

    Parameters
    ----------
    input_dims  : list of input dims per modality, e.g. [512, 512, 512]
    hidden_dims : hidden layer sizes for each StructureEncoder FFN
    r_list      : list of 7 ints, latent dimension for each structure
    """

    def __init__(
        self,
        input_dims: List[int],
        hidden_dims: List[int],
        r_list: List[int],
    ):
        assert len(r_list) == _NUM_STRUCTURES, (
            f"r_list must have {_NUM_STRUCTURES} elements, got {len(r_list)}"
        )
        # out_dim = sum of all per-structure dims
        total_r = sum(r_list)
        super().__init__(out_dim=total_r, has_pretrain=True)

        self.r_list         = r_list
        self.num_modalities = _NUM_MODALITIES
        self.num_levels     = _NUM_MODALITIES  # one encoder per hierarchy level

        # For each modality m and each level s, we need the output dim.
        # Level s corresponds to structures of size (M - s).
        # Multiple structures can share the same level, but each modality
        # only has ONE encoder per level. The encoder output dim for
        # modality m at level s must be a single value.
        #
        # Strategy: for each (m, s), use the MAX r among all structures
        # at that level where modality m participates. Then when expanding
        # to the 7 structures, slice or zero-pad to match r_list[k].
        self._encoder_r = {}  # (m, s) -> int
        for m in range(_NUM_MODALITIES):
            for s in range(self.num_levels):
                # Find all structures at level s where modality m participates
                candidates = []
                for k in range(_NUM_STRUCTURES):
                    if k in _STRUCTURE_MASK[m] and _STRUCTURE_MASK[m][k] == s:
                        candidates.append(r_list[k])
                if candidates:
                    self._encoder_r[(m, s)] = max(candidates)
                else:
                    self._encoder_r[(m, s)] = r_list[0]  # fallback, won't be used

        # Build encoders: num_modalities x num_levels
        self.encoders = nn.ModuleList([
            nn.ModuleList([
                StructureEncoder(
                    input_dim=input_dims[m],
                    hidden_dims=hidden_dims,
                    output_dim=self._encoder_r[(m, s)],
                )
                for s in range(self.num_levels)
            ])
            for m in range(self.num_modalities)
        ])

    def _run_encoders(self, x_list: List[torch.Tensor]) -> list:
        """
        Run all per-modality, per-level encoders and expand to 7 structures.

        For each structure k and modality m:
          - If m participates in k at level s, take encoder output h_raw[m][s]
            and slice it to r_list[k] dims (since encoder may output more).
          - Otherwise, fill with zeros of size r_list[k].

        Returns
        -------
        h_full : list[list[Tensor]]  h_full[m][k] : [B, r_list[k]]
        """
        B      = x_list[0].size(0)
        device = x_list[0].device
        dtype  = x_list[0].dtype

        # Run all encoders
        h_raw = [
            [self.encoders[m][s](x_list[m]) for s in range(self.num_levels)]
            for m in range(self.num_modalities)
        ]

        # Expand to 7 structures
        h_full = []
        for m in range(_NUM_MODALITIES):
            hm = []
            for k in range(_NUM_STRUCTURES):
                rk = self.r_list[k]
                if k in _STRUCTURE_MASK[m]:
                    s = _STRUCTURE_MASK[m][k]
                    raw = h_raw[m][s]  # [B, encoder_r]
                    hm.append(raw[:, :rk])  # slice to r_list[k]
                else:
                    hm.append(torch.zeros(B, rk, device=device, dtype=dtype))
            h_full.append(hm)
        return h_full

    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse modality encodings via HCL structure averaging.

        Parameters
        ----------
        x_list : list of [B, d_m] tensors, one per modality

        Returns
        -------
        [B, sum(r_list)] tensor
        """
        h_full = self._run_encoders(x_list)
        h_avg  = _average_hierarchical_outputs(h_full)   # dict k -> [B, r_list[k]]
        return torch.cat([h_avg[k] for k in range(_NUM_STRUCTURES)], dim=1)

    def compute_pretrain_loss(
        self,
        x_list: List[torch.Tensor],
        hcl_lam: float = 1.0,
        **kwargs,
    ) -> Optional[dict]:
        """
        Compute HCL pretraining loss.

        Parameters
        ----------
        x_list  : list of [B, d_m] tensors
        hcl_lam : regularisation weight for the HCL loss

        Returns
        -------
        dict:
            total : HCL loss (scalar Tensor)
            hcl   : HCL loss value (float)
        """
        h_full   = self._run_encoders(x_list)
        loss_hcl = _hcl_loss(h_full, lam=hcl_lam)

        return {
            "total": loss_hcl,
            "hcl"  : loss_hcl.item(),
        }