import torch
from torch.utils.data import DataLoader
import numpy as np
@torch.no_grad()
def extract_all_representations(model, loader, device):
    """
    Freeze RNN encoders, extract per-patient modality representations.

    Returns
    -------
    x_code_all  : torch.Tensor [N, hidden_size]
    x_note_all  : torch.Tensor [N, hidden_size]
    x_lab_all   : torch.Tensor [N, hidden_size]
    labels_all  : torch.Tensor [N]
    demos_all   : torch.Tensor [N, demo_dim]
    pids_all    : list of patient_ids
    """
    from utils import move_batch_to_device
    model.eval()
    x_codes, x_notes, x_labs = [], [], []
    labels, demos, pids = [], [], []

    from tqdm import tqdm
    for batch in tqdm(loader, desc="Extracting representations", leave=True):
        batch  = move_batch_to_device(batch, device)

        med    = batch["medical"]
        x_code = model.code_enc(med["codes"], med["mask"])

        if batch["notes"]:
            x_note = model.note_enc(
                batch["notes"]["embeddings"], batch["notes"]["mask"]
            )
        else:
            x_note = torch.zeros_like(x_code)

        lab   = batch["labs"]
        x_lab = model.lab_enc(
            lab["values"], lab["times"], lab["types"].long(), lab["mask"]
        )

        x_codes.append(x_code.cpu())
        x_notes.append(x_note.cpu())
        x_labs.append(x_lab.cpu())
        labels.append(batch["label"].cpu())
        demos.append(batch["demographic"].cpu())
        pids.extend(batch["patient_id"])

    return (
        torch.cat(x_codes, dim=0),
        torch.cat(x_notes, dim=0),
        torch.cat(x_labs,  dim=0),
        torch.cat(labels,  dim=0),
        torch.cat(demos,   dim=0),
        pids,
    )

def run_slide_on_full_data(model, x_code, x_note, x_lab, device):
    """
    Run SLIDE factorization on the full training representation matrix.
    Should only be called on training data.

    Parameters
    ----------
    x_code, x_note, x_lab : [N, hidden_size] CPU tensors

    Returns
    -------
    U : [N, 7*r] CPU tensor — score matrix for training samples
    V : [p, 7*r] CPU tensor — loading matrix to be reused for val/test projection
    """
    model.fusion._ensure_S(device)

    x_list = [
        x_code.to(device),
        x_note.to(device),
        x_lab.to(device),
    ]
    x_centered = [xi - xi.mean(dim=0, keepdim=True) for xi in x_list]

    U, V = _slide_fit_from_fusion(model.fusion, x_centered, device)
    return U.cpu(), V.cpu()
def project_with_V(
    x_code: torch.Tensor,
    x_note: torch.Tensor,
    x_lab: torch.Tensor,
    V_train: torch.Tensor,
    col_means: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Project out-of-sample representations onto the training SLIDE space.
    Uses the closed-form least-squares projection: U = X @ V @ (V^T @ V)^{-1}

    Parameters
    ----------
    x_code, x_note, x_lab : [N, hidden_size] CPU tensors for val or test split
    V_train   : [p, 7*r] CPU tensor — loading matrix saved from training SLIDE fit
    col_means : [p]      CPU tensor — per-feature column means from training data,
                                      used to apply the same centering as training
    device    : torch.device

    Returns
    -------
    U_proj : [N, 7*r] CPU tensor
    """
    X = torch.cat([x_code, x_note, x_lab], dim=1).to(device)  # [N, p]
    V = V_train.to(device)                                     # [p, 7*r]
    mu = col_means.to(device)                                  # [p]

    # Apply the same column centering used during training
    X_centered = X - mu.unsqueeze(0)

    # Closed-form projection: U = X_centered @ V @ (V^T V)^{-1}
    VtV     = V.T @ V                                          # [7*r, 7*r]
    VtV_inv = torch.linalg.pinv(VtV)                          # [7*r, 7*r]
    U_proj  = X_centered @ V @ VtV_inv                        # [N, 7*r]

    return U_proj.cpu()

def _slide_fit_from_fusion(fusion_module, x_centered, device):
    """
    Call internal SLIDE fit using the fusion module's S buffer and config.

    Returns
    -------
    U : [B, total_r]
    V : [p, total_r]
    """
    from model.fusion.SLIDE import _slide_fit
    return _slide_fit(
        X_list  = x_centered,
        S       = fusion_module.S,
        total_r = fusion_module.total_r,
        max_iter= fusion_module.max_iter,
        tol     = fusion_module.tol,
    )
def run_hnn_on_full_data(model, x_code, x_note, x_lab, device):
    """
    Run HNN (Algorithm 1 + Algorithm 2) on the full training representation matrix.
    Stores loading matrices inside model.fusion for later out-of-sample projection.

    Parameters
    ----------
    x_code, x_note, x_lab : [N_train, hidden_size] CPU tensors

    Returns
    -------
    U_train : [N_train, 7*r] CPU tensor — per-structure score matrix for training
    """
    x_list = [
        x_code.to(device),
        x_note.to(device),
        x_lab.to(device),
    ]
    return model.fusion.fit_transform(x_list)   # returns CPU tensor [N_train, 7*r]

def project_hnn_with_loadings(
    x_code: torch.Tensor,
    x_note: torch.Tensor,
    x_lab: torch.Tensor,
    fusion_module,
    device: torch.device,
) -> torch.Tensor:
    """
    Project out-of-sample representations onto the training HNN space.
    Uses loading matrices stored in fusion_module by fit_transform().

    Parameters
    ----------
    x_code, x_note, x_lab : [N, hidden_size] CPU tensors for val or test
    fusion_module          : HNNFusion instance (already fit on training data)
    device                 : torch.device

    Returns
    -------
    scores : [N, 7*r] CPU tensor
    """
    x_list = [
        x_code.to(device),
        x_note.to(device),
        x_lab.to(device),
    ]
    return fusion_module.transform(x_list)      # returns CPU tensor [N, 7*r]


def make_repr_loader(
    x_code: torch.Tensor,
    x_note: torch.Tensor,
    x_lab: torch.Tensor,
    labels: torch.Tensor,
    demos: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """
    Wrap pre-extracted representations into a DataLoader.
    Each batch yields a dict: x_code, x_note, x_lab, label, demographic.
    """
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(x_code, x_note, x_lab, labels, demos)

    def collate(batch):
        xc, xn, xl, lbl, dem = zip(*batch)
        return {
            "x_code"     : torch.stack(xc),
            "x_note"     : torch.stack(xn),
            "x_lab"      : torch.stack(xl),
            "label"      : torch.stack(lbl),
            "demographic": torch.stack(dem),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

def run_mmfl_on_full_data(
    fusion_module,
    x_code: torch.Tensor,
    x_note: torch.Tensor,
    x_lab: torch.Tensor,
    labels: torch.Tensor,
) -> np.ndarray:
    """
    Run MMFL fit on the full training data.
    Labels are required because MMFL is a supervised decomposition.

    Parameters
    ----------
    fusion_module : MMFLFusion instance
    x_code, x_note, x_lab : [N_train, hidden_size] CPU tensors
    labels : [N_train] CPU tensor, binary {0, 1}

    Returns
    -------
    U_train : np.ndarray [N_train, total_r]
    """
    import numpy as np
    x_list = [x_code, x_note, x_lab]
    return fusion_module.fit(x_list, labels)


def project_mmfl(
    fusion_module,
    x_code: torch.Tensor,
    x_note: torch.Tensor,
    x_lab: torch.Tensor,
) -> np.ndarray:
    """
    Project out-of-sample data onto the training MMFL space.

    Parameters
    ----------
    fusion_module : MMFLFusion instance (already fit on training data)
    x_code, x_note, x_lab : [N, hidden_size] CPU tensors

    Returns
    -------
    U_new : np.ndarray [N, total_r]
    """
    x_list = [x_code, x_note, x_lab]
    return fusion_module.transform(x_list)
def run_jive_on_full_data(
    fusion_module,
    x_code: torch.Tensor,
    x_note: torch.Tensor,
    x_lab: torch.Tensor,
) -> np.ndarray:
    """
    Run JIVE fit on the full training representation matrix.

    Parameters
    ----------
    fusion_module : JIVEFusion instance
    x_code, x_note, x_lab : [N_train, hidden_size] CPU tensors

    Returns
    -------
    U_train : np.ndarray [N_train, total_r]
    """
    x_list = [x_code, x_note, x_lab]
    return fusion_module.fit(x_list)


def project_jive(
    fusion_module,
    x_code: torch.Tensor,
    x_note: torch.Tensor,
    x_lab: torch.Tensor,
) -> np.ndarray:
    """
    Project out-of-sample data onto the training JIVE space.

    Parameters
    ----------
    fusion_module : JIVEFusion instance (already fit on training data)
    x_code, x_note, x_lab : [N, hidden_size] CPU tensors

    Returns
    -------
    U_new : np.ndarray [N, total_r]
    """
    x_list = [x_code, x_note, x_lab]
    return fusion_module.transform(x_list)
def run_sjive_on_full_data(
    fusion_module,
    x_code: torch.Tensor,
    x_note: torch.Tensor,
    x_lab: torch.Tensor,
    labels: torch.Tensor,
) -> np.ndarray:
    """
    Run sJIVE fit on the full training representation matrix.
    Labels are required because sJIVE is a supervised decomposition.

    Parameters
    ----------
    fusion_module : sJIVEFusion instance
    x_code, x_note, x_lab : [N_train, hidden_size] CPU tensors
    labels : [N_train] CPU tensor, binary {0, 1}

    Returns
    -------
    U_train : np.ndarray [N_train, total_r]
    """
    x_list = [x_code, x_note, x_lab]
    return fusion_module.fit(x_list, labels)


def project_sjive(
    fusion_module,
    x_code: torch.Tensor,
    x_note: torch.Tensor,
    x_lab: torch.Tensor,
) -> np.ndarray:
    """
    Project out-of-sample data onto the training sJIVE space.

    Parameters
    ----------
    fusion_module : sJIVEFusion instance (already fit on training data)
    x_code, x_note, x_lab : [N, hidden_size] CPU tensors

    Returns
    -------
    U_new : np.ndarray [N, total_r]
    """
    x_list = [x_code, x_note, x_lab]
    return fusion_module.transform(x_list)