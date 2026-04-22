import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
def _pad_1d(arr: np.ndarray, max_len: int, pad_val):
    n = arr.shape[0]
    if n >= max_len:
        out  = arr[-max_len:]
        mask = np.ones(max_len, bool)
    else:
        pad  = max_len - n
        out  = np.concatenate([arr, np.full(pad, pad_val, arr.dtype)], axis=0)
        mask = np.zeros(max_len, bool); mask[:n] = True
    return out, mask

def _pad_2d(mat: np.ndarray, max_len: int):
    n, d = mat.shape
    if n >= max_len:
        out  = mat[-max_len:]
        mask = np.ones(max_len, bool)
    else:
        pad  = max_len - n
        out  = np.vstack([mat, np.zeros((pad, d), mat.dtype)])
        mask = np.zeros(max_len, bool); mask[:n] = True
    return out, mask

def _pad_nd_firstdim(arr: np.ndarray, max_len: int, target_feat_shape: Tuple[int, ...], pad_val: float = 0.0):
    """
    将形状 (n, *feat) 的数组在首维补到 max_len；内部特征维若与 target_feat_shape 不同，
    会在右下方向做“裁剪/零填充”以适配 target_feat_shape。
    返回: (out, mask) 其中 out.shape == (max_len, *target_feat_shape)
    """
    # 统一到至少 1 维（N 在 axis=0）
    if arr.ndim == 0:
        arr = np.zeros((0,) + target_feat_shape, dtype=np.float32)

    # 若 arr 是单个样本而非序列，比如 (C,H,W) / (C,F,T)，则视作 1 个事件
    if arr.ndim == len(target_feat_shape):
        arr = arr[None, ...]  # 变成 (1, *feat)

    n = arr.shape[0]
    out = np.full((max_len, *target_feat_shape), pad_val, dtype=arr.dtype)
    mask = np.zeros(max_len, dtype=bool)

    if n == 0:
        return out, mask

    # 只取尾部最近 max_len 个
    take = min(n, max_len)
    src = arr[-take:]  # (take, *src_feat)

    # 将 src“对齐”到 target_feat_shape：按各维做 min 裁剪，剩余区间自动保持零填充
    slices_dst = [slice(0, take)]
    slices_src = [slice(0, take)]
    for d in range(len(target_feat_shape)):
        sd = src.shape[1 + d]   # src feat dim
        td = target_feat_shape[d]
        use = min(sd, td)
        slices_dst.append(slice(0, use))
        slices_src.append(slice(0, use))

    out[tuple(slices_dst)] = src[tuple(slices_src)]
    mask[:take] = True
    return out, mask

def ehr_collate_fn(
    batch: List[Dict],
    fixed_lengths: Optional[Dict[str, int]] = None,
    *,
    digit_num_classes: int = 10,
    make_onehot_label: bool = True
):
    batch = [b for b in batch if b]
    if not batch:
        return {}

    if fixed_lengths is None:
        fixed_lengths = {
            "medical_events": 512,
            "note_events"   : 20,
            "lab_events"    : 512,
            "cxr_events"    : 50,
            "mimic_events"  : 8,   # 新增：每个样本最多保留多少个 mimic 事件（visit）
        }

    # demographics & label
    out = {
        "demographic": torch.stack([b["demographic"] for b in batch]).contiguous(),
        "label"      : torch.stack([b["label"] for b in batch]).contiguous(),
        "patient_id" : [b["patient_id"] for b in batch],
    }

    # ----------------- 额外输出：digit 的 one-hot -----------------
    if make_onehot_label:
        # 保持 float（便于与 BCE/CE 以外损失组合），若你只在 CE 下用可保留 long→one_hot→float 的流程
        out["label_onehot"] = F.one_hot(
            out["label"].to(torch.long),
            num_classes=digit_num_classes
        ).float().contiguous()

    # -------------------------- medical --------------------------
    max_med = fixed_lengths["medical_events"]
    codes, times, types, m_masks, m_len = [], [], [], [], []
    for b in batch:
        c, t, ty = b["medical_codes"], b["medical_times"], b["medical_types"]
        pc, mc = _pad_1d(c, max_med, 0)
        pt, _  = _pad_1d(t, max_med, 0.0)
        pty, _ = _pad_1d(ty, max_med, 99)
        codes.append(pc); times.append(pt); types.append(pty); m_masks.append(mc)
        m_len.append(len(c))
    out["medical"] = {
        "codes"  : torch.from_numpy(np.stack(codes)).contiguous(),
        "times"  : torch.from_numpy(np.stack(times)).float().contiguous(),
        "types"  : torch.from_numpy(np.stack(types)).contiguous(),
        "mask"   : torch.from_numpy(np.stack(m_masks)).contiguous(),
        "lengths": torch.as_tensor(m_len, dtype=torch.long).contiguous(),
    }

    # --------------------------- notes ---------------------------
    has_note = any(b["note_embs"].size != 0 for b in batch)
    if not has_note:
        out["notes"] = {}
    else:
        max_note = fixed_lengths["note_events"]
        embs, ntimes, ntypes, nmask, nlen = [], [], [], [], []
        emb_dim  = next(b["note_embs"].shape[1] for b in batch
                        if b["note_embs"].ndim == 2 and b["note_embs"].shape[0] > 0)

        for b in batch:
            e, t, ty = b["note_embs"], b["note_times"], b["note_types"]

            if e.ndim != 2 or e.shape[1] != emb_dim:
                e = np.zeros((0, emb_dim), dtype=np.float32)

            pe, me = _pad_2d(e, max_note)

            if t.ndim != 1:
                t = np.zeros((0,), dtype=np.float32)
            pt, _  = _pad_1d(t, max_note, 0.0)

            if ty.ndim != 1:
                ty = np.zeros((0,), dtype=np.int32)
            pty, _ = _pad_1d(ty, max_note, 99)

            embs.append(pe); ntimes.append(pt); ntypes.append(pty); nmask.append(me)
            nlen.append(e.shape[0])

        out["notes"] = {
            "embeddings": torch.from_numpy(np.stack(embs)).float().contiguous(),
            "times"     : torch.from_numpy(np.stack(ntimes)).float().contiguous(),
            "types"     : torch.from_numpy(np.stack(ntypes)).contiguous(),
            "mask"      : torch.from_numpy(np.stack(nmask)).contiguous(),
            "lengths"   : torch.as_tensor(nlen, dtype=torch.long).contiguous(),
        }

    # ---------------------------- labs ---------------------------
    max_lab = fixed_lengths["lab_events"]
    ltimes, lvals, ltypes, lmask, llen = [], [], [], [], []
    for b in batch:
        t, v, ty = b["lab_times"], b["lab_vals"], b["lab_types"]
        pt, mt = _pad_1d(t, max_lab, 0.0)
        pv, _  = _pad_1d(v, max_lab, 0.0)
        pty, _ = _pad_1d(ty, max_lab, 99)
        ltimes.append(pt); lvals.append(pv); ltypes.append(pty); lmask.append(mt)
        llen.append(len(t))
    out["labs"] = {
        "times"  : torch.from_numpy(np.stack(ltimes)).float().contiguous(),
        "values" : torch.from_numpy(np.stack(lvals)).float().contiguous(),
        "types"  : torch.from_numpy(np.stack(ltypes)).contiguous(),
        "mask"   : torch.from_numpy(np.stack(lmask)).contiguous(),
        "lengths": torch.as_tensor(llen, dtype=torch.long).contiguous(),
    }

    # ----------------------------- cxr ---------------------------
    has_cxr = any(b["cxr_embs"].size != 0 for b in batch)
    if not has_cxr:
        out["cxr"] = {}
    else:
        max_cxr = fixed_lengths["cxr_events"]
        cembs, ctimes, ctypes, cmask, clen = [], [], [], [], []
        cxr_dim = next(b["cxr_embs"].shape[1] for b in batch
                       if b["cxr_embs"].ndim == 2 and b["cxr_embs"].shape[0] > 0)

        for b in batch:
            e, t, ty = b["cxr_embs"], b["cxr_times"], b["cxr_types"]
            if e.ndim != 2 or e.shape[1] != cxr_dim:
                e = np.zeros((0, cxr_dim), dtype=np.float32)

            pe, me = _pad_2d(e, max_cxr)
            pt, _  = _pad_1d(t, max_cxr, 0.0)
            pty, _ = _pad_1d(ty, max_cxr, 99)
            cembs.append(pe); ctimes.append(pt); ctypes.append(pty); cmask.append(me)
            clen.append(e.shape[0])

        out["cxr"] = {
            "embeddings": torch.from_numpy(np.stack(cembs)).float().contiguous(),
            "times"     : torch.from_numpy(np.stack(ctimes)).float().contiguous(),
            "types"     : torch.from_numpy(np.stack(ctypes)).contiguous(),
            "mask"      : torch.from_numpy(np.stack(cmask)).contiguous(),
            "lengths"   : torch.as_tensor(clen, dtype=torch.long).contiguous(),
        }

    # --------------------------- mimic ---------------------------
    has_mimic_img = any(b["mimic_image"].size != 0 for b in batch)
    has_mimic_aud = any(b["mimic_audio"].size != 0 for b in batch)

    if not (has_mimic_img or has_mimic_aud):
        out["mimic"] = {}
    else:
        max_mimic = fixed_lengths["mimic_events"]

        # 基准形状（feat shape，不含首维 N）
        # 图像：优先从非空样本提取 (C,H,W)，否则默认 MNIST 风格 (1,28,28)
        if has_mimic_img:
            try:
                img_feat_shape = next(
                    (x["mimic_image"].shape[1:] if x["mimic_image"].ndim >= 3 and x["mimic_image"].shape[0] > 0
                     else x["mimic_image"].shape if x["mimic_image"].ndim == 3 else None)
                    for x in batch if x["mimic_image"].size != 0
                )
                if img_feat_shape is None or len(img_feat_shape) < 3:
                    img_feat_shape = (1, 28, 28)
            except StopIteration:
                img_feat_shape = (1, 28, 28)
        else:
            img_feat_shape = None

        # 音频：从非空样本提取 (C,F,T)，否则给一个安全默认 (1, 20, 20)
        if has_mimic_aud:
            try:
                aud_feat_shape = next(
                    (x["mimic_audio"].shape[1:] if x["mimic_audio"].ndim >= 3 and x["mimic_audio"].shape[0] > 0
                     else x["mimic_audio"].shape if x["mimic_audio"].ndim == 3 else None)
                    for x in batch if x["mimic_audio"].size != 0
                )
                if aud_feat_shape is None or len(aud_feat_shape) < 3:
                    aud_feat_shape = (1, 20, 20)
            except StopIteration:
                aud_feat_shape = (1, 20, 20)
        else:
            aud_feat_shape = None

        # --- 逐样本补齐 ---
        img_list, img_mask_list, img_len = [], [], []
        aud_list, aud_mask_list, aud_len = [], [], []

        for b in batch:
            # image
            if has_mimic_img:
                arr_img = b["mimic_image"]
                if not isinstance(arr_img, np.ndarray):
                    arr_img = np.asarray(arr_img, dtype=np.float32)
                if arr_img.size == 0:
                    arr_img = np.zeros((0,) + img_feat_shape, dtype=np.float32)
                pi, mi = _pad_nd_firstdim(arr_img, max_mimic, img_feat_shape, pad_val=0.0)
                img_list.append(pi); img_mask_list.append(mi)
                img_len.append(0 if arr_img.ndim == 0 else (arr_img.shape[0] if arr_img.ndim >= 4 else 1))

            # audio
            if has_mimic_aud:
                arr_aud = b["mimic_audio"]
                if not isinstance(arr_aud, np.ndarray):
                    arr_aud = np.asarray(arr_aud, dtype=np.float32)
                if arr_aud.size == 0:
                    arr_aud = np.zeros((0,) + aud_feat_shape, dtype=np.float32)
                pa, ma = _pad_nd_firstdim(arr_aud, max_mimic, aud_feat_shape, pad_val=0.0)
                aud_list.append(pa); aud_mask_list.append(ma)
                aud_len.append(0 if arr_aud.ndim == 0 else (arr_aud.shape[0] if arr_aud.ndim >= 4 else 1))

        mimic_out = {}
        if has_mimic_img:
            mimic_out["images"]   = torch.from_numpy(np.stack(img_list)).float().contiguous()    # (B, L, C, H, W)
            mimic_out["img_mask"] = torch.from_numpy(np.stack(img_mask_list)).contiguous()       # (B, L)
            mimic_out["img_lens"] = torch.as_tensor(img_len, dtype=torch.long).contiguous()

        if has_mimic_aud:
            mimic_out["audio"]    = torch.from_numpy(np.stack(aud_list)).float().contiguous()    # (B, L, C, F, T)
            mimic_out["aud_mask"] = torch.from_numpy(np.stack(aud_mask_list)).contiguous()       # (B, L)
            mimic_out["aud_lens"] = torch.as_tensor(aud_len, dtype=torch.long).contiguous()

        out["mimic"] = mimic_out

    return out
