import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle

##############################################################################
#               ----------  一些工具函数  ----------
##############################################################################
def one_hot(idx: int, length: int) -> List[int]:
    vec = [0] * length
    vec[idx] = 1
    return vec

def log1p_or_zero(x):
    try:
        return np.log1p(float(x))
    except (TypeError, ValueError):
        return 0.0

# #############################################################################
#               ----------  加速版 Dataset ----------
# #############################################################################
class EHRDataset(Dataset):

    # -------------- 常量 --------------
    _TYPE2ID_MED  = {"diagnosis": 0, "medication": 1, "drg": 2}
    _TYPE2ID_NOTE = {"discharge": 3, "radiology": 4}
    _CXR_ID       = 5
    _DEATH_CODE_EXCLUDE_SET = {7296, 4225}

    _MODAL_CFG = {
        "image_only":  ({"image"}, {"note", "code"}),
        "note_only":   ({"note"},  {"image", "code"}),
        "code_only":   ({"code"},  {"image", "note"}),
        "image+note":  ({"image", "note"}, {"code"}),
        "image+code":  ({"image", "code"}, {"note"}),
        "note+code":   ({"note", "code"}, {"image"}),
        "all_exist":   ({"note", "code"}, set()),   # only note and code required; image is always dropped
    }

    def __init__(
        self,
        pkl_paths: List[str],
        cxr_embeddings_path: str,
        mappings: Dict[str, Dict[str, int]],
        index_set: Set[int],
        task: str,
        missing_mode: Optional[str] = None,
        strict_match = False,
        missing_prob: float = 0.0,
        label_file: Optional[str] = None,
        label_task: Optional[str] = None,
        label_task_type: Optional[str] = None,
        label_visit_policy: Optional[str] = None,
        label_pid_col: str = "patient_id",
        exclude_death_codes_from_code_branch: bool = False,
    ):
        super().__init__()
        if missing_mode is not None and missing_mode.lower() not in self._MODAL_CFG:
            raise ValueError(
                f"missing_mode={missing_mode!r} is invalid, options: {list(self._MODAL_CFG)}"
            )
        self.label_file = label_file
        self.label_task = label_task
        self.label_task_type = None if label_task_type is None else label_task_type.lower()
        self.label_visit_policy = None if label_visit_policy is None else label_visit_policy.lower()
        self.label_pid_col = label_pid_col
        self.use_external_label = bool(label_file and label_task)
        if self.use_external_label:
            if self.label_task_type not in {"classification", "regression"}:
                raise ValueError("label_task_type must be 'classification' or 'regression'")
            if self.label_visit_policy not in {"all_visits", "history_before_last"}:
                raise ValueError("label_visit_policy must be 'all_visits' or 'history_before_last'")
        elif any(x is not None for x in [label_file, label_task, label_task_type, label_visit_policy]):
            raise ValueError(
                "label_file, label_task, label_task_type, and label_visit_policy must be provided together"
            )
        self.missing_mode  = None if missing_mode is None else missing_mode.lower()
        self.task          = task
        self.index_set     = index_set
        self.mappings      = mappings
        self.strict_match  = strict_match
        self.missing_prob  = missing_prob
        self.exclude_code_indices = (
            set(self._DEATH_CODE_EXCLUDE_SET)
            if exclude_death_codes_from_code_branch
            else set()
        )
        self.all_pkl_paths = pkl_paths  # store full path list for chunked loading
        self.stats = {
            "total_patients": 0,
            "kept": 0,
            "skipped_short_history": 0,
            "skipped_missing_label": 0,
            "skipped_nan_label": 0,
        }

        self.label_lookup = None
        if self.use_external_label:
            label_df = self._load_label_df(label_file)
            if self.label_pid_col not in label_df.columns:
                raise ValueError(f"{self.label_pid_col!r} not found in label file: {label_file}")
            if self.label_task not in label_df.columns:
                raise ValueError(f"{self.label_task!r} not found in label file: {label_file}")
            label_df = label_df[[self.label_pid_col, self.label_task]].copy()
            label_df[self.label_pid_col] = label_df[self.label_pid_col].astype(str)
            label_df = label_df.drop_duplicates(subset=[self.label_pid_col], keep="last")
            self.label_lookup = dict(zip(label_df[self.label_pid_col], label_df[self.label_task]))

        # ------- 1) load CXR embeddings -------
        self.cxr_dict: Dict[Tuple[int, float], List[dict]] = {}
        cxr_df = pd.read_pickle(cxr_embeddings_path)
        for row in cxr_df.itertuples(index=False):
            self.cxr_dict.setdefault((row.subject_id, row.visit), []).append(
                {"embedding": row.embedding, "relative_time": row.relative_time}
            )

        # ------- 2) build demographic max-category lookup -------
        self._max_cat = {
            ft: max(v for v in mappings[ft].values() if isinstance(v, int))
            for ft in ["gender", "race", "marital_status", "language"]
        }

        # data is empty until load_chunk() is called
        self.data: List[dict] = []
        print(f"[EHRDataset] initialized — missing_mode={self.missing_mode}, "
            f"strict_match={self.strict_match}, total pkl files={len(pkl_paths)}")

    @staticmethod
    def _load_label_df(label_file: str) -> pd.DataFrame:
        ext = os.path.splitext(label_file)[1].lower()
        if ext == ".csv":
            return pd.read_csv(label_file)
        if ext in {".parquet", ".pq"}:
            return pd.read_parquet(label_file)
        if ext in {".pkl", ".pickle"}:
            return pd.read_pickle(label_file)
        raise ValueError(f"Unsupported label file extension: {ext}")

    def load_chunk(self, pkl_paths: List[str]) -> List[dict]:
        """
        Load and preprocess patients from a list of pkl files.
        Returns the processed sample list WITHOUT modifying self.data.
        Safe to call from a background thread.
        """
        chunk_data = []
        for pkl_path in pkl_paths:
            with open(pkl_path, "rb") as f:
                raw_patients = pickle.load(f)
            for pat in raw_patients:
                self.stats["total_patients"] += 1
                cached = self._preprocess_patient(pat)
                if cached is not None:
                    chunk_data.append(cached)
                    self.stats["kept"] += 1
        return chunk_data

    def replace_data(self, new_data: List[dict]):
        """
        Replace the current in-memory samples with new_data.
        Call this on the main thread after the background load finishes.
        """
        self.data = new_data
        print(f"[EHRDataset] chunk replaced — {len(self.data)} samples loaded")
    # -------------- 核心：预处理单个患者 --------------
    @property
    def task_type(self) -> str:
        return self.label_task_type if self.use_external_label else "classification"

    def _get_target_visits(self, visits):
        if self.use_external_label:
            if self.label_visit_policy == "all_visits":
                return visits
            return visits[:-1]
        if self.task != 'mimic':
            return visits[:-1]
        return visits

    def _resolve_label(self, patient: dict, visits):
        if self.use_external_label:
            patient_id = str(patient["patient_id"])
            if patient_id not in self.label_lookup:
                self.stats["skipped_missing_label"] += 1
                return None
            label = self.label_lookup[patient_id]
            if pd.isna(label):
                self.stats["skipped_nan_label"] += 1
                return None
            return label

        if self.task == "readmission":
            return visits[-2].get("30_days_readmission")
        if self.task == "mortality":
            return visits[-1].get("in_hospital_mortality")
        if self.task == "next_visit_diseases":
            return visits[-2].get("next_visit_diseases")
        if self.task == 'mimic':
            return visits[0].get("digit_label")
        raise NotImplementedError(f"Unsupported task: {self.task}")

    def _preprocess_patient(self, patient: dict) -> Optional[dict]:
        visits = patient["visits"]
        min_visits = 1 if (self.use_external_label and self.label_visit_policy == "all_visits") else 2
        if len(visits) < min_visits:
            self.stats["skipped_short_history"] += 1
            return None

        # ------- 标签 --------
        label = self._resolve_label(patient, visits)
        if label is None:
            if not self.use_external_label:
                self.stats["skipped_missing_label"] += 1
            return None

        demo_feat = self._build_demo_vec(patient.get("demographics", {}))
        target_visits = self._get_target_visits(visits)

        med_codes, med_times, med_types = self._collect_medical(target_visits)
        note_embs, note_times, note_types = self._collect_notes(target_visits)
        lab_times, lab_vals, lab_types   = self._collect_labs(target_visits)
        cxr_embs, cxr_times, cxr_types   = self._collect_cxr(patient["patient_id"], target_visits)

        mimic_images, mimic_audio = self._collect_mimic(visits)

        # ---------- 按缺失模式保留 / 舍弃 ----------
        if self.missing_mode is not None:
            required, drop = self._MODAL_CFG[self.missing_mode]
            has = {
                "image": len(cxr_embs)  > 0,
                "note":  len(note_embs) > 0,
                "code":  len(med_codes) > 0,
            }
            if self.strict_match:  # ★ 新增：完全匹配
                # 必须恰好拥有 required，且其他模态必须缺失
                if not (all(has[m] for m in required) and
                        all(not has[m] for m in drop)):
                    return None
                # 严格模式下不需要再“裁剪多余模态”，因为已经保证它们本来就不存在
            else:  # ★ 旧逻辑：包含即可
                if not all(has[m] for m in required):
                    return None
                # 去掉多余模态
                if "image" in drop:
                    cxr_embs = cxr_times = cxr_types = np.empty((0,), dtype=np.float32)
                if "note" in drop:
                    note_embs = note_times = note_types = np.empty((0,), dtype=np.float32)
                if "code" in drop:
                    med_codes = med_times = med_types = np.empty((0,), dtype=np.int32)

        return {
            "patient_id"   : patient["patient_id"],
            "label"        : torch.as_tensor(
                float(label) if self.task_type == "regression" else int(label),
                dtype=torch.float32 if self.task_type == "regression" else torch.long,
            ),
            "demographic"  : demo_feat,
            "medical_codes": med_codes, "medical_times": med_times, "medical_types": med_types,
            "note_embs"    : note_embs, "note_times"  : note_times, "note_types"  : note_types,
            "lab_times"    : lab_times, "lab_vals"    : lab_vals,  "lab_types"   : lab_types,
            "cxr_embs"     : cxr_embs,  "cxr_times"   : cxr_times, "cxr_types"   : cxr_types,
            "mimic_image" : mimic_images, "mimic_audio" : mimic_audio,
        }

    # -------------------  各采集函数  --------------------
    def _build_demo_vec(self, demo: dict) -> torch.Tensor:
        feats: List[float] = [1.0]            # bias
        feats.append(log1p_or_zero(demo.get("age", 0.0)))

        for feat_name in ["gender", "race", "marital_status", "language"]:
            raw_val = demo.get(feat_name, "nan")
            if pd.isna(raw_val):
                raw_val = "nan"
            idx = self.mappings[feat_name].get(
                raw_val, self.mappings[feat_name]["nan"]
            )
            feats.extend(one_hot(idx, self._max_cat[feat_name] + 1))
        return torch.tensor(feats, dtype=torch.float32)

    def _collect_medical(self, visits):
        events = []
        for v in visits:
            for gname, tdict in {
                "diagnosis": ["ccs_events", "icd10_events", "icd9_events", "phecode_events"],
                "medication": ["rxnorm_events"],
                "drg": ["drg_APR_events", "drg_HCFA_events"],
            }.items():
                for t in tdict:
                    for e in v.get(t, []):
                        if ("code_index" in e and "relative_time" in e
                            and not pd.isna(e["relative_time"])
                            and (not self.index_set or e["code_index"] in self.index_set)):
                            if e["code_index"] in self.exclude_code_indices:
                                continue
                            events.append((
                                e["relative_time"],
                                e["code_index"],
                                self._TYPE2ID_MED[gname],
                            ))
        if not events:
            return (
                np.empty((0,), np.int32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32),
            )
        events.sort(key=lambda x: x[0])
        times, codes, types = zip(*events)
        return (
            np.asarray(codes, np.int32),
            np.asarray(times, np.float32),
            np.asarray(types, np.int32),
        )

    def _collect_notes(self, visits):
        notes = []
        for v in visits:
            for key, typ in [("dis_embeddings", "discharge"), ("rad_embeddings", "radiology")]:
                for e in v.get(key, []):
                    notes.append(
                        (e["relative_time"], np.array(e["embedding"], np.float32), typ)
                    )
        if not notes:
            # ★ 修正：返回 1-D 空数组，避免直接带列数 1
            return (
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32),
            )
        notes.sort(key=lambda x: x[0])
        times, embs, typs = zip(*notes)
        return (
            np.stack(embs, axis=0),
            np.asarray(times, np.float32),
            np.asarray([self._TYPE2ID_NOTE[t] for t in typs], np.int32),
        )

    def _collect_labs(self, visits):
        labs = []
        for v in visits:
            for e in v.get("lab_events", []):
                if ("code_index" in e and "relative_time" in e
                    and "standardized_value" in e and not pd.isna(e["relative_time"])):
                    labs.append((
                        e["relative_time"],
                        e["standardized_value"],
                        100 + e["code_index"],
                    ))
        if not labs:
            return (
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32),
            )
        labs.sort(key=lambda x: x[0])
        times, vals, typs = zip(*labs)
        return (
            np.asarray(times, np.float32),
            np.asarray(vals,  np.float32),
            np.asarray(typs,  np.int32),
        )

    def _collect_cxr(self, patient_id: int, visits):
        emb_list = []
        for v in visits:
            vnum = v.get("visit_number")
            if vnum is None:
                continue
            for e in self.cxr_dict.get((patient_id, float(vnum)), []):
                emb_list.append(
                    (e["relative_time"], np.array(e["embedding"], np.float32))
                )
        if not emb_list:
            # ★ 修正：返回 1-D 空数组
            return (
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int32),
            )
        emb_list.sort(key=lambda x: x[0])
        times, embs = zip(*emb_list)
        return (
            np.stack(embs, axis=0),
            np.asarray(times, np.float32),
            np.full(len(embs), self._CXR_ID, np.int32),
        )

    def _collect_mimic(self, visits):
        """
        收集 audio-vision（MNIST风格）模态：
          - 从每个 visit 中读取 'image_data' 与 'audio_data'
          - 若存在 relative_time，则按时间排序后堆叠
          - 用 self.missing_prob 按模态随机整体置空以模拟缺失
        返回:
          mimic_images: np.ndarray，若有数据通常为 (N, C, H, W) 或 (N, H, W)（保留原数据形状）
          mimic_audio : np.ndarray，若有数据通常为 (N, C, F, T) 或 (N, F, T)
          若对应模态不存在/被置空，则返回 np.empty((0,), dtype=np.float32)
        """
        rng = np.random.default_rng()

        img_buf = []
        aud_buf = []

        for v in visits:
            t = float(0)

            # ---- image ----
            img = v.get("image_data", None)
            if img is not None:
                img_arr = np.asarray(img, dtype=np.float32)
                if img_arr.size > 0:
                    # 统一到至少3维，便于堆叠：(H,W)->(1,H,W)
                    if img_arr.ndim == 2:
                        img_arr = img_arr[None, ...]
                    img_buf.append((t, img_arr))

            # ---- audio ----
            aud = v.get("audio_data", None)
            if aud is not None:
                aud_arr = np.asarray(aud, dtype=np.float32)
                if aud_arr.size > 0:
                    # 同理，至少3维：(F,T)->(1,F,T)
                    if aud_arr.ndim == 2:
                        aud_arr = aud_arr[None, ...]
                    aud_buf.append((t, aud_arr))

        # 时间排序并堆叠
        img_buf.sort(key=lambda x: x[0])
        aud_buf.sort(key=lambda x: x[0])

        if img_buf:
            mimic_images = np.stack([a for _, a in img_buf], axis=0)
        else:
            mimic_images = np.empty((0,), dtype=np.float32)

        if aud_buf:
            mimic_audio = np.stack([a for _, a in aud_buf], axis=0)
        else:
            mimic_audio = np.empty((0,), dtype=np.float32)

        # 随机构造缺失（按模态整体置空）
        if mimic_images.size > 0 and self.missing_prob > 0.0 and rng.random() < self.missing_prob:
            mimic_images = np.empty((0,), dtype=np.float32)

        if mimic_audio.size > 0 and self.missing_prob > 0.0 and rng.random() < self.missing_prob:
            mimic_audio = np.empty((0,), dtype=np.float32)

        return mimic_images, mimic_audio

    # -------------- Dataset API --------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
