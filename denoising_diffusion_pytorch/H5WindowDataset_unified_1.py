#%%
"""
Unified dataset that supports both:
- Single-stream windowed HDF5 loading (like H5WindowDataset)
- Paired target/cond ControlNet loading (like H5WindowDataset_CLDM)

Backwards-compatible wrappers are provided at the bottom so existing code that
imports H5WindowDataset or H5WindowDataset_CLDM keeps working without changes.

Design goals:
- Keep all public behaviors/flags from both originals
- Reuse original logic where possible (helpers kept mostly verbatim)
- Preserve return shapes and meta payloads
"""
from __future__ import annotations

import re, random, math
from typing import Optional, Tuple, List, Dict, Set, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# ============================= helpers (kept from originals) =============================

def slugify_group(gpath: str) -> str:
    s = gpath.strip('/')
    return re.sub(r'[^\w\-\.\+]+' ,'_', s)

def list_h5_groups_with_dataset(h5file: h5py.File, dataset_key: str):
    """Yield (group_path, dataset) for group_path/dataset_key"""
    pending = ['/']; seen = set()
    while pending:
        gpath = pending.pop()
        if gpath in seen:
            continue
        seen.add(gpath)
        g = h5file[gpath]
        dkey = f"{gpath.rstrip('/')}/{dataset_key}".lstrip('/')
        if dkey in h5file and isinstance(h5file[dkey], h5py.Dataset):
            yield (gpath, h5file[dkey])
        for k, v in g.items():
            if isinstance(v, h5py.Group):
                sub = f"{gpath.rstrip('/')}/{k}".replace('//','/')
                if not sub.startswith('/'):
                    sub = '/' + sub
                pending.append(sub)

def _maybe_pad_edge(x: np.ndarray, need_h: int, need_w: int) -> np.ndarray:
    """x: (C,H,W). Edge-pad to at least (need_h, need_w)"""
    C,H,W = x.shape
    pad_h = max(0, need_h - H)
    pad_w = max(0, need_w - W)
    if pad_h == 0 and pad_w == 0:
        return x
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return np.pad(x, ((0,0),(top,bottom),(left,right)), mode='edge')

def _crop_offsets(H:int, W:int, ch:int, cw:int, mode:str, rng:Optional[random.Random]):
    ch = int(ch); cw = int(cw)
    top = (H - ch) // 2
    left = (W - cw) // 2
    if mode == 'random':
        assert rng is not None, "rng required for random crop"
        if H - ch > 0: top = rng.randint(0, H - ch)
        if W - cw > 0: left = rng.randint(0, W - cw)
    return top, left, ch, cw

def crop2d(x: np.ndarray, crop_size, mode='center', rng=None) -> np.ndarray:
    C,H,W = x.shape
    if isinstance(crop_size, int):
        ch, cw = crop_size, crop_size
    else:
        ch, cw = crop_size
    x = _maybe_pad_edge(x, ch, cw)
    _,H,W = x.shape
    top, left, ch, cw = _crop_offsets(H, W, ch, cw, mode, rng)
    return x[:, top:top+ch, left:left+cw]

def crop2d_with_offsets(x: np.ndarray, crop_size, mode='center', rng=None, offsets=None):
    """Return cropped (C,ch,cw) and (top,left,ch,cw)."""
    C,H,W = x.shape
    if isinstance(crop_size, int):
        ch, cw = crop_size, crop_size
    else:
        ch, cw = crop_size
    x = _maybe_pad_edge(x, ch, cw)
    _,H,W = x.shape
    if offsets is None:
        top, left, ch, cw = _crop_offsets(H, W, ch, cw, mode, rng)
    else:
        top, left, ch, cw = offsets
    return x[:, top:top+ch, left:left+cw], (top, left, ch, cw)

def _ensure_chw_strict(arr: np.ndarray) -> np.ndarray:
    """
    (H,W) -> (1,H,W)
    (C,H,W) -> 그대로
    (H,W,C) with C in {1,3} -> (C,H,W)
    """
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[-1] in (1,3) and arr.shape[0] not in (1,3):
            arr = np.transpose(arr, (2,0,1))
    else:
        raise ValueError(f"Unsupported array ndim: {arr.shape}")
    return arr.astype(np.float32, copy=False)

def _rescale_chw_channelwise(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'none':
        return x.astype(np.float32, copy=False)
    C = x.shape[0]
    out = np.empty_like(x, dtype=np.float32)
    for c in range(C):
        v = x[c]
        vmin, vmax = float(v.min()), float(v.max())
        if vmax == vmin:
            out[c] = np.zeros_like(v, dtype=np.float32)
        elif mode == 'zero_one':
            out[c] = (v - vmin) / (vmax - vmin)
        elif mode == 'neg_one_one':
            v01 = (v - vmin) / (vmax - vmin)
            out[c] = v01 * 2.0 - 1.0
        else:
            raise ValueError(f"Unknown rescale mode: {mode}")
    return out

def _read_zs_3d(dset: h5py.Dataset, zs: List[int]) -> np.ndarray:
    """dset: (Z,H,W), zs -> (K,H,W) with uniqueness trick"""
    zs_arr = np.asarray(zs, dtype=np.intp)
    uniq, inv = np.unique(zs_arr, return_inverse=True)
    arr_u = dset[uniq, ...]         # (U,H,W)
    arr   = arr_u[inv, ...]         # (K,H,W)
    return arr.astype(np.float32, copy=False)

def _read_zs_4d_EZHW(dset: h5py.Dataset, zs: List[int]) -> np.ndarray:
    """dset: (E,Z,H,W), zs -> (E,K,H,W)"""
    zs_arr = np.asarray(zs, dtype=np.intp)
    uniq, inv = np.unique(zs_arr, return_inverse=True)
    arr_u = dset[:, uniq, ...]      # (E,U,H,W)
    arr   = arr_u[:, inv, ...]      # (E,K,H,W)
    return arr.astype(np.float32, copy=False)

def _stack_window_general(dset: h5py.Dataset, zc: Optional[int], Z:int, neighbors:int,
                          z_axis:int, is4d:bool, rescale_mode:str) -> np.ndarray:
    """
    반환: (C,H,W)
      - 3D: C = K
      - 4D: C = E*K  (에코×인접슬라이스)
    """
    def _load_single(_z):
        if _z is None:
            arr = dset[...]
        else:
            if is4d:
                if z_axis == 1:
                    arr = dset[:, _z, :, :]       # (E,H,W)
                else:
                    slicer = [slice(None)] * dset.ndim
                    slicer[z_axis] = _z
                    arr = dset[tuple(slicer)]
                    if arr.ndim != 3:
                        arr = np.asarray(arr)
            else:
                arr = dset[_z]                    # (H,W)
        x = _ensure_chw_strict(arr)               # (C,H,W)
        return _rescale_chw_channelwise(x, rescale_mode)

    if Z == 1 or zc is None or neighbors == 0:
        return _load_single(zc)

    # 인접 z들 (경계 클램핑)
    K = 2*neighbors + 1
    zs = [min(max(zc+off, 0), Z-1) for off in range(-neighbors, neighbors+1)]

    if not is4d:
        arr = _read_zs_3d(dset, zs)                     # (K,H,W)
        return _rescale_chw_channelwise(arr, rescale_mode)

    # 4D
    if z_axis == 1:
        arr = _read_zs_4d_EZHW(dset, zs)                # (E,K,H,W)
    else:
        stacks = [_load_single(zi) for zi in zs]        # [(E,H,W) ...] K개
        arr = np.stack(stacks, axis=1)                  # (E,K,H,W)
    arr = np.transpose(arr, (1,0,2,3)).reshape(-1, arr.shape[2], arr.shape[3])  # (E*K,H,W)
    return _rescale_chw_channelwise(arr, rescale_mode)

# ============================= Unified dataset =============================
class H5WindowDatasetUnified(Dataset):
    """
    If cond_key is None -> single-stream behavior (like H5WindowDataset)
    If cond_key is not None -> paired behavior returning (x, cond) (like H5WindowDataset_CLDM)
    """
    def __init__(self,
        h5_path: str,
        # Key params: accept both legacy names
        dataset_key: Optional[str] = 'data',     # single-stream
        target_key: Optional[str] = None,        # paired target (overrides dataset_key if set)
        cond_key: Optional[str] = None,          # paired cond (None => single-stream)
        image_size: Optional[Tuple[int,int]] = None,
        neighbors: int = 0,
        split: str = 'train',                    # 'train' | 'val' | 'test'  #----------수정 2025-10-20: 주석만 보완
        # splitting
        val_ratio: float = 0.05,
        val_subject_count: Optional[int] = None,
        val_subject_list: Optional[List[str]] = None,
        # ▼ test 전용 추가
        test_ratio: float = 0.10,
        test_subject_count: Optional[int] = None,
        test_subject_list: Optional[List[str]] = None,
        seed: int = 42,
        # scaling & aug
        rescale: str = 'none',
        cond_rescale: Optional[str] = None,      # paired 전용. None이면 rescale과 동일
        horizontal_flip: bool = False,
        # cropping
        crop_size: Optional[Union[Tuple[int, int], int]] = None,
        crop_mode_train: str = 'random',
        crop_mode_val:   str = 'center',
        # misc
        return_meta: bool = False,
        # ### ---------------------------------수정: z 슬라이스 제한 옵션 추가
        z_start: Optional[int] = None,           # 예: 30 (윈도우 전체가 이 값보다 작아지면 제외)
        z_end:   Optional[int] = None,           # 예: 80 (윈도우 전체가 이 값보다 커지면 제외)
        # ### ---------------------------------수정 끝
    ):
        assert split in ('train','val','test')   # ★ test 허용
        if val_subject_count is not None:
            assert val_subject_count > 0
        if val_ratio is not None:
            assert 0.0 < val_ratio < 1.0
        if test_subject_count is not None:
            assert test_subject_count > 0
        if test_ratio is not None:
            assert 0.0 < test_ratio < 1.0

        self.h5_path = str(h5_path)
        # Mode detection
        if target_key is None and cond_key is None:
            self.mode = 'single'
            self.key = dataset_key or 'data'
        else:
            self.mode = 'paired'
            self.target_key = (target_key or dataset_key or 'data')
            self.cond_key   = cond_key or 'cond'

        self.image_size = image_size
        self.neighbors  = int(neighbors)
        self.K          = 2*self.neighbors + 1
        self.split      = split
        self.rescale    = rescale
        self.cond_rescale = (cond_rescale if cond_rescale is not None else rescale)
        self.horizontal_flip = horizontal_flip
        self.crop_size  = crop_size
        self.crop_mode_train = crop_mode_train
        self.crop_mode_val   = crop_mode_val
        self.return_meta = return_meta

        # ### ---------------------------------수정: z_start / z_end 멤버로 저장
        self.z_start = z_start
        self.z_end   = z_end
        # ### ---------------------------------수정 끝

        self.seed = seed
        self._rng = random.Random(seed)

        #----------수정 2025-10-20: split 파라미터를 멤버에 반드시 저장 (build_index에서 getattr로 읽음)
        self.val_ratio = float(val_ratio)
        self.val_subject_count = val_subject_count
        self.val_subject_list = val_subject_list

        self.test_ratio = float(test_ratio) if test_ratio is not None else None
        self.test_subject_count = test_subject_count
        self.test_subject_list = test_subject_list
        #----------수정 2025-10-20 끝

        with h5py.File(self.h5_path, 'r') as f:
            if self.mode == 'single':
                self._build_index_single(f, dataset_key=self.key,
                                         val_ratio=val_ratio,
                                         val_subject_count=val_subject_count,
                                         val_subject_list=val_subject_list)
            else:
                self._build_index_paired(f,
                                         target_key=self.target_key,
                                         cond_key=self.cond_key,
                                         val_ratio=val_ratio,
                                         val_subject_count=val_subject_count,
                                         val_subject_list=val_subject_list)

        self._h5_cache = None  # worker-local file handle

    # ---------------- index builders ----------------
    def _inspect_dataset(self, d: h5py.Dataset) -> dict:
        if d.ndim == 4:
            E, Z, H, W = d.shape
            return dict(is4d=True, z_axis=1, E=E, Z=Z, H=H, W=W)
        elif d.ndim == 3:
            Z, H, W = d.shape
            return dict(is4d=False, z_axis=0, E=1, Z=Z, H=H, W=W)
        elif d.ndim == 2:
            H, W = d.shape
            return dict(is4d=False, z_axis=0, E=1, Z=1, H=H, W=W)
        else:
            raise ValueError(f"Unsupported dataset ndim: {d.ndim}")
        
        
    def _build_index_single(self, f: h5py.File, dataset_key: str,
                            val_ratio: float, val_subject_count: Optional[int], val_subject_list: Optional[List[str]]):
        subj: Dict[str, List[dict]] = {}

        # ### ---------------------------------수정: neighbors/윈도우 반폭 가져오기
        half = self.neighbors
        # ### ---------------------------------수정 끝

        for gpath, d in list_h5_groups_with_dataset(f, dataset_key):
            subject = slugify_group(gpath)
            if d.ndim == 3:
                Z,H,W = d.shape
                L = subj.setdefault(subject, [])
                for z in range(Z):
                    # ### ---------------------------------수정: z 슬라이스 범위 / 경계 체크
                    # 윈도우가 볼륨 경계를 벗어나면 제외
                    if (z - half) < 0 or (z + half) >= Z:
                        continue
                    # z_start/z_end가 설정된 경우 그 범위 밖인 윈도우는 제외
                    if self.z_start is not None:
                        if (z - half) < self.z_start:
                            continue
                    if self.z_end is not None:
                        if (z + half) > self.z_end:
                            continue
                    # ### ---------------------------------수정 끝

                    L.append({'gpath': gpath, 'z': z, 'H': H, 'W': W, 'Z': Z,
                              'subject': subject, 'is4d': False, 'z_axis': 0, 'E': 1})

            elif d.ndim == 4:
                E, Z, H, W = d.shape
                L = subj.setdefault(subject, [])
                for z in range(Z):
                    # ### ---------------------------------수정: z 슬라이스 범위 / 경계 체크
                    if (z - half) < 0 or (z + half) >= Z:
                        continue
                    if self.z_start is not None:
                        if (z - half) < self.z_start:
                            continue
                    if self.z_end is not None:
                        if (z + half) > self.z_end:
                            continue
                    # ### ---------------------------------수정 끝

                    L.append({'gpath': gpath, 'z': z, 'H': H, 'W': W, 'Z': Z,
                              'subject': subject, 'is4d': True, 'z_axis': 1, 'E': E})

            elif d.ndim == 2:
                H, W = d.shape
                L = subj.setdefault(subject, [])
                L.append({'gpath': gpath, 'z': None, 'H': H, 'W': W, 'Z': 1,
                          'subject': subject, 'is4d': False, 'z_axis': 0, 'E': 1})
            else:
                continue

        subjects = list(subj.keys())
        # if self.split == 'train':
        self._rng.shuffle(subjects)####################################################################################################################################심각한오류

        if val_subject_list:
            val_set = set(val_subject_list)
        elif val_subject_count is not None:
            val_set = set(subjects[:val_subject_count])
        else:
            n_val = max(1, int(math.ceil(len(subjects) * val_ratio)))
            n_val = min(n_val, max(1, len(subjects) - 1))
            val_set = set(subjects[:n_val])

        # (기존 val_set 계산 바로 아래에 추가)
        #----------수정 2025-10-20 (디버그)
        print(f"[SPLIT DEBUG][single] total_subjects={len(subjects)}  "
              f"val={len(val_set)}  remaining={len([s for s in subjects if s not in val_set])}  "
              f"split={self.split}  "
              f"test_ratio={getattr(self,'test_ratio',None)} "
              f"test_count={getattr(self,'test_subject_count',None)} "
              f"has_test_list={bool(getattr(self,'test_subject_list',None))}")
        #----------수정 2025-10-20 (디버그) 끝

        #----------수정 2025-10-20: test 분할 지원(있는 경우만 활성화; 없으면 기존 2-way 유지)
        # self에 test 관련 설정이 있을 때만 사용. 없으면 빈 집합으로 두어 기존 동작과 완전 동일.
        test_ratio = getattr(self, 'test_ratio', None)
        test_subject_count = getattr(self, 'test_subject_count', None)
        test_subject_list = getattr(self, 'test_subject_list', None)

        test_set: set[str] = set()  # 경고 제거 위해 빌트인 제네릭 사용
        # 남은 subject 중에서 test를 선택 (val과 겹치지 않도록)
        remaining_subjects = [s for s in subjects if s not in val_set]

        if test_subject_list:
            # 사용자가 명시한 리스트를 우선 적용 (remaining에 존재하는 것만)
            test_set = set([s for s in test_subject_list if s in remaining_subjects])
        elif test_subject_count is not None:
            # 개수 기반 선택
            if test_subject_count > 0:
                test_set = set(remaining_subjects[:test_subject_count])
        elif (test_ratio is not None) and (test_ratio > 0.0):
            # 비율 기반 선택
            # 전체 대비 비율이지만, 실제 선택은 remaining에서 수행
            n_test = int(math.ceil(len(subjects) * float(test_ratio)))
            # train이 최소 1개는 남도록 상한 조정 (가능한 경우)
            max_allow = max(0, len(remaining_subjects) - 1)  # train 최소 1
            if max_allow > 0:
                n_test = min(n_test, max_allow)
            else:
                n_test = 0
            if n_test > 0:
                test_set = set(remaining_subjects[:n_test])
        #----------수정 2025-10-20 끝

        # (기존 test_set 계산 블록 끝난 다음, pick 만들기 전에 추가)
        #----------수정 2025-10-20 (디버그)
        train_subjects_est = [s for s in subjects if (s not in val_set) and (s not in test_set)]
        print(f"[SPLIT DEBUG][single] test={len(test_set)}  train≈{len(train_subjects_est)}")
        # 필요하면 일부 샘플 이름도 확인
        # print("[SAMPLE] val:", list(val_set)[:5], "test:", list(test_set)[:5])
        #----------수정 2025-10-20 (디버그) 끝

        #----------수정 2025-10-20: split에 'test' 분기 추가
        if self.split == 'val':
            pick = (lambda s: s in val_set)
        elif getattr(self, 'split', None) == 'test':
            pick = (lambda s: s in test_set)
        else:
            pick = (lambda s: (s not in val_set) and (s not in test_set))
        #----------수정 2025-10-20 끝

        self.index = [rec for s in subjects if pick(s) for rec in subj[s]]
        self._single_key = dataset_key


    def _build_index_paired(self, f: h5py.File, target_key: str, cond_key: str,
                            val_ratio: float, val_subject_count: Optional[int], val_subject_list: Optional[List[str]]):
        groups: Dict[str, Dict[str, dict]] = {}
        for gpath, d in list_h5_groups_with_dataset(f, target_key):
            subject = slugify_group(gpath)
            meta = self._inspect_dataset(d)
            groups.setdefault(subject, {})['t'] = dict(meta, gpath=gpath)
        for gpath, d in list_h5_groups_with_dataset(f, cond_key):
            subject = slugify_group(gpath)
            meta = self._inspect_dataset(d)
            groups.setdefault(subject, {})['c'] = dict(meta, gpath=gpath)

        subjects_all = [s for s, mc in groups.items() if ('t' in mc and 'c' in mc)]
        # if self.split == 'train':#######################################################################################################################################################심각한오류
        self._rng.shuffle(subjects_all)

        if val_subject_list:
            val_set = set(val_subject_list)
        elif val_subject_count is not None:
            val_set = set(subjects_all[:val_subject_count])
        else:
            n_val = max(1, int(math.ceil(len(subjects_all) * val_ratio)))
            n_val = min(n_val, max(1, len(subjects_all) - 1))
            val_set = set(subjects_all[:n_val])

        # (기존 val_set 계산 바로 아래에 추가)
        #----------수정 2025-10-20 (디버그)
        print(f"[SPLIT DEBUG][paired] total_subjects={len(subjects_all)}  "
              f"val={len(val_set)}  remaining={len([s for s in subjects_all if s not in val_set])}  "
              f"split={self.split}  "
              f"test_ratio={getattr(self,'test_ratio',None)} "
              f"test_count={getattr(self,'test_subject_count',None)} "
              f"has_test_list={bool(getattr(self,'test_subject_list',None))}")
        #----------수정 2025-10-20 (디버그) 끝


        #----------수정 2025-10-20: test 분할 지원(있는 경우만 활성화; 없으면 기존 2-way 유지)
        test_ratio = getattr(self, 'test_ratio', None)
        test_subject_count = getattr(self, 'test_subject_count', None)
        test_subject_list = getattr(self, 'test_subject_list', None)

        test_set = set()
        remaining_subjects = [s for s in subjects_all if s not in val_set]

        if test_subject_list:
            test_set = set([s for s in test_subject_list if s in remaining_subjects])
        elif test_subject_count is not None:
            if test_subject_count > 0:
                test_set = set(remaining_subjects[:test_subject_count])
        elif (test_ratio is not None) and (float(test_ratio) > 0.0):
            n_test = int(math.ceil(len(subjects_all) * float(test_ratio)))
            # train이 최소 1개는 남도록 제한
            max_allow = max(0, len(remaining_subjects) - 1)
            if max_allow > 0:
                n_test = min(n_test, max_allow)
            else:
                n_test = 0
            if n_test > 0:
                test_set = set(remaining_subjects[:n_test])
        #----------수정 2025-10-20 끝

        # (기존 test_set 계산 블록 끝난 다음, pick 만들기 전에 추가)
        #----------수정 2025-10-20 (디버그)
        train_subjects_est = [s for s in subjects_all if (s not in val_set) and (s not in test_set)]
        print(f"[SPLIT DEBUG][paired] test={len(test_set)}  train≈{len(train_subjects_est)}")
        # 필요하면 일부 이름도
        # print("[SAMPLE] val:", list(val_set)[:5], "test:", list(test_set)[:5])
        #----------수정 2025-10-20 (디버그) 끝


        #----------수정 2025-10-20: split에 'test' 분기 추가
        if self.split == 'val':
            pick = (lambda s: s in val_set)
        elif getattr(self, 'split', None) == 'test':
            pick = (lambda s: s in test_set)
        else:
            pick = (lambda s: (s not in val_set) and (s not in test_set))
        #----------수정 2025-10-20 끝

        index = []

        # ### ---------------------------------수정: neighbors/윈도우 반폭 가져오기
        half = self.neighbors
        # ### ---------------------------------수정 끝

        for s in subjects_all:
            if not pick(s):
                continue
            mt = groups[s]['t']; mc = groups[s]['c']
            Z = min(mt['Z'], mc['Z'])
            H = min(mt['H'], mc['H'])
            W = min(mt['W'], mc['W'])
            for z in range(Z):
                # ### ---------------------------------수정: z 슬라이스 범위 / 경계 체크
                # 윈도우가 볼륨 범위를 벗어나면 스킵
                if (z - half) < 0 or (z + half) >= Z:
                    continue
                # z_start / z_end가 지정된 경우 해당 범위 밖 윈도우는 스킵
                if self.z_start is not None:
                    if (z - half) < self.z_start:
                        continue
                if self.z_end is not None:
                    if (z + half) > self.z_end:
                        continue
                # ### ---------------------------------수정 끝

                index.append({
                    'subject': s,
                    't_gpath': mt['gpath'], 'c_gpath': mc['gpath'],
                    'z': z, 'Z': Z, 'H': H, 'W': W,
                    't_is4d': mt['is4d'], 't_z_axis': mt['z_axis'], 't_E': mt['E'],
                    'c_is4d': mc['is4d'], 'c_z_axis': mc['z_axis'], 'c_E': mc['E'],
                })
        self.index = index


    # ---------------- h5 open ----------------
    def _get_h5(self):
        if self._h5_cache is None:
            try:
                self._h5_cache = h5py.File(self.h5_path, 'r', swmr=True, libver='latest')
            except Exception:
                self._h5_cache = h5py.File(self.h5_path, 'r')
        return self._h5_cache

    def __len__(self):
        return len(self.index)

    # ---------------- item loader ----------------
    # ---------------- item loader ----------------
    def __getitem__(self, idx: int):
        f = self._get_h5()
        rec = self.index[idx]

        if self.mode == 'single':
            gpath, zc, Z = rec['gpath'], rec['z'], rec['Z']
            dset = f[f"{gpath.rstrip('/')}/{self._single_key}".lstrip('/')]
            if rec.get('is4d', False) or dset.ndim == 4:
                x = _stack_window_general(dset, zc, Z, neighbors=self.neighbors,
                                          z_axis=rec.get('z_axis', 0), is4d=True,
                                          rescale_mode=self.rescale)
            else:
                x = _stack_window_general(dset, zc, Z, neighbors=self.neighbors,
                                          z_axis=rec.get('z_axis', 0), is4d=False,
                                          rescale_mode=self.rescale)

            # crop
            if self.crop_size is not None:
                mode = self.crop_mode_train if self.split=='train' else self.crop_mode_val
                x = crop2d(x, self.crop_size, mode=mode, rng=(self._rng if mode=='random' else None))

            if self.horizontal_flip and self.split=='train' and (self._rng.random() < 0.5):
                x = x[:, :, ::-1].copy()

            out = torch.from_numpy(x)
            if not self.return_meta:
                return out
            meta = {
                'subject': rec.get('subject', slugify_group(gpath)),
                'gpath': gpath,
                'z': zc,
                'Z': Z,
                'is4d': rec.get('is4d', False),
                'z_axis': rec.get('z_axis', 0),
                'E': rec.get('E', 1),
                'K': self.K,
                'C': int(out.shape[0]),
            }
            return out, meta

        else:
            # ---------------- paired (CLDM) 경로 ----------------
            zc, Z = rec['z'], rec['Z']
            tset = f[f"{rec['t_gpath'].rstrip('/')}/{self.target_key}".lstrip('/')]
            cset = f[f"{rec['c_gpath'].rstrip('/')}/{self.cond_key}".lstrip('/')]

            x = _stack_window_general(tset, zc, Z, neighbors=self.neighbors,
                                      z_axis=rec.get('t_z_axis', 0), is4d=rec.get('t_is4d', False),
                                      rescale_mode=self.rescale)
            cond = _stack_window_general(cset, zc, Z, neighbors=self.neighbors,
                                         z_axis=rec.get('c_z_axis', 0), is4d=rec.get('c_is4d', False),
                                         rescale_mode=self.cond_rescale)


            
            # <<< 여기 추가: 무조건 H/W 전치 >>>
            x   = np.transpose(x,   (0, 2, 1))   # (C,H,W) -> (C,W,H)
            cond= np.transpose(cond,(0, 2, 1))   # (C,H,W) -> (C,W,H)

            # 동일 오프셋으로 crop
            if self.crop_size is not None:
                mode = self.crop_mode_train if self.split=='train' else self.crop_mode_val
                x, offs = crop2d_with_offsets(x, self.crop_size, mode=mode,
                                              rng=(self._rng if mode=='random' else None))
                cond, _ = crop2d_with_offsets(cond, self.crop_size, mode=mode, offsets=offs)

            # 좌우 플립도 동일하게
            if self.horizontal_flip and self.split=='train' and (self._rng.random() < 0.5):
                x = x[:, :, ::-1].copy()
                cond = cond[:, :, ::-1].copy()

            out_x = torch.from_numpy(x)
            out_c = torch.from_numpy(cond)

            if not self.return_meta:
                return out_x, out_c

            meta = {
                'subject': rec['subject'],
                't_gpath': rec['t_gpath'],
                'c_gpath': rec['c_gpath'],
                'z': zc,
                'Z': Z,
                'H': rec['H'],
                'W': rec['W'],
                't_is4d': rec.get('t_is4d', False), 't_z_axis': rec.get('t_z_axis', 0), 't_E': rec.get('t_E', 1),
                'c_is4d': rec.get('c_is4d', False), 'c_z_axis': rec.get('c_z_axis', 0), 'c_E': rec.get('c_E', 1),
                'K': self.K,
                'Cx': int(out_x.shape[0]),
                'Cc': int(out_c.shape[0]),
            }
            return (out_x, out_c), meta


        # # paired mode
        # dt = f[f"{rec['t_gpath'].rstrip('/')}/{self.target_key}".lstrip('/')]
        # dc = f[f"{rec['c_gpath'].rstrip('/')}/{self.cond_key}".lstrip('/')]
        # z, Z = rec['z'], rec['Z']
        # x = _stack_window_general(dt, z, Z, neighbors=self.neighbors,
        #                           z_axis=rec['t_z_axis'], is4d=rec['t_is4d'],
        #                           rescale_mode=self.rescale)
        # cond = _stack_window_general(dc, z, Z, neighbors=self.neighbors,
        #                              z_axis=rec['c_z_axis'], is4d=rec['c_is4d'],
        #                              rescale_mode=self.cond_rescale)
        # if self.image_size is not None:
        #     expH, expW = self.image_size
        #     # assertion intentionally relaxed like original
        #     # assert x.shape[-2:] == (expH, expW) and cond.shape[-2:] == (expH, expW)

        # if self.crop_size is not None:
        #     mode = self.crop_mode_train if self.split=='train' else self.crop_mode_val
        #     x, offs = crop2d_with_offsets(x, self.crop_size, mode=mode, rng=(self._rng if mode=='random' else None))
        #     cond, _ = crop2d_with_offsets(cond, self.crop_size, mode=mode, rng=None, offsets=offs)

        # if self.horizontal_flip and self.split=='train' and (self._rng.random() < 0.5):
        #     x    = x[:, :, ::-1].copy()
        #     cond = cond[:, :, ::-1].copy()

        # out_x, out_cond = torch.from_numpy(x), torch.from_numpy(cond)
        # if not self.return_meta:
        #     return out_x, out_cond
        # meta = {
        #     'subject': rec['subject'],
        #     'z': rec['z'], 'Z': rec['Z'],
        #     't_gpath': rec['t_gpath'], 'c_gpath': rec['c_gpath'],
        #     't_is4d': rec['t_is4d'], 't_E': rec['t_E'],
        #     'c_is4d': rec['c_is4d'], 'c_E': rec['c_E'],
        #     'K': self.K,
        #     'Ct': int(out_x.shape[0]), 'Cc': int(out_cond.shape[0]),
        #     'crop_size': (out_x.shape[-2], out_x.shape[-1]),
        # }
        # return (out_x, out_cond), meta

# ============================= Backwards-compatible wrappers =============================

# ============================= Backwards-compatible wrappers =============================

class H5WindowDataset(H5WindowDatasetUnified):
    """Drop-in replacement for the original single-stream dataset."""
    def __init__(self,
        h5_path: str,
        dataset_key: str = 'data',
        image_size: Optional[Tuple[int,int]] = None,
        neighbors: int = 0,
        split: str = 'train',
        val_ratio: float = 0.05,
        val_subject_count: Optional[int] = None,
        val_subject_list: Optional[List[str]] = None,
        #----------수정 2025-10-20
        test_ratio: float = 0.10,
        test_subject_count: Optional[int] = None,
        test_subject_list: Optional[List[str]] = None,
        #----------수정 2025-10-20 끝
        seed: int = 42,
        rescale: str = 'none',
        horizontal_flip: bool = False,
        crop_size: Optional[Union[Tuple[int, int], int]] = None,
        crop_mode_train: str = 'random',
        crop_mode_val:   str = 'center',
        return_meta: bool = False,
        # ### ---------------------------------수정: wrapper에서도 z_start / z_end 인자 노출
        z_start: Optional[int] = None,
        z_end:   Optional[int] = None,
        # ### ---------------------------------수정 끝
    ):
        super().__init__(
            h5_path=h5_path,
            dataset_key=dataset_key,
            target_key=None, cond_key=None,
            image_size=image_size,
            neighbors=neighbors,
            split=split,
            val_ratio=val_ratio,
            val_subject_count=val_subject_count,
            val_subject_list=val_subject_list,
            #----------수정 2025-10-20
            test_ratio=test_ratio,
            test_subject_count=test_subject_count,
            test_subject_list=test_subject_list,
            #----------수정 2025-10-20 끝
            seed=seed,
            rescale=rescale,
            cond_rescale=None,
            horizontal_flip=horizontal_flip,
            crop_size=crop_size,
            crop_mode_train=crop_mode_train,
            crop_mode_val=crop_mode_val,
            return_meta=return_meta,
            # ### ---------------------------------수정: super로 전달
            z_start=z_start,
            z_end=z_end,
            # ### ---------------------------------수정 끝
        )

class H5WindowDataset_CLDM(H5WindowDatasetUnified):
    """Drop-in replacement for the original paired ControlNet dataset."""
    def __init__(self,
        h5_path: str,
        target_key: str = 'tof',
        cond_key: str   = 'mag',
        image_size: Optional[Tuple[int,int]] = None,
        neighbors: int = 0,
        split: str = 'train',
        val_ratio: float = 0.05,
        val_subject_count: Optional[int] = None,
        val_subject_list: Optional[List[str]] = None,
        #----------수정 2025-10-20
        test_ratio: float = 0.10,
        test_subject_count: Optional[int] = None,
        test_subject_list: Optional[List[str]] = None,
        #----------수정 2025-10-20 끝
        seed: int = 42,
        rescale: str = 'none',
        cond_rescale: Optional[str] = None,
        horizontal_flip: bool = False,
        crop_size: Optional[Union[Tuple[int, int], int]] = None,
        crop_mode_train: str = 'random',
        crop_mode_val:   str = 'center',
        return_meta: bool = False,
        # ### ---------------------------------수정: wrapper에서도 z_start / z_end 인자 노출
        z_start: Optional[int] = None,
        z_end:   Optional[int] = None,
        # ### ---------------------------------수정 끝
    ):
        super().__init__(
            h5_path=h5_path,
            dataset_key=None,
            target_key=target_key,
            cond_key=cond_key,
            image_size=image_size,
            neighbors=neighbors,
            split=split,
            val_ratio=val_ratio,
            val_subject_count=val_subject_count,
            val_subject_list=val_subject_list,
            #----------수정 2025-10-20
            test_ratio=test_ratio,
            test_subject_count=test_subject_count,
            test_subject_list=test_subject_list,
            #----------수정 2025-10-20 끝
            seed=seed,
            rescale=rescale,
            cond_rescale=cond_rescale,
            horizontal_flip=horizontal_flip,
            crop_size=crop_size,
            crop_mode_train=crop_mode_train,
            crop_mode_val=crop_mode_val,
            return_meta=return_meta,
            # ### ---------------------------------수정: super로 전달
            z_start=z_start,
            z_end=z_end,
            # ### ---------------------------------수정 끝
        )













if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- 기존 show_zyx 그대로 사용 ----
    def show_zyx(arr_ZHW, title, transpose_hw=False):
        Z, H, W = arr_ZHW.shape
        z0, y0, x0 = Z//2, H//2, W//2
        if transpose_hw:
            arr_ZHW = arr_ZHW.transpose(0,2,1)
        axial   = arr_ZHW[z0]
        coronal = arr_ZHW[:, y0, :]
        sagitt  = arr_ZHW[:, :, x0]
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(axial,   cmap="gray"); axs[0].set_title(f"{title}\naxial z={z0}")
        axs[1].imshow(coronal, cmap="gray"); axs[1].set_title(f"{title}\ncoronal y={y0}")
        axs[2].imshow(sagitt,  cmap="gray"); axs[2].set_title(f"{title}\nsagittal x={x0}")
        for ax in axs: ax.axis("off")
        plt.tight_layout(); plt.show()

    # ================== 경로 설정 ==================
    pretraining_tof_h5 = "/SSD5_8TB/Yeon/data/merged_250916.h5"       # 단일 로더
    paired_tof_h5      = "/SSD5_8TB/Yeon/data/mgre_tof_m1to1_fast.h5" # CLDM 로더

    # ================== 1) 단일 로더로 볼륨 복원 & 시각화 ==================
    ds_single = H5WindowDataset(
        h5_path=pretraining_tof_h5,
        dataset_key="data",
        neighbors=0,              # z 중앙 한 장만 채널 1로 받기
        split="val",
        return_meta=True,
    )

    # 같은 subject의 모든 z를 모아 (Z,H,W)로 쌓기
    # (첫 샘플의 subject로 고정. 필요하면 원하는 subject 필터링)
    first_x, first_meta = ds_single[0]
    subject = first_meta["subject"]
    Z = first_meta["Z"]; H = first_x.shape[1]; W = first_x.shape[2]

    vol_single = np.zeros((Z, H, W), dtype=np.float32)
    filled = np.zeros(Z, dtype=bool)
    for i in range(len(ds_single)):
        x, m = ds_single[i]
        if m["subject"] != subject: 
            continue
        z = m["z"]
        vol_single[z] = x[0]  # neighbors=0 → C=1
        filled[z] = True
        if filled.all():
            break

    show_zyx(vol_single, title=f"{pretraining_tof_h5}\nkey=data")

    # ================== 2) CLDM 로더로 타깃/컨디션 볼륨 복원 & 시각화 ==================
    ds_cldm = H5WindowDataset_CLDM(
        h5_path=paired_tof_h5,
        target_key="tof",
        cond_key="mag",
        neighbors=0,
        split="val",
        return_meta=True,
    )

    # 동일하게 한 subject의 전 z를 쌓기
    (x0, c0), m0 = ds_cldm[0]
    subject2 = m0["subject"]
    Z2 = m0["Z"]; H2 = x0.shape[1]; W2 = x0.shape[2]

    vol_t = np.zeros((Z2, H2, W2), dtype=np.float32)
    vol_c = np.zeros((Z2, H2, W2), dtype=np.float32)
    filled2 = np.zeros(Z2, dtype=bool)

    for i in range(len(ds_cldm)):
        (xt, xc), m = ds_cldm[i]
        if m["subject"] != subject2:
            continue
        z = m["z"]
        vol_t[z] = xt[0]  # neighbors=0 → C=1
        vol_c[z] = xc[0]
        filled2[z] = True
        if filled2.all():
            break

    # 로더를 통과한 결과이므로 transpose_hw=False가 정상이어야 함
    show_zyx(vol_t, title=f"{paired_tof_h5}\nTARGET key=tof", transpose_hw=False)
    show_zyx(vol_c, title=f"{paired_tof_h5}\nCOND   key=mag", transpose_hw=False)
