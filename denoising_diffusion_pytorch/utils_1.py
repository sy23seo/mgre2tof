from pathlib import Path  # 이미 있으면 생략
import numpy as np
import matplotlib.pyplot as plt

def _natural_key(path):
    """자연스러운 정렬 키 (숫자/문자 혼합 파일명도 올바른 순서로)."""
    import re
    s = str(path.name)
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def _load_center_volume(dir_path: Path, prefix: str, expected_len: int = None):
    """
    dir_path: .../val_samples/inputs | samples | conditions
    prefix  : 'val-input-center' | 'val-sample-center' | 'val-condition-center'
    expected_len: 기대 개수(예: 150). None이면 있는 만큼만 사용.
    """
    files = sorted(dir_path.glob(f"{prefix}-*.npy"), key=_natural_key)
    if expected_len is not None and len(files) != expected_len:
        print(f"[WARN] {dir_path.name}: expected {expected_len}, found {len(files)}")

    vol = []
    for p in files:
        arr = np.load(p)  # 보통 shape (1,1,H,W)
        # 안전하게 squeeze: (1,1,H,W) -> (H,W); (1,H,W) or (H,W)도 대응
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected ndim ({arr.ndim}) for {p.name}; expected 2D (H,W)")
        vol.append(arr.astype('float32', copy=False))
    if not vol:
        raise FileNotFoundError(f"No NPY files matching {prefix}-*.npy in {dir_path}")
    return np.stack(vol, axis=0)  # (N,H,W)


def _load_condition_echo_center_volumes_from_full(
    dir_path: Path,
    expected_len: int = None,
    K: int = 5,
    split_tag: str = 'val',  #----------수정 2025-10-20: val/test 등 접두어 선택
):
    """
    conditions 폴더의 '{split_tag}-condition-slice-full-*.npy'에서
    중앙 z(k_center=K//2)의 에코별 슬라이스만 뽑아 echo별 (N,H,W) 볼륨들로 반환.

    - 각 파일: (1, C, H, W) 또는 (C, H, W), C = E*K
    - 채널 매핑: ch = k*E + e (k: 0..K-1, e: 0..E-1)
    - 중앙 z 채널들: ch_center_e = (K//2)*E + e
    """
    #----------수정 2025-10-20: val 하드코딩 제거 → split_tag 사용
    pattern = f"{split_tag}-condition-slice-full-*.npy"
    files = sorted(dir_path.glob(pattern), key=_natural_key)
    if expected_len is not None and len(files) != expected_len:
        print(f"[WARN] conditions(full): expected {expected_len}, found {len(files)} (pattern='{pattern}')")
    if not files:
        raise FileNotFoundError(f"No NPY files matching '{pattern}' in {dir_path}")
    #----------수정 2025-10-20 끝

    # 첫 파일로 E(에코 수) 추론
    arr0 = np.squeeze(np.load(files[0]))   # (C,H,W)
    if arr0.ndim != 3:
        raise ValueError(f"Unexpected ndim ({arr0.ndim}) for {files[0].name}; expected 3D (C,H,W)")
    C, H, W = arr0.shape
    if C % K != 0:
        raise ValueError(f"C({C}) not divisible by K({K}); expected C=E*K")
    E = C // K
    k_center = K // 2
    center_ch = [k_center * E + e for e in range(E)]  # 예: [10,11,12,13,14] when K=5,E=5

    # echo별 스택 리스트 준비
    stacks = [[] for _ in range(E)]

    for p in files:
        a = np.squeeze(np.load(p))  # (C,H,W)
        if a.ndim != 3:
            raise ValueError(f"Unexpected ndim ({a.ndim}) for {p.name}; expected 3D (C,H,W)")
        if a.shape[0] != C:
            raise ValueError(f"Channel size mismatch: first file C={C}, but {p.name} has C={a.shape[0]}")
        for e in range(E):
            sl = a[center_ch[e]]  # (H,W)
            stacks[e].append(sl.astype('float32', copy=False))

    echo_vols = [np.stack(stacks[e], axis=0) for e in range(E)]  # 각 (N,H,W)
    return echo_vols


# ---------------------- 공통 체크포인트 선택 유틸(신규 교체) ----------------------
from pathlib import Path

def _select_resume_filename(models_dir: Path,
                            priority=("latest", "best", "step")) -> str | None:
    """
    models_dir 안에서 사용할 체크포인트 파일명을 고른다.
    우선순위(priority)는 ('latest','best','step') 튜플로 지정할 수 있음:
      - "latest" : model-latest.pt
      - "best"   : model-best.pt
      - "step"   : model-<가장 큰 숫자>.pt
    반환: filename(str) 또는 None
    """
    if not models_dir.exists():
        return None

    files = {p.name for p in models_dir.glob("model-*.pt")}

    # step 후보 수집
    step_candidates = []
    for fname in files:
        suf = fname.removeprefix("model-").removesuffix(".pt")
        if suf.isdigit():
            step_candidates.append((int(suf), fname))
    step_candidates.sort(reverse=True)

    for key in priority:
        if key == "latest" and "model-latest.pt" in files:
            return "model-latest.pt"
        if key == "best" and "model-best.pt" in files:
            return "model-best.pt"
        if key == "step" and step_candidates:
            return step_candidates[0][1]

    return None


# # ---------------------- 공통 체크포인트 선택 유틸(신규 추가) ----------------------
# def _select_resume_filename(models_dir: Path):
#     """
#     models_dir 안에서 재개/이식에 사용할 체크포인트 파일명을 고른다.
#     우선순위: model-best.pt -> model-latest.pt -> 숫자 최대(model-12345.pt)
#     반환: filename(str) 또는 None
#     """
#     if not models_dir.exists():
#         return None

#     best = models_dir / "model-best.pt"
#     latest = models_dir / "model-latest.pt"
#     if best.exists():
#         return best.name
#     if latest.exists():
#         return latest.name

#     cand = []
#     for p in models_dir.glob("model-*.pt"):
#         stem = p.stem  # 예) "model-12" / "model-latest"
#         suffix = stem.replace("model-", "")
#         if suffix.isdigit():
#             cand.append((int(suffix), p.name))
#     if cand:
#         cand.sort(reverse=True)
#         return cand[0][1]
#     return None


def _mips(vol: np.ndarray):
    """
    vol: (Z, Y, X) = (N, H, W)
    return: (mip_z, mip_y, mip_x) 각각 2D
    """
    assert vol.ndim == 3, f"expect (Z,Y,X), got {vol.shape}"
    mip_z = vol.max(axis=0)  # (H, W)  : 슬라이스 축(Z) MIP
    mip_y = vol.max(axis=1)  # (Z, W)  : 세로축(Y) MIP
    mip_x = vol.max(axis=2)  # (Z, H)  : 가로축(X) MIP
    return mip_z, mip_y, mip_x

def _imshow_with_percentile(ax, img, title=None, q_low=0.01, q_high=0.99):
    lo, hi = np.quantile(img, [q_low, q_high])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max() if img.max() > img.min() else img.min() + 1e-6)
    img_n = np.clip((img - lo) / (hi - lo), 0, 1)
    ax.imshow(img_n, cmap="gray", interpolation="nearest")
    if title: ax.set_title(title, fontsize=10)
    ax.axis("off")

def _save_mip_subplot(vol: np.ndarray, out_path: Path, title_prefix: str):
    print(vol.shape)
    mip_z, mip_y, mip_x = _mips(vol)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, constrained_layout=True)
    _imshow_with_percentile(axes[0], mip_z, f"{title_prefix} • MIP-Z (over slices)")
    _imshow_with_percentile(axes[1], np.flipud(mip_y), f"{title_prefix} • MIP-Y (over height)")
    _imshow_with_percentile(axes[2], np.flipud(mip_x), f"{title_prefix} • MIP-X (over width)")
    fig.savefig(out_path)
    plt.close(fig)

# def _save_mip_subplot_multi_echo(volumes_3d_list: list, out_path: Path, title_prefix: str = "Echo-wise"):
#     """
#     에코별 (N,H,W) 볼륨 리스트를 받아 5행×3열(z/y/x) 한 장으로 저장.
#     """
#     import matplotlib.pyplot as plt
#     E = len(volumes_3d_list)
#     fig, axes = plt.subplots(E, 3, figsize=(12, 2.6 * E))

#     # 공통 스케일
#     all_flat = np.concatenate([V.reshape(-1) for V in volumes_3d_list])
#     vmin, vmax = np.percentile(all_flat, 1), np.percentile(all_flat, 99)

#     for e in range(E):
#         V = volumes_3d_list[e]    # (N,H,W)
#         mip_z = V.max(axis=0)     # (H,W)
#         mip_y = V.max(axis=1)     # (N,W)
#         mip_x = V.max(axis=2)     # (N,H)

#         imgs = [mip_z, mip_y, mip_x]
#         titles = ["Axial (z MIP)", "MIP y", "MIP x"]
#         for j in range(3):
#             ax = axes[e, j] if E > 1 else axes[j]
#             ax.imshow(imgs[j], cmap="gray")
#             # ax.imshow(imgs[j], cmap="gray", vmin=vmin, vmax=vmax,
#             #           interpolation="nearest", aspect="auto")
#             if j == 0:
#                 ax.set_ylabel(f"Echo {e+1}", fontsize=9)
#             ax.set_title(titles[j], fontsize=9)
#             ax.axis("off")

#     fig.suptitle(f"{title_prefix} MIPs (rows: echo1~{E}, cols: z/y/x)", fontsize=11)
#     fig.tight_layout(rect=[0, 0, 1, 0.98])
#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)

def _save_mip_subplot_multi_echo(
    volumes_3d_list: list,
    out_path: Path,
    title_prefix: str = "Echo-wise",
    q_low: float = 0.01,
    q_high: float = 0.99,
):
    import numpy as np
    import matplotlib.pyplot as plt

    E = len(volumes_3d_list)
    fig, axes = plt.subplots(E, 3, figsize=(12, 2.6 * E), dpi=150, constrained_layout=True)

    # >>> 추가: axes를 항상 (E,3) 형태로 만들기
    axes = np.array(axes)
    if axes.ndim == 1:          # E == 1 인 경우, axes.shape == (3,)
        axes = axes.reshape(1, 3)

    for e in range(E):
        V = volumes_3d_list[e]             # (N,H,W)
        mip_z, mip_y, mip_x = _mips(V)

        _imshow_with_percentile(axes[e, 0], mip_z,            f"Axial • MIP-Z", q_low, q_high)
        _imshow_with_percentile(axes[e, 1], np.flipud(mip_y), f"    MIP-Y",     q_low, q_high)
        _imshow_with_percentile(axes[e, 2], np.flipud(mip_x), f"    MIP-X",     q_low, q_high)

        axes[e, 0].set_ylabel(f"Echo {e+1}", fontsize=9)

    fig.suptitle(f"{title_prefix} MIPs (rows: echo1~{E}, cols: z / y / x)", fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def _center_slices(vol: np.ndarray):
    """
    vol: (Z, Y, X) = (N, H, W)
    return: (axial, coronal, sagittal) 각 2D
    """
    assert vol.ndim == 3, f"expect (Z,Y,X), got {vol.shape}"
    zc = vol.shape[0] // 2
    yc = vol.shape[1] // 2
    xc = vol.shape[2] // 2

    axial    = vol[zc, :, :]   # (H, W)
    coronal  = vol[:, yc, :]   # (Z, W)
    sagittal = vol[:, :, xc]   # (Z, H)
    return axial, coronal, sagittal, (zc, yc, xc)


def _save_center_slice_subplot(
    vol: np.ndarray,
    out_path: Path,
    title_prefix: str,
    q_low: float = 0.01,
    q_high: float = 0.99,
):
    """
    MIP 대신 축별 '중앙 슬라이스'를 1x3 subplot으로 저장.
    (좌) Axial(z=mid), (중) Coronal(y=mid), (우) Sagittal(x=mid)
    coronal/sagittal은 기존 MIP와 동일하게 위아래 뒤집어 시각적 방향을 맞춤.
    """
    axial, coronal, sagittal, (zc, yc, xc) = _center_slices(vol)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150, constrained_layout=True)
    _imshow_with_percentile(axes[0], axial,                 f"{title_prefix} • Axial (z={zc})", q_low, q_high)
    _imshow_with_percentile(axes[1], np.flipud(coronal),    f"{title_prefix} • Coronal (y={yc})", q_low, q_high)
    _imshow_with_percentile(axes[2], np.flipud(sagittal),   f"{title_prefix} • Sagittal (x={xc})", q_low, q_high)
    fig.savefig(out_path)
    plt.close(fig)


def _save_center_slice_subplot_multi_echo(
    volumes_3d_list: list,
    out_path: Path,
    title_prefix: str = "Echo-wise (Center Slices)",
    q_low: float = 0.01,
    q_high: float = 0.99,
):
    import numpy as np
    import matplotlib.pyplot as plt

    E = len(volumes_3d_list)
    fig, axes = plt.subplots(E, 3, figsize=(12, 2.6 * E), dpi=150, constrained_layout=True)

    # >>> 추가
    axes = np.array(axes)
    if axes.ndim == 1:          # 단일 에코일 때 (3,) -> (1,3)
        axes = axes.reshape(1, 3)

    for e in range(E):
        V = volumes_3d_list[e]  # (N,H,W)
        axial, coronal, sagittal, (zc, yc, xc) = _center_slices(V)

        _imshow_with_percentile(axes[e, 0], axial,               f"Axial (z={zc})",     q_low, q_high)
        _imshow_with_percentile(axes[e, 1], np.flipud(coronal),  f"Coronal (y={yc})",   q_low, q_high)
        _imshow_with_percentile(axes[e, 2], np.flipud(sagittal), f"Sagittal (x={xc})",  q_low, q_high)

        axes[e, 0].set_ylabel(f"Echo {e+1}", fontsize=9)

    fig.suptitle(
        f"{title_prefix} • rows: echo1~{E}, cols: axial/coronal/sagittal",
        fontsize=11
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)







#%% evaluation
'''
모델 성능 평가를 위한 코드
'''
# ======================= Metrics utils (추가) =======================
def _to_float01(a, q_low=0.0, q_high=1.0):
    """퍼센타일 정규화(평가용). q_low=0, q_high=1이면 그대로 min-max 안함."""
    a = a.astype('float32', copy=False)
    if not (0.0 <= q_low < q_high <= 1.0):
        q_low, q_high = 0.0, 1.0
    if q_low == 0.0 and q_high == 1.0:
        lo, hi = float(a.min()), float(a.max() if a.max() > a.min() else a.min()+1e-6)
    else:
        lo, hi = np.quantile(a, [q_low, q_high])
        if hi <= lo:
            lo, hi = float(a.min()), float(a.max() if a.max() > a.min() else a.min()+1e-6)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

def _mse(a, b): return float(np.mean((a - b) ** 2))
def _mae(a, b): return float(np.mean(np.abs(a - b)))
def _psnr(a, b, data_range=1.0):
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12: return float('inf')
    import math
    return float(20 * math.log10(data_range) - 10 * math.log10(mse))

def _ncc(a, b):
    a_ = a - a.mean()
    b_ = b - b.mean()
    denom = (np.linalg.norm(a_) * np.linalg.norm(b_)) + 1e-12
    return float((a_ * b_).sum() / denom)

def _ssim_safe(a, b, data_range=1.0):
    """skimage가 있으면 SSIM, 없으면 None."""
    try:
        from skimage.metrics import structural_similarity as ssim
        # SSIM은 2D/3D 모두 지원(버전에 따라 3D가 안 되면 slice 평균으로 대체)
        if a.ndim == 2 and b.ndim == 2:
            return float(ssim(a, b, data_range=data_range))
        elif a.ndim == 3 and b.ndim == 3:
            try:
                return float(ssim(a, b, data_range=data_range))
            except Exception:
                # 축0(=Z) slice별 평균
                vals = []
                for z in range(min(a.shape[0], b.shape[0])):
                    vals.append(ssim(a[z], b[z], data_range=data_range))
                return float(np.mean(vals))
        else:
            return None
    except Exception:
        return None

def evaluate_volume(gt_vol, pred_vol, normalize=("percentile", 0.01, 0.99)):
    """
    gt_vol, pred_vol: (N,H,W) float/uint계열
    normalize: ("percentile", ql, qh) | ("none",) | ("minmax",)
    return dict: 전체/슬라이스별 통계
    """
    assert gt_vol.shape == pred_vol.shape, f"shape mismatch: {gt_vol.shape} vs {pred_vol.shape}"
    if normalize[0] == "percentile":
        ql, qh = normalize[1], normalize[2]
        G = _to_float01(gt_vol, ql, qh)
        P = _to_float01(pred_vol, ql, qh)
    elif normalize[0] == "minmax":
        G = _to_float01(gt_vol, 0.0, 1.0)
        P = _to_float01(pred_vol, 0.0, 1.0)
    else:
        G = gt_vol.astype('float32', copy=False)
        P = pred_vol.astype('float32', copy=False)

    # 전체 볼륨(3D)에 대한 지표
    out = {}
    out["volume"] = {
        "MSE":  _mse(G, P),
        "MAE":  _mae(G, P),
        "PSNR": _psnr(G, P, data_range=1.0),
        "NCC":  _ncc(G, P),
        "SSIM": _ssim_safe(G, P, data_range=1.0),
    }

    # slice별(축0)로도 기록
    rows = []
    for z in range(G.shape[0]):
        g2, p2 = G[z], P[z]
        rows.append({
            "z": z,
            "MSE":  _mse(g2, p2),
            "MAE":  _mae(g2, p2),
            "PSNR": _psnr(g2, p2),
            "NCC":  _ncc(g2, p2),
            "SSIM": _ssim_safe(g2, p2, data_range=1.0),
        })
    out["per_slice"] = rows
    return out

def evaluate_mip(gt_vol, pred_vol, normalize=("percentile", 0.01, 0.99)):
    """
    gt_vol, pred_vol: (N,H,W)
    MIP-Z/Y/X 각각 2D로 평가 + 평균
    """
    assert gt_vol.shape == pred_vol.shape, f"shape mismatch: {gt_vol.shape} vs {pred_vol.shape}"
    if normalize[0] == "percentile":
        ql, qh = normalize[1], normalize[2]
        G = _to_float01(gt_vol, ql, qh)
        P = _to_float01(pred_vol, ql, qh)
    elif normalize[0] == "minmax":
        G = _to_float01(gt_vol, 0.0, 1.0)
        P = _to_float01(pred_vol, 0.0, 1.0)
    else:
        G = gt_vol.astype('float32', copy=False)
        P = pred_vol.astype('float32', copy=False)

    gz, gy, gx = _mips(G)
    pz, py, px = _mips(P)

    # 시각화 규칙과 맞추려면 gy/py, gx/px에 flipud를 적용해도 되지만
    # 수치평가는 좌표변환 없이 직접 비교하는 게 일반적이라 그대로 둠.
    def _eval2d(a, b):
        return {
            "MSE":  _mse(a, b),
            "MAE":  _mae(a, b),
            "PSNR": _psnr(a, b, data_range=1.0),
            "NCC":  _ncc(a, b),
            "SSIM": _ssim_safe(a, b, data_range=1.0),
        }

    res = {
        "MIP_Z": _eval2d(gz, pz),
        "MIP_Y": _eval2d(gy, py),
        "MIP_X": _eval2d(gx, px),
    }
    # 간단 평균(SSIM 등 None이면 제외)
    avg = {}
    for k in ["MSE","MAE","PSNR","NCC","SSIM"]:
        vals = [res["MIP_Z"][k], res["MIP_Y"][k], res["MIP_X"][k]]
        vals = [v for v in vals if v is not None]
        avg[k] = float(np.mean(vals)) if vals else None
    res["mean"] = avg
    return res

# --- soft-MIP utilities -------------------------------------------------------
def _softmip(vol: np.ndarray, axis: int = 0, tau: float = 1.0) -> np.ndarray:
    """
    Soft-MIP along a given axis using softmax weights with temperature tau.
    vol: (N,H,W) 3D array; axis in {0,1,2}
    returns: 2D projection with the same two remaining dims
    """
    assert vol.ndim == 3, f"expect 3D, got {vol.shape}"
    eps = 1e-12
    tau = max(float(tau), 1e-6)  # avoid div-by-zero
    vmax = np.max(vol, axis=axis, keepdims=True)
    e = np.exp((vol - vmax) / tau)            # stable softmax
    w = e / (np.sum(e, axis=axis, keepdims=True) + eps)
    return np.sum(w * vol, axis=axis)

def _softmips(vol: np.ndarray, tau: float = 1.0):
    """
    Return soft-MIP along Z/Y/X (i.e., axis 0/1/2) in the same order
    as _mips(vol) does for hard-MIP.
    """
    gz = _softmip(vol, axis=0, tau=tau)  # soft-MIP over Z -> (H,W)
    gy = _softmip(vol, axis=1, tau=tau)  # soft-MIP over Y -> (N,W)
    gx = _softmip(vol, axis=2, tau=tau)  # soft-MIP over X -> (N,H)
    return gz, gy, gx

def evaluate_mip_soft(gt_vol, pred_vol, tau: float = 1.0,
                      normalize=("percentile", 0.01, 0.99)):
    """
    soft-MIP(τ)로 Z/Y/X 축 투영 후 SSIM 등 계산.
    gt_vol, pred_vol: (N,H,W)
    """
    assert gt_vol.shape == pred_vol.shape, f"shape mismatch: {gt_vol.shape} vs {pred_vol.shape}"
    # 동일 정규화 규칙 적용 (hard와 같은 경로)
    if normalize[0] == "percentile":
        ql, qh = normalize[1], normalize[2]
        G = _to_float01(gt_vol, ql, qh)
        P = _to_float01(pred_vol, ql, qh)
    elif normalize[0] == "minmax":
        G = _to_float01(gt_vol, 0.0, 1.0)
        P = _to_float01(pred_vol, 0.0, 1.0)
    else:
        G = gt_vol.astype('float32', copy=False)
        P = pred_vol.astype('float32', copy=False)

    gz, gy, gx = _softmips(G, tau=tau)
    pz, py, px = _softmips(P, tau=tau)

    def _eval2d(a, b):
        return {
            "MSE":  _mse(a, b),
            "MAE":  _mae(a, b),
            "PSNR": _psnr(a, b, data_range=1.0),
            "NCC":  _ncc(a, b),
            "SSIM": _ssim_safe(a, b, data_range=1.0),
        }

    tag = f"softMIP(tau={tau:g})"
    res = {
        f"{tag}_Z": _eval2d(gz, pz),
        f"{tag}_Y": _eval2d(gy, py),
        f"{tag}_X": _eval2d(gx, px),
    }
    # 축 평균
    avg = {}
    for k in ["MSE","MAE","PSNR","NCC","SSIM"]:
        vals = [res[f"{tag}_Z"][k], res[f"{tag}_Y"][k], res[f"{tag}_X"][k]]
        vals = [v for v in vals if v is not None]
        avg[k] = float(np.mean(vals)) if vals else None
    res["mean"] = avg
    return res

def evaluate_mip_both(gt_vol, pred_vol, tau: float = 1.0,
                      normalize=("percentile", 0.01, 0.99)):
    """
    편의용: hard/soft-MIP 결과를 한 번에 반환
    """
    hard = evaluate_mip(gt_vol, pred_vol, normalize=normalize)
    soft = evaluate_mip_soft(gt_vol, pred_vol, tau=tau, normalize=normalize)
    return {"hard": hard, "soft": soft}

