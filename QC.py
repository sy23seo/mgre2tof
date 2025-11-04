#%%
import numpy as np
from pathlib import Path

# utils_1에서 우리가 쓸 함수들만 가져옴
from denoising_diffusion_pytorch.utils_1 import (
    _save_center_slice_subplot,
    _save_mip_subplot,
)

def main():
    # 1) 네 볼륨 NPY 경로
    npy_path = "/home/milab/SSD5_8TB/Yeon/code/denoising-diffusion-pytorch2/inputs_center_volume.npy"

    # 2) 저장할 폴더
    out_dir = Path("./debug_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) 로드
    vol = np.load(npy_path)  # shape 예상: (N, H, W) = (Z, Y, X)
    print("[INFO] loaded volume:")
    print("  path:", npy_path)
    print("  dtype:", vol.dtype)
    print("  shape:", vol.shape)  # (Z,H,W)
    print("  min:", float(vol.min()), "max:", float(vol.max()))
    print("  mean:", float(vol.mean()), "std:", float(vol.std()))

    # 4) center slice 3-view (axial / coronal / sagittal)
    center_png = out_dir / "center_views.png"
    _save_center_slice_subplot(
        vol=vol,
        out_path=center_png,
        title_prefix="Input(center)",
        # q_low, q_high 기본값(0.01, 0.99) 이미 utils_1에서 쓰는 값이랑 동일하게 전달 가능
        q_low=0.01,
        q_high=0.99,
    )
    print(f"[SAVE] center views -> {center_png}")

    # 5) MIP (Z, Y, X 방향 max projection)
    mip_png = out_dir / "mip_views.png"
    _save_mip_subplot(
        vol=vol,
        out_path=mip_png,
        title_prefix="Input(center)",
    )
    print(f"[SAVE] MIP views -> {mip_png}")

if __name__ == "__main__":
    main()








































#%%
import h5py
import numpy as np
from pathlib import Path

# utils_1.py에서 필요한 시각화 함수들 임포트
from denoising_diffusion_pytorch.utils_1 import (
    _save_center_slice_subplot,
    _save_mip_subplot,
    _save_center_slice_subplot_multi_echo,
    _save_mip_subplot_multi_echo,
)

def _center_crop_3d(vol_3d: np.ndarray, crop_hw: int = 320):
    """
    vol_3d: (Z, H, W)
    가운데 기준으로 H,W를 crop_hw x crop_hw로 잘라서 반환.
    """
    assert vol_3d.ndim == 3, f"expected (Z,H,W), got {vol_3d.shape}"
    z, h, w = vol_3d.shape
    if h < crop_hw or w < crop_hw:
        raise ValueError(f"volume is smaller ({h},{w}) than crop {crop_hw}")
    y0 = (h - crop_hw) // 2
    x0 = (w - crop_hw) // 2
    return vol_3d[:, y0:y0+crop_hw, x0:x0+crop_hw]

# -------------------------------------------------
# 1. H5 파일 경로
# -------------------------------------------------
paired_tof_h5 = "/SSD5_8TB/Yeon/data/registered_all_subjects_m1to1.h5"

# -------------------------------------------------
# 2. 출력 폴더 만들기 (PNG 저장 위치)
# -------------------------------------------------
out_dir = Path("./debug_viz")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 3. H5 열어서 한 subject 고르고 데이터 불러오기
# -------------------------------------------------
with h5py.File(paired_tof_h5, "r") as f:
    # subject 그룹들 이름 모으기
    subject_ids = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
    if len(subject_ids) == 0:
        raise RuntimeError("H5 파일 안에서 subject group을 못 찾았어요.")

    subject_id = subject_ids[-1]  # 마지막 subject 선택
    print(f"[INFO] using subject:", subject_id)

    grp = f[subject_id]

    # ToF 볼륨: (Z,H,W)
    tof_vol = grp["tof"][...].astype(np.float32)

    # mGRE magnitude 전체 에코: (E,Z,H,W)
    mag_4d = grp["mgre"][...].astype(np.float32)

print("tof_vol shape   :", tof_vol.shape)
print("mgre_4d shape    :", mag_4d.shape)

# -------------------------------------------------
# 4. mGRE 첫번째 에코(에코 인덱스 0)만 가져오기
# -------------------------------------------------
echo0_vol = mag_4d[0]  # (Z,H,W)

# === 추가: 320x320 중앙 크롭 ===
tof_vol_cropped   = _center_crop_3d(tof_vol,   crop_hw=320)
echo0_vol_cropped = _center_crop_3d(echo0_vol, crop_hw=320)

# 모든 에코에 대해서도 crop 적용
volumes_3d_list = [
    _center_crop_3d(mag_4d[e], crop_hw=320) for e in range(mag_4d.shape[0])
]

print("tof_vol_cropped shape   :", tof_vol_cropped.shape)
print("echo0_vol_cropped shape :", echo0_vol_cropped.shape)

# -------------------------------------------------
# 5. center slice subplot 저장 (cropped 사용)
# -------------------------------------------------
_save_center_slice_subplot(
    vol=echo0_vol_cropped,  # mGRE echo0 (320x320 crop)
    out_path=out_dir / f"{subject_id}_mgre_echo0_center.png",
    title_prefix=f"{subject_id} mGRE echo0 (center)",
)

_save_center_slice_subplot(
    vol=tof_vol_cropped,    # TOF (320x320 crop)
    out_path=out_dir / f"{subject_id}_tof_center.png",
    title_prefix=f"{subject_id} TOF (center)",
)

# 모든 에코 한 장(행=에코)
_save_center_slice_subplot_multi_echo(
    volumes_3d_list=volumes_3d_list,  # 이미 crop된 리스트
    out_path=out_dir / f"{subject_id}_mgre_allEcho_center.png",
    title_prefix=f"{subject_id} mGRE all echoes (center slices)",
)

# -------------------------------------------------
# 6. MIP subplot 저장 (cropped 사용)
# -------------------------------------------------
_save_mip_subplot(
    vol=echo0_vol_cropped,
    out_path=out_dir / f"{subject_id}_mgre_echo0_mip.png",
    title_prefix=f"{subject_id} mGRE echo0 (MIP)",
)

_save_mip_subplot(
    vol=tof_vol_cropped,
    out_path=out_dir / f"{subject_id}_tof_mip.png",
    title_prefix=f"{subject_id} TOF (MIP)",
)

# 모든 에코에 대해 echo별 MIP를 행으로 쌓은 이미지
_save_mip_subplot_multi_echo(
    volumes_3d_list=volumes_3d_list,  # crop된 echo별 볼륨 리스트
    out_path=out_dir / f"{subject_id}_mgre_allEcho_mip.png",
    title_prefix=f"{subject_id} mGRE all echoes (MIP)",
)

print("done. 이미지들은", out_dir, "아래에 저장됐습니다.")


# -------------------------------------------------
# 7. Registration quality check - vessels only (추가 코드)
#    ※ 위의 기존 코드는 건드리지 않고, 여기부터 추가.
# -------------------------------------------------

import matplotlib.pyplot as plt

def _get_vessel_mask(vol_3d: np.ndarray, perc: float = 99.5):
    """
    밝은 혈관만 추출하기 위한 마스크 생성.
    아이디어:
      - 혈관은 TOF나 mGRE(특히 초기 echo)에서 매우 밝다.
      - 상위 perc 퍼센타일 (예: 99%) 이상만 1로 두고 나머지는 0으로 둔다.
    vol_3d: (Z,H,W) float
    return: mask_3d (Z,H,W) bool
    """
    thr = np.percentile(vol_3d, perc)
    mask = vol_3d >= thr
    return mask

def _dice_coef(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-8):
    """
    Dice similarity coefficient = 2|A∩B| / (|A|+|B|)
    mask_a, mask_b: bool arrays with same shape
    """
    inter = np.logical_and(mask_a, mask_b).sum(dtype=np.float64)
    size_a = mask_a.sum(dtype=np.float64)
    size_b = mask_b.sum(dtype=np.float64)
    dice = (2.0 * inter) / (size_a + size_b + eps)
    return float(dice), float(inter), float(size_a), float(size_b)

def _mips_binary(mask_3d: np.ndarray):
    """
    단순 max projection (binary). shape: (Z,H,W)
    return:
      mip_axial   : (H,W)   <- Z 방향으로 max
      mip_coronal : (Z,W)   <- Y(H) 방향으로 max
      mip_sagittal: (Z,H)   <- X(W) 방향으로 max
    """
    mip_axial    = mask_3d.max(axis=0)   # collapse Z -> (H,W)
    mip_coronal  = mask_3d.max(axis=1)   # collapse H -> (Z,W)
    mip_sagittal = mask_3d.max(axis=2)   # collapse W -> (Z,H)
    return mip_axial, mip_coronal, mip_sagittal

def _save_vessel_overlap_debug(
    tof_vol_3d: np.ndarray,
    gre_vol_3d: np.ndarray,
    out_png: Path,
    perc: float = 99.0,
    subject_tag: str = ""
):
    """
    - 혈관 마스크(밝은 픽셀 상위 perc%)를 TOF와 mGRE 각각 구함
    - Dice 출력
    - 세 가지 투영축(Z/Y/X 비슷)을 시각화해서,
      R채널=TOF 혈관 / G채널=mGRE 혈관 / 노랑=겹침 으로 저장
    - 또한 텍스트 리포트도 저장
    """
    # 1) vessel mask 생성
    mask_tof = _get_vessel_mask(tof_vol_3d, perc=perc)
    mask_gre = _get_vessel_mask(gre_vol_3d, perc=perc)

    # 2) Dice
    dice, inter, sz_t, sz_g = _dice_coef(mask_tof, mask_gre)
    print(f"[REG CHECK] vessel Dice (perc={perc}): {dice:.4f}")
    print(f"[REG CHECK] intersection={inter} tof_vox={sz_t} gre_vox={sz_g}")

    # 텍스트 리포트 저장
    report_path = out_png.with_suffix(".txt")
    with open(report_path, "w") as rf:
        rf.write(f"subject: {subject_tag}\n")
        rf.write(f"percentile threshold: {perc}\n")
        rf.write(f"dice: {dice:.6f}\n")
        rf.write(f"intersection voxels: {inter}\n")
        rf.write(f"tof vessel voxels:  {sz_t}\n")
        rf.write(f"mgre vessel voxels: {sz_g}\n")
    print(f"[SAVE] vessel overlap report -> {report_path}")

    # 3) MIP들 (binary mask 기준)
    tof_mip_ax, tof_mip_co, tof_mip_sa = _mips_binary(mask_tof.astype(np.uint8))
    gre_mip_ax, gre_mip_co, gre_mip_sa = _mips_binary(mask_gre.astype(np.uint8))

    # RGB overlay: R=TOF, G=mGRE
    def make_rgb_overlay(r_mask, g_mask):
        r = r_mask.astype(np.float32)
        g = g_mask.astype(np.float32)
        b = np.zeros_like(r, dtype=np.float32)
        rgb = np.stack([r, g, b], axis=-1)  # (H,W,3) or (Z,W,3)... depends on view
        rgb = np.clip(rgb, 0.0, 1.0)
        return rgb

    ov_ax = make_rgb_overlay(tof_mip_ax, gre_mip_ax)
    ov_co = make_rgb_overlay(tof_mip_co, gre_mip_co)
    ov_sa = make_rgb_overlay(tof_mip_sa, gre_mip_sa)

    # -------------------------------------------------
    # 여기서부터: 그림만 np.flipud 적용
    # -------------------------------------------------
    tof_mip_ax_f = np.flipud(tof_mip_ax)
    gre_mip_ax_f = np.flipud(gre_mip_ax)
    ov_ax_f      = np.flipud(ov_ax)

    tof_mip_co_f = np.flipud(tof_mip_co)
    gre_mip_co_f = np.flipud(gre_mip_co)
    ov_co_f      = np.flipud(ov_co)

    tof_mip_sa_f = np.flipud(tof_mip_sa)
    gre_mip_sa_f = np.flipud(gre_mip_sa)
    ov_sa_f      = np.flipud(ov_sa)

    # 4) figure 저장 (flip된 것들을 플롯)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle(f"{subject_tag} vessel overlap (R=TOF,G=mGRE) perc={perc}", fontsize=12)

    # row 0: axial(Z-proj)
    axes[0,0].imshow(tof_mip_ax_f, cmap="Reds")
    axes[0,0].set_title("TOF axial MIP")
    axes[0,1].imshow(gre_mip_ax_f, cmap="Greens")
    axes[0,1].set_title("mGRE axial MIP")
    axes[0,2].imshow(ov_ax_f)
    axes[0,2].set_title("overlap axial")

    # row 1: coronal(Y-proj)
    axes[1,0].imshow(tof_mip_co_f, cmap="Reds")
    axes[1,0].set_title("TOF coronal MIP")
    axes[1,1].imshow(gre_mip_co_f, cmap="Greens")
    axes[1,1].set_title("mGRE coronal MIP")
    axes[1,2].imshow(ov_co_f)
    axes[1,2].set_title("overlap coronal")

    # row 2: sagittal(X-proj)
    axes[2,0].imshow(tof_mip_sa_f, cmap="Reds")
    axes[2,0].set_title("TOF sagittal MIP")
    axes[2,1].imshow(gre_mip_sa_f, cmap="Greens")
    axes[2,1].set_title("mGRE sagittal MIP")
    axes[2,2].imshow(ov_sa_f)
    axes[2,2].set_title("overlap sagittal")

    # axes 미관 정리
    for axrow in axes:
        for ax in axrow:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[SAVE] vessel overlap figure -> {out_png}")
    print(f"[REG CHECK] Done vessel registration check for {subject_tag}")


# 실제 실행 (cropped된 echo0 vs tof 사용)
_overlap_png = out_dir / f"{subject_id}_vessel_overlap.png"
_save_vessel_overlap_debug(
    tof_vol_cropped,
    echo0_vol_cropped,
    out_png=_overlap_png,
    perc=99.5,               # 상위 1%만 혈관으로 간주; 필요하면 98~97로 낮춰도 됨
    subject_tag=subject_id,
)
