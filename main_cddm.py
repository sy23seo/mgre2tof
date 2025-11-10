#%%
"""
내용설명 — 2025.10.13 작성
이 스크립트는 확산 모델(DDPM/DDIM 계열)과 ControlNet을 이용해
의료영상(TOF/mGRE) 기반의 사전학습(pretrain) 및 컨트롤 단계(control) 학습을 수행합니다.
"""

# ============================== 1) 환경 & 임포트 ==============================
import os
from pathlib import Path
import numpy as np

import math
import json, csv

os.environ["CUDA_VISIBLE_DEVICES"] = str(7)
print("Using CUDA devices:", os.environ["CUDA_VISIBLE_DEVICES"])

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_SEO_cldm_1 import (
    Unet,
    GaussianDiffusion,
    Trainer,
)

from denoising_diffusion_pytorch.utils_1 import (
    _natural_key,
    _load_center_volume,
    _select_resume_filename,
    _mips,
    _save_mip_subplot,
    _imshow_with_percentile,
    _load_condition_echo_center_volumes_from_full,
    _save_mip_subplot_multi_echo,
    _save_center_slice_subplot,
    _save_center_slice_subplot_multi_echo,
    evaluate_volume,
    evaluate_mip,
    evaluate_mip_soft,
)



def main():

    # ============================ 2) 전역 설정(모드/데이터) ============================
    STAGE = "pretrain"  # "pretrain" | "control"-----------------------------------------------------------------------------------------------------------------------------------

    pretraining_tof_h5 = "/SSD5_8TB/Yeon/data/merge_oasis_openneuro.h5"
    paired_tof_h5      = "/SSD5_8TB/Yeon/data/registered_all_subjects_m1to1.h5"
    # paired_tof_h5      = "/SSD5_8TB/Yeon/data/mgre_tof_m1to1_fast.h5"
    paired_mgre_h5     = paired_tof_h5  # ControlNet은 같은 파일에서 키만 다르게 읽음(워커당 I/O 1회)

    # -------- 입력/해상도 --------
    neighbors = 2                       # 위/아래 2장 → 총 5채널---------------------------------------------------------------------------------------------------------------------
    K = 2 * neighbors + 1               # 입력 채널 수
    IMG_SIZE = 512

    # -------- 전처리 정책 --------
    RESCALE_MODE   = "none"             # "none" | "zero_one" | "neg_one_one"
    AUTO_NORMALIZE = False

    # -------- 결과 저장 --------
    RESULTS_DIR_PRE  = "./rlt_stage1_5slices_FOV512"#---------------------------------------------------------------------------------------------------------------------------------
    RESULTS_DIR_CTRL = "./rlt_stage2_5slices"#---------------------------------------------------------------------------------------------------------------------------------
    # RESULTS_DIR_CTRL = "./results_cldm_stage2_5slices_last_80_150"


    # =============================== 3) 모델(U-Net) ===============================
    use_control = (STAGE == "control")
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
        channels=K,                          # 입력 채널 수(중앙 + 상하 이웃)
        use_control=use_control,             # ControlNet 사용 여부
        cond_channels=(5 * K if use_control else 0),  # 조건 채널 수(컨트롤 시 5배 스택)
        control_inject_down=True,            # 다운 경로에 컨트롤 주입
        control_inject_mid=True,             # 미드블록에 컨트롤 주입
        control_inject_up=False,             # 업 경로 주입 비활성화
        freeze_locked=(True if use_control else False),  # control 단계에서 백본 동결
    )

    # ========================== 4) Diffusion 래퍼 구성 ==========================
    mip_options = {
        "mip_loss": {"temp": 1.0},
        "data": {"use_slice": K},            # MIP 쪽에도 입력 슬라이스 개수 전달
    }

    diffusion = GaussianDiffusion(
        model,
        image_size=IMG_SIZE,                 # 학습/샘플링 이미지 크기
        timesteps=1000,                      # 학습 노이즈 스텝 수(β 스케줄 길이)
        sampling_timesteps=250,              # 샘플링(추론) 스텝 수
        auto_normalize=AUTO_NORMALIZE,       # 자동 정규화 사용 여부
        mip_options=mip_options,             # MIP/데이터 옵션 -----------------------------------------------------------------------------------------------------------------------------------
    )

    # =========================== 5) 기능 활성 로그 ===========================
    print('MIP enabled? ', hasattr(diffusion, 'mip_loss') and diffusion.mip_loss is not None)
    print('AWL enabled? ', hasattr(diffusion, 'awl') and diffusion.awl is not None)
    print('MIP num_slice:', getattr(diffusion.mip_loss, 'num_slice', None))

    # ==================== 6) Trainer 공통 학습 하이퍼파라미터 ====================
    common_trainer_kwargs = dict(
        train_batch_size=2,              # 배치 크기------------------------------------------------------------------------------------------------------------------------------------------ # 12 (FOV224), 8 (FOV320), 2 (FOV512)
        train_lr=8e-5,                       # 학습률
        train_num_steps=700000,              # 총 학습 스텝
        gradient_accumulate_every=9,         # 그래디언트 누적---------------------------------------------------------------------------------------------------------------------------------
        ema_decay=0.995,                     # EMA 감쇠
        amp=True,                            # 혼합정밀(AMP)
        calculate_fid=False,                 # FID 비계산
        use_val_loss_for_best=True,          # 밸리데이션 로스 기준 best 선정
        save_best_and_latest_only=True,     # best/latest만 보존하지 않음
        save_and_sample_every=1000,          # 저장/샘플 주기----------------------------------------------------------------------------------------------------------------------------------------
        num_samples=4,                       # 샘플 생성 개수
        neighbors=neighbors,                 # 데이터셋에 이웃 슬라이스 수 전달
        rescale=RESCALE_MODE,                # 스케일링 정책
        seed=42,                             # 시드
        crop_size=(IMG_SIZE, IMG_SIZE),      # 크롭 사이즈
        crop_mode_train="random",            # 학습: 랜덤 크롭``
        crop_mode_val="center",              # 검증: 센터 크롭
        augment_horizontal_flip=True,        # 수평 플립 증강

        # ----------수정 2025-10-20: 분할 비율 명시 (val/test 모두 0.05)
        # val_ratio=1,
        # test_ratio=1,
        val_subject_count = 1,
        test_subject_count = 1,
        eval_use_test_split=True,            # 테스트 DataLoader 생성·사용
        # ----------수정 2025-10-20 끝
        z_start=None,
        z_end=None, # 295
        # z_start=80,
        # z_end=150, # 295
    )

    # ========================== 7) Trainer 인스턴스 생성 ==========================
    if STAGE == "pretrain":
        trainer = Trainer(
            diffusion,
            folder=pretraining_tof_h5,       # 프리트레인 H5
            results_folder=RESULTS_DIR_PRE,  # 결과 폴더
            dataset_key="data",              # H5 데이터셋 키
            best_policy="any",   # ← ["val_loss", "mip", "base", "total_simple", "any"]
            **common_trainer_kwargs,
        )
    elif STAGE == "control":
        # Control 단계: 정합 보호(학습時 센터 크롭, 수평 플립 비활성화)
        common_trainer_kwargs.update(dict(crop_mode_train="center", augment_horizontal_flip=True))
        trainer = Trainer(
            diffusion,
            folder=paired_tof_h5,            # 타깃=TOF
            results_folder=RESULTS_DIR_CTRL, # 결과 폴더
            dataset_key="tof",               # 타깃 키
            cond_h5_path=paired_mgre_h5,     # 조건 H5 경로(동일 파일)
            cond_dataset_key="mag",          # 조건 키(mGRE magnitude)
            cond_rescale=RESCALE_MODE,       # 조건 스케일링 정책
            best_policy="any",   # ← ["val_loss", "mip", "base", "total_simple", "any"]
            **common_trainer_kwargs,
        )
    else:
        raise ValueError("STAGE must be 'pretrain' or 'control'")

    # ===================== 8) 체크포인트 로드(재개/가중치 이식) =====================
    # --- 신규 로직: pretrain도 재개 지원, control도 처음 시작 시 best/latest/숫자 최대로 이식 ---
    if STAGE == "pretrain":
        pre_models = Path(RESULTS_DIR_PRE) / "models"
        resume_name = _select_resume_filename(pre_models)

        if resume_name is not None:
            milestone = resume_name.replace("model-","").replace(".pt","")  # "best"|"latest"|숫자
            trainer.load(
                milestone,
                checkpoint_dir=RESULTS_DIR_PRE,
                strict=True,            # 아키텍처/하이퍼 동일 가정(엄격 로드)
                load_optimizer=True,    # 옵티마이저 상태까지 복원(이어달리기)
                load_ema=True,          # EMA 가중치 복원
            )
            print(f"[RESUME] Pretrain resumed from {resume_name}; step={trainer.step}")
        else:
            print("[RESUME] No pretrain checkpoint found. Starting pretrain from scratch.")

    elif STAGE == "control":
        ctrl_models = Path(RESULTS_DIR_CTRL) / "models"
        resume_name = _select_resume_filename(ctrl_models)# - "latest" : model-latest.pt  - "best"   : model-best.pt - "step"   : model-<가장 큰 숫자>.pt
        # resume_name = "model-35.pt" #----------------------------------------------------------------------------------------------------------------------------------------------------------------------

        if resume_name is not None:
            # ---- control 자체 재개 ----
            milestone = resume_name.replace("model-","").replace(".pt","")  # "best"|"latest"|숫자
            trainer.load(
                milestone,
                checkpoint_dir=RESULTS_DIR_CTRL,
                strict=True,
                load_optimizer=True,
                load_ema=True,
            )
            print(f"[RESUME] Control resumed from {resume_name}; step={trainer.step}")
        else:
            # ---- control 처음 시작: 프리트레인에서 best/latest/숫자 최대를 이식 ----
            pre_models = Path(RESULTS_DIR_PRE) / "models"
            init_name = _select_resume_filename(pre_models)
            if init_name is None:
                raise FileNotFoundError(
                    f"[INIT] No control checkpoints in {ctrl_models} and no pretrain checkpoints in {pre_models}. "
                    "Nothing to load for control initialization."
                )
            milestone = init_name.replace("model-","").replace(".pt","")  # "best"|"latest"|숫자
            trainer.load(
                milestone,
                checkpoint_dir=RESULTS_DIR_PRE,
                strict=False,            # 컨트롤 헤드 등 키 불일치 허용(백본 이식)
                load_optimizer=False,    # 새 옵티마이저로 시작
                load_ema=False,          # 새 EMA로 시작
            )
            print(f"[INIT] Control initialized from PRETRAIN {init_name}; step reset: {trainer.step}")



    # ============================== 9) 학습 시작 ===============================
    trainer.train()

    # ============================== 10) 평가 호출 ===============================
    # 학습 끝난 뒤, 밸리데이션 셋에서 샘플링/저장
    # trainer.sample_on_val(max_items=150)   # None이면 전체, 예시는 150개만
    # print("=== Sampling on Test sets ===")
    trainer.sample_on_test(max_items=None)





    # ============================== 11) NPY 볼륨 합치기 ===============================
    # ============================== 11) NPY 볼륨 합치기 ===============================
    flag = True  # 필요시 수동 토글
    if flag == True:
        print("=== Combining center slice NPYs into volumes ===")
        #----------수정 2025-10-20: val, test 모두 처리하도록 공통 루프화
        # sample_on_val / sample_on_test의 저장 규칙을 그대로 따릅니다(센터 채널만 사용).
        # splits_to_process = ["val", "test"]  # 필요시 ["val"]만 남겨도 됨
        splits_to_process = ["test"]  # 필요시 ["val"]만 남겨도 됨
        for SPLIT_TAG in splits_to_process:
            if STAGE == "control":
                save_root = Path(RESULTS_DIR_CTRL) / f"{SPLIT_TAG}_samples"
            else:
                save_root = Path(RESULTS_DIR_PRE) / f"{SPLIT_TAG}_samples"

            inputs_dir   = save_root / "inputs"
            samples_dir  = save_root / "samples"
            conds_dir    = save_root / "conditions"
            volumes_dir  = save_root / "volumes"
            volumes_dir.mkdir(parents=True, exist_ok=True)

            # 기존 코드 그대로 두되, prefix만 split에 맞게 치환
            # (예: val → "val-input-center", test → "test-input-center")
            # 예상 개수는 상황에 맞게 수정. 자동 추정이 필요하면 별도 유틸로 세도 됨.
            EXPECTED = 150  # [VAL]/[TEST] len(...) 로그에 맞춰 필요시 변경
            #----------수정 2025-10-20 끝

            # 각 폴더에서 center NPY만 모아서 (N,H,W)로 스택 (inputs/samples는 기존 그대로)
            inputs_vol  = _load_center_volume(inputs_dir,  prefix=f"{SPLIT_TAG}-input-center",  expected_len=EXPECTED)   #----------수정 2025-10-20
            samples_vol = _load_center_volume(samples_dir, prefix=f"{SPLIT_TAG}-sample-center", expected_len=EXPECTED)   #----------수정 2025-10-20

            # ### ---------------------------------수정: pretrain(STAGE!="control")일 때는 conditions 없이 진행
            if STAGE == "control":
                # ---------- 조건(cond)만: 블록별이 아니라 '에코별' 중앙 z 슬라이스를 (N,H,W)로 스택 ----------
                # K=5(이웃 5장), 에코 수 E는 파일에서 추론. 중앙 z = k_center = K//2
                # sample_on_val/test 둘 다 conditions 폴더에 동일 규칙("...-condition-slice-full-*.npy")으로 저장된다고 가정
                conditions_echo_vols = _load_condition_echo_center_volumes_from_full(
                    conds_dir, expected_len=EXPECTED, K=5, split_tag=SPLIT_TAG  # 'val' 또는 'test'
                )
                
                # 검증: 크기 일치 확인 (inputs/samples vs 첫번째 echo)
                assert inputs_vol.shape[1:] == samples_vol.shape[1:] == conditions_echo_vols[0].shape[1:], \
                    f"Shape mismatch: {inputs_vol.shape}, {samples_vol.shape}, {conditions_echo_vols[0].shape}"
            # ### ---------------------------------수정 끝

            # 저장 (float32 유지)
            np.save(volumes_dir / "inputs_center_volume.npy",   inputs_vol.astype('float32', copy=False))
            np.save(volumes_dir / "samples_center_volume.npy",  samples_vol.astype('float32', copy=False))

            # conds는 에코별로 개별 저장
            # ### ---------------------------------수정: control인 경우에만 conditions_* 저장
            if STAGE == "control":
                E = len(conditions_echo_vols)
                conds_echo_paths = []
                for e in range(E):
                    outp = volumes_dir / f"conditions_center_echo{e+1}_volume.npy"
                    np.save(outp, conditions_echo_vols[e].astype('float32', copy=False))
                    conds_echo_paths.append(outp)

                print(f"[OK] Saved volumes ({SPLIT_TAG}):",                                #----------수정 2025-10-20
                    (volumes_dir / "inputs_center_volume.npy"),
                    (volumes_dir / "samples_center_volume.npy"),
                    *conds_echo_paths)
            else:
                # pretrain 모드에서는 conditions 없음
                print(f"[OK] Saved volumes ({SPLIT_TAG}):",                                #----------수정 2025-10-20
                    (volumes_dir / "inputs_center_volume.npy"),
                    (volumes_dir / "samples_center_volume.npy"),
                    "(no conditions in pretrain)")
            # ### ---------------------------------수정 끝

            # ============================== 12.1) Center slice 플롯 저장 ===============================
            _save_center_slice_subplot(inputs_vol,  volumes_dir / "inputs_center_Slices.png",    "Inputs(center)")
            _save_center_slice_subplot(samples_vol, volumes_dir / "samples_center_Slices.png",   "Samples(center)")
            # ❌ (지우기) _save_center_slice_subplot(conditions_echo_vols,   volumes_dir / "conditions_center_Slices.png","Conditions(center)")

            # --- ADD #1: 에코별(행) × 3뷰(열)로 '가운데 슬라이스' 한 장에 저장 ---
            # ### ---------------------------------수정: control일 때만 conditions 시각화
            if STAGE == "control":
                _save_center_slice_subplot_multi_echo(
                    conditions_echo_vols,
                    volumes_dir / "conditions_center_Slices_echowise.png",
                    title_prefix="Conditions(center) — Echo-wise"
                )

                # --- ADD #2: 에코별로 개별(1×3) '가운데 슬라이스' PNG 저장 (원하면) ---
                for e, V in enumerate(conditions_echo_vols, start=1):
                    _save_center_slice_subplot(
                        V,
                        volumes_dir / f"conditions_echo{e}_center_Slices.png",
                        title_prefix=f"Conditions(center) — Echo {e}"
                    )
            # ### ---------------------------------수정 끝

            # ============================== 12.2) MIP 플롯 저장 ===============================
            _save_mip_subplot(inputs_vol,   volumes_dir / "inputs_center_MIPs.png",   "Inputs(center)")
            _save_mip_subplot(samples_vol,  volumes_dir / "samples_center_MIPs.png",  "Samples(center)")

            # ### ---------------------------------수정: control일 때만 conditions MIP 플롯 저장
            if STAGE == "control":
                _save_mip_subplot_multi_echo(
                    conditions_echo_vols,
                    volumes_dir / "conditions_center_MIPs_echowise.png",
                    title_prefix="Conditions(center) — Echo-wise"
                )

                # 에코별로 개별 MIP PNG 저장
                for e, V in enumerate(conditions_echo_vols, start=1):
                    _save_mip_subplot(V, volumes_dir / f"conditions_echo{e}_center_MIPs.png",
                                    title_prefix=f"Conditions(center) — Echo {e}")
                
                print(f"[OK] Saved MIP figures ({SPLIT_TAG}):",                            #----------수정 2025-10-20
                    volumes_dir / "inputs_center_MIPs.png",
                    volumes_dir / "samples_center_MIPs.png",
                    volumes_dir / "conditions_center_MIPs_echowise.png")
            else:
                print(f"[OK] Saved MIP figures ({SPLIT_TAG}):",                            #----------수정 2025-10-20
                    volumes_dir / "inputs_center_MIPs.png",
                    volumes_dir / "samples_center_MIPs.png",
                    "(no conditions in pretrain)")
            # ### ---------------------------------수정 끝
        #----------수정 2025-10-20 끝



    # ============================== 13) 정량평가 (추가) ===============================
    #----------수정 2025-10-20: 평가도 val/test 모두 수행하도록 루프화
    import json, csv
    for SPLIT_TAG in ["val", "test"]:
        if STAGE == "control":
            save_root = Path(RESULTS_DIR_CTRL) / f"{SPLIT_TAG}_samples"
        else:
            save_root = Path(RESULTS_DIR_PRE) / f"{SPLIT_TAG}_samples"

        targets_dir    = save_root / "inputs"                     # GT로 쓸 입력(또는 별도 targets 디렉토리)   # 필요시 변경
        target_prefix  = f"{SPLIT_TAG}-input-center"              # "val-input-center" | "test-input-center"
        volumes_dir    = save_root / "volumes"

        try:
            gt_vol = _load_center_volume(targets_dir, target_prefix, expected_len=EXPECTED)  # (N,H,W)
        except Exception as e:
            print(f"[WARN] Cannot load GT targets ({SPLIT_TAG}): {e}")
            gt_vol = None

        # 13.2) 샘플 vs GT 평가 (볼륨 / MIP)
        if gt_vol is not None:
            try:
                # samples_center_volume.npy 는 위 11)에서 이미 저장됨
                samples_vol = np.load(volumes_dir / "samples_center_volume.npy")

                # 볼륨 단위 기본 화질지표
                vol_metrics  = evaluate_volume(gt_vol, samples_vol, normalize=("percentile", 0.01, 0.99))

                # (기존) 하드 MIP 지표
                mip_metrics_hard  = evaluate_mip(gt_vol, samples_vol, normalize=("percentile", 0.01, 0.99))

                # --------------------soft MIP SSIM추가: 소프트 MIP(τ) 지표 계산--------------------
                # 필요 시 tau 스윕도 가능: [0.5, 1.0, 2.0] 등
                tau = 1.0
                mip_metrics_soft  = evaluate_mip_soft(gt_vol, samples_vol, tau=tau, normalize=("percentile", 0.01, 0.99))
                # ----------------------------------------------------------------------------------

                # 저장 (split별 파일명 분리)
                with open(volumes_dir / f"metrics_volume_{SPLIT_TAG}.json", "w") as f:
                    json.dump(vol_metrics["volume"], f, indent=2)

                with open(volumes_dir / f"metrics_mip_hard_{SPLIT_TAG}.json", "w") as f:
                    json.dump(mip_metrics_hard, f, indent=2)

                # --------------------soft MIP SSIM추가: 소프트 MIP 결과 저장-----------------------
                with open(volumes_dir / f"metrics_mip_soft_tau{str(tau).replace('.','p')}_{SPLIT_TAG}.json", "w") as f:
                    json.dump(mip_metrics_soft, f, indent=2)
                # ----------------------------------------------------------------------------------

                # 슬라이스별 CSV (볼륨 지표)
                per_slice_csv = volumes_dir / f"metrics_per_slice_{SPLIT_TAG}.csv"
                with open(per_slice_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["z","MSE","MAE","PSNR","NCC","SSIM"])
                    writer.writeheader()
                    for row in vol_metrics["per_slice"]:
                        writer.writerow(row)

                print(f"[OK] Saved metrics ({SPLIT_TAG}):",
                      volumes_dir / f"metrics_volume_{SPLIT_TAG}.json",
                      volumes_dir / f"metrics_mip_hard_{SPLIT_TAG}.json",
                      volumes_dir / f"metrics_mip_soft_tau{str(tau).replace('.','p')}_{SPLIT_TAG}.json",  # --------------------soft MIP SSIM추가
                      per_slice_csv)
            except Exception as e:
                print(f"[WARN] Evaluation failed for {SPLIT_TAG}: {e}")
        else:
            print(f"[WARN] Skipped metrics ({SPLIT_TAG}): GT not available")
    #----------수정 2025-10-20 끝


    # 13.3) (선택) Inputs/Conditions도 GT와 비교하고 싶다면 아래처럼 추가 가능
    # if gt_vol is not None:
    #     inp_metrics = evaluate_volume(gt_vol, inputs_vol)
    #     con_metrics = [evaluate_volume(gt_vol, V) for V in conditions_echo_vols]
    #     with open(volumes_dir / "metrics_inputs_volume.json", "w") as f:
    #         json.dump(inp_metrics["volume"], f, indent=2)
    #     with open(volumes_dir / "metrics_conditions_volume.json", "w") as f:
    #         json.dump([m["volume"] for m in con_metrics], f, indent=2)






# ======================= 10) 스크립트 진입점 & 멀티프로세싱 =======================
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("fork")  # 리눅스에서 성능/호환 유리(이미 설정된 경우 예외 무시)
    except RuntimeError:
        pass
    main()

