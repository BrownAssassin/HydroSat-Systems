# HydroSat Systems

Full-data training repo for the **ITU Ingenuity Cup: AI and Space Computing Challenge** preliminary round submission by **HydroSat Systems**.

This repository is focused on **Track 2: Space IntelligenceS Promoting Water Quality**. The preliminary-round task is binary pixel-level water extraction from remote-sensing imagery. The competition metrics are:

- `mIoU`
- `Kappa`

## Team

- **Team name:** HydroSat Systems
- **Team leader:** [Arv Bali](https://github.com/ArvBali2101)
- **Team members:**
  - [Arv Bali](https://github.com/ArvBali2101)
  - [Mrinank Sivakumar](https://github.com/BrownAssassin)
  - Myles Liu
  - Kenji Baritua

## Final Winning Result

- **Model family:** SegFormer-B5
- **Initialization:** ADE20K pretrained backbone
- **Winning ensemble members** (kept in the shared Google Drive bundle, not committed to Git):
  - `work_dirs/segformer_b5_train__prelim_water_fresh_ade_seed3407/best_mIoU_iter_20000.pth`
  - `work_dirs/segformer_b5_train__prelim_water_fresh_ade_seed6143/best_mIoU_iter_16000.pth`
- **Winning inference recipe:**
  - TTA scales: `0.75`, `1.0`, `1.25`
  - TTA flips: `none`, `horizontal`, `vertical`, `both`
  - threshold: `0.45`
  - min connected-component size: `0`
  - fill small holes up to area `512`
- **Best local validation result:**
  - `mIoU = 0.889628192968432`
  - `Kappa = 0.8801566605387882`

Locked references:

- `artifacts/final_selection/segformer_current_champion.json`
- `artifacts/tuning/20260328_wave2_tta12_ms075_100_125/best.json`

Final submission artifact:

- `submission/segformer_ensemble_3407_6143_tta12_ms075_100_125_thr045_hole512/`

That folder contains the checked-in competition-named ZIP plus the final `water/` mask folder. The submission ZIP is flat and contains only the `216` PNG mask files required by Zero2X.

## What Is Versioned

This repo intentionally keeps the minimum Git-tracked assets needed to retrain and document the winning result:

- raw competition dataset in `dataset-preliminary round/`
- all winning-workflow code, including the MMseg configs, in `hydrosat/`
- winning tuning and selection metadata in `artifacts/final_selection/` and `artifacts/tuning/`
- final mask folder and final ZIP in `submission/`

This repo intentionally does **not** keep large or derived artifacts that are better shared outside Git:

- pretrained and winning checkpoints under `checkpoints/` and `work_dirs/`
- local environments such as `openmmlab_env/`
- normalized dataset copy under `artifacts/datasets/`
- probability dumps under `artifacts/predictions/`
- verbose logs
- stale checkpoints
- older draft submissions

Because the checkpoints are no longer committed, a plain Git clone supports **full retraining from the raw dataset** out of the box. A faster inference-only recreation of the winning submission is still possible, but only if you also have the shared Google Drive bundle with the saved checkpoints.

## Repository Layout

```text
hydrosat/
  cli/        # Python entrypoints
  configs/    # mmseg dataset + model configs for the winning SegFormer run
  core/       # shared helpers, metrics, mask ops
  tools/      # environment bootstrap and validation helpers
requirements/ # pinned environment notes for OpenMMLab on Windows
dataset-preliminary round/
checkpoints/  # optional local / shared-Drive checkpoint bundle, not versioned
work_dirs/    # optional local / shared-Drive winning checkpoints, not versioned
artifacts/
submission/
tests/
```

## Windows Install

This project targets native Windows with NVIDIA CUDA and OpenMMLab.

### Prerequisites

- Python `3.10`
- Git
- NVIDIA GPU with CUDA available to `nvidia-smi`
- CUDA Toolkit `13.0`
- Visual Studio Build Tools 2022 with the C++ workload

### Bootstrap The Environment

Run the bootstrap helper from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\hydrosat\tools\bootstrap_openmmlab_env.ps1
```

This script:

- creates `openmmlab_env/`
- installs the CUDA-enabled PyTorch stack
- builds the Windows MMCV ops setup we use
- installs `mmengine`, `mmsegmentation`, and `mmdetection`
- installs this repo in editable mode as `hydrosat`

`hydrosat.cli.train_model` will automatically download the ADE20K SegFormer initialization checkpoint if `checkpoints/segformer_mit_b5_ade20k.pth` is missing.

Validate the environment:

```powershell
.\openmmlab_env\Scripts\python.exe .\hydrosat\tools\check_openmmlab_env.py --require-cuda
```

The bootstrap installs the repo in editable mode with the small development extras needed for the local regression tests.

Run the lightweight regression tests:

```powershell
.\openmmlab_env\Scripts\python.exe -m pytest
```

## Optional Quick Reproduction With Shared Drive Checkpoints

This path is **not** available from the Git clone alone anymore.

Before running the commands below, copy the shared Google Drive checkpoint bundle into the repo so these files exist locally:

- `work_dirs/segformer_b5_train__prelim_water_fresh_ade_seed3407/best_mIoU_iter_20000.pth`
- `work_dirs/segformer_b5_train__prelim_water_fresh_ade_seed6143/best_mIoU_iter_16000.pth`

If you do not have the Drive bundle, skip this section and use the full retraining flow instead.

### 1. Prepare the normalized binary dataset

```powershell
.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.prepare_preliminary_round_dataset `
  --raw-root ".\\dataset-preliminary round" `
  --output-root .\artifacts\datasets\preliminary_round_water
```

### 2. Export test probabilities for each winning checkpoint

```powershell
.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.predict_probs `
  --model segformer `
  --config .\hydrosat\configs\segformer_water_binary_train.py `
  --checkpoint .\work_dirs\segformer_b5_train__prelim_water_fresh_ade_seed3407\best_mIoU_iter_20000.pth `
  --split test `
  --data-root .\artifacts\datasets\preliminary_round_water `
  --tta `
  --tta-scales 0.75 1.0 1.25 `
  --tta-flips none horizontal vertical both `
  --save-probs `
  --output-dir .\artifacts\predictions\segformer_fresh_seed3407_test_tta12_ms075_100_125

.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.predict_probs `
  --model segformer `
  --config .\hydrosat\configs\segformer_water_binary_train.py `
  --checkpoint .\work_dirs\segformer_b5_train__prelim_water_fresh_ade_seed6143\best_mIoU_iter_16000.pth `
  --split test `
  --data-root .\artifacts\datasets\preliminary_round_water `
  --tta `
  --tta-scales 0.75 1.0 1.25 `
  --tta-flips none horizontal vertical both `
  --save-probs `
  --output-dir .\artifacts\predictions\segformer_fresh_seed6143_test_tta12_ms075_100_125
```

### 3. Export the final binary masks from the ensemble

```powershell
.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.export_ensemble_water_masks `
  --probs-dir .\artifacts\predictions\segformer_fresh_seed3407_test_tta12_ms075_100_125\probs `
  --probs-dir .\artifacts\predictions\segformer_fresh_seed6143_test_tta12_ms075_100_125\probs `
  --threshold 0.45 `
  --fill-hole-area 512 `
  --output-root .\submission\segformer_ensemble_3407_6143_tta12_ms075_100_125_thr045_hole512
```

### 4. Package the final flat Zero2X ZIP

```powershell
.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.package_submission `
  --pred-dir .\submission\segformer_ensemble_3407_6143_tta12_ms075_100_125_thr045_hole512\water `
  --team-name "<team-name>" `
  --leader-name "<team-leader>" `
  --email "<leader-email>" `
  --phone "<leader-phone>" `
  --output-dir .\submission\segformer_ensemble_3407_6143_tta12_ms075_100_125_thr045_hole512
```

## Retrain The Winning Model From Scratch

This is the default reproduction path for anyone cloning the repo without the shared Drive artifacts.

### 1. Prepare the normalized binary dataset

Use the same dataset prep command from the previous section.

### 2. Train the two winning seeds

```powershell
.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.train_model `
  --model segformer `
  --config .\hydrosat\configs\segformer_water_binary_train.py `
  --data-root .\artifacts\datasets\preliminary_round_water `
  --run-name prelim_water_fresh_ade_seed3407 `
  --seed 3407 `
  --max-iters 20000 `
  --val-interval 2000 `
  --device cuda

.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.train_model `
  --model segformer `
  --config .\hydrosat\configs\segformer_water_binary_train.py `
  --data-root .\artifacts\datasets\preliminary_round_water `
  --run-name prelim_water_fresh_ade_seed6143 `
  --seed 6143 `
  --max-iters 20000 `
  --val-interval 2000 `
  --device cuda
```

### 3. Recreate the winning validation comparison

Run validation TTA exports for the two winning checkpoints:

```powershell
.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.predict_probs `
  --model segformer `
  --config .\hydrosat\configs\segformer_water_binary_train.py `
  --checkpoint .\work_dirs\segformer_b5_train__prelim_water_fresh_ade_seed3407\best_mIoU_iter_20000.pth `
  --split val `
  --data-root .\artifacts\datasets\preliminary_round_water `
  --tta `
  --tta-scales 0.75 1.0 1.25 `
  --tta-flips none horizontal vertical both `
  --save-probs `
  --output-dir .\artifacts\predictions\segformer_fresh_seed3407_val_tta12_ms075_100_125

.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.predict_probs `
  --model segformer `
  --config .\hydrosat\configs\segformer_water_binary_train.py `
  --checkpoint .\work_dirs\segformer_b5_train__prelim_water_fresh_ade_seed6143\best_mIoU_iter_16000.pth `
  --split val `
  --data-root .\artifacts\datasets\preliminary_round_water `
  --tta `
  --tta-scales 0.75 1.0 1.25 `
  --tta-flips none horizontal vertical both `
  --save-probs `
  --output-dir .\artifacts\predictions\segformer_fresh_seed6143_val_tta12_ms075_100_125
```

Tune the winning ensemble recipe:

```powershell
.\openmmlab_env\Scripts\python.exe -m hydrosat.cli.tune_segformer_ensemble `
  --probs-dir .\artifacts\predictions\segformer_fresh_seed3407_val_tta12_ms075_100_125\probs `
  --probs-dir .\artifacts\predictions\segformer_fresh_seed6143_val_tta12_ms075_100_125\probs `
  --mask-dir .\artifacts\datasets\preliminary_round_water\val\masks `
  --output-dir .\artifacts\tuning\20260328_wave2_tta12_ms075_100_125 `
  --threshold-start 0.45 `
  --threshold-stop 0.55 `
  --threshold-step 0.005 `
  --min-component-sizes 0 512 1024 1536 `
  --fill-hole-areas 0 128 256 512
```

## Dataset Layout

The committed raw dataset is expected to keep this layout:

```text
dataset-preliminary round/
  Train/
    Images/
    Masks/
  Val/
    Images/
    Masks/
  Test/
    Images/
```

The normalized working layout generated by `hydrosat.cli.prepare_preliminary_round_dataset` is:

```text
artifacts/datasets/preliminary_round_water/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
  splits/
    train.txt
    val.txt
    test.txt
```

## Notes

- This repo is intentionally heavy because it versions the raw dataset directly, but the large checkpoints are shared outside Git through Google Drive.
- The README does not publish the leader phone number or email; those details belong only in the competition ZIP filename.
- The authoritative winning references are `artifacts/final_selection/segformer_current_champion.json` and `artifacts/tuning/20260328_wave2_tta12_ms075_100_125/best.json`.
- The repo containing our first attempt at tackling this challenge can be found [here](https://github.com/ArvBali2101/hydrosat-system).
