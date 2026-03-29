from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"classes": ["background", "water"], "num_classes": 2}, indent=2),
        encoding="utf-8",
    )


def _write_prob(path: Path, water: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probs = np.stack([1.0 - water, water], axis=0).astype(np.float16)
    np.save(path, probs)


def test_tune_and_export_segformer_ensemble(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    probs_a = tmp_path / "member_a" / "probs"
    probs_b = tmp_path / "member_b" / "probs"
    masks = tmp_path / "masks"

    _write_manifest(probs_a.parent / "manifest.json")
    _write_manifest(probs_b.parent / "manifest.json")

    target_1 = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    target_2 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    masks.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(masks / "tile_1.png"), target_1)
    cv2.imwrite(str(masks / "tile_2.png"), target_2)

    _write_prob(probs_a / "tile_1.npy", np.array([[0.2, 0.8], [0.3, 0.8]], dtype=np.float32))
    _write_prob(probs_b / "tile_1.npy", np.array([[0.3, 0.9], [0.2, 0.7]], dtype=np.float32))
    _write_prob(probs_a / "tile_2.npy", np.array([[0.8, 0.7], [0.2, 0.3]], dtype=np.float32))
    _write_prob(probs_b / "tile_2.npy", np.array([[0.7, 0.8], [0.3, 0.2]], dtype=np.float32))

    tune_dir = tmp_path / "tuning"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hydrosat.cli.tune_segformer_ensemble",
            "--probs-dir",
            str(probs_a),
            "--probs-dir",
            str(probs_b),
            "--mask-dir",
            str(masks),
            "--output-dir",
            str(tune_dir),
            "--threshold-start",
            "0.4",
            "--threshold-stop",
            "0.6",
            "--threshold-step",
            "0.1",
            "--min-component-sizes",
            "0",
            "--fill-hole-areas",
            "0",
        ],
        cwd=repo_root,
        check=True,
    )

    best = json.loads((tune_dir / "best.json").read_text(encoding="utf-8"))
    assert best["best"]["miou"] == 1.0
    assert (tune_dir / "results.csv").exists()

    export_root = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hydrosat.cli.export_ensemble_water_masks",
            "--probs-dir",
            str(probs_a),
            "--probs-dir",
            str(probs_b),
            "--threshold",
            "0.5",
            "--output-root",
            str(export_root),
        ],
        cwd=repo_root,
        check=True,
    )

    exported = sorted((export_root / "water").glob("*.png"))
    assert [path.name for path in exported] == ["tile_1.png", "tile_2.png"]
    assert set(np.unique(cv2.imread(str(exported[0]), cv2.IMREAD_UNCHANGED)).tolist()) <= {0, 1}
