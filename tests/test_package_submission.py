from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hydrosat.cli.package_submission import validate_prediction_folder


def test_package_validation_accepts_binary_pngs(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for index in range(3):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[:, index : index + 2] = 1
        cv2.imwrite(str(pred_dir / f"tile_{index}.png"), mask)

    pngs = validate_prediction_folder(pred_dir, expected_count=3)

    assert len(pngs) == 3
