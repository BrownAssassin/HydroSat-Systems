from __future__ import annotations

import numpy as np

from hydrosat.core.metrics import BinaryConfusionMatrix


def test_binary_metrics_perfect_prediction() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    prediction = target.copy()
    confusion = BinaryConfusionMatrix()
    confusion.update(prediction, target)
    metrics = confusion.compute()

    assert metrics["miou"] == 1.0
    assert metrics["kappa"] == 1.0
    assert metrics["iou_foreground"] == 1.0
    assert metrics["iou_background"] == 1.0
