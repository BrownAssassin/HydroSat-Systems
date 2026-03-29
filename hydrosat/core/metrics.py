from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_iou(intersection: int, union: int) -> float:
    if union == 0:
        return 1.0
    return float(intersection / union)


@dataclass
class BinaryConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = prediction.astype(bool)
        truth = target.astype(bool)
        self.tp += int(np.logical_and(pred, truth).sum())
        self.fp += int(np.logical_and(pred, np.logical_not(truth)).sum())
        self.fn += int(np.logical_and(np.logical_not(pred), truth).sum())
        self.tn += int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())

    def compute(self) -> dict[str, float]:
        total = self.tp + self.fp + self.fn + self.tn
        if total == 0:
            return {
                "iou_background": 1.0,
                "iou_foreground": 1.0,
                "miou": 1.0,
                "kappa": 1.0,
                "accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
            }

        iou_foreground = _safe_iou(self.tp, self.tp + self.fp + self.fn)
        iou_background = _safe_iou(self.tn, self.tn + self.fp + self.fn)
        po = (self.tp + self.tn) / total
        pred_positive = self.tp + self.fp
        pred_negative = self.tn + self.fn
        true_positive = self.tp + self.fn
        true_negative = self.tn + self.fp
        pe = ((pred_positive * true_positive) + (pred_negative * true_negative)) / (total * total)
        if abs(1.0 - pe) < 1e-12:
            kappa = 1.0
        else:
            kappa = (po - pe) / (1.0 - pe)
        precision = 1.0 if (self.tp + self.fp) == 0 else self.tp / (self.tp + self.fp)
        recall = 1.0 if (self.tp + self.fn) == 0 else self.tp / (self.tp + self.fn)
        return {
            "iou_background": float(iou_background),
            "iou_foreground": float(iou_foreground),
            "miou": float((iou_background + iou_foreground) / 2.0),
            "kappa": float(kappa),
            "accuracy": float(po),
            "precision": float(precision),
            "recall": float(recall),
        }


def is_better(candidate: dict[str, float], best: dict[str, float] | None) -> bool:
    if best is None:
        return True
    return (candidate["miou"], candidate["kappa"]) > (best["miou"], best["kappa"])
