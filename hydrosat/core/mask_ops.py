from __future__ import annotations

import cv2
import numpy as np


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_size:
            kept[labels == label] = 1
    return kept


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return mask

    inverse = (mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse, connectivity=8
    )
    filled = mask.copy()
    height, width = mask.shape

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if not touches_border and area <= max_area:
            filled[labels == label] = 1
    return filled

