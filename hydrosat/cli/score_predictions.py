from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hydrosat.core.common import (
    build_output_dir,
    manifest_path,
    normalize_data_root,
    read_json,
    relpath,
    require_python_310,
    resolve_prediction_dir,
    resolve_split,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Score saved predictions against a labeled split.')
    parser.add_argument('--prediction-dir', required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--output')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_python_310()

    prediction_dir = resolve_prediction_dir(Path(args.prediction_dir).resolve())
    data_root = normalize_data_root(args.data_root)
    _, mask_root = resolve_split(args.split, data_root=data_root)

    predictions = sorted(prediction_dir.glob('*.png'))
    if not predictions:
        raise SystemExit(f'No PNG predictions found in {prediction_dir}')

    manifest_file = manifest_path(prediction_dir)
    source_manifest = read_json(manifest_file) if manifest_file.is_file() else {}
    class_names = tuple(source_manifest.get('classes') or ('background', 'water'))

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    for index, pred_path in enumerate(predictions, start=1):
        gt_path = mask_root / pred_path.name
        if not gt_path.exists():
            raise SystemExit(f'Missing ground-truth mask for {pred_path.name}')
        pred = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if pred is None or gt is None:
            raise SystemExit(f'Unable to read pair for {pred_path.name}')
        if pred.shape != gt.shape:
            raise SystemExit(f'Shape mismatch for {pred_path.name}: pred={pred.shape}, gt={gt.shape}')

        encoded = len(class_names) * gt.astype(np.int64) + pred.astype(np.int64)
        confusion += np.bincount(
            encoded.ravel(),
            minlength=len(class_names) * len(class_names)).reshape(len(class_names), len(class_names))
        print(f'[{index:>4}/{len(predictions)}] {pred_path.name}')

    intersection = np.diag(confusion).astype(np.float64)
    gt_total = confusion.sum(axis=1).astype(np.float64)
    pred_total = confusion.sum(axis=0).astype(np.float64)
    union = gt_total + pred_total - intersection

    iou = np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union > 0)
    acc = np.divide(intersection, gt_total, out=np.full_like(intersection, np.nan), where=gt_total > 0)

    payload = {
        'kind': 'prediction_score',
        'split': args.split,
        'data_root': str(data_root),
        'prediction_dir': str(prediction_dir),
        'source_manifest': source_manifest,
        'classes': list(class_names),
        'metrics': {
            'aAcc': float(intersection.sum() / gt_total.sum()),
            'mIoU': float(np.nanmean(iou)),
            'mAcc': float(np.nanmean(acc)),
        },
        'class_iou': {class_name: float(iou[idx]) for idx, class_name in enumerate(class_names)},
        'class_acc': {class_name: float(acc[idx]) for idx, class_name in enumerate(class_names)},
    }

    output_path = Path(args.output) if args.output else (
        build_output_dir('metrics', f'{prediction_dir.parent.name}_{args.split}')
        / 'metrics.json')
    write_json(output_path, payload)
    print(f'Saved metrics to {relpath(output_path.resolve())}')


if __name__ == '__main__':
    main()
