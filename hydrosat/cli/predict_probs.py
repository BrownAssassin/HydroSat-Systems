from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from hydrosat.core.common import (
    DEFAULT_TTA_FLIPS,
    DEFAULT_TTA_SCALES,
    allow_mmengine_checkpoint_globals,
    build_output_dir,
    disable_torch_weights_only_default,
    list_images_from_ann,
    list_images,
    manifest_path,
    model_spec,
    normalize_data_root,
    normalize_input_root,
    relpath,
    resolve_existing_path,
    require_python_310,
    resolve_split,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run inference and save probabilities/predictions.')
    parser.add_argument('--model', choices=('segformer',), required=True)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=('val', 'test'))
    parser.add_argument('--input-root')
    parser.add_argument('--ann-file')
    parser.add_argument('--data-root')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--tta-scales', nargs='+', type=float)
    parser.add_argument(
        '--tta-flips',
        nargs='+',
        choices=('none', 'horizontal', 'vertical', 'both'))
    parser.add_argument('--save-probs', action='store_true')
    parser.add_argument('--save-preds', action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--limit', type=int)
    return parser.parse_args()


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def apply_flip(image: np.ndarray, direction: str) -> np.ndarray:
    if direction == 'none':
        return image
    if direction == 'horizontal':
        return cv2.flip(image, 1)
    if direction == 'vertical':
        return cv2.flip(image, 0)
    if direction == 'both':
        return cv2.flip(image, -1)
    raise ValueError(f'Unsupported flip direction: {direction}')


def extract_logits(sample, torch_module):
    logits = getattr(sample, 'seg_logits', None)
    if logits is None and isinstance(sample, dict):
        logits = sample.get('seg_logits')
    if logits is None:
        raise RuntimeError('Model output did not include seg_logits.')
    tensor = logits.data if hasattr(logits, 'data') else logits
    if isinstance(tensor, np.ndarray):
        tensor = torch_module.from_numpy(tensor)
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise RuntimeError(f'Expected [C,H,W] logits, got shape {tuple(tensor.shape)}')
    return tensor


def reverse_flip(logits, direction: str, torch_module):
    if direction == 'none':
        return logits
    if direction == 'horizontal':
        return torch_module.flip(logits, dims=[2])
    if direction == 'vertical':
        return torch_module.flip(logits, dims=[1])
    if direction == 'both':
        return torch_module.flip(logits, dims=[1, 2])
    raise ValueError(f'Unsupported flip direction: {direction}')


def main() -> None:
    args = parse_args()
    require_python_310()

    if not args.split and not args.input_root:
        raise SystemExit('Pass either --split or --input-root.')
    if args.split and args.input_root:
        raise SystemExit('Use only one of --split or --input-root.')
    if not args.save_probs and not args.save_preds:
        args.save_probs = True
        args.save_preds = True

    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    import torch
    from mmseg.apis import inference_model, init_model

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA inference requested but torch.cuda.is_available() is False.')

    allow_mmengine_checkpoint_globals()
    disable_torch_weights_only_default()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise SystemExit(f'Missing checkpoint: {checkpoint}')

    data_root = normalize_data_root(args.data_root)
    selected_ann = None
    if args.split:
        image_root, _ = resolve_split(args.split, data_root=data_root)
        if args.ann_file:
            selected_ann = resolve_existing_path(args.ann_file, data_root=data_root)
        input_label = f'{args.split}_{Path(selected_ann).stem}' if selected_ann else args.split
    else:
        image_root = normalize_input_root(Path(args.input_root).resolve())
        if args.ann_file:
            selected_ann = resolve_existing_path(args.ann_file)
        input_label = f'{image_root.name}_{Path(selected_ann).stem}' if selected_ann else image_root.name

    image_paths = (
        list_images_from_ann(image_root, selected_ann)
        if selected_ann else list_images(image_root)
    )
    if args.limit is not None:
        image_paths = image_paths[:args.limit]

    output_dir = Path(args.output_dir) if args.output_dir else build_output_dir(
        'predictions', f"{args.model}_{input_label}_{'tta' if args.tta else 'base'}")
    probs_dir = output_dir / 'probs'
    preds_dir = output_dir / 'preds'
    if args.save_probs:
        probs_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)

    spec = model_spec(args.model)
    config_path = resolve_existing_path(args.config) if args.config else Path(spec['config'])
    device = 'cuda:0' if args.device == 'cuda' else 'cpu'
    model = init_model(str(config_path), str(checkpoint), device=device)
    class_names = tuple(getattr(model, 'dataset_meta', {}).get('classes', ()))
    num_classes = len(class_names)
    tta_scales = tuple(args.tta_scales) if args.tta_scales else DEFAULT_TTA_SCALES
    tta_flips = tuple(args.tta_flips) if args.tta_flips else DEFAULT_TTA_FLIPS

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f'Unable to read image: {image_path}')
        original_height, original_width = image.shape[:2]

        if args.tta:
            probs_sum = None
            num_views = 0
            for scale in tta_scales:
                resized = resize_image(image, scale)
                for flip in tta_flips:
                    augmented = apply_flip(resized, flip)
                    sample = inference_model(model, augmented)
                    logits = extract_logits(sample, torch)
                    logits = reverse_flip(logits, flip, torch)
                    logits = torch.nn.functional.interpolate(
                        logits.unsqueeze(0),
                        size=(original_height, original_width),
                        mode='bilinear',
                        align_corners=False)[0]
                    view_probs = torch.softmax(logits, dim=0)
                    if probs_sum is None:
                        probs_sum = torch.zeros_like(view_probs)
                    probs_sum += view_probs
                    num_views += 1
            assert probs_sum is not None
            probs = probs_sum / float(num_views)
        else:
            sample = inference_model(model, str(image_path))
            logits = extract_logits(sample, torch)
            probs = torch.softmax(logits, dim=0)

        num_classes = int(probs.shape[0])
        prediction = probs.argmax(dim=0).to(torch.uint8).cpu().numpy()

        if args.save_probs:
            np.save(probs_dir / f'{image_path.stem}.npy', probs.numpy().astype(np.float16))
        if args.save_preds:
            cv2.imwrite(str(preds_dir / f'{image_path.stem}.png'), prediction)

        print(f'[{index:>4}/{len(image_paths)}] {image_path.name}')

    manifest = {
        'kind': 'prediction',
        'model': args.model,
        'checkpoint': str(checkpoint),
        'config': relpath(Path(config_path)),
        'split': args.split,
        'ann_file': str(selected_ann) if selected_ann else None,
        'data_root': str(data_root) if args.split or args.data_root else None,
        'input_root': str(image_root),
        'num_images': len(image_paths),
        'num_classes': num_classes,
        'classes': list(class_names) if class_names else None,
        'tta_enabled': bool(args.tta),
        'tta_scales': list(tta_scales),
        'tta_flips': list(tta_flips),
        'probs_dir': str(probs_dir) if args.save_probs else None,
        'preds_dir': str(preds_dir) if args.save_preds else None,
    }
    write_json(output_dir / 'manifest.json', manifest)
    print(f'Saved manifest to {relpath(manifest_path(output_dir).resolve())}')


if __name__ == '__main__':
    main()
