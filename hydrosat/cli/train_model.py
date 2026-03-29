from __future__ import annotations

import argparse
import os
from pathlib import Path

from hydrosat.core.common import (
    ARTIFACTS_DIR,
    DEFAULT_SEED,
    allow_mmengine_checkpoint_globals,
    apply_crop_size,
    apply_data_root,
    apply_learning_rate,
    apply_seed,
    disable_torch_weights_only_default,
    derive_work_dir,
    ensure_dir,
    ensure_pretrained_checkpoint,
    model_spec,
    normalize_data_root,
    parse_crop_size,
    relpath,
    resolve_existing_path,
    resolve_split_file,
    require_python_310,
    set_loader_ann_file,
    write_json,
)
from hydrosat.cli.generate_split_files import main as generate_split_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a Hydrosat segmentation model.')
    parser.add_argument('--model', choices=('segformer',), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--run-name')
    parser.add_argument('--max-iters', type=int)
    parser.add_argument('--val-interval', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--crop-size', type=parse_crop_size)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--train-ann')
    parser.add_argument('--val-ann')
    parser.add_argument('--load-from')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-root')
    parser.add_argument('--work-dir')
    parser.add_argument('--skip-pretrained-download', action='store_true')
    return parser.parse_args()


def build_smoke_split(source_path: Path, output_path: Path, limit: int) -> Path:
    lines = [
        line.strip()
        for line in source_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    ensure_dir(output_path.parent)
    output_path.write_text('\n'.join(lines[:limit]) + '\n', encoding='utf-8')
    return output_path


def main() -> None:
    args = parse_args()
    require_python_310()

    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmseg.utils import register_all_modules
    import torch

    allow_mmengine_checkpoint_globals()
    disable_torch_weights_only_default()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA training requested but torch.cuda.is_available() is False.')

    spec = model_spec(args.model)
    config_path = resolve_existing_path(args.config) if args.config else Path(spec['config'])
    cfg = Config.fromfile(str(config_path))
    cfg.launcher = 'none'
    data_root = apply_data_root(cfg, normalize_data_root(args.data_root))
    generate_split_files(data_root)

    default_work_dir = derive_work_dir(Path(spec['work_dir']), args.run_name)
    cfg.work_dir = args.work_dir or str(default_work_dir)
    apply_seed(cfg, args.seed)

    if args.lr is not None:
        apply_learning_rate(cfg, args.lr)
    if args.crop_size is not None:
        apply_crop_size(cfg, args.crop_size)

    cfg.resume = bool(args.resume)

    if args.load_from:
        cfg.load_from = str(resolve_existing_path(args.load_from, data_root=data_root))
    elif not args.resume and not args.skip_pretrained_download:
        cfg.load_from = str(ensure_pretrained_checkpoint(args.model))

    if args.smoke_test:
        cfg.train_cfg.max_iters = 1
        cfg.train_cfg.val_interval = 1
        cfg.default_hooks.checkpoint.interval = 1
        cfg.default_hooks.logger.interval = 1
        smoke_root = ensure_dir(ARTIFACTS_DIR / 'smoke_splits')
        train_source = resolve_existing_path(
            args.train_ann or resolve_split_file('train', data_root), data_root=data_root)
        val_source = resolve_existing_path(
            args.val_ann or resolve_split_file('val', data_root), data_root=data_root)
        train_ann = build_smoke_split(
            train_source,
            smoke_root / f'{args.model}_train.txt',
            limit=8)
        val_ann = build_smoke_split(
            val_source,
            smoke_root / f'{args.model}_val.txt',
            limit=4)
        cfg.train_dataloader.dataset.ann_file = str(train_ann.resolve())
        cfg.val_dataloader.dataset.ann_file = str(val_ann.resolve())
        cfg.test_dataloader.dataset.ann_file = str(val_ann.resolve())
    else:
        train_ann = None
        val_ann = None
        if args.train_ann:
            train_ann = set_loader_ann_file(cfg, 'train_dataloader', args.train_ann, data_root=data_root)
        if args.val_ann:
            val_ann = set_loader_ann_file(cfg, 'val_dataloader', args.val_ann, data_root=data_root)

    if args.max_iters is not None:
        cfg.train_cfg.max_iters = args.max_iters
    if args.val_interval is not None:
        cfg.train_cfg.val_interval = args.val_interval
        cfg.default_hooks.checkpoint.interval = args.val_interval

    register_all_modules(init_default_scope=True)

    print(f'Training {args.model} from {relpath(Path(cfg.filename))}')
    print(f'Work dir: {relpath(Path(cfg.work_dir))}')
    print(f'Checkpoint source: {cfg.load_from}')
    print(f'Resume mode: {cfg.resume}')
    print(f'Device: {args.device}')

    write_json(Path(cfg.work_dir) / 'train_manifest.json', {
        'kind': 'training_run',
        'model': args.model,
        'config': relpath(Path(config_path)),
        'work_dir': str(Path(cfg.work_dir).resolve()),
        'device': args.device,
        'run_name': args.run_name,
        'seed': args.seed,
        'data_root': str(data_root),
        'load_from': cfg.load_from,
        'resume': bool(cfg.resume),
        'max_iters': cfg.train_cfg.max_iters,
        'val_interval': cfg.train_cfg.val_interval,
        'learning_rate': cfg.optim_wrapper.optimizer.lr,
        'crop_size': list(cfg.model.data_preprocessor.size),
        'train_ann_file': str(train_ann) if train_ann else cfg.train_dataloader.dataset.ann_file,
        'val_ann_file': str(val_ann) if val_ann else cfg.val_dataloader.dataset.ann_file,
        'smoke_test': bool(args.smoke_test),
    })

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
