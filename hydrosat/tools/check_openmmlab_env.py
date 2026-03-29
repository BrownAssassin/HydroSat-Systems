from __future__ import annotations

import argparse
import importlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate the OpenMMLab environment.')
    parser.add_argument('--require-cuda', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    checks = {
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mmcv._ext': 'ok',
        'mmseg': 'ok',
        'mmdet': 'ok',
    }

    try:
        importlib.import_module('mmcv._ext')
    except Exception as exc:  # pragma: no cover - direct environment check
        checks['mmcv._ext'] = f'FAIL: {type(exc).__name__}: {exc}'

    for module_name in ('mmseg', 'mmdet'):
        try:
            module = importlib.import_module(module_name)
            checks[module_name] = getattr(module, '__version__', 'ok')
        except Exception as exc:  # pragma: no cover - direct environment check
            checks[module_name] = f'FAIL: {type(exc).__name__}: {exc}'

    for key, value in checks.items():
        print(f'{key}: {value}')

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit('CUDA is required but torch.cuda.is_available() is False.')
    if isinstance(checks['mmcv._ext'], str) and checks['mmcv._ext'].startswith('FAIL'):
        raise SystemExit('mmcv._ext is unavailable.')
    if isinstance(checks['mmseg'], str) and checks['mmseg'].startswith('FAIL'):
        raise SystemExit('mmseg is unavailable.')
    if isinstance(checks['mmdet'], str) and checks['mmdet'].startswith('FAIL'):
        raise SystemExit('mmdet is unavailable.')


if __name__ == '__main__':
    sys.exit(main())
