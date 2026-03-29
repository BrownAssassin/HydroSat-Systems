dataset_type = 'BaseSegDataset'
data_root = '.'

classes = ('background', 'water')
palette = [[0, 0, 0], [0, 0, 255]]

crop_size = (512, 512)
train_batch_size = 1
eval_batch_size = 1
num_workers = 0

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=90, pad_val=0, seg_pad_val=255),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=16,
        contrast_range=(0.9, 1.1),
        saturation_range=(0.9, 1.1),
        hue_delta=4),
    dict(type='PackSegInputs'),
]

eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='PackSegInputs'),
]

tta_scales = [1.0]
tta_flips = ['none', 'horizontal', 'vertical', 'both']

hydrosat_inference = dict(tta_scales=tta_scales, tta_flips=tta_flips)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/train.txt',
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=dict(classes=classes, palette=palette),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=eval_batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/val.txt',
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=dict(classes=classes, palette=palette),
        pipeline=eval_pipeline))

test_dataloader = dict(
    batch_size=eval_batch_size,
    num_workers=num_workers,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/test.txt',
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=dict(classes=classes, palette=palette),
        pipeline=eval_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'),
)

vis_backends = []
visualizer = dict(type='Visualizer', vis_backends=[])
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
