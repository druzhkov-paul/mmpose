log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
cudnn_benchmark = True
workflow = [('train', 1)]
checkpoint_config = dict(interval=50)
evaluation = dict(interval=5, metric='mAP', key_indicator='AP')

optimizer = dict(
    type='Adam',
    lr=0.001,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[200, 260])
total_epochs = 300
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

image_size = 448
data_cfg = dict(
    image_size=image_size,
    base_size=image_size // 2,
    base_sigma=2,
    heatmap_size=[image_size // 4, image_size // 2],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    num_scales=2,
    scale_aware_sigma=False,
)

# model settings
model = dict(
    type='BottomUp',
    # pretrained='/media/cluster_fs/user/pdruzhko/experiments/mmdetection/hpe/0038/initial.pth',
    # pretrained='/media/paul/cluster_fs/user/pdruzhko/experiments/mmdetection/hpe/0038/initial.pth',
    backbone=dict(
        type='EHRNet',
        version='efficienthrnet_bm2',
        norm_eval=False
    ),
    keypoint_head=dict(
        type='BottomUpHigherResolutionHeadDecoupled',
        in_channels=21,
        num_joints=17,
        tag_per_joint=True,
        extra=dict(final_conv_kernel=1, ),
        num_deconv_layers=1,
        num_deconv_filters=[32],
        num_deconv_kernels=[4],
        num_basic_blocks=1,
        cat_output=[True],
        with_ae_loss=[True, False]),
    train_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        img_size=data_cfg['image_size']),
    test_cfg=dict(
        num_joints=channel_cfg['dataset_joints'],
        max_num_people=30,
        scale_factor=[1],
        with_heatmaps=[True, True],
        with_ae=[True, False],
        project2image=False,
        nms_kernel=5,
        nms_padding=2,
        tag_per_joint=True,
        detection_threshold=0.1,
        tag_threshold=1,
        use_detection_val=True,
        ignore_too_much=False,
        adjust=True,
        refine=True,
        dist_reweight=True,
        flip_test=False,
        use_udp=False),
    loss_pose=dict(
        type='MultiLossFactory',
        num_joints=17,
        num_stages=2,
        ae_loss_type='exp',
        with_ae_loss=[True, False],
        push_loss_factor=[0.001, 0.001],
        pull_loss_factor=[0.001, 0.001],
        with_heatmaps_loss=[True, True],
        heatmaps_loss_factor=[1.0, 1.0],
    ),
)

albu_train_transforms = [
    dict(type='SmallestMaxSize',
         max_size=image_size,
         always_apply=True),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=(-0.25, 0.5),
        rotate_limit=30,
        interpolation=1,
        border_mode=0,
        value=0,
        mask_value=0,
        p=0.5),
    dict(type='PadIfNeeded',
         min_height=image_size,
         min_width=image_size,
         border_mode=0,
         value=0,
         mask_value=0,
         always_apply=True),
    dict(type='RandomCrop',
         height=image_size,
         width=image_size,
         always_apply=True)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Albu',
         transforms=albu_train_transforms,
         bbox_params=None,
         kp_params=dict(
             type='KeypointParams',
             format='xy',
             label_fields=[],
             remove_invisible=True,
             filter_lost_elements=True),
         keymap={
             'img': 'image',
             'mask': 'masks',
             'joints': 'keypoints'
         },
         update_pad_shape=False,
         skip_img_without_anno=True),
    dict(type='BottomUpRandomFlip', flip_prob=0.5),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='BottomUpGenerateTarget',
        sigma=2,
        max_num_people=30,
        use_udp=False  # Not used in the current code patch
    ),
    dict(
        type='Collect',
        keys=['img', 'joints', 'targets', 'masks'],
        meta_keys=[]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BottomUpGetImgSize', test_scale_factor=[1], use_udp=False, size_divisor=32),
    dict(
        type='BottomUpResizeAlign',
        transforms=[
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ],
        use_udp=False,
        size_divisor=32),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'image_file', 'aug_data', 'test_scale_factor', 'base_size',
            'center', 'scale', 'flip_index'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/coco'
data = dict(
    samples_per_gpu=14,
    workers_per_gpu=2,
    train=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/images/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/images/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='BottomUpCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/images/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
)
