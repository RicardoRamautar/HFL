custom_imports = dict(
    imports=[
        'projects.mmdet3d_plugin.models.detectors.cmt',
        'projects.mmdet3d_plugin.datasets.pipelines.loading',
        'HFL.datasets.nuscenes_dataset'
    ],
    allow_failed_imports=False)
scenes = [
    'scene-0002', 'scene-0060', 'scene-0124', 'scene-0126', 'scene-0372',
    'scene-0357', 'scene-0351', 'scene-0120', 'scene-0359', 'scene-0716',
    'scene-0542', 'scene-0397', 'scene-0741', 'scene-0247', 'scene-0751',
    'scene-0663', 'scene-0681', 'scene-0717', 'scene-0700', 'scene-0242',
    'scene-0209', 'scene-0705', 'scene-0068', 'scene-0696', 'scene-0670',
    'scene-0880', 'scene-0648', 'scene-0878', 'scene-0902', 'scene-0649',
    'scene-0873', 'scene-0897', 'scene-0458', 'scene-0436', 'scene-0184',
    'scene-0790', 'scene-1012', 'scene-1018', 'scene-0402', 'scene-1099',
    'scene-1090', 'scene-1108', 'scene-0381', 'scene-0975', 'scene-0149',
    'scene-0005', 'scene-0020', 'scene-0949', 'scene-0355', 'scene-0979',
    'scene-0011', 'scene-0196', 'scene-0666', 'scene-0508', 'scene-0709',
    'scene-0715', 'scene-0803', 'scene-0237', 'scene-0287', 'scene-0299',
    'scene-0260', 'scene-0710', 'scene-0251', 'scene-0255', 'scene-0171',
    'scene-0222', 'scene-0230', 'scene-0233', 'scene-0677', 'scene-0588',
    'scene-0578', 'scene-0883', 'scene-0876', 'scene-0645', 'scene-0592',
    'scene-0889', 'scene-0887', 'scene-0866', 'scene-0185', 'scene-0178',
    'scene-0863', 'scene-1000', 'scene-1025', 'scene-0413', 'scene-1096',
    'scene-1080', 'scene-1093', 'scene-1074'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.1, 0.1, 0.2]
out_size_factor = 8
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='pts_bbox_NuScenes/mAP',
    rule='greater')
dataset_type = 'HFLNuScenesDataset'
data_root = '/opt/src/mmdetection3d/data/nuscenes/'
pkl_root = '/tudelft.net/staff-umbrella/rdramautar/datasets/nuscenes_infos/new/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)
ida_aug_conf = dict(
    resize_lim=(0.47, 0.625),
    final_dim=(320, 800),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True)
train_pipeline = [
    dict(
        type='CustomLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4]),
    dict(
        type='CustomLoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4]),
    dict(type='CustomLoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='UnifiedObjectSample',
        sample_2d=True,
        mixup_rate=0.5,
        db_sampler=dict(
            type='UnifiedDataBaseSampler',
            data_root=
            '/tudelft.net/staff-umbrella/rdramautar/datasets/nuscenes_infos/new/',
            info_path=
            '/tudelft.net/staff-umbrella/rdramautar/datasets/nuscenes_infos/new/CustomNuScenesDataset_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='CustomLoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4]))),
    dict(type='ModalMask3D', mode='train'),
    dict(
        type='GlobalRotScaleTransAll',
        rot_range=[-0.785, 0.785],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='CustomRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='PointShuffle'),
    dict(
        type='ResizeCropFlipImage',
        data_aug_conf=dict(
            resize_lim=(0.47, 0.625),
            final_dim=(320, 800),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True),
        training=True),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                   'transformation_3d_flow', 'rot_degree', 'gt_bboxes_3d',
                   'gt_labels_3d'))
]
test_pipeline = [
    dict(
        type='CustomLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4]),
    dict(
        type='CustomLoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4]),
    dict(type='CustomLoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.625),
                    final_dim=(320, 800),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=True),
                training=False),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type='HFLNuScenesDataset',
            data_root='/opt/src/mmdetection3d/data/nuscenes/',
            ann_file=
            '/tudelft.net/staff-umbrella/rdramautar/datasets/nuscenes_infos/new/nuscenes_infos_train_2.pkl',
            load_interval=1,
            pipeline=[
                dict(
                    type='CustomLoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=[0, 1, 2, 3, 4]),
                dict(
                    type='CustomLoadPointsFromMultiSweeps',
                    sweeps_num=10,
                    use_dim=[0, 1, 2, 3, 4]),
                dict(type='CustomLoadMultiViewImageFromFiles'),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='UnifiedObjectSample',
                    sample_2d=True,
                    mixup_rate=0.5,
                    db_sampler=dict(
                        type='UnifiedDataBaseSampler',
                        data_root=
                        '/tudelft.net/staff-umbrella/rdramautar/datasets/nuscenes_infos/new/',
                        info_path=
                        '/tudelft.net/staff-umbrella/rdramautar/datasets/nuscenes_infos/new/CustomNuScenesDataset_dbinfos_train.pkl',
                        rate=1.0,
                        prepare=dict(
                            filter_by_difficulty=[-1],
                            filter_by_min_points=dict(
                                car=5,
                                truck=5,
                                bus=5,
                                trailer=5,
                                construction_vehicle=5,
                                traffic_cone=5,
                                barrier=5,
                                motorcycle=5,
                                bicycle=5,
                                pedestrian=5)),
                        classes=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        sample_groups=dict(
                            car=2,
                            truck=3,
                            construction_vehicle=7,
                            bus=4,
                            trailer=6,
                            barrier=2,
                            motorcycle=6,
                            bicycle=6,
                            pedestrian=2,
                            traffic_cone=2),
                        points_loader=dict(
                            type='CustomLoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=5,
                            use_dim=[0, 1, 2, 3, 4]))),
                dict(type='ModalMask3D', mode='train'),
                dict(
                    type='GlobalRotScaleTransAll',
                    rot_range=[-0.785, 0.785],
                    scale_ratio_range=[0.9, 1.1],
                    translation_std=[0.5, 0.5, 0.5]),
                dict(
                    type='CustomRandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
                dict(
                    type='ObjectNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(type='PointShuffle'),
                dict(
                    type='ResizeCropFlipImage',
                    data_aug_conf=dict(
                        resize_lim=(0.47, 0.625),
                        final_dim=(320, 800),
                        bot_pct_lim=(0.0, 0.0),
                        rot_lim=(0.0, 0.0),
                        H=900,
                        W=1600,
                        rand_flip=True),
                    training=True),
                dict(
                    type='NormalizeMultiviewImage',
                    mean=[103.53, 116.28, 123.675],
                    std=[57.375, 57.12, 58.395],
                    to_rgb=False),
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'lidar2img', 'depth2img', 'cam2img',
                               'pad_shape', 'scale_factor', 'flip',
                               'pcd_horizontal_flip', 'pcd_vertical_flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                               'pcd_rotation', 'pts_filename',
                               'transformation_3d_flow', 'rot_degree',
                               'gt_bboxes_3d', 'gt_labels_3d'))
            ],
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            modality=dict(
                use_lidar=True,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            test_mode=False,
            enable_hfl=True,
            scenes=[
                'scene-0002', 'scene-0060', 'scene-0124', 'scene-0126',
                'scene-0372', 'scene-0357', 'scene-0351', 'scene-0120',
                'scene-0359', 'scene-0716', 'scene-0542', 'scene-0397',
                'scene-0741', 'scene-0247', 'scene-0751', 'scene-0663',
                'scene-0681', 'scene-0717', 'scene-0700', 'scene-0242',
                'scene-0209', 'scene-0705', 'scene-0068', 'scene-0696',
                'scene-0670', 'scene-0880', 'scene-0648', 'scene-0878',
                'scene-0902', 'scene-0649', 'scene-0873', 'scene-0897',
                'scene-0458', 'scene-0436', 'scene-0184', 'scene-0790',
                'scene-1012', 'scene-1018', 'scene-0402', 'scene-1099',
                'scene-1090', 'scene-1108', 'scene-0381', 'scene-0975',
                'scene-0149', 'scene-0005', 'scene-0020', 'scene-0949',
                'scene-0355', 'scene-0979', 'scene-0011', 'scene-0196',
                'scene-0666', 'scene-0508', 'scene-0709', 'scene-0715',
                'scene-0803', 'scene-0237', 'scene-0287', 'scene-0299',
                'scene-0260', 'scene-0710', 'scene-0251', 'scene-0255',
                'scene-0171', 'scene-0222', 'scene-0230', 'scene-0233',
                'scene-0677', 'scene-0588', 'scene-0578', 'scene-0883',
                'scene-0876', 'scene-0645', 'scene-0592', 'scene-0889',
                'scene-0887', 'scene-0866', 'scene-0185', 'scene-0178',
                'scene-0863', 'scene-1000', 'scene-1025', 'scene-0413',
                'scene-1096', 'scene-1080', 'scene-1093', 'scene-1074'
            ],
            scene_translation=
            '/tudelft.net/staff-umbrella/rdramautar/HFL/data/scene_name_to_token.json',
            box_type_3d='LiDAR')),
    val=dict(
        type='HFLNuScenesDataset',
        data_root='/opt/src/mmdetection3d/data/nuscenes/',
        ann_file=
        '/tudelft.net/staff-umbrella/rdramautar/datasets/nuscenes_infos/new/nuscenes_infos_val_2.pkl',
        load_interval=1,
        pipeline=[
            dict(
                type='CustomLoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4]),
            dict(
                type='CustomLoadPointsFromMultiSweeps',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4]),
            dict(type='CustomLoadMultiViewImageFromFiles'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='ResizeCropFlipImage',
                        data_aug_conf=dict(
                            resize_lim=(0.47, 0.625),
                            final_dim=(320, 800),
                            bot_pct_lim=(0.0, 0.0),
                            rot_lim=(0.0, 0.0),
                            H=900,
                            W=1600,
                            rand_flip=True),
                        training=False),
                    dict(
                        type='NormalizeMultiviewImage',
                        mean=[103.53, 116.28, 123.675],
                        std=[57.375, 57.12, 58.395],
                        to_rgb=False),
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        enable_hfl=False,
        box_type_3d='LiDAR'),
    test=dict(
        type='HFLNuScenesDataset',
        data_root='/opt/src/mmdetection3d/data/nuscenes/',
        ann_file=
        '/tudelft.net/staff-umbrella/rdramautar/datasets/nuscenes_infos/new/nuscenes_infos_val_2.pkl',
        load_interval=1,
        pipeline=[
            dict(
                type='CustomLoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4]),
            dict(
                type='CustomLoadPointsFromMultiSweeps',
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4]),
            dict(type='CustomLoadMultiViewImageFromFiles'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='ResizeCropFlipImage',
                        data_aug_conf=dict(
                            resize_lim=(0.47, 0.625),
                            final_dim=(320, 800),
                            bot_pct_lim=(0.0, 0.0),
                            rot_lim=(0.0, 0.0),
                            H=900,
                            W=1600,
                            rand_flip=True),
                        training=False),
                    dict(
                        type='NormalizeMultiviewImage',
                        mean=[103.53, 116.28, 123.675],
                        std=[57.375, 57.12, 58.395],
                        to_rgb=False),
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img'])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        enable_hfl=False,
        box_type_3d='LiDAR'))
model = dict(
    type='CmtDetector',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN', in_channels=[1024, 2048], out_channels=256, num_outs=2),
    pts_voxel_layer=dict(
        num_point_features=5,
        max_num_points=10,
        voxel_size=[0.1, 0.1, 0.2],
        max_voxels=(120000, 160000),
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CmtHead',
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(
                num_class=10,
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ])
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            max_num=300,
            voxel_size=[0.1, 0.1, 0.2],
            num_classes=10),
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='CmtTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=1024,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction='mean',
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0)),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
                code_weights=[
                    2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2
                ]),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1024, 1024, 40],
            voxel_size=[0.1, 0.1, 0.2],
            out_size_factor=8,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1024, 1024, 40],
            out_size_factor=8,
            pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.1, 0.1, 0.2],
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200)))
optimizer = dict(type='AdamW', lr=7e-05, weight_decay=0.01)
optimizer_config = dict(
    type='CustomFp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
    custom_fp16=dict(
        pts_voxel_encoder=False, pts_middle_encoder=False,
        pts_bbox_head=False))
lr_config = dict(
    policy='cyclic',
    target_ratio=(6, 0.001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/tudelft.net/staff-umbrella/rdramautar/HFL/experiments/exp_040'
resume_from = '/tudelft.net/staff-umbrella/rdramautar/HFL/experiments/exp_040/epoch_13.pth'
workflow = [('train', 1)]
gpu_ids = [0]
