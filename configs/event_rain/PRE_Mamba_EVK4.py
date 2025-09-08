_base_ = ["../_base_/default_runtime_EVK4.py"]

# misc custom setting
batch_size = 4  # bs: total bs in all gpus  # 1 batch/gpu
lr_per_sample = 0.00008
# gride_size = 0.1
# gride_size = 3
gride_x = 1
gride_y = 1
gride_size = 0.2

event_size = [1280 , 720]

gather_num = 5
epoch = 200
eval_epoch = 20


seed = 1024  # train process will init a random seed and record
num_worker = 24
empty_cache = False
enable_amp = False

# dataset settings
data_root = "/data/real/EVK4_DATA_DIR/scene2"  # The last directory name (e.g., scene2) is arbitrary and only required as a placeholder.
num_classes = 2
ignore_index = -1
names = ["background","rain"]
dataset_type = "EVK4MultiScansDataset"
size = (720, 1280)      # frame size

shuffle_orders = False
scan_modulation = False

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=16,
    backbone=dict(
        type="PRE_Mamba_EVK4",
        in_channels=5,  # x, y, z, i, t
        gather_num=gather_num,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2),
        enc_depths=(2,4),
        enc_channels=(16, 32),
        dec_depths=(2),
        dec_channels=(16,),
        mlp_ratio=4,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=shuffle_orders,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
        dict(type="FrequencyLoss", loss_weight=1.0),
    ],
)

# scheduler settings
optimizer = dict(type="AdamW", lr=lr_per_sample * batch_size, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[lr_per_sample * batch_size, lr_per_sample * batch_size * 0.1],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=lr_per_sample * batch_size * 0.1)]

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        event_size = event_size,
        data_width = event_size[0],
        data_height = event_size[1],
        gather_num=gather_num,
        scan_modulation=scan_modulation,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=gride_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "tn"),
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "tn"),
                feat_keys=("coord", "strength", "tn"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        event_size = event_size,
        gather_num=gather_num,
        scan_modulation=scan_modulation,
        transform=[
            dict(
                type="GridSample",
                grid_size=gride_size,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "tn"),
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "tn"),
                feat_keys=("coord", "strength", "tn"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        event_size = event_size,
        gather_num=gather_num,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment", "tn": "origin_tn"}),
            dict(
                type="GridSample",
                grid_size=gride_size / 2,
                hash_type="fnv",
                mode="train",
                keys=("coord", "strength", "segment", "tn"),
                return_inverse=True
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            scale2kitti=None,
            voxelize=dict(
                type="GridSample",
                grid_size=gride_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "strength", "tn"),
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "tn"),
                    feat_keys=("coord", "strength", "tn"),
                ),
            ],
            aug_transform=[
                [
                    dict(type="Add")
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
