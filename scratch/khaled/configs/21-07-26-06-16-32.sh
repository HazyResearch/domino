#!/usr/bin/env bash

# 21-07-26-06-16-32
# 964489c

# 6 configurations

python -m 07-23_train_cxr wandb.group=21-07-26-06-16-32 train.wd=1e-05 train.loss.gdro=True dataset.datapanel_pth=/media/4tb_hdd/siim/gazeslicer_dp_07-23-21.dp 'dataset.subgroup_columns=['gazeslicer_time']'
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-26-06-16-32 train.wd=1e-05 train.loss.gdro=False dataset.datapanel_pth=/media/4tb_hdd/siim/gazeslicer_dp_07-23-21.dp 'dataset.subgroup_columns=['gazeslicer_time']'
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-26-06-16-32 train.wd=0.001 train.loss.gdro=True dataset.datapanel_pth=/media/4tb_hdd/siim/gazeslicer_dp_07-23-21.dp 'dataset.subgroup_columns=['gazeslicer_time']'
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-26-06-16-32 train.wd=0.001 train.loss.gdro=False dataset.datapanel_pth=/media/4tb_hdd/siim/gazeslicer_dp_07-23-21.dp 'dataset.subgroup_columns=['gazeslicer_time']'
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-26-06-16-32 train.wd=0.1 train.loss.gdro=True dataset.datapanel_pth=/media/4tb_hdd/siim/gazeslicer_dp_07-23-21.dp 'dataset.subgroup_columns=['gazeslicer_time']'
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-26-06-16-32 train.wd=0.1 train.loss.gdro=False dataset.datapanel_pth=/media/4tb_hdd/siim/gazeslicer_dp_07-23-21.dp 'dataset.subgroup_columns=['gazeslicer_time']'
sleep 10s

# def gazeslicer_time_sweep():
# 
#     sweep = prod(
#         [
#             flag("train.wd", [1e-5, 1e-3, 1e-1]),
#             flag("train.loss.gdro", [True, False]),
#             flag(
#                 "dataset.datapanel_pth",
#                 ["/media/4tb_hdd/siim/gazeslicer_dp_07-23-21.dp"],
#             ),
#             flag("dataset.subgroup_columns", [["gazeslicer_time"]]),
#         ]
#     )
# 
#     return sweep
#