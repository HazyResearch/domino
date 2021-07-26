#!/usr/bin/env bash

# 21-07-25-18-52-23
# 9b4fc97

# 6 configurations

python -m 07-23_train_cxr wandb.group=21-07-25-18-52-23 train.wd=1e-05 train.loss.gdro=True
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-25-18-52-23 train.wd=1e-05 train.loss.gdro=False
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-25-18-52-23 train.wd=0.001 train.loss.gdro=True
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-25-18-52-23 train.wd=0.001 train.loss.gdro=False
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-25-18-52-23 train.wd=0.1 train.loss.gdro=True
sleep 10s
python -m 07-23_train_cxr wandb.group=21-07-25-18-52-23 train.wd=0.1 train.loss.gdro=False
sleep 10s

# def gdro_sweep():
# 
#     sweep = prod(
#         [
#             flag("train.wd", [1e-5, 1e-3, 1e-1]),
#             flag("train.loss.gdro", [True, False]),
#         ]
#     )
# 
#     return sweep
# 