#!/bin/bash

python experiments/run_train_aug.py --config configs/train_custom_resnet18.yaml --method custom
python experiments/run_train_aug.py --config configs/train_custom_resnet50.yaml --method custom
python experiments/run_train_aug.py --config configs/train_mvdg_resnet18.yaml --method mvdg
python experiments/run_train_aug.py --config configs/train_mvdg_resnet50.yaml --method mvdg