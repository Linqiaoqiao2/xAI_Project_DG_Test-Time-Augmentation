#!/bin/bash

python experiments/run_train_aug.py --config configs/test_tta_on_trained_custom_resnet18.yaml --method custom
python experiments/run_train_aug.py --config configs/test_tta_on_trained_custom_resnet50.yaml --method custom
python experiments/run_train_aug.py --config configs/test_tta_on_trained_mvdg_resnet18.yaml --method mvdg
python experiments/run_train_aug.py --config configs/test_tta_on_trained_mvdg_resnet50.yaml --method mvdg