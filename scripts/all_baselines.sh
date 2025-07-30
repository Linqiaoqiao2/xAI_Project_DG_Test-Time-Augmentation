#!/bin/bash

python experiments/run_baselines.py --config configs/baseline_resnet18_npt.yaml
python experiments/run_baselines.py --config configs/baseline_resnet18_pt.yaml
python experiments/run_baselines.py --config configs/baseline_resnet50_npt.yaml
python experiments/run_baselines.py --config configs/baseline_resnet50_pt.yaml