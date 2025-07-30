#!/bin/bash

python experiments/run_test_tta.py --config configs/test_tta_custom_resnet18_npt.yaml
python experiments/run_test_tta.py --config configs/test_tta_custom_resnet18_pt.yaml
python experiments/run_test_tta.py --config configs/test_tta_custom_resnet50_npt.yaml
python experiments/run_test_tta.py --config configs/test_tta_custom_resnet50_pt.yaml
python experiments/run_test_tta.py --config configs/test_tta_mvdg_resnet18_npt.yaml
python experiments/run_test_tta.py --config configs/test_tta_mvdg_resnet18_pt.yaml
python experiments/run_test_tta.py --config configs/test_tta_mvdg_resnet50_npt.yaml
python experiments/run_test_tta.py --config configs/test_tta_mvdg_resnet50_pt.yaml
python experiments/run_test_tta.py --config configs/test_tta_tencrop_resnet18_npt.yaml
python experiments/run_test_tta.py --config configs/test_tta_tencrop_resnet18_pt.yaml
python experiments/run_test_tta.py --config configs/test_tta_tencrop_resnet50_npt.yaml
python experiments/run_test_tta.py --config configs/test_tta_tencrop_resnet50_pt.yaml