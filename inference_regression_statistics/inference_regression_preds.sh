#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python inference_regression_statistics/inference_regression_preds.py --device 0 --use_anglez_only
