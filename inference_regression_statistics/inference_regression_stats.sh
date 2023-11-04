#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python inference_regression_statistics/inference_regression_statistics.py --device 0 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --use_anglez_only
