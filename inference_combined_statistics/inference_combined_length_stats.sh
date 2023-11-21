#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python inference_combined_statistics/inference_combined_length_stats.py
