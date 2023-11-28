#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python inference_density_combined_statistics/inference_density_combined_detailed_stats.py
