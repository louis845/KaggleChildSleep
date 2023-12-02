#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python inference_density_combined_statistics/inference_density_grid_search.py
