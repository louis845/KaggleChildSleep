#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python ./inference_confidence_statistics/inference_confidence_statistics_visualization.py