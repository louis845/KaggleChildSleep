#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python time_binning_bootstrapping/scores_distribution_plot.py

