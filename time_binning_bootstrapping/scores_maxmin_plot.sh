#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python time_binning_bootstrapping/scores_maxmin_plot.py

