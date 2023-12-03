#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python time_binning_bootstrapping/time_bootstrapping.py
