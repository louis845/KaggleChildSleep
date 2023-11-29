#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

while [ ! -f "./TEMPFILE_DONE1.txt" ]
do
    sleep 10
done

while [ ! -f "./TEMPFILE_DONE2.txt" ]
do
    sleep 10
done

while [ ! -f "./TEMPFILE_DONE3.txt" ]
do
    sleep 10
done

while [ ! -f "./TEMPFILE_DONE4.txt" ]
do
    sleep 10
done

while [ ! -f "./TEMPFILE_DONE5.txt" ]
do
    sleep 10
done

python inference_regression_statistics/inference_regression_preds.py --device 0
