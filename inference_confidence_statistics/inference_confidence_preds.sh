#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..

python ./inference_confidence_statistics/inference_confidence_preds.py --attention_blocks 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --expand 8640 --upconv_channels_override 8