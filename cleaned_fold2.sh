python training_clean_data.py --device 0 --memory_limit 0.495 --model model_fold2_clean --train_data fold_2_train --val_data fold_2_val --num_extra_steps 9 --hidden_channels 8 --hidden_blocks 4 6 8 12 8 --squeeze_excitation --use_mixup_training --dropout 0.5 --optimizer sgd