python training_resnet.py --device 1 --model model_fold3_cutmix_dropout_fast --train_data fold_3_train --val_data fold_3_val --num_extra_steps 7 --hidden_channels 8 --hidden_blocks 4 6 8 12 8 --squeeze_excitation --use_cutmix_training --dropout 0.2