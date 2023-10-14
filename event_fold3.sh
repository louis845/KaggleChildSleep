python training_clean_data_3f.py --device 1 --model 3f_3cv_all_model_fold3_clean_deep_adam --train_data fold_3_train --val_data fold_3_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --do_not_exclude --use_decay_schedule --use_batch_norm --use_deep_supervision --dropout 0.75 --optimizer adam

python training_event_resnet.py --device 1 --model model_fold3_event --train_data fold_3_train --val_data fold_3_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75
python training_event_resnet.py --device 1 --model model_fold3_event_dropout --train_data fold_3_train --val_data fold_3_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75 --dropout_pos_embeddings
python training_event_resnet.py --device 1 --model model_fold3_event_dropout_beta --train_data fold_3_train --val_data fold_3_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75 --dropout_pos_embeddings
python training_event_resnet.py --device 1 --model model_fold3_event_dropout_beta_ce --train_data fold_3_train --val_data fold_3_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75 --dropout_pos_embeddings --use_ce_loss
python training_event_resnet.py --epochs 35 --device 1 --model model_fold3_event_random --train_data fold_3_train --val_data fold_3_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75 --random_shift 2160
python training_event_resnet.py --epochs 35 --device 1 --model model_fold3_event_dropout_random --train_data fold_3_train --val_data fold_3_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75 --dropout_pos_embeddings --random_shift 2160
python training_event_resnet.py --epochs 35 --device 1 --model model_fold3_event_random_decay --train_data fold_3_train --val_data fold_3_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75 --random_shift 2160 --weight_decay 0.01
#python training_event_resnet.py --device 1 --model model_fold3_event_dropout_random --train_data fold_3_train --val_data fold_3_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75 --dropout_pos_embeddings --random_shift 720