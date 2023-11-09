python training_event_resnet.py --epochs 30 --device 1 --model event5fold_expanded_allflip_model_fold5 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --always_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 30 --device 1 --model event5fold_expanded_noflip_model_fold5 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 30 --device 1 --model event5fold_expanded_model_fold5 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 30 --device 1 --model event5fold_expanded_elastic_model_fold5 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 30 --device 1 --model event5fold_expanded_elastic_clean_model_fold5 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --exclude_bad_series_from_training --log_average_precision
python training_event_resnet.py --epochs 30 --device 1 --model event5fold_expanded_all_elastic_model_fold5 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss
