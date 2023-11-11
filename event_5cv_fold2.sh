python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_allflip_model_fold2 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --always_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_noflip_model_fold2 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_model_fold2 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_elastic_model_fold2 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_elastic_clean_model_fold2 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --exclude_bad_series_from_training --log_average_precision
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_all_elastic_model_fold2 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_elastic_length_model_fold2 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --log_average_precision
python training_event_resnet.py --epochs 60 --device 0 --memory_limit 0.495 --model event5fold_expanded_elastic_length2_mix_model_fold2 --attention_blocks 2 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --mix_anglez_enmo --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --log_average_precision
python training_event_resnet.py --epochs 60 --device 0 --memory_limit 0.495 --model event5fold_expanded_elastic_length2_model_fold2 --attention_blocks 2 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --log_average_precision
python training_event_resnet.py --epochs 10 --device 0 --memory_limit 0.495 --model event5fold_expanded_elastic_local_model_fold2 --attention_blocks 0 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 10 --device 0 --memory_limit 0.495 --model event5fold_expanded_elastic_local_vrflip_model_fold2 --attention_blocks 0 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --flip_value --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
python training_event_resnet.py --epochs 10 --device 0 --memory_limit 0.495 --model event5fold_expanded_2elastic_local_vrflip_model_fold2 --attention_blocks 0 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --random_flip --flip_value --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss --log_average_precision
