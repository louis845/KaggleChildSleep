python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold3 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --use_elastic_deformation --expand 8640 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold3_flip --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold3_nodrop_flip --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --upconv_channels_override 8 --use_ce_loss

python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold3_center_flip --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_only --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold3_center_drop --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_only --dropout 0.1 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_center_model_fold3 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --predict_center_only --dropout 0.1 --upconv_channels_override 8 --use_ce_loss