python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold1 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --use_elastic_deformation --expand 8640 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold1_flip --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold1_nodrop_flip --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --upconv_channels_override 8 --use_ce_loss

python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold1_center_flip --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_only --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_elastic_model_fold1_center_drop --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_only --dropout 0.1 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_center_model_fold1 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --predict_center_only --dropout 0.1 --upconv_channels_override 8 --use_ce_loss

python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_elastic_model_fold1 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_anglez_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_enmo_elastic_model_fold1 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_enmo_only --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_expanded_all_elastic_model_fold1 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.0 --random_shift 2160 --use_elastic_deformation --random_flip --expand 8640 --predict_center_mode expanded --dropout 0.1 --upconv_channels_override 8 --use_ce_loss

python inference_confidence_preds.py --device 0 --memory_limit 0.495 --load_model event5fold_expanded_elastic_model_fold1 --attention_blocks 1 --batch_size 32 --train_data fold_1_train_5cv --val_data fold_1_val_5cv --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --use_anglez_only --expand 8640 --upconv_channels_override 8