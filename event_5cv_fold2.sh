python training_event_resnet.py --epochs 45 --device 0 --memory_limit 0.495 --model event5fold_2elastic_length3_drop_model_fold2 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640 --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length
python training_event_resnet.py --epochs 45 --device 0 --memory_limit 0.495 --model event5fold_2elastic_length2_time_drop_model_fold2 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640 --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --use_time_information

python training_event_resnet.py --epochs 45 --device 0 --memory_limit 0.495 --model event5fold_enmo_2elastic_length3_drop_model_fold2 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_enmo_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640 --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length