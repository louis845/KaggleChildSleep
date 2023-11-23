python training_event_resnet.py --epochs 45 --device 1 --model event5fold_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length
python training_event_resnet.py --epochs 45 --device 1 --model event5fold_2elastic_length2_time_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --use_time_information

python training_event_resnet.py --epochs 45 --device 1 --model event5fold_enmo_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length

python training_event_resnet.py --epochs 60 --device 1 --model event5fold_swa_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --use_swa
python training_event_resnet.py --epochs 35 --device 1 --model event5fold_swa_2elastic_length2_time_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --use_time_information --use_swa

python training_event_resnet.py --epochs 40 --device 1 --model event5fold_laplace_2elastic_length2_time_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --use_time_information --prediction_kernel_mode laplace
python training_event_resnet.py --epochs 40 --device 1 --model event5fold_swa_laplace_2elastic_length2_time_drop_model_fold5 --load_model event5fold_laplace_2elastic_length2_time_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --use_time_information --use_swa --swa_start 0 --prediction_kernel_mode laplace

python training_event_resnet.py --epochs 35 --device 1 --model swa_2elastic_length3_drop_model5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_all --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --use_swa

python training_event_resnet.py --epochs 45 --device 1 --model event5fold_small_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_ce_loss --attention_mode length --disable_IOU_converter --prediction_tolerance_width 60
python training_event_resnet.py --epochs 30 --device 1 --model event5fold_focal15_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --positive_weight 1.5 --attention_mode length --disable_IOU_converter --prediction_tolerance_width 36
python training_event_resnet.py --epochs 30 --device 1 --model event5fold_focal3_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 1 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --positive_weight 3.0 --attention_mode length --disable_IOU_converter --prediction_tolerance_width 60