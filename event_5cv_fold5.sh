python training_event_density_resnet.py --epochs 40 --device 1 --model event5fold_density_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_swa --swa_start 20
python training_event_density_resnet.py --epochs 40 --device 1 --model event5fold_density_time_2elastic_length2_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_time_information

python training_event_density_resnet.py --epochs 50 --device 1 --model event5fold_density_new_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8
python training_event_density_resnet.py --epochs 50 --device 1 --model event5fold_density_new_time_2elastic_length2_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_time_information

python training_event_density_resnet.py --epochs 35 --device 1 --model event5fold_density_swa_2elastic_length3_drop_model_fold5 --load_model_epoch 20 --load_model event5fold_density_new_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_swa --swa_start 0
python training_event_density_resnet.py --epochs 35 --device 1 --model event5fold_density_swa_time_2elastic_length2_drop_model_fold5 --load_model_epoch 20 --load_model event5fold_density_new_time_2elastic_length2_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_time_information --use_swa --swa_start 0

python training_event_density_resnet.py --epochs 50 --device 1 --model event5fold_density_enmo_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_enmo_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8
python training_event_density_resnet.py --epochs 50 --device 1 --model event5fold_density_time_enmo_2elastic_length2_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_enmo_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_time_information

python training_event_density_resnet.py --epochs 50 --device 1 --model event5fold_density_bdfix_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8
python training_event_density_resnet.py --epochs 50 --device 1 --model event5fold_density_bdfix_time_2elastic_length2_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_time_information

python training_event_density_resnet.py --epochs 35 --device 1 --model event5fold_density_bdfix_swa_2elastic_length3_drop_model_fold5 --load_model_epoch 20 --load_model event5fold_density_bdfix_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_swa --swa_start 0
python training_event_density_resnet.py --epochs 35 --device 1 --model event5fold_density_bdfix_swa_time_2elastic_length2_drop_model_fold5 --load_model_epoch 10 --load_model event5fold_density_bdfix_time_2elastic_length2_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_time_information --use_swa --swa_start 0

python training_event_density_resnet.py --epochs 40 --device 1 --model event5fold_density_bdfix_enmo_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_enmo_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8



python training_event_density_resnet.py --epochs 40 --device 1 --model event5fold_density_focal_2elastic_length3_drop_model_fold5 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_focal_loss
python training_event_density_resnet.py --epochs 40 --device 1 --model event5fold_density_focal_time_2elastic_length2_drop_model_fold5 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640  --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_time_information --use_focal_loss