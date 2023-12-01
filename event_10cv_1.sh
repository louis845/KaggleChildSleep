python training_event_density_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model event10fold_2elastic_length3_drop_model_fold1 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_10cv --val_data fold_1_val_10cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640 --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8
python training_event_density_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model event10fold_time_2elastic_length2_drop_model_fold1 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_10cv --val_data fold_1_val_10cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640 --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_time_information
python training_event_density_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model event10fold_enmo_2elastic_length3_drop_model_fold1 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_1_train_10cv --val_data fold_1_val_10cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_enmo_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640 --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8

python training_event_density_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model event10fold_2elastic_length3_drop_model_fold6 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_6_train_10cv --val_data fold_6_val_10cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640 --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8
python training_event_density_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model event10fold_time_2elastic_length2_drop_model_fold6 --attention_blocks 2 --batch_size 32 --learning_rate 1e-3 --train_data fold_6_train_10cv --val_data fold_6_val_10cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_anglez_only --use_elastic_deformation --use_velastic_deformation --flip_value --expand 8640 --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8 --use_time_information
python training_event_density_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model event10fold_enmo_2elastic_length3_drop_model_fold6 --attention_blocks 3 --batch_size 32 --learning_rate 1e-3 --train_data fold_6_train_10cv --val_data fold_6_val_10cv --num_extra_steps 0 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --random_shift 2160 --use_enmo_only --use_elastic_deformation --use_velastic_deformation --flip_value --random_flip --expand 8640 --dropout 0.1 --attn_dropout 0.1 --upconv_channels_override 8