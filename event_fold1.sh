python training_event_resnet.py --epochs 35 --device 0 --memory_limit 0.495 --model model_fold1_event_random_decay --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75 --dropout_pos_embeddings --random_shift 2160 --weight_decay 0.01
python training_event_resnet.py --epochs 35 --device 0 --memory_limit 0.495 --model model_fold1_event_random_decay_deep --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --use_batch_norm --dropout 0.75 --dropout_pos_embeddings --random_shift 2160 --weight_decay 0.01 --use_deep_supervision 9600000

python training_clean_data_3f.py --device 0 --memory_limit 0.495 --model 3f_3cv_all_model_fold1_clean_11 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --kernel_size 11 --squeeze_excitation --do_not_exclude --use_batch_norm --use_deep_supervision --dropout 0.75 --optimizer adam --weight_decay 0.01
python training_clean_data_3f.py --device 0 --memory_limit 0.495 --model 3f_3cv_all_model_fold1_clean_9 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --kernel_size 9 --squeeze_excitation --do_not_exclude --use_batch_norm --use_deep_supervision --dropout 0.75 --optimizer adam --weight_decay 0.01
python training_clean_data_3f.py --device 0 --memory_limit 0.495 --model 3f_3cv_all_model_fold1_clean_7 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --kernel_size 7 --squeeze_excitation --do_not_exclude --use_batch_norm --use_deep_supervision --dropout 0.75 --optimizer adam --weight_decay 0.01
python training_clean_data_3f.py --device 0 --memory_limit 0.495 --model 3f_3cv_all_model_fold1_clean_5 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --kernel_size 5 --squeeze_excitation --do_not_exclude --use_batch_norm --use_deep_supervision --dropout 0.75 --optimizer adam --weight_decay 0.01
python training_clean_data_3f.py --device 0 --memory_limit 0.495 --model 3f_3cv_all_model_fold1_clean_3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --kernel_size 3 --squeeze_excitation --do_not_exclude --use_batch_norm --use_deep_supervision --dropout 0.75 --optimizer adam --weight_decay 0.01
