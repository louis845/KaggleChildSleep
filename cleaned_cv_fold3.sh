#python training_clean_data_3f.py --device 1 --model 3f_3cv_model_fold3_clean_huge --train_data fold_3_train --val_data fold_3_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 6 8 8 8 8 --squeeze_excitation --use_decay_schedule --use_batch_norm --dropout 0.75 --optimizer sgd --record_best_model
#python training_clean_data_3f.py --device 1 --model 3f_3cv_model_fold3_clean_large --train_data fold_3_train --val_data fold_3_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 6 8 8 8 --squeeze_excitation --use_decay_schedule --use_batch_norm --dropout 0.75 --optimizer sgd --record_best_model
#python training_clean_data_3f.py --device 1 --model 3f_3cv_model_fold3_clean --train_data fold_3_train --val_data fold_3_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_decay_schedule --use_batch_norm --dropout 0.75 --optimizer sgd --record_best_model

python training_clean_data_3f.py --device 1 --model 3f_3cv_all_model_fold3_clean_deep --train_data fold_3_train --val_data fold_3_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --do_not_exclude --use_decay_schedule --use_batch_norm --use_deep_supervision --dropout 0.75 --optimizer sgd
python training_clean_data_3f.py --device 1 --model 3f_3cv_all_model_fold3_clean_huge --train_data fold_3_train --val_data fold_3_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 4 --squeeze_excitation --do_not_exclude --use_decay_schedule --use_batch_norm --dropout 0.75 --optimizer sgd
python training_clean_data_3f.py --device 1 --model 3f_3cv_all_model_fold3_clean_large --train_data fold_3_train --val_data fold_3_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 4 --squeeze_excitation --do_not_exclude --use_decay_schedule --use_batch_norm --dropout 0.75 --optimizer sgd
python training_clean_data_3f.py --device 1 --model 3f_3cv_all_model_fold3_clean --train_data fold_3_train --val_data fold_3_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 4 4 4 4 --squeeze_excitation --do_not_exclude --use_decay_schedule --use_batch_norm --dropout 0.75 --optimizer sgd
