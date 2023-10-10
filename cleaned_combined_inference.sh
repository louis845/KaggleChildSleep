python inference_clean_preds.py --device 0 --load_model model_fold1_clean --data_to_infer fold_1_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model
python inference_clean_preds.py --device 0 --load_model model_fold2_clean --data_to_infer fold_2_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model
python inference_clean_preds.py --device 0 --load_model model_fold3_clean --data_to_infer fold_3_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model
python inference_clean_preds.py --device 0 --load_model model_fold4_clean --data_to_infer fold_4_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model
python inference_clean_preds.py --device 0 --load_model model_fold5_clean --data_to_infer fold_5_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model
python inference_clean_preds.py --device 0 --load_model model_fold6_clean --data_to_infer fold_6_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model
python inference_clean_preds.py --device 0 --load_model model_fold7_clean --data_to_infer fold_7_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model
python inference_clean_preds.py --device 0 --load_model model_fold8_clean --data_to_infer fold_8_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model
python inference_clean_preds.py --device 0 --load_model model_fold9_clean --data_to_infer fold_9_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model
python inference_clean_preds.py --device 0 --load_model model_fold10_clean --data_to_infer fold_10_train_10cv --hidden_channels 8 --hidden_blocks 4 6 8 8 --squeeze_excitation --use_batch_norm --use_best_model

python cleaned_combine_pseudo_labels.py
