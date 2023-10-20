python training_event_resnet.py --epochs 150 --device 0 --memory_limit 0.495 --model model_fold2_event_random_small_evenlessreg --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.2 --random_shift 2160
#python training_event_resnet.py --epochs 50 --device 0 --memory_limit 0.495 --model model_fold2_event_cutmix_anglez --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --random_flip --use_anglez_only --attention_bottleneck 1 --use_cutmix --cutmix_skip 480 --cutmix_length 720
python training_event_resnet.py --epochs 150 --device 0 --memory_limit 0.495 --model model_fold2_event_anglez --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only
python training_event_resnet.py --epochs 200 --device 0 --memory_limit 0.495 --model model_fold2_event_anglez_bottleneck --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --attention_bottleneck 1

python training_clean_data_3f.py --device 0 --memory_limit 0.495 --model 3f_3cv_all_model_fold2_clean_anglez --train_data fold_2_train --val_data fold_2_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --do_not_exclude --use_batch_norm --use_deep_supervision --dropout 0.4 --optimizer adam --weight_decay 0.01 --use_anglez_only
python training_event_resnet.py --epochs 100 --device 0 --memory_limit 0.495 --model model_fold2_event_anglez_bottleneck2 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --attention_bottleneck 1

python training_clean_data_3f.py --device 0 --memory_limit 0.495 --model 3f_3cv_all_model_fold2_clean_anglez_compressed --train_data fold_2_train --val_data fold_2_val --num_extra_steps 9 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --do_not_exclude --use_batch_norm --use_deep_supervision --dropout 0.4 --optimizer adam --weight_decay 0.01 --use_anglez_only
#python training_event_resnet.py --epochs 100 --device 0 --memory_limit 0.495 --model model_fold2_event_anglez_compressed_bottleneck --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --attention_bottleneck 1

python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model model_fold2_event_noflip --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model model_fold2_event_allflip --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --always_flip
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model model_fold2_event_randflip --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip

python training_event_resnet.py --epochs 80 --device 0 --memory_limit 0.495 --model smallbatch_model_fold2_event --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640
python training_event_resnet.py --epochs 80 --device 0 --memory_limit 0.495 --model smallbatch_model_fold2_event_ce --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --use_ce_loss
python training_event_resnet.py --epochs 80 --device 0 --memory_limit 0.495 --model smallbatch_model_fold2_event_iou --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --use_iou_loss
