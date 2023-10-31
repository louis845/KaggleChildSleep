#python training_event_resnet.py --epochs 150 --device 0 --memory_limit 0.495 --model model_fold2_event_random_small_evenlessreg --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.2 --random_shift 2160
#python training_event_resnet.py --epochs 50 --device 0 --memory_limit 0.495 --model model_fold2_event_cutmix_anglez --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --random_flip --use_anglez_only --attention_bottleneck 1 --use_cutmix --cutmix_skip 480 --cutmix_length 720
#python training_event_resnet.py --epochs 150 --device 0 --memory_limit 0.495 --model model_fold2_event_anglez --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only
#python training_event_resnet.py --epochs 200 --device 0 --memory_limit 0.495 --model model_fold2_event_anglez_bottleneck --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --attention_bottleneck 1

#python training_clean_data_3f.py --device 0 --memory_limit 0.495 --model 3f_3cv_all_model_fold2_clean_anglez --train_data fold_2_train --val_data fold_2_val --num_extra_steps 9 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --do_not_exclude --use_batch_norm --use_deep_supervision --dropout 0.4 --optimizer adam --weight_decay 0.01 --use_anglez_only
#python training_event_resnet.py --epochs 100 --device 0 --memory_limit 0.495 --model model_fold2_event_anglez_bottleneck2 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --attention_bottleneck 1

#python training_clean_data_3f.py --device 0 --memory_limit 0.495 --model 3f_3cv_all_model_fold2_clean_anglez_compressed --train_data fold_2_train --val_data fold_2_val --num_extra_steps 9 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --do_not_exclude --use_batch_norm --use_deep_supervision --dropout 0.4 --optimizer adam --weight_decay 0.01 --use_anglez_only
#python training_event_resnet.py --epochs 100 --device 0 --memory_limit 0.495 --model model_fold2_event_anglez_compressed_bottleneck --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --attention_bottleneck 1

#python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model model_fold2_event_noflip --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only
#python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model model_fold2_event_allflip --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --always_flip
#python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model model_fold2_event_randflip --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 --hidden_blocks 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip

#python training_event_resnet.py --epochs 80 --device 0 --memory_limit 0.495 --model smallbatch_model_fold2_event --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640
#python training_event_resnet.py --epochs 80 --device 0 --memory_limit 0.495 --model smallbatch_model_fold2_event_ce --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --use_ce_loss

#python training_event_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model event5fold_model_fold3 --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640
#python training_event_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model event5fold_iou_model_fold3 --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --use_iou_loss

#python training_event_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model event5fold_detailed_model_fold3 --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --use_iou_loss --use_detailed_probas
#python training_event_resnet.py --epochs 40 --device 0 --memory_limit 0.495 --model smallbatch_model_fold2_event_iou --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --use_iou_loss

#python training_event_resnet.py --epochs 8 --device 0 --memory_limit 0.495 --model smallbatch_model_fold2_event_regress --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --deep_upconv_kernel 5 --disable_deep_upconv_contraction --upconv_channels_override 16 --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --use_iou_loss --use_deep_supervision 90 180 360
#python training_event_resnet.py --epochs 80 --device 0 --memory_limit 0.495 --model smallbatch_model_fold2_event_regress_wide --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --deep_upconv_kernel 5 --disable_deep_upconv_contraction --upconv_channels_override 24 --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --use_iou_loss --use_deep_supervision 120 240 480

#python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_1block_model_fold3 --use_iou_loss --attention_blocks 1 --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --upconv_channels_override 8

#python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_1block_model_fold3_nbm --use_iou_loss --attention_blocks 1 --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --upconv_channels_override 8

# fliptest
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_fliptest_model_fold2 --attention_blocks 1 --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --expand 8640 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_fliptest_model_fold3 --attention_blocks 1 --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --expand 8640 --upconv_channels_override 8 --use_ce_loss

python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_fliptest_model_fold3_flip --attention_blocks 1 --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --upconv_channels_override 8 --use_ce_loss

# larger prediction test
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_lpredtest_model_fold2 --attention_blocks 1 --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --expand 4320 --prediction_length 25920 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_lpredtest_model_fold3 --attention_blocks 1 --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --expand 4320 --prediction_length 25920 --upconv_channels_override 8 --use_ce_loss

python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_lpredtest_model_fold3_flip --attention_blocks 1 --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --expand 4320 --prediction_length 25920 --upconv_channels_override 8 --use_ce_loss

# local test
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_local_model_fold2 --attention_blocks 0 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_local_model_fold3 --attention_blocks 0 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --upconv_channels_override 8 --use_ce_loss

# all test
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_alltest_model_fold2 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_2_train_5cv --val_data fold_2_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --expand 8640 --upconv_channels_override 8 --use_ce_loss
python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_alltest_model_fold3 --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --expand 8640 --upconv_channels_override 8 --use_ce_loss

python training_event_resnet.py --epochs 30 --device 0 --memory_limit 0.495 --model event5fold_alltest_model_fold3_flip --attention_blocks 1 --include_all_events --batch_size 32 --learning_rate 1e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --hidden_channels 2 2 4 8 16 32 32 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --squeeze_excitation --use_batch_norm --dropout 0.1 --random_shift 2160 --use_anglez_only --random_flip --expand 8640 --upconv_channels_override 8 --use_ce_loss