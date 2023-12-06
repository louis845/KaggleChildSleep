#python training_resnet_regression.py --epochs 100 --device 0 --memory_limit 0.495 --loss huber --model regress_standard5cv_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_standard5cv_wide_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_standard5cv_mid_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_standard5cv_small_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 60 60 60 --use_standard_model --hidden_blocks 2 2 2 2 --hidden_channels 8 8 16 32
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_standard5cv_tiny_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 30 30 30 --use_standard_model --hidden_blocks 2 2 2 --hidden_channels 16 16 32

#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss mse --model regress_standard5cv_gaussian_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss mse --model regress_standard5cv_gaussian_wide_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss mse --model regress_standard5cv_gaussian_mid_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64

#python training_resnet_regression.py --epochs 25 --device 0 --memory_limit 0.495 --loss huber --model regress_swa_standard5cv_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model --use_swa
#python training_resnet_regression.py --epochs 25 --device 0 --memory_limit 0.495 --loss huber --model regress_swa_standard5cv_wide_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64 --use_swa
#python training_resnet_regression.py --epochs 25 --device 0 --memory_limit 0.495 --loss huber --model regress_swa_standard5cv_mid_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64 --use_swa

#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_standardBal5cv_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data balanced5cv_fold_3_train --val_data balanced5cv_fold_3_val --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_standardBal5cv_wide_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data balanced5cv_fold_3_train --val_data balanced5cv_fold_3_val --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_standardBal5cv_mid_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data balanced5cv_fold_3_train --val_data balanced5cv_fold_3_val --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64

#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss mse --model regress_mse_standard5cv_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss mse --model regress_mse_standard5cv_wide_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss mse --model regress_mse_standard5cv_mid_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64

#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_enmo_standard5cv_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_enmo_only --regression_width 120 240 480 --use_standard_model
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_enmo_standard5cv_wide_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_enmo_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
#python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber --model regress_enmo_standard5cv_mid_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_enmo_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64

touch TEMPFILE_DONE3.txt
