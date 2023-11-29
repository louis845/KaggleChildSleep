#python training_resnet_regression.py --epochs 100 --device 1 --loss huber --model regress_standard5cv_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model
#python training_resnet_regression.py --epochs 50 --device 1 --loss huber --model regress_standard5cv_wide_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
#python training_resnet_regression.py --epochs 50 --device 1 --loss huber --model regress_standard5cv_mid_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64

#python training_resnet_regression.py --epochs 50 --device 1 --loss mse --model regress_standard5cv_gaussian_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model
#python training_resnet_regression.py --epochs 50 --device 1 --loss mse --model regress_standard5cv_gaussian_wide_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
#python training_resnet_regression.py --epochs 50 --device 1 --loss mse --model regress_standard5cv_gaussian_mid_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64

#python training_resnet_regression.py --epochs 25 --device 1 --loss huber --model regress_swa_standard5cv_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model --use_swa
#python training_resnet_regression.py --epochs 25 --device 1 --loss huber --model regress_swa_standard5cv_wide_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64 --use_swa
#python training_resnet_regression.py --epochs 25 --device 1 --loss huber --model regress_swa_standard5cv_mid_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64 --use_swa

python training_resnet_regression.py --epochs 50 --device 1 --loss huber --model regress_standardBal5cv_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data balanced5cv_fold_5_train --val_data balanced5cv_fold_5_val --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model
python training_resnet_regression.py --epochs 50 --device 1 --loss huber --model regress_standardBal5cv_wide_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data balanced5cv_fold_5_train --val_data balanced5cv_fold_5_val --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
python training_resnet_regression.py --epochs 50 --device 1 --loss huber --model regress_standardBal5cv_mid_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data balanced5cv_fold_5_train --val_data balanced5cv_fold_5_val --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64

python training_resnet_regression.py --epochs 50 --device 1 --loss mse --model regress_mse_standard5cv_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 120 240 480 --use_standard_model
python training_resnet_regression.py --epochs 50 --device 1 --loss mse --model regress_mse_standard5cv_wide_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
python training_resnet_regression.py --epochs 50 --device 1 --loss mse --model regress_mse_standard5cv_mid_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64

python training_resnet_regression.py --epochs 50 --device 1 --loss huber --model regress_enmo_standard5cv_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_enmo_only --regression_width 120 240 480 --use_standard_model
python training_resnet_regression.py --epochs 50 --device 1 --loss huber --model regress_enmo_standard5cv_wide_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_enmo_only --regression_width 240 240 240 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64
python training_resnet_regression.py --epochs 50 --device 1 --loss huber --model regress_enmo_standard5cv_mid_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_enmo_only --regression_width 180 180 180 --use_standard_model --hidden_blocks 2 2 2 2 2 3 --hidden_channels 4 4 8 16 32 64

touch TEMPFILE_DONE5.txt
