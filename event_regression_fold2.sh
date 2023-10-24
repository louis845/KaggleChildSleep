python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --model regress_model_fold2_low --batch_size 32 --learning_rate 5e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 60 120 240
python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --model regress_model_fold2 --batch_size 32 --learning_rate 5e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 90 180 360
python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --model regress_model_fold2_high --batch_size 32 --learning_rate 5e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 120 240 480

python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --model regress_huber_model_fold2_low --batch_size 32 --learning_rate 5e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 60 120 240
python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --model regress_huber_model_fold2 --batch_size 32 --learning_rate 5e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 90 180 360
python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --model regress_huber_model_fold2_high --batch_size 32 --learning_rate 5e-3 --train_data fold_2_train --val_data fold_2_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 120 240 480