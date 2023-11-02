# Experiments
python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss huber --model regress_model_fold1_low --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 60 120 240
python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss huber --model regress_model_fold1_high --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 120 240 480

python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss mse --model regress_mse_model_fold1_low --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 60 120 240
python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss mse --model regress_mse_model_fold1_high --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 120 240 480

python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss huber_mse --model regress_hmse_model_fold1_low --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 60 120 240
python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss huber_mse --model regress_hmse_model_fold1_high --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 120 240 480

python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss ce --regression_kernel 12 --model regress_ce_model_fold1_high --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 120 240 480
python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss focal --regression_kernel 12 --model regress_focal_model_fold1_high --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 120 240 480

python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss ce --regression_kernel 12 --model regress_ce_model_fold1_low --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 60 120 240
python training_resnet_regression.py --epochs 20 --device 0 --memory_limit 0.495 --loss focal --regression_kernel 12 --model regress_focal_model_fold1_low --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --hidden_channels 4 4 8 16 32 64 64 --hidden_blocks 2 2 2 2 2 2 2 --kernel_size 3 --deep_upconv_kernel 5 --disable_deep_upconv_contraction --dropout 0.0 --use_anglez_only --regression_width 60 120 240

# Final model
python training_resnet_regression.py --epochs 200 --device 0 --memory_limit 0.495 --loss huber --model regress_standard_model_fold1 --batch_size 32 --learning_rate 5e-3 --train_data fold_1_train --val_data fold_1_val --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model
# Inference
python inference_regression_preds.py --device 0 --memory_limit 0.495 --load_model regress_standard_model_fold1 --train_data fold_1_train --val_data fold_1_val --use_anglez_only