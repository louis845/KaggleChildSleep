python training_resnet_regression.py --epochs 100 --device 1 --loss huber --model regress_standard5cv_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model
python training_resnet_regression.py --epochs 100 --device 1 --loss huber --model regress_standard5cv_model_fold5_continue --load_model regress_standard5cv_model_fold5 --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model
python training_resnet_regression.py --epochs 100 --device 1 --loss huber --model regress_standard5cv_model_fold5_continue2 --load_model regress_standard5cv_model_fold5_continue --batch_size 32 --learning_rate 5e-3 --train_data fold_5_train_5cv --val_data fold_5_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model