python training_resnet_regression.py --epochs 100 --device 0 --memory_limit 0.495 --loss huber --model regress_standard5cv_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model
python training_resnet_regression.py --epochs 100 --device 0 --memory_limit 0.495 --loss huber --model regress_standard5cv_model_fold3_continue --load_model regress_standard5cv_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model
python training_resnet_regression.py --epochs 100 --device 0 --memory_limit 0.495 --loss huber --model regress_standard5cv_model_fold3_continue2 --load_model regress_standard5cv_model_fold3_continue --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model

python training_resnet_regression.py --epochs 100 --device 0 --memory_limit 0.495 --loss huber_sigma --model regress_standard5cv_sigma_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model
python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber_sigma --model regress_standard5cv_sigma_velastic_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model  --flip_value --use_velastic_deformation
python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber_sigma --model regress_standard5cv_sigma_elastic_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model --use_elastic_deformation
python training_resnet_regression.py --epochs 50 --device 0 --memory_limit 0.495 --loss huber_sigma --model regress_standard5cv_sigma_2elastic_model_fold3 --batch_size 32 --learning_rate 5e-3 --train_data fold_3_train_5cv --val_data fold_3_val_5cv --num_extra_steps 4 --use_anglez_only --use_decay_schedule --regression_width 120 240 480 --use_standard_model --flip_value --use_velastic_deformation --use_elastic_deformation