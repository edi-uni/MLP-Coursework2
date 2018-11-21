#!/bin/sh
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --dim_reduction_type strided_convolution --experiment_name results/strided_convolution/batch_size/100 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 200 --dim_reduction_type strided_convolution --experiment_name results/strided_convolution/batch_size/200 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 400 --dim_reduction_type strided_convolution --experiment_name results/strided_convolution/batch_size/400 --use_gpu True
