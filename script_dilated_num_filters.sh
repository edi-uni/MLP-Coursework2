#!/bin/sh
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_filters 16 --dim_reduction_type dilated_convolution --experiment_name results/dilated_convolution/num_filters/16 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_filters 32 --dim_reduction_type dilated_convolution --experiment_name results/dilated_convolution/num_filters/32 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_filters 128 --dim_reduction_type dilated_convolution --experiment_name results/dilated_convolution/num_filters/128 --use_gpu True
