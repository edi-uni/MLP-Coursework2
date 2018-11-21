#!/bin/sh
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_layers 2 --dim_reduction_type strided_convolution --experiment_name results/strided_convolution/num_layers/2 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_layers 6 --dim_reduction_type strided_convolution --experiment_name results/strided_convolution/num_layers/6 --use_gpu True
