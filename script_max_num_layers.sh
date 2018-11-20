#!/bin/sh
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_layers 2 --dim_reduction_type max_pooling --experiment_name results/max_pooling/num_layers/2 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_layers 6 --dim_reduction_type max_pooling --experiment_name results/max_pooling/num_layers/6 --use_gpu True
