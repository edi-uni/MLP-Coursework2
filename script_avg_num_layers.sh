#!/bin/sh
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_layers 2 --dim_reduction_type avg_pooling --experiment_name results/avg_pooling/num_layers/2 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_layers 6 --dim_reduction_type avg_pooling --experiment_name results/avg_pooling/num_layers/6 --use_gpu True
