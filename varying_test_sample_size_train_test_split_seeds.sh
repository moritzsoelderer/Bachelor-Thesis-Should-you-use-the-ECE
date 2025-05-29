#!/usr/bin/env bash

. .venv/Scripts/activate
pip install -r requirements.txt

dataset_name="gummy_worm"
dataset_size=40000
min_sample_size=100
max_sample_size=20000
num_steps=200
true_ece_sample_size=400000

python -m src.experiments.varying_test_sample_size_train_test_split_seeds.run_experiment \
            "$dataset_name" "$dataset_size" "$min_sample_size" "$max_sample_size" "$num_steps" "$true_ece_sample_size"

# Push plots and logs
git add ./src/experiments/varying_test_sample_size_train_test_split_seeds/plots ./src/experiments/varying_test_sample_size_train_test_split_seeds/logs
git commit -m "VTSSTTS-SKRIPT: Executed with parameters dataset_name=$dataset_name dataset_size=$dataset_size min_sample_size=$min_sample_size max_sample_size=$max_sample_size num_steps=$num_steps true_ece_sample_size=$true_ece_sample_size"
git push
