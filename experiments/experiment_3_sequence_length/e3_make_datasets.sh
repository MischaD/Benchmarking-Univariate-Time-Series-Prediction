#!/bin/bash -l
# script for generating dataset

# load python
cd masterthesis
module load python/3.8-anaconda
source activate /home/woody/iwso/iwso039h/envs/mt_env

# define environment variables for training
experiment_name=experiment_3_sequence_length

# define environement variables for dataset
seq_lengths_train="128,256,1024,2048"
seq_lengths_test="128,256,512,1024,2048"
delay_lengths="0,32"
sigmas="2,5"
period_lengths="16,32"

sampling_size_train=10000
sampling_size_val=2000
sampling_size_test=4000

sampling_strategy_train=default_mean_normal
sampling_strategy_val=default_mean_uniform
sampling_strategy_test=default_mean_uniform

base_dir=./experiments/${experiment_name}

python ./src/data/make_dataset.py --sample_sequence_lengths=$seq_lengths_train --sigmas=$sigmas --delay_lengths=$delay_lengths --period_lengths=$period_lengths --sampling_strategy=$sampling_strategy_train --sample_size=$sampling_size_train ${base_dir}/data/train.pkl
python ./src/data/make_dataset.py --sample_sequence_lengths=$seq_lengths_train --sigmas=$sigmas --delay_lengths=$delay_lengths --period_lengths=$period_lengths --sampling_strategy=$sampling_strategy_val   --sample_size=$sampling_size_val   ${base_dir}/data/val.pkl
python ./src/data/make_dataset.py --sample_sequence_lengths=$seq_lengths_test  --sigmas=$sigmas --delay_lengths=$delay_lengths --period_lengths=$period_lengths --sampling_strategy=$sampling_strategy_test  --sample_size=$sampling_size_test  ${base_dir}/data/test.pkl
