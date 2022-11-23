#!/bin/bash -l
#PBS -l nodes=1:ppn=4:anytesla,walltime=24:00:00
#PBS -N trafo
#PBS -M mischa.dombrowski@fau.de -m done

# load python
cd masterthesis
module load python/3.8-anaconda
source activate /home/woody/iwso/iwso039h/envs/mt_env

# define environment variables for training
experiment_name=experiment_2_frequency
n_trials=25
max_epochs=100
batch_size=128
max_cnn_model_depth=2
max_cnn_d_model=56

# final training parameters
final_training_max_epochs=300
final_training_patience=15

seq_lengths="512"
delay_lengths="0,64,128"
sigmas="2,3,4,5,6,7,8,9"
period_lengths_train="8,16,64,128"
period_lengths_val="8,16,64,128"
period_lengths_test="8,16,32,64,128"

sampling_size_train=10000
sampling_size_val=2000
sampling_size_test=4000

sampling_strategy_train=default_mean_normal
sampling_strategy_val=default_mean_uniform
sampling_strategy_test=default_mean_uniform

base_dir=./experiments/${experiment_name}

python ./src/data/make_dataset.py --sample_sequence_lengths=$seq_lengths --sigmas=$sigmas --delay_lengths=$delay_lengths --period_lengths=$period_lengths_train --sampling_strategy=$sampling_strategy_train --sample_size=$sampling_size_train ${base_dir}/data/train.pkl
python ./src/data/make_dataset.py --sample_sequence_lengths=$seq_lengths --sigmas=$sigmas --delay_lengths=$delay_lengths --period_lengths=$period_lengths_val   --sampling_strategy=$sampling_strategy_val   --sample_size=$sampling_size_val   ${base_dir}/data/val.pkl
python ./src/data/make_dataset.py --sample_sequence_lengths=$seq_lengths --sigmas=$sigmas --delay_lengths=$delay_lengths --period_lengths=$period_lengths_test --sampling_strategy=$sampling_strategy_test  --sample_size=$sampling_size_test  ${base_dir}/data/test.pkl

python ./src/tuning/hyperparameter_tuning.py cnn         --max_epochs=$max_epochs --batch_size=$batch_size  --max_model_depth=$max_cnn_model_depth --max_d_model=$max_cnn_d_model --n_trials=$n_trials  ${base_dir}/data/train.pkl ${base_dir}/data/val.pkl $experiment_name
python ./src/tuning/hyperparameter_tuning.py transformer --max_epochs=$max_epochs --batch_size=$batch_size                                                                        --n_trials=$n_trials  ${base_dir}/data/train.pkl ${base_dir}/data/val.pkl $experiment_name
python ./src/tuning/hyperparameter_tuning.py lstm        --max_epochs=$max_epochs --batch_size=$batch_size                                                                        --n_trials=$n_trials  ${base_dir}/data/train.pkl ${base_dir}/data/val.pkl $experiment_name

python ./src/models/train_model.py optuna --max_epochs=$final_training_max_epochs  --patience=$final_training_patience ${base_dir}/data/train.pkl ${base_dir}/data/val.pkl ./models/${experiment_name}/cnn
python ./src/models/train_model.py optuna --max_epochs=$final_training_max_epochs  --patience=$final_training_patience ${base_dir}/data/train.pkl ${base_dir}/data/val.pkl ./models/${experiment_name}/transformer
python ./src/models/train_model.py optuna --max_epochs=$final_training_max_epochs  --patience=$final_training_patience ${base_dir}/data/train.pkl ${base_dir}/data/val.pkl ./models/${experiment_name}/lstm