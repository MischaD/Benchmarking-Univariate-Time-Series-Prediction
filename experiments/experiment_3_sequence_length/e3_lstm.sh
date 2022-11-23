#!/bin/bash -l
#PBS -l nodes=1:ppn=8:anytesla,walltime=24:00:00
#PBS -N trafo
#PBS -M mischa.dombrowski@fau.de -m done

# load python
cd masterthesis
module load python/3.8-anaconda
source activate /home/woody/iwso/iwso039h/envs/mt_env

# define environment variables for training
experiment_name=experiment_3_sequence_length
n_trials=25
max_epochs=100
batch_size=64

# final training parameters
final_training_max_epochs=300
final_training_patience=15

base_dir=./experiments/${experiment_name}

# hp optimization
python ./src/tuning/hyperparameter_tuning.py lstm  --max_epochs=$max_epochs --batch_size=$batch_size  --n_trials=$n_trials  ${base_dir}/data/train.pkl ${base_dir}/data/val.pkl $experiment_name

# Training best model until convergence
python ./src/models/train_model.py optuna --max_epochs=$final_training_max_epochs  --patience=$final_training_patience ${base_dir}/data/train.pkl ${base_dir}/data/val.pkl ./models/${experiment_name}/lstm