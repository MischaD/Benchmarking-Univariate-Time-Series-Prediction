# Data-Centric Benchmarking of Neural Network Architectures for the Univariate Time Series Prediction Task 

## Installation 
Clone repository: 

    git clone https://mad-srv.informatik.uni-erlangen.de/MadLab/industry-4.0/student-projects/analysis-of-the-robustness-of-transformer-models.git
    cd analysis-of-the-robustness-of-transformer-models

Install requirements: 

    pip install -r requirements.txt

Install Cuda enabled PyTorch for Windows: 

    pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchaudio===0.10.0+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

Install Cuda enabled PyTorch for Linux: 

    pip install torch torchvision torchaudio

Installation with conda sometimes requires manually reinstalling the local repo:
    
    pip install -e .


# Run Experiment Manually: 
Create three separate datasets with different amounts of overall samples for training, validation, and testing. 
You may also load the datasets from ./experiments/<experiment_num>/data  

    python ./src/data/make_dataset.py --sample_sequence_lengths="128,256" --sigmas="1,5" --sampling_strategy="default" --sample_size=10000 ./train.pkl
    python ./src/data/make_dataset.py --sample_sequence_lengths="128,256" --sigmas="1,5" --sampling_strategy="default" --sample_size=2000 ./val.pkl
    python ./src/data/make_dataset.py --sample_sequence_lengths="128,256" --sigmas="1,5" --sampling_strategy="default" --sample_size=4000 ./test.pkl

Train model for a predefined amount of epochs, best model according to validation loss will be saved. 

    python ./src/models/train_model.py transformer --epochs=10 ./train.pkl ./val.pkl

Look at the test performance of a model:

    python ./src/models/predict_model.py ./test.pkl <path_to_best_model> 

Visualize some predictions:

     python ./src/visualization/show_prediction.py ./test.pkl <path_to_best_model> <output_file_save_path> 

## Run Hyperparameter Tuning 
Start a new study with name <study_name> that will be saved in ./models/<study_name>

    python ./src/tuning/hyperparameter_tuning.py transformer ./test.pkl ./val.pkl <study_name> --max_epochs=100 --n_trials=25


## Reproducing Experiments 
### Experiment Description 
All of the experiments were performed in the same way. We ran them by using the shell scripts located in ./experiments/<experiment_name>/<training_name>.sh. 
All of them do the same thing: 

First, they define a few environmental variables that are used as input parameters for the scripts. 
Then they run the scripts for the creation of the datasets. These datasets are then used for the hyperparameter optimization of all the models. We also saved the datasets for increased reproducibility.

There are three experiments. They are located in ./experiments. 

- experiment 1: Experiment looking at the impact of delay lengths on the performance
- experiment 2: Experiment looking at the impact of dynamics (frequency) and noise levels on the performance
- experiment 3: Experiment looking at the impact of the sequence length on the performance

To reproduce the visualizations, look at the corresponding notebooks in ./experiments/<experiment_name>/

The entire script to reproduce the results from all the experiments as well as the 
data and the notebooks that visualize the results can be found in:

    .\experiments\<experiment_name>

However, the results may be different due to the stochasticity aspects, such as the hyperparameter
optimization, the model initializations, the generation of the dataset, etc. 


## Testing 
To verify the correct implementation of the models and the data synthesization, we implemented some unittest. 
Run them with the following command: 

    pytest . 

##  Importance Visualization 

The notebook for the hyperparameter optimization importance visualization can be found in    

experiments/experiment_3_sequence_length/0.4.4-mnd-hp_opt_importance_visualization.ipynb
