# Experiments Directory
Directory contains: 

- visualization notebooks for all visualizations in the Experiments Chapter
- Shell scripts that show how the training was performed 

## Experiment 1 delay length: 
- Experiment with different delay lengths
- Visualization of ambiguous input data

## Experiment 2 frequency: 
- Experiment with different frequencies

## Experiment 3 sequence length: 
- Experiment with different frequencies
- Notebook that analyzes hypereparameter optimization


Note: Due to a bug during training time we had to manually export the loss curves for the experiments from the 
tensorboard logs. The bug is fixed, but looking at loss curves other than the best models of each run is only 
possible via tensorboard --logdir=<path_to_tb_logs> and manually exporting them.
