import os
import yaml
import click
from src.models.utils import load_model
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.data.synthetic_dataset import SyntheticDataset
import pytorch_lightning as pl


@click.command()
@click.option('--model',
              type=click.Choice(["cnn", "lstm", "transformer"]),
              default=None,
              help='Only do testing of one model')
@click.option(
    '--batch_size',
    type=click.INT,
    default=32,
    help='Batch Size during testing. Only relevant for memory issues.')
@click.option('--output_filepath',
              type=click.Path(),
              default=None,
              help='Directory where output of model will be saved to ')
@click.option('--test_path',
              type=click.Path(),
              default=None,
              help='Path to test dataset')
@click.argument('experiment_name', type=click.STRING)
def cli(model, batch_size, output_filepath, test_path, experiment_name):
    """ This script can be used to test the performance after performing hyperparametre optimization using
    src.tuning.hyperparameter_tuning.

    The script loads the best MODEL from the experiment with name EXPERIMENT_NAME (all models are evaluated if
    MODEL option is not provided). It evaluates the mae on the samples and saves it for each sample.
    The evaluated samples will be saved to OUTPUT_FILEPATH with the model name attached
    """

    # only once model if it is specified, otherwise all three
    if model is not None:
        study_paths = {model: f"./models/{experiment_name}/{model}"}
    else:
        study_paths = {
            "cnn": f"./models/{experiment_name}/cnn",
            "lstm": f"./models/{experiment_name}/lstm",
            "transformer": f"./models/{experiment_name}/transformer"
        }

    summaries = {}
    models = {}
    for model, path in study_paths.items():
        with open(os.path.join(path, "study_summary.yaml"), "r") as inp:
            try:
                summary = yaml.load(inp, Loader=yaml.BaseLoader)
            except yaml.YAMLError as e:
                print(e)
        summaries[model] = summary

        # load best model according to val loss
        best_model_path = summary["best_model"]["checkpoint_path"]
        models[model] = load_model(best_model_path)

    test_dataloader = DataLoader(SyntheticDataset.load(test_path),
                                 batch_size=batch_size)

    if output_filepath is None:
        Path(f"./experiments/{experiment_name}/test").mkdir(parents=True,
                                                            exist_ok=True)
        output_filepath = f"./experiments/{experiment_name}/test"

    if not test_dataloader.dataset.is_normalized:
        scaler = [*models.values()][0].train_dataset.scaler
        test_dataloader.dataset.normalize(scaler=scaler)

    for model_name, model in models.items():
        tf_trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                                accelerator="dp",
                                logger=None)
        tf_trainer.test(model=model, test_dataloaders=test_dataloader)
        predictions = pd.DataFrame(
            tf_trainer.lightning_module.evaluated_samples)
        pd.to_pickle(
            predictions,
            os.path.join(output_filepath, f"{model_name}_test_output"))


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli()
