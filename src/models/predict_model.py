import click
import torch.cuda

from log import logger_models
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader
from src.models.utils import load_model
from src.data.synthetic_dataset import SyntheticDataset
import pytorch_lightning as pl


def start_testing(model, test_dataset_path, batch_size):
    test_dataloader = DataLoader(SyntheticDataset.load(test_dataset_path),
                                 batch_size=batch_size)
    if not test_dataloader.dataset.is_normalized:
        logger_models.info("Normalization of test dataset with train scaler.")
        test_dataloader.dataset.normalize(scaler=model.train_dataset.scaler)

    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         accelerator="dp",
                         logger=None)
    trainer.test(model=model, test_dataloaders=test_dataloader)
    return trainer


@click.command()
@click.option('--batch_size', type=click.INT, default=100, help='Batch size')
@click.argument('test_dataset_path', type=click.Path())
@click.argument('model_path', type=click.Path())
def cli(batch_size, test_dataset_path, model_path):
    """ This script can be used to test the performance of a pre-trained model.

    Given by a checkpoint saved at MODEL_PATH. The prediction performance of this model will be evaluated on a dataset
    located at TEST_DATASET_PATH. This loads the scaler that was used for normalization at training time.
    """
    model = load_model(model_path)
    start_testing(model=model,
                  test_dataset_path=test_dataset_path,
                  batch_size=batch_size)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli()
