import click
import os
import yaml
from log import logger_models
from src.models.utils import load_model, build_model_kwargs_from_optuna_summary, get_lightning_class
from src.models.lightning_models import Transformer, SeqToSeqLSTM, SeqToSeqCNN
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch.nn as nn
import torch
import numpy as np

LOSS_FUNCTION_CHOICES = {
    "l1": nn.L1Loss(),
    "l2": nn.MSELoss(),
}


@click.group()
def cli():
    """ This script is used to train a model of type COMMAND."""
    pass


def start_training(model, epochs, name, patience=50, tb_logs_save_dir=None):
    """
    Starts model training

    :param model: lightnign model to be used for training
    :param epochs: amount of training epochs
    :param name: tensorboard logger name
    :param patience: training patience until no longer improving validation loss makes training stop
    :param tb_logs_save_dir: where to save tensorboard logs to
    :return: path to best model
    """
    logger_models.info(
        f"Starting training with {len(model.train_dataset)} training samples, Validation on {len(model.val_dataset)} samples."
    )

    tb_logs_save_dir = "tb_logs" if tb_logs_save_dir is None else tb_logs_save_dir
    logger = TensorBoardLogger(tb_logs_save_dir, name=name)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    gradient_clip_val = 0.0 if isinstance(model, Transformer) else 0.5
    logger_models.info(
        f"Train Model Gradient clipping (0 means no clipping): {gradient_clip_val}"
    )

    # set early stopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss",
                                                     min_delta=0.001,
                                                     patience=patience,
                                                     verbose=False)
    logger_models.info(f"Using early stop callback with patience {patience}")

    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         accelerator="dp",
                         logger=logger,
                         max_epochs=epochs,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         limit_train_batches=1.0,
                         fast_dev_run=False,
                         log_gpu_memory=None,
                         gradient_clip_val=gradient_clip_val)

    trainer.fit(model)
    best_path = checkpoint_callback.best_model_path

    # save train and val loss into log directory
    np.save(os.path.join(logger.log_dir, "train_loss"),
            arr=np.array(model.train_loss))
    np.save(os.path.join(logger.log_dir, "val_loss"),
            arr=np.array(model.val_loss))

    logger_models.info(f"Path to best model: {best_path}")
    return best_path


@click.command()
@click.option('--epochs',
              type=click.INT,
              default=100,
              help='Number of training epochs')
@click.option('--batch_size', type=click.INT, default=100, help='Batch size')
@click.option('--n',
              type=click.INT,
              default=4,
              help='Number of Transformers layers')
@click.option('--d_model',
              type=click.INT,
              default=64,
              help='Latent model dimensionality')
@click.option('--d_ff',
              type=click.INT,
              default=2048,
              help='Hidden dimensionality of positionwise feedforward layer')
@click.option('--h',
              type=click.INT,
              default=8,
              help='Number of attention heads in each layer')
@click.option('--dropout', type=click.FLOAT, default=0.2, help='Dropout')
@click.option('--opt_factor',
              type=click.FLOAT,
              default=1,
              help='Transformer optimizer factor for learning rate')
@click.option(
    '--opt_warmup',
    type=click.FLOAT,
    default=400,
    help=
    'Transformer optimizer number of warmup steps (linearly increasing learning rate)'
)
@click.option('--loss_function',
              type=click.Choice(LOSS_FUNCTION_CHOICES.keys()),
              default="l1",
              help='Loss function. Either l1 or l2')
@click.argument('train_dataset_filepath', type=click.Path())
@click.argument('validation_dataset_filepath', type=click.Path())
def transformer(epochs, batch_size, n, d_model, d_ff, h, dropout, opt_factor,
                opt_warmup, loss_function, train_dataset_filepath,
                validation_dataset_filepath):
    """Train a Transformer model.

    Script needs a training set given by TRAIN_DATASET_FILEPATH
    and a dataset for validation given by VALIDATION_DATASET_FILEPATH.
    """

    loss_function = LOSS_FUNCTION_CHOICES[loss_function]
    transformer = Transformer(
        batch_size=batch_size,
        train_dataset=train_dataset_filepath,
        val_dataset=validation_dataset_filepath,
        loss_function=loss_function,
        d_input=1,
        d_output=1,
        N=n,
        d_model=d_model,
        d_ff=d_ff,
        h=h,
        dropout=dropout,
        opt_factor=opt_factor,
        opt_warmup=opt_warmup,
    )
    path = start_training(model=transformer, epochs=epochs, name="transformer")


@click.command()
@click.option('--epochs',
              type=click.INT,
              default=100,
              help='Number of training epochs')
@click.option('--batch_size', type=click.INT, default=100, help='Batch size')
@click.option('--n',
              type=click.INT,
              default=4,
              help='Depth of the architecture')
@click.option('--d_model',
              type=click.INT,
              default=64,
              help='Latent model dimensionality')
@click.option('--loss_function',
              type=click.Choice(LOSS_FUNCTION_CHOICES.keys()),
              default="l1",
              help='Loss function. Either l1 or l2')
@click.option('--dropout', type=click.FLOAT, default=0.2, help='Dropout')
@click.option('--learning_rate',
              type=click.FLOAT,
              default=1e-3,
              help='Learning rate of optimizer')
@click.argument('train_dataset_filepath', type=click.Path())
@click.argument('validation_dataset_filepath', type=click.Path())
def lstm(epochs, batch_size, n, d_model, loss_function, dropout, learning_rate,
         train_dataset_filepath, validation_dataset_filepath):
    """Train a lstm model.

    Script needs a training set given by TRAIN_DATASET_FILEPATH
    and a dataset for validation given by VALIDATION_DATASET_FILEPATH.
    """
    logger_models.info(f"Rnn Model training for {epochs} epochs")
    loss_function = LOSS_FUNCTION_CHOICES[loss_function]
    s2s_lstm = SeqToSeqLSTM(
        batch_size=batch_size,
        N=n,
        dropout=dropout,
        d_model=d_model,
        d_input=1,
        d_output=1,
        loss_function=loss_function,
        learning_rate=learning_rate,
        train_dataset=train_dataset_filepath,
        val_dataset=validation_dataset_filepath,
    )
    best_path = start_training(model=s2s_lstm, epochs=epochs, name="lstm")


@click.command()
@click.option('--epochs',
              type=click.INT,
              default=100,
              help='Number of training epochs')
@click.option('--batch_size', type=click.INT, default=100, help='Batch size')
@click.option('--n',
              type=click.INT,
              default=10,
              help='Number of Transformers layers')
@click.option('--d_model',
              type=click.INT,
              default=64,
              help='Latent model dimensionality')
@click.option('--kernel_size',
              type=click.INT,
              default=7,
              help='cnn kernel size, has to be odd')
@click.option(
    '--max_dilation',
    type=click.INT,
    default="32",
    help=
    'maximum dilation length. Has to be power of 2. ld(max_dilation) is the depth of each Block. E.g. '
    '--max_dilation=8 means that the dilations within one block will be [1,2,4,8]'
)
@click.option('--loss_function',
              type=click.Choice(LOSS_FUNCTION_CHOICES.keys()),
              default="l1",
              help='Loss function. Either l1 or l2')
@click.option('--dropout', type=click.FLOAT, default=0.2, help='Dropout')
@click.option('--learning_rate',
              type=click.FLOAT,
              default=1e-3,
              help='Learning rate of optimizer')
@click.argument('train_dataset_filepath', type=click.Path())
@click.argument('validation_dataset_filepath', type=click.Path())
def cnn(epochs, batch_size, n, d_model, loss_function, kernel_size,
        max_dilation, dropout, learning_rate, train_dataset_filepath,
        validation_dataset_filepath):
    """Train a cnn model.

    Script needs a training set given by TRAIN_DATASET_FILEPATH
    and a dataset for validation given by VALIDATION_DATASET_FILEPATH."""

    loss_function = LOSS_FUNCTION_CHOICES[loss_function]
    s2s_cnn = SeqToSeqCNN(batch_size=batch_size,
                          N=n,
                          d_model=d_model,
                          kernel_size=kernel_size,
                          max_dilation=max_dilation,
                          dropout=dropout,
                          train_dataset=train_dataset_filepath,
                          val_dataset=validation_dataset_filepath,
                          loss_function=loss_function,
                          learning_rate=learning_rate)
    best_path = start_training(model=s2s_cnn, epochs=epochs, name="cnn")


@click.command()
@click.option('--max_epochs',
              type=click.INT,
              default=500,
              help='Maximum number of epochs')
@click.option('--patience',
              type=click.INT,
              default=10,
              help='Patience for the validation loss')
@click.option('--loss_function',
              type=click.Choice(LOSS_FUNCTION_CHOICES.keys()),
              default="l1",
              help='Loss function. Either l1 or l2')
@click.argument('train_dataset_filepath', type=click.Path())
@click.argument('validation_dataset_filepath', type=click.Path())
@click.argument('optuna_study', type=click.Path())
def optuna(max_epochs, patience, loss_function, train_dataset_filepath,
           validation_dataset_filepath, optuna_study):
    """ Train model from best model of optuna study
    """
    loss_function = LOSS_FUNCTION_CHOICES[loss_function]
    with open(os.path.join(optuna_study, "study_summary.yaml"), "r") as inp:
        summary = yaml.load(inp, Loader=yaml.BaseLoader)

    model_type = summary["model"]
    Klaas = get_lightning_class(model_type)

    # create model from study_summary.yaml file
    kwargs = build_model_kwargs_from_optuna_summary(summary)
    assert kwargs.get("max_dilation") is None
    model = Klaas(train_dataset=train_dataset_filepath,
                  val_dataset=validation_dataset_filepath,
                  loss_function=loss_function,
                  **kwargs)

    final_model_path = os.path.join(os.path.dirname(optuna_study), "final")
    best_path = start_training(model,
                               epochs=max_epochs,
                               patience=patience,
                               name=model_type,
                               tb_logs_save_dir=final_model_path)


cli.add_command(lstm)
cli.add_command(transformer)
cli.add_command(cnn)
cli.add_command(optuna)

if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli()
