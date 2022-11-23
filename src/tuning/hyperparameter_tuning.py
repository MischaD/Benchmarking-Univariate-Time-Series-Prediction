import gc
import logging
import sys
import yaml
import optuna
from log import logger_hp_optim, formatter
import click
from pathlib import Path
from src.tuning.optuna_models import TransformerHpOptimizer, CnnHpOptimizer, LstmHpOptimizer
from dotenv import find_dotenv, load_dotenv
import torch.nn as nn


@click.command()
@click.option('--max_epochs',
              type=click.INT,
              default=10,
              help='Maximum number of training epochs')
@click.option('--n_trials',
              type=click.INT,
              default=10,
              help='Number of training trials')
@click.option(
    '--n_startup_trials',
    type=click.INT,
    default=10,
    help=
    'Number trials in the beginning that use random sampling instead of TPE scheduler'
)
@click.option('--batch_size',
              type=click.INT,
              default=None,
              help='Fix the training batch size')
@click.option('--max_model_depth',
              type=click.INT,
              default=5,
              help='Enforce maximum on model depth N')
@click.option('--max_d_model',
              type=click.INT,
              default=64,
              help='Enforce maximum on model dimensionality dmodel')
@click.option(
    '--pruning',
    type=click.BOOL,
    default=False,
    help=
    'Whether or not to use median pruner after N_STARTUP_TRIAL to prune unpromising trials'
)
@click.argument('model', type=click.Choice(["cnn", "lstm", "transformer"]))
@click.argument('train_dataset_filepath', type=click.Path())
@click.argument('validation_dataset_filepath', type=click.Path())
@click.argument('study_name', type=click.STRING)
def cli(max_epochs, n_trials, n_startup_trials, batch_size, max_model_depth,
        max_d_model, pruning, model, train_dataset_filepath,
        validation_dataset_filepath, study_name):
    """Start a optuna study for Bayesian Hyperparameter Tuning.

    Script needs a training set given by TRAIN_DATASET_FILEPATH to train on
    and a dataset for validation given by VALIDATION_DATASET_FILEPATH to optimize on.
    Additionally it needs a STUDY_NAME. The final study will be saved in ./models/STUDY_NAME/optuna_study/.

    The study is conducted using TPE sampling.
    """
    # create directories for this study
    Path(f"models/{study_name}/{model}/optuna_study").mkdir(parents=True,
                                                            exist_ok=True)
    # add file handler to the hp logger to have logs available after training
    file_handler = logging.FileHandler(
        f"models/{study_name}/{model}/optuna.log")
    file_handler.setFormatter(formatter)
    logger_hp_optim.addHandler(file_handler)

    if model == "transformer":
        optimizer_class = TransformerHpOptimizer
    elif model == "cnn":
        optimizer_class = CnnHpOptimizer
    elif model == "lstm":
        optimizer_class = LstmHpOptimizer

    optimizer = optimizer_class(
        train_dataset=train_dataset_filepath,
        val_dataset=validation_dataset_filepath,
        loss_function=nn.L1Loss(),
        d_input=1,
        d_output=1,
        max_epochs=max_epochs,
        model_dir_path=Path(f"models/{study_name}/{model}"),
        max_model_depth=max_model_depth,
        max_d_model=max_d_model,
    )

    if batch_size is not None:
        #set max model batch size
        optimizer.set_batch_size(batch_size)

    logger_hp_optim.info(f"Starting study on {model} model")

    # Initializes storage db for the study to look at it later
    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout))
    optuna.logging.get_logger("optuna").addHandler(
        logging.FileHandler(f"models/{study_name}/{model}/optuna.log"))
    storage_name = f"sqlite:///models/{study_name}/{model}/optuna_study/{study_name}.db"

    if pruning:
        pruner = optuna.pruners.MedianPruner(n_min_trials=n_startup_trials,
                                             n_warmup_steps=200)
        logger_hp_optim.info(
            f"Using Median Pruner with {n_startup_trials} startup trials")
    else:
        pruner = optuna.pruners.NopPruner()
        logger_hp_optim.info(
            f"Using no pruner for hyperparameter optimization")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=n_startup_trials
        ),  # startup trials only if new study or study has less than n_startup_trials successfull trials
        pruner=pruner)
    # in case study fails, i can restart, reuse trials and don't even have change my count s.t. all end up having
    # the same amount of trials
    n_trials = n_trials - len(study.trials)
    logger_hp_optim.info(f"Remaining trials: {n_trials}")
    if n_trials <= 0:
        return
    study.optimize(optimizer.objective,
                   n_trials=n_trials,
                   callbacks=[lambda study, trial: gc.collect()])

    # Log some metadata like best parameters, trials and evaluation loss
    logger_hp_optim.info("=" * 80 + "study phase over! " + "=" * 80)
    logger_hp_optim.info(f"Best Model: {optimizer.best_run}")
    logger_hp_optim.info(
        f"Best parameters of {study_name}: {study.best_params}")
    logger_hp_optim.info(f"Best value of {study_name}: {study.best_value}")
    logger_hp_optim.info(f"Best trial of {study_name}: {study.best_trial}")

    # yaml dump most important things into study_summary.yaml
    summary = dict(
        model=model,
        study_name=study_name,
        storage_name=storage_name,
        best_model=optimizer.best_run,
        best_params=study.best_params,
        best_val_loss=study.best_value,
        best_trial_train_time_s=(
            study.best_trial.datetime_complete -
            study.best_trial.datetime_start).total_seconds(),
        best_trial=study.best_trial,
        best_run_tb_path=optimizer.best_run["tensorboard_logs_dir"],
        dataset=dict(
            sampling_strategy=optimizer.model.train_dataset.sampling_strategy,
            train_dataset_length=len(optimizer.model.train_dataset),
            val_dataset_length=len(optimizer.model.val_dataset),
        ))

    with open(f"models/{study_name}/{model}/study_summary.yaml", "w") as fp:
        yaml.dump(summary, fp)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    cli()
