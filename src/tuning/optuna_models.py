from abc import abstractmethod, ABC
import os
import numpy as np
import sqlalchemy.exc
import torch.cuda
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from src.models.lightning_models import Transformer, SeqToSeqLSTM, SeqToSeqCNN
from log import logger_hp_optim
from pytorch_lightning.callbacks import ModelCheckpoint


class Optimizer(ABC):
    """
    Optuna optimizer model. The difference to a lightning model is, that some hyperparameters are not fixed but
    only a range is suggested and multiple of them will be compared to test for the best hyperparameter settings.

    Child class of this has to be implemented that suggests the hypeparameters for a specific model.
    """
    def __init__(
        self,
        train_dataset,
        val_dataset,
        loss_function,
        d_input,
        d_output,
        model_dir_path,
        min_batch_size,
        max_batch_size,
        max_model_depth,
        max_d_model,
        max_epochs=30,
    ):
        """Abstract Hyperparameter optimization class.

        :param train_dataset: path to train dataset. Used for normalization
        :param val_dataset: path to val dataset.
        :param loss_function: loss function to minimize
        :param d_input: dimensionality of input time series
        :param d_output: dimensionality of output time series
        :param model_dir_path: path to save the trained models in
        :param min_batch_size: minimum value for search space of batch size
        :param max_batch_size: maximum value for search space of batch size
        :param max_model_depth: maximum depth of model
        :param max_d_model: maximum model dimensionality
        :param max_epochs: maximum amount of epochs to train for
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_function = loss_function
        self.d_input = d_input
        self.d_output = d_output
        self.model_dir_path = model_dir_path
        self.max_epochs = max_epochs

        self.best_run = {
            "val_loss": np.inf,
            "checkpoint_path": "",
            "tensorboard_logs_dir": ""
        }

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_d_model = max_d_model
        self.max_model_depth = max_model_depth

    def set_batch_size(self, batch_size):
        self.min_batch_size = batch_size
        self.max_batch_size = batch_size

    def update_best(self, val_loss, checkpoint_path, tensorboard_logs_dir):
        """
        Checks if the current model is not only the best with the current setting but also the best across all runs.
        If so update theses values. Will be used at the end of the optimization session to see log the best model.

        :param val_loss: best val loss of model
        :param checkpoint_path: checkpoint that reached this val_loss
        :param tensorboard_logs_dir: path to where the tensorboard logs are
        :return:
        """
        if val_loss <= self.best_run["val_loss"]:
            self.best_run["val_loss"] = val_loss
            self.best_run["checkpoint_path"] = checkpoint_path
            self.best_run["tensorboard_logs_dir"] = tensorboard_logs_dir
        logger_hp_optim.info \
            (f"Path to best model with this Hyperparameter setting: {checkpoint_path} with val_loss:{val_loss} and Tensorboard Logs at {tensorboard_logs_dir}")

    @abstractmethod
    def get_model(self, trial):
        """
        :param trial: Optuna trial object
        :return: Pytorch-Lightning model with suggested hyperparameters
        """
        pass

    def objective(self, trial):
        """Objective to minimize. In this case self.loss_funcition of the model output

        :param trial: Optuna trial object
        :return: validation loss
        """
        model = self.get_model(trial)
        logger_hp_optim.info(f"Params in current trial: {trial.params}")
        early_stop_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_loss")
        checkpoint_callback = ModelCheckpoint(dirpath=self.model_dir_path,
                                              monitor='val_loss',
                                              filename=self.train_dataset +
                                              '-{epoch:02d}-{val_loss:.4f}')
        logger = TensorBoardLogger(self.model_dir_path, name="tb_logs")
        gradient_clip_val = 0.0 if isinstance(self,
                                              TransformerHpOptimizer) else 0.5
        logger_hp_optim.info(
            f"Gradient clipping (0 means no clipping): {gradient_clip_val}")
        trainer = pl.Trainer(
            logger=logger,
            gpus=torch.cuda.device_count(),
            accelerator="dp",
            max_epochs=self.max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            limit_train_batches=1.0,
            fast_dev_run=False,
            log_gpu_memory=None,
            gradient_clip_val=gradient_clip_val,
        )

        try:
            trainer.fit(model)
            val_loss = float(trainer.logged_metrics["val_loss"])
        except RuntimeError as e:
            logger_hp_optim.error("cuda out of memory")
            torch.cuda.empty_cache()
            val_loss = 1
        except sqlalchemy.exc.IntegrityError as e:
            logger_hp_optim.error("val_loss is nan")
            val_loss = float(trainer.logged_metrics["val_loss"])

        # check if current model is best of study and if so save
        self.update_best(val_loss=val_loss,
                         checkpoint_path=checkpoint_callback.best_model_path,
                         tensorboard_logs_dir=trainer.log_dir)

        # save train and val loss into log directory
        np.save(os.path.join(logger.log_dir, "train_loss"),
                arr=np.array(model.train_loss))
        np.save(os.path.join(logger.log_dir, "val_loss"),
                arr=np.array(
                    model.val_loss)[1:])  # 1st entry from gut-check trial

        return val_loss


class LstmHpOptimizer(Optimizer):
    """ Optuna model of a LSTM Network that can be used for hyperparameter optimization. """
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 loss_function,
                 d_input,
                 d_output,
                 model_dir_path,
                 min_batch_size=32,
                 max_batch_size=128,
                 max_model_depth=6,
                 max_d_model=64,
                 max_epochs=30):
        """
        :param train_dataset: Train dataset
        :param val_dataset: Validatoin dataset
        :param loss_function: Loss function to be used for training/evaluation
        :param d_input: number of input dimensions (1 for univariate)
        :param d_output: number of output dimmensions (1 for univariate)
        :param model_dir_path: path to save the trained models in
        :param min_batch_size: minimum search space value for batch size
        :param max_batch_size: maximum search space value for batch size
        :param max_model_depth: maximum depth of the model
        :param max_d_model: maximum latent dimensionality of the model
        :param max_epochs: maximum amount of epochs
        """
        super().__init__(train_dataset, val_dataset, loss_function, d_input,
                         d_output, model_dir_path, min_batch_size,
                         max_batch_size, max_model_depth, max_d_model,
                         max_epochs)
        self.model = None

    def get_model(self, trial):
        batch_size = trial.suggest_int("batch_size", self.min_batch_size,
                                       self.max_batch_size)
        N = trial.suggest_int("N", 1, self.max_model_depth)
        d_model = trial.suggest_int("d_model", 16, self.max_d_model)
        learning_rate = trial.suggest_float("learning_rate",
                                            5e-5,
                                            5e-3,
                                            log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        weight_decay = trial.suggest_float("weight_decay",
                                           0.00001,
                                           0.01,
                                           log=True)

        model = SeqToSeqLSTM(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            loss_function=self.loss_function,
            d_input=self.d_input,
            d_output=self.d_output,
            batch_size=batch_size,
            N=N,
            d_model=d_model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
        )
        self.model = model
        return model


class CnnHpOptimizer(Optimizer):
    """ Optuna model of a CNN Network that can be used for hyperparameter optimization. """
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 loss_function,
                 d_input,
                 d_output,
                 model_dir_path,
                 min_batch_size=32,
                 max_batch_size=128,
                 max_model_depth=6,
                 max_d_model=64,
                 max_epochs=30):
        """
        :param train_dataset: Train dataset
        :param val_dataset: Validation dataset
        :param loss_function: Loss function to be used for training/evaluation
        :param d_input: Input dimensionality of the model (1 for univariate)
        :param d_output: Output dimensionality of the model (1 for univariate)
        :param model_dir_path: path to save models in
        :param min_batch_size: minimum of batch size search space
        :param max_batch_size: maximum of batch size search space
        :param max_model_depth: maximum model depth
        :param max_d_model: maximum latent dimensionality of model
        :param max_epochs: maximum amount of epochs each model trains
        """
        super().__init__(train_dataset, val_dataset, loss_function, d_input,
                         d_output, model_dir_path, min_batch_size,
                         max_batch_size, max_model_depth, max_d_model,
                         max_epochs)
        self.model = None

    def get_model(self, trial):
        batch_size = trial.suggest_int("batch_size", self.min_batch_size,
                                       self.max_batch_size)
        N = trial.suggest_int("N", 1, self.max_model_depth)
        d_model = trial.suggest_int("d_model", 1, self.max_d_model)
        kernel_size = trial.suggest_int("kernel_size", 3, 5, step=2)
        learning_rate = trial.suggest_float("learning_rate",
                                            3e-5,
                                            3e-4,
                                            log=True)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        weight_decay = trial.suggest_float("weight_decay",
                                           0.00001,
                                           0.01,
                                           log=True)

        model = SeqToSeqCNN(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            loss_function=self.loss_function,
            batch_size=batch_size,
            d_model=d_model,
            kernel_size=kernel_size,
            N=N,
            dropout=dropout,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
        )
        self.model = model
        return model


class TransformerHpOptimizer(Optimizer):
    """Optuna model of a Transformer Network that can be used for hyperparameter optimization. """
    def __init__(
        self,
        train_dataset,
        val_dataset,
        loss_function,
        d_input,
        d_output,
        model_dir_path,
        min_batch_size=32,
        max_batch_size=128,
        max_model_depth=6,
        max_d_model=64,
        max_epochs=30,
    ):
        super().__init__(train_dataset, val_dataset, loss_function, d_input,
                         d_output, model_dir_path, min_batch_size,
                         max_batch_size, max_model_depth, max_d_model,
                         max_epochs)
        self.model = None
        if self.max_model_depth is None:
            self.max_model_depth = 6
        if self.max_d_model is None:
            self.max_d_model = 64

    def get_model(self, trial):
        N = trial.suggest_int("N", 1, self.max_model_depth)
        # restrict d_model to be power of 2
        log_d_model = trial.suggest_int("log_d_model", 3,
                                        int(np.log2(self.max_d_model)))
        d_model = 2**log_d_model

        # restrict h to be power of 2
        log_h = trial.suggest_int("log_h", 1, 3)
        h = 2**log_h

        batch_size = trial.suggest_int("batch_size", self.min_batch_size,
                                       self.max_batch_size)
        d_ff = trial.suggest_int("d_ff", d_model, 512, log=True)
        opt_factor = trial.suggest_float("opt_factor", 1e-1, 1e1, log=True)
        opt_warmup = trial.suggest_int("opt_warmup", 200, 800)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        weight_decay = trial.suggest_float("weight_decay",
                                           0.00001,
                                           0.01,
                                           log=True)

        model = Transformer(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            loss_function=self.loss_function,
            d_input=self.d_input,
            d_output=self.d_output,
            batch_size=batch_size,
            d_model=d_model,
            d_ff=d_ff,
            N=N,
            h=h,
            dropout=dropout,
            opt_factor=opt_factor,
            opt_warmup=opt_warmup,
            weight_decay=weight_decay,
        )
        self.model = model
        return model
