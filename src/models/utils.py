import os
import yaml
import numpy as np
from src.models.lightning_models import Transformer, SeqToSeqCNN, SeqToSeqLSTM
from log import logger_models

MODEL_CLASSES = {
    "transformer": Transformer,
    "lstm": SeqToSeqLSTM,
    "cnn": SeqToSeqCNN
}


class TrainedModels:
    model_names = ["transformer", "cnn", "lstm"]

    def __init__(self, models_path):
        """Helper class that provides access to all the trainined models at once. Assumes that all the models were
        trained usinging src.tuning.hyperparameter_optimization with the same EXPERIMENT_NAME.

        :param models_path: Path to folder that has trained cnn, rnn and transformer models in it
        """
        self.cnn = {}
        self.lstm = {}
        self.transformer = {}
        self.model_dict = {}

        for model_name in self.model_names:
            self.model_dict[model_name] = {}
            self.model_dict[model_name]["summary"] = self._load_summary(
                models_path, model_name)
            self.model_dict[model_name]["path"] = self.model_dict[model_name][
                "summary"]["best_model"]["checkpoint_path"].split("/.")[-1]
            self.model_dict[model_name]["model"] = load_model(
                os.path.join(models_path, model_name) +
                self.model_dict[model_name]["path"])
            tb_logs_path = self.model_dict[model_name]["summary"][
                "best_model"]["tensorboard_logs_dir"]

            self.model_dict[model_name]["val_loss"] = get_loss(
                base_path=tb_logs_path, loss_name="val_loss.npy")
            self.model_dict[model_name]["train_loss"] = get_loss(
                base_path=tb_logs_path, loss_name="train_loss.npy")

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.model_dict[item]
        return self.model_dict[self.model_names[item]]

    def _load_summary(self, models_path, model_name):
        with open(os.path.join(models_path, model_name, "study_summary.yaml"),
                  "r") as inp:
            try:
                summary = yaml.load(inp, Loader=yaml.BaseLoader)
            except yaml.YAMLError as e:
                print(e)
            return summary

    def get_val_loss(self, model_name):
        return self.model_dict[model_name]["val_loss"]

    def get_train_loss(self, model_name):
        return self.model_dict[model_name]["train_loss"]


def get_lightning_class(model_type):
    return MODEL_CLASSES[model_type]


def build_model_kwargs_from_optuna_summary(optuna_summary):
    """Builds model keyword arguments for a optuna_study.yaml file. No knowledge about the model
    type required.

    :param optuna_summary: optuna_study.yaml file created by src.tuning.hyperparameter_optimization.py
    :return: dictionary with kwargs to create lightning model
    """
    kwargs = {}
    model_type = optuna_summary["model"]
    best_params = optuna_summary["best_params"]

    kwargs["batch_size"] = int(best_params["batch_size"])
    kwargs["N"] = int(best_params["N"])
    kwargs["dropout"] = float(best_params["dropout"])
    kwargs["weight_decay"] = float(best_params["weight_decay"])

    # add model specific kwargs
    if model_type == "cnn":
        kwargs["kernel_size"] = int(best_params["kernel_size"])
        kwargs["learning_rate"] = float(best_params["learning_rate"])
        kwargs["d_model"] = int(best_params["d_model"])
    elif model_type == "lstm":
        kwargs["learning_rate"] = float(best_params["learning_rate"])
        kwargs["d_model"] = int(best_params["d_model"])
        kwargs["d_input"] = 1
        kwargs["d_output"] = 1
    elif model_type == "transformer":
        kwargs["d_input"] = 1
        kwargs["d_output"] = 1
        kwargs["d_ff"] = int(best_params["d_ff"])
        kwargs["d_model"] = 2**int(best_params["log_d_model"])
        kwargs["h"] = 2**int(best_params["log_h"])
        kwargs["opt_factor"] = float(best_params["opt_factor"])
        kwargs["opt_warmup"] = float(best_params["opt_warmup"])
    return kwargs


def load_model(model_path):
    for ModelClass in MODEL_CLASSES.values():
        try:
            model = ModelClass.load_from_checkpoint(model_path)
            logger_models.info(
                f"Provided path successfully loaded Model of type: {ModelClass}"
            )
            return model
        except:
            continue
    raise ValueError(
        f"No fitting model class found for path {model_path}. Make sure that the model path is correct, the checkpoint exists and that the model was trained on the same version as it is tried to be tested on."
    )


def get_best_model_path(basepath, model_name):
    """ Get best model after training with train_model.py optuna
    basepath: path to tb logs
    model_name: model_name
    """
    path = os.path.join(basepath, model_name, "version_0/checkpoints")
    ckpt = os.listdir(path)[0]
    return os.path.join(path, ckpt)


def get_loss(base_path, loss_name):
    """Get loss from model that was logged and saved during training/validation time.
    Assumes training done by hyperparameter_tuning.py therefore best model is logged in summary.

    basepath: path to best model tb_logs
    model_name: name of model - cnn, lstm or transformer
    """
    return np.load(os.path.join(base_path, loss_name))
