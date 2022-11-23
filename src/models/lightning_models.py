from src.models.transformer import make_encoder_only_time_series_model
from src.models.rnn import SeqToSeqLSTMModule
from src.models.cnn import TCN
from src.models.optimizer import NoamOpt
from src.models.cnn.utils import CNNValidityChecker
from src.data.synthetic_dataset import SyntheticDataset
from src.data.utils import get_sample_seq_len
from log import logger_models
from torch.utils.data import DataLoader, Dataset
from torch.optim.adam import Adam
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class SyntheticDatasetModel(pl.LightningModule):
    """Lightning Model parent class for all models that implements model independent methods

    Args:
        :param batch_size: batch_size
        :param train_dataset: train dataset or path to train dataset
        :param val_dataset: val dataset or path to val dataset
        :param test_dataset: test dataset or path to test dataset
    """
    def __init__(
        self,
        batch_size,
        train_dataset,
        val_dataset,
        test_dataset=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        if self.test_dataset is None:
            logger_models.info(
                "Initializing Lightning model without test dataset")

        self.load_datasets()
        self.sequence_length = self.train_dataset.sequence_length

        self.normalize_data()

        # saves the characteristics together with the mae and mse for every test sample once trainer.test() has be invoked
        self.evaluated_samples = {
            "d": [],
            "l": [],
            "s": [],
            "f": [],
            "a1": [],
            "a2": [],
            "a3": [],
            "mean": [],
            "mae": [],
            "mse": []
        }
        # saves output predictions for this model
        self.output_predictions = []

        # log loss for easy access in visualization
        self.train_loss = []
        self.val_loss = []

    def load_datasets(self):
        """Load datasets if the datasets were provided as paths
        """
        if isinstance(self.train_dataset, str):
            self.train_dataset = SyntheticDataset.load(self.train_dataset)
        if isinstance(self.val_dataset, str):
            self.val_dataset = SyntheticDataset.load(self.val_dataset)
        if self.test_dataset:
            if isinstance(self.test_dataset, str):
                self.test_dataset = SyntheticDataset.load(self.test_dataset)

    def normalize_data(self):
        """Normalize train, validation and test data with the normalizer used for train dataset normlization.
        """
        self.train_dataloader().dataset.normalize()

        # use train scaler to scale validation and test set
        scaler = self.train_dataset.scaler
        self.val_dataset.normalize(scaler=scaler)
        if self.test_dataset:
            self.test_dataset.normalize(scaler=scaler)

    def train_dataloader(self):
        if not self.train_dataset.is_shuffled:
            self.train_dataset.shuffle(self.batch_size)

        dataloader_train = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=False,
                                      drop_last=False)
        return dataloader_train

    def val_dataloader(self):
        if not self.val_dataset.is_shuffled:
            # shuffling doesn't change the loss but this makes sure that the batches returned all have the same
            # sample_sequence_length
            self.val_dataset.shuffle(self.batch_size)

        dataloader_val = DataLoader(self.val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    drop_last=False)
        return dataloader_val

    def test_dataloader(self):
        if not self.test_dataset.is_shuffled:
            # shuffling doesn't change the loss but this makes sure that the batches returned all have the same
            # sample_sequence_length
            self.test_dataset.shuffle(self.batch_size)

        dataloader_test = DataLoader(self.test_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     drop_last=False)
        return dataloader_test

    def forward(self, src, sample_seq_len):
        """Forward pass of model.

        Only returns the part of the sequence that is relevant for predicting.
        """
        seq_len = self.train_dataset.sequence_length
        if sample_seq_len == seq_len:
            return self.model.forward(src, sample_seq_len)
        else:
            one_sided_padding = int((seq_len - sample_seq_len) // 2)
            return self.model.forward()[:, one_sided_padding:, :]

    def prune_ground_truth(self, y_out, sample_seq_len):
        """Depending on the actual sample sequence lenght of the smaples in the current batch it may be necessary
        to prune the ground trouth values in such a way that the loss is only evaluated over relevant values.
        """
        if sample_seq_len == self.train_dataset.sequence_length:
            # entire model 'width' used i.e. full attention mask
            return y_out
        else:
            # part of the model not used, sequence smaller than maximum
            # i.e not the entire output sequence has to be predicted
            one_sided_padding = int(
                (self.train_dataset.sequence_length - sample_seq_len) // 2)
            return y_out[:, :-one_sided_padding, :]

    def training_step(self, batch, batch_idx):
        if not self.train_dataset.is_shuffled:
            logger_models.warn("Train dataset not shuffled")

        x, y_out, characteristics = batch
        # sample sequence length in the current batch
        sample_seq_len = get_sample_seq_len(characteristics)
        pruned_y_out = self.prune_ground_truth(y_out=y_out,
                                               sample_seq_len=sample_seq_len)

        y_out_pred = self(x, sample_seq_len)
        loss = self.loss_function(y_out_pred, pruned_y_out)

        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_loss.append(train_loss.item())
        self.log('train_loss_epoch', train_loss)

    def validation_step(self, batch, batch_idx):
        x, y_out, characteristics = batch
        sample_seq_len = get_sample_seq_len(characteristics)
        y_out_pred = self(x, sample_seq_len)

        pruned_y_out = self.prune_ground_truth(y_out=y_out,
                                               sample_seq_len=sample_seq_len)
        loss = self.loss_function(y_out_pred, pruned_y_out)

        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.val_loss.append(avg_loss.item())
        self.log('val_loss_epoch', avg_loss)

    def test_step(self, batch, batch_idx):
        x, y_out, characteristics = batch
        sample_seq_len = get_sample_seq_len(characteristics)

        y_out_pred = self(x, sample_seq_len)
        pruned_y_out = self.prune_ground_truth(y_out=y_out,
                                               sample_seq_len=sample_seq_len)

        mse_crit = nn.MSELoss()
        mae_crit = nn.L1Loss()

        mse = mse_crit(y_out_pred, pruned_y_out).item()
        mae = mae_crit(y_out_pred, pruned_y_out).item()

        #characteristics
        characteristics = [x.cpu().numpy() for x in characteristics]
        self.evaluated_samples["a1"].extend(characteristics[2][:, 0, 0])
        self.evaluated_samples["a2"].extend(characteristics[2][:, 0, 1])
        self.evaluated_samples["a3"].extend(characteristics[2][:, 0, 2])
        self.evaluated_samples["mean"].extend(characteristics[1][:, 0, 0])

        self.evaluated_samples["f"].extend(characteristics[0][:, 0, 0])
        self.evaluated_samples["d"].extend(characteristics[0][:, 0, 1])
        self.evaluated_samples["l"].extend(characteristics[0][:, 0, 2])
        self.evaluated_samples["s"].extend(characteristics[0][:, 0, 3])

        self.evaluated_samples["mae"].extend(
            nn.L1Loss(reduction="none")(
                y_out_pred,
                pruned_y_out).mean(axis=1).cpu().numpy().reshape(-1))
        self.evaluated_samples["mse"].extend(
            nn.MSELoss(reduction="none")(
                y_out_pred,
                pruned_y_out).mean(axis=1).cpu().numpy().reshape(-1))

        # save output prediction for analysis/visualization
        self.output_predictions.extend(y_out_pred.cpu().numpy())
        return {'mse': mse, 'mae': mae}

    def test_step_end(self, outputs):
        if not isinstance(outputs, list):
            outputs = [
                outputs,
            ]

        avg_mae = np.array([float(x['mae']) for x in outputs]).mean()
        avg_rmse = np.array([float(x['mse']) for x in outputs]).mean()
        self.log('test_mae', avg_mae)
        self.log('test_mse', avg_rmse)


class Transformer(SyntheticDatasetModel):
    """
    Pytorch Lightning Transformer Model

    Args:
        :param batch_size: batch size
        :param train_dataset (str, Dataset): Either path to or dataset that will be used for training
        :param val_dataset (str, Dataset): Either path to or dataset that will be used for validation
        :param test_dataset (str, Dataset): Either path to or dataset that will be used for testing
        :param loss_function: loss function
        :param d_input: input dimensionality
        :param d_output: output dimensionality
        :param N: Number of layers
        :param d_model: model dimensionality
        :param d_ff: dimensionality of feedforward network after attention layers
        :param h: amount of heads in each layer
        :param dropout: dropout
        :param opt_factor: optimizer learning rate scaler dependant on
        :param opt_warmup: optimizer number of warmup steps
        :
    """
    def __init__(
        self,
        batch_size,
        train_dataset,
        val_dataset,
        loss_function,
        d_input,
        d_output,
        test_dataset=None,
        N=4,
        d_model=64,
        d_ff=2048,
        h=8,
        dropout=0.2,
        opt_factor=1,
        opt_warmup=400,
        weight_decay=0,
    ):

        # implements all the methods that are independant of the model
        super().__init__(batch_size, train_dataset, val_dataset, test_dataset)

        self.loss_function = loss_function

        self.d_input = d_input
        self.d_output = d_output

        self.N = N

        self.d_model = d_model
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout
        self.weight_decay = weight_decay

        #optimizer params
        self.opt_factor = opt_factor
        self.opt_warmup = opt_warmup

        # Init Model
        self.model = make_encoder_only_time_series_model(
            d_input=self.d_input,
            d_output=self.d_output,
            N=self.N,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout)
        # No masking of 'future' values because they are actually past values
        self.mask = torch.ones(1, 1, self.sequence_length // 2).to("cuda")
        self.save_hyperparameters()
        logger_models.info(
            f"Training size: {len(self.train_dataset)}, Validation size: {len(self.val_dataset)}"
        )

        self.check_complexity()

    def check_complexity(self):
        """Rough estimate how memory consuming this Transformer is going to be."""
        complexity = self.N * self.batch_size * self.h * (
            self.sequence_length)**2
        logger_models.info(
            f"Transformer complexity: {complexity}")  # still 286e6 works

    def configure_optimizers(self):
        opt_beta_1 = 0.9
        opt_beta_2 = 0.98
        optimizer = NoamOpt(self.parameters(),
                            self.d_model,
                            self.opt_factor,
                            self.opt_warmup,
                            opt_beta_1,
                            opt_beta_2,
                            eps=1e-9,
                            weight_decay=self.weight_decay)
        return optimizer

    def get_mask(self, sample_seq_len):
        if sample_seq_len == self.sequence_length:
            return torch.ones(1, self.sequence_length // 2,
                              self.sequence_length // 2).to(device=self.device)
        else:
            # sample seq shorter than model itself. This means that parts of the input can be ignored. This can be
            # done using the attention mask.
            mask = torch.zeros(1, self.sequence_length // 2,
                               self.sequence_length //
                               2).to(device=self.device)
            mask[0] = torch.diag(torch.ones(self.sequence_length // 2))
            # assert torch.all([sample_seq_len == sample_seq_len[0]])
            start_of_relevant_points = int(sample_seq_len.item() // 2)
            mask[0, -start_of_relevant_points:, -start_of_relevant_points:] = 1
            return mask

    def forward(self, src, sample_seq_len):
        """Transformer forward pass

        Transformer needs the input and the actual sample sequence length of each sequence in the batch.
        It is recommended that each sample sequence has the same sequence.
        The mask for every inputs sample looks slitghtly different depending on its sample sequence length.
        See method Transformer.get_mask() for more detail.
        """

        mask = torch.ones(1, int(sample_seq_len.item() // 2),
                          int(sample_seq_len.item() //
                              2)).to(device=self.device)
        if sample_seq_len == self.sequence_length:
            output = self.model.forward(src, mask)
            return output
        else:
            one_sided_padding = int(
                (self.sequence_length - sample_seq_len) // 2)
            return self.model.forward(src[:, one_sided_padding:, :], mask)


class SeqToSeqLSTM(SyntheticDatasetModel):
    """
    Sequence to sequence lstm architecture

    Args:
        :param batch_size: batch size
        :param train_dataset: train dataset
        :param val_dataset: validation dataset
        :param loss_function: loss function
        :param d_input: input dimensionality
        :param d_output: output dimenstionality
        :param test_dataset: test dataset
        :param N: layer depth of lstm
        :param d_model: model dimensionality or hidden dimensionality of lstm
        :param dropout: dropout
        :param learning_rate: optimizer learning rate
        :param weight_decay: weight decay or l2 regularization of weights in the optimizer
    """
    def __init__(
        self,
        batch_size,
        train_dataset,
        val_dataset,
        loss_function,
        d_input,
        d_output,
        test_dataset=None,
        N=1,
        d_model=64,
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=0,
    ):

        super().__init__(batch_size, train_dataset, val_dataset, test_dataset)

        self.loss_function = loss_function

        self.d_input = d_input
        self.d_output = d_output
        self.N = N
        self.d_model = d_model
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model = SeqToSeqLSTMModule(
            d_input=d_input,
            d_hidden=d_model,
            N=self.N,
            dropout=self.dropout,
            d_output=self.d_output,
            device=self.device,
        )

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.learning_rate,
                         weight_decay=self.weight_decay)
        return optimizer

    def forward(self, src, sample_seq_len):
        src = src.transpose(0, 1)  # put batch second
        batch_size = src.shape[1]
        start_of_non_padded_sequence = int(
            (self.sequence_length - sample_seq_len) // 2)
        if start_of_non_padded_sequence != 0:
            src = src[start_of_non_padded_sequence:]

        # initialize empty memeory and decoder input and output arrays
        hidden = (torch.zeros(2 * self.N, batch_size,
                              self.d_model).to(self.device),
                  torch.zeros(2 * self.N, batch_size,
                              self.d_model).to(self.device))
        #outputs = torch.zeros((sequence_len, self.batch_size, self.d_output * 2 )).to(self.device)  # Initializing this in nn.Module doesn't work

        # forward
        outputs = self.model(src, hidden)

        decoder_outputs = outputs.transpose(0, 1)  # put batch first
        return decoder_outputs


class SeqToSeqCNN(SyntheticDatasetModel):
    """
    Sequence to sequence cnn architecture (TCN)

    Args:
        :param batch_size: batch size
        :param loss_function: loss function
        :param N: layer depth of tcn. N TemporalBlocks are stacked after each other
        :param d_model: model dimensionality or hidden dimensionality of cnn
        :param kernel_size: cnn kernel size
        :param max_dilation:maximum dilation length. Has to be power of 2. ld(max_dilation) is the depth of each Block. E.g.
        max_dilation=8 means that the dilations within one block will be [1,2,4,8]
        :param dropout: dropout
        :param weight_decay: weight decay or l2 regularization of weights in the optimizer
        :param train_dataset: train dataset
        :param val_dataset: validation dataset
        :param test_dataset: test dataset
        :param learning_rate: optimizer learning rate
    """
    def __init__(self,
                 batch_size,
                 loss_function,
                 N,
                 d_model,
                 kernel_size,
                 dropout,
                 train_dataset,
                 val_dataset,
                 learning_rate=1e-3,
                 max_dilation=None,
                 weight_decay=0,
                 test_dataset=None):
        super().__init__(batch_size, train_dataset, val_dataset, test_dataset)
        self.loss_function = loss_function

        self.N = N
        self.d_model = d_model
        self.kernel_size = kernel_size
        if max_dilation is None:
            # model width will be self.sequence_length // 2 and with dilated_convs we reach all nodes
            max_dilation = self.sequence_length // 4
        self.dilations = [2**i for i in range(int(np.log2(max_dilation)) + 1)]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        logger_models.info(f"dilations: {self.dilations}")
        self.model = TCN(
            input_size=1,
            output_size=1,
            num_channels=[
                d_model,
            ] * N,
            dilations=self.dilations,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.save_hyperparameters()
        self.check_validity()

    def check_validity(self):
        """Sends out a warning if the receptive field of the model is too short.
        """

        cnn_validity_checker = CNNValidityChecker(
            N=self.N,
            dilation_factors=self.dilations,
            kernel_size=self.kernel_size,
            sequence_length=self.train_dataset.sequence_length // 2,
            causal_conv_only=False,
        )
        if not cnn_validity_checker.is_valid:
            logger_models.warn(
                "Receptive field of TCN smaller than sequence length.")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.learning_rate,
                         weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x, sample_seq_len):
        if sample_seq_len == self.sequence_length:
            out = self.model(x)
            return out
        else:
            # part of the input is just zero. We can prune this part away to make the model smaller
            one_sided_padding = int(
                (self.sequence_length - sample_seq_len) // 2)
            return self.model(x[:, one_sided_padding:, :])
