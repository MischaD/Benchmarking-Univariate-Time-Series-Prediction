import pytest
import torch.nn as nn
import torch
from src.models.lightning_models import SyntheticDatasetModel, Transformer, SeqToSeqLSTM, SeqToSeqCNN
from src.data.make_dataset import make_mixed_dataset
from src.data.sampling_strategy import DefaultSamplingStrategy
import numpy as np


class TestLightningModels:
    @pytest.fixture()
    def sequence_length(self):
        return 10

    @pytest.fixture()
    def dataset(self, sequence_length):
        return make_mixed_dataset(
            sample_size=32,
            sample_sequence_lengths=[
                sequence_length,
            ],
            sampling_strategy_class=DefaultSamplingStrategy,
            sigmas=[0, 5],
            frequencies=[1 / 8, 1 / 32],
            delays=[0, 32])

    def test_synthetic_dataset_model(self, dataset):
        model = SyntheticDatasetModel(
            batch_size=10,
            train_dataset=dataset,
            val_dataset=dataset,
        )
        # make sure that this kind of model does not have any kind of model it can used for the forward pass
        assert not hasattr(model, "model")  # model.model used in forward

    def test_transformer(self, dataset, sequence_length):
        batch_size = 8
        transformer = Transformer(
            batch_size=batch_size,
            train_dataset=dataset,
            val_dataset=dataset,
            loss_function=nn.L1Loss(),
            d_input=1,
            d_output=1,
            N=4,
            d_model=64,
            d_ff=2048,
            h=8,
            dropout=0.2,
        )
        # input length is only half of the sequence length
        sequence_length = torch.tensor([
            sequence_length,
        ])
        inp = torch.zeros(batch_size, sequence_length // 2, 1)
        outp = transformer(inp, sequence_length)
        assert outp.shape == inp.shape

        x, y, chars = next(iter(transformer.train_dataloader()))
        assert x.shape == inp.shape
        assert y.shape == outp.shape

    def test_lstm(self, dataset, sequence_length):
        batch_size = 8
        s2s_lstm = SeqToSeqLSTM(
            batch_size=batch_size,
            train_dataset=dataset,
            val_dataset=dataset,
            loss_function=nn.L1Loss(),
            d_input=1,
            d_output=1,
            N=5,
            d_model=64,
            dropout=0.2,
        )

        inp = torch.zeros(batch_size, sequence_length // 2, 1)
        outp = s2s_lstm(inp, sequence_length)
        assert outp.shape == inp.shape

        x, y, chars = next(iter(s2s_lstm.train_dataloader()))
        assert x.shape == inp.shape
        assert y.shape == outp.shape

    def test_cnn(self, dataset, sequence_length):
        batch_size = 8
        s2s_cnn = SeqToSeqCNN(
            batch_size=batch_size,
            train_dataset=dataset,
            val_dataset=dataset,
            loss_function=nn.L1Loss(),
            N=5,
            d_model=64,
            kernel_size=3,
            max_dilation=int(np.log2(sequence_length)) + 1,
            dropout=0.2,
        )

        inp = torch.zeros(batch_size, sequence_length // 2, 1)
        outp = s2s_cnn(inp, sequence_length)
        assert outp.shape == inp.shape

        x, y, chars = next(iter(s2s_cnn.train_dataloader()))
        assert x.shape == inp.shape
        assert y.shape == outp.shape
