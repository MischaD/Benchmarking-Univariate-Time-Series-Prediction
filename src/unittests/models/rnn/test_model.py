from src.models.rnn.model import SeqToSeqLSTMModule
import torch


class TestRNN:
    def test_s2s_lstm(self):
        N = 5
        d_model = 64
        lstm = SeqToSeqLSTMModule(d_input=1,
                                  d_output=1,
                                  d_hidden=d_model,
                                  N=N,
                                  dropout=0.2)

        src = torch.zeros(20, 10, 1)  # length of 10
        hidden = (torch.zeros(2 * N, src.shape[1], d_model),
                  torch.zeros(2 * N, src.shape[1], d_model))

        out = lstm(src, hidden)
        assert out.shape == (20, 10, 1)
