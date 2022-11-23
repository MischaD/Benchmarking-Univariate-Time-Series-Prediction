import torch.nn as nn


class SeqToSeqLSTMModule(nn.Module):
    """Bidirectional Sequence to Sequence LSTM with FC Layer after the outputs.
     """
    def __init__(self, d_input, d_output, d_hidden, N, dropout, device="cpu"):
        """

        :param d_input: dimensionality of input time series
        :param d_output: dimensionality of output time series
        :param d_hidden: hidden dimensionality of the lstm
        :param N: depth of stacked lstm
        :param dropout: dropout probability
        :param device: pytorch device
        """
        super(SeqToSeqLSTMModule, self).__init__()
        self.num_layers = N
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output

        self.sequence_len = None
        self.batch_size = None
        self.device = device

        self.lstm = nn.LSTM(
            input_size=self.d_input,
            hidden_size=self.d_hidden,
            num_layers=self.num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.d_hidden * 2, 1)  # 2 for bidirectional

    def forward(self, inputs, hidden):
        out, _ = self.lstm(inputs, hidden)
        dropout_out = self.dropout(out)
        fc_out = self.fc(dropout_out)
        return fc_out
