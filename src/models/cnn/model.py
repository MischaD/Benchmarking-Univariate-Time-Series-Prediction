import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """
    Single convolutional block consisting of convolutions, weight normalizations, activation functions and dropout
    """
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 dilation,
                 dropout=0.2):
        """
        :param n_inputs: number of inputs
        :param n_outputs: number of outputs
        :param kernel_size: kernel size
        :param dilation:  size of dilation
        :param dropout: dropout probability
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs,
                      n_outputs,
                      kernel_size,
                      padding='same',
                      padding_mode='zeros',
                      dilation=dilation))
        self.silu1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs,
                      n_outputs,
                      kernel_size,
                      padding='same',
                      padding_mode='zeros',
                      dilation=dilation))
        self.silu2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.silu1, self.dropout1,
                                 self.conv2, self.silu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.silu = nn.SiLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.silu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_channels,
                 dilations,
                 kernel_size=2,
                 dropout=0.2):
        """
        :param num_inputs: input dimensionality
        :param num_channels: model dimensionality of each block, should be list of length model_depth
        :param dilations: dilation parameters for the convolutional layers
        :param kernel_size: kernel sizes
        :param dropout: dropout layer
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)  # how many blocks with all dilations
        for i in range(num_levels):
            out_channels = num_channels[i]
            for j, dilation_size in enumerate(dilations):
                if j == 0:
                    in_channels = num_inputs if i == 0 else num_channels[i - 1]
                else:
                    in_channels = num_channels[i]

                layers += [
                    TemporalBlock(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  dilation=dilation_size,
                                  dropout=dropout)
                ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """
    Temporal Convolutional Neural Network. Slightly modified version of https://github.com/locuslab/TCN
    Added dropout layer prior to final linear prediction layer.
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size,
                 dilations, dropout):
        """
        :param input_size: input time series dimension
        :param output_size: output time series dimension
        :param num_channels: tcn model dimensionality of each TCN block (determines depth)
        :param kernel_size: odd kernel size of tcn
        :param dilations: dilation sizes
        :param dropout: dropout
        """
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size,
                                   num_channels,
                                   dilations=dilations,
                                   kernel_size=kernel_size,
                                   dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x.transpose(1, 2))
        dropout_out = self.dropout(y1.transpose(1, 2))
        lin_out = self.linear(dropout_out)
        return torch.squeeze(lin_out, dim=1)  # squeeze middle dimension
