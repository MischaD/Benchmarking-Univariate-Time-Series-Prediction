from src.models.cnn.model import TCN, TemporalBlock, TemporalConvNet
import torch


class TestCNN:
    def test_temporal_block(self):
        tb = TemporalBlock(n_inputs=1,
                           n_outputs=1,
                           kernel_size=3,
                           dilation=2,
                           dropout=0.2)

        inp = torch.zeros(5, 1, 10)
        outp = tb(inp)
        assert outp.shape == (5, 1, 10)

    def test_temporal_conv_net(self):
        d_model = 12
        N = 3
        num_channels = [
            d_model,
        ] * N
        dilations = [1, 2, 4, 8]
        tcn = TemporalConvNet(num_inputs=1,
                              num_channels=num_channels,
                              kernel_size=3,
                              dilations=dilations,
                              dropout=0.2)
        assert len(tcn.network) == len(num_channels) * len(dilations)
        for i in range(len(num_channels)):
            # dilation increases with network depth
            idx = i * len(dilations)
            assert tcn.network[idx].net[0].dilation[0] == dilations[0]
            assert tcn.network[idx + 1].net[0].dilation[0] == dilations[1]
            assert tcn.network[idx + 2].net[0].dilation[0] == dilations[2]
            assert tcn.network[idx + 3].net[0].dilation[0] == dilations[3]

        inp = torch.zeros(5, 1, 10)
        outp = tcn(inp)
        assert outp.shape == (5, num_channels[-1], 10)

    def test_tcn(self):
        tcn = TCN(input_size=1,
                  output_size=1,
                  num_channels=[4, 4, 4],
                  kernel_size=3,
                  dilations=[1, 2, 4],
                  dropout=0.2)

        inp_shape = (5, 10, 1)
        inp = torch.zeros(inp_shape)
        outp = tcn(inp)
        assert outp.shape == inp.shape

    def test_tcn_num_channels(self):
        num_channels = [4, 8, 4, 4]
        dilations = [1, 2, 4]
        tcn = TCN(input_size=1,
                  output_size=1,
                  num_channels=num_channels,
                  kernel_size=3,
                  dilations=dilations,
                  dropout=0.2)

        for i, layer in enumerate(tcn.tcn.network):
            assert layer.net[0].dilation[0] == dilations[i % len(dilations)]
            assert layer.net[0].out_channels == num_channels[i //
                                                             len(dilations)]
