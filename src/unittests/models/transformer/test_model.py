import torch
from src.models.transformer import make_encoder_only_time_series_model


class TestTransformer:
    def test_make_encoder_only_time_series_model(self):
        transformer = make_encoder_only_time_series_model(d_input=1,
                                                          d_output=1,
                                                          N=10,
                                                          d_model=64,
                                                          d_ff=2048,
                                                          h=8,
                                                          dropout=0.2)

        inp = torch.zeros(100, 10, 1)
        mask = torch.ones((1, 10, 10))
        outp = transformer(inp, mask)
        assert outp.shape == (100, 10, 1)
