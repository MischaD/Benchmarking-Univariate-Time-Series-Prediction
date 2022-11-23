import pytest
from src.data.data_generators import PiecewiseSinusoidal
from src.data.sampler import MeanSamplerConstant, AmplitudeSamplerConstant, AmplitudeSamplerUniformInteger, NoiseSamplerNormal
from src.data.sampling_strategy import SamplingStrategy
import numpy as np


class TestPiecewiseSinusoidalDataset:
    def test_piecewise_sinusoidal(self):
        seq_len = 40
        sample_size = 10
        pws = PiecewiseSinusoidal(sequence_length=seq_len,
                                  sample_sequence_length=seq_len)
        pws.sample(sample_size)
        assert len(pws.data) == sample_size
        assert len(pws.data[0]) == seq_len

    def test_constant_sampler(self):
        seq_len = 128
        sigma = 0
        fixed_amplitudes = [30, 20, 10]
        fixed_mean = 10
        f = 1 / 8

        expected_amplitudes = [30, 20, 10, 30,
                               20]  # 4th value max, 5th value min

        # define a sampling strategy that returns constant values
        sampling_strategy = SamplingStrategy(
            mean_sampler=MeanSamplerConstant(mean_value=fixed_mean),
            amplitude_sampler=AmplitudeSamplerConstant(
                amplitudes=fixed_amplitudes),
            noise_sampler=NoiseSamplerNormal(sigma=sigma,
                                             sequence_length=seq_len))
        # monkey patch abstract method
        sampling_strategy.get_descriptor = lambda x: "dummy"
        pws = PiecewiseSinusoidal(sequence_length=seq_len,
                                  sampling_strategy=sampling_strategy,
                                  frequency=f)
        pws.sample(2)
        assert np.array_equal(pws.amplitudes[0], expected_amplitudes)
        assert np.array_equal(
            pws.amplitudes[1],
            expected_amplitudes)  # amplitudes of all samples look the same
        assert np.array_equal(pws.data[0],
                              pws.data[1])  # all samples are the same

        # check that the maximums have the right values and are on the correct places
        assert pws.data[0][2] == fixed_mean + fixed_amplitudes[0]
        assert pws.data[0][2 + 16] == fixed_mean + fixed_amplitudes[1]
        assert pws.data[0][2 + 32] == fixed_mean + fixed_amplitudes[2]

    def test_uniform_sampler(self):
        np.random.seed(1)

        seq_len = 128
        sampling_strategy = SamplingStrategy(
            mean_sampler=MeanSamplerConstant(mean_value=0),
            amplitude_sampler=AmplitudeSamplerUniformInteger(),
            noise_sampler=NoiseSamplerNormal(sigma=0, sequence_length=seq_len),
        )
        # monkey patch abstract method
        sampling_strategy.get_descriptor = lambda x: "dummy"
        pws = PiecewiseSinusoidal(sequence_length=seq_len,
                                  sampling_strategy=sampling_strategy)
        pws.sample(2)
        # verify that this creates two differnt looking sinusoidals
        assert not np.array_equal(pws.data[0], pws.data[1])
