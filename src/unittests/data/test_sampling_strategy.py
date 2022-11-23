import numpy as np
from src.data.sampling_strategy import SamplingStrategy, DefaultSamplingStrategy
from src.data.sampler import AmplitudeSampler, MeanSampler, NoiseSampler


class TestSamplingStrategy:
    def test_init(self):
        sam_strat = SamplingStrategy(AmplitudeSampler(), MeanSampler(),
                                     NoiseSampler(1, 0, output_noise=False))
        assert hasattr(sam_strat, "mean_sampler")
        assert hasattr(sam_strat, "amplitude_sampler")
        assert hasattr(sam_strat, "noise_sampler")


class TestDefaultSamplingStrategy:
    def test_init(self):
        sam_strat = DefaultSamplingStrategy(10, sigma=0)
        assert hasattr(sam_strat, "mean_sampler")
        assert hasattr(sam_strat, "amplitude_sampler")
        assert hasattr(sam_strat, "noise_sampler")

    def test_default_samplers(self):
        sam_strat = DefaultSamplingStrategy(10, sigma=0)
        assert isinstance(sam_strat.mean_sampler, MeanSampler)
        assert isinstance(sam_strat.amplitude_sampler, AmplitudeSampler)
        assert isinstance(sam_strat.noise_sampler, NoiseSampler)

        # noise should be zero vector
        assert np.array_equal(sam_strat.noise_sampler.sample(5),
                              np.zeros((5, 10)))
        assert sam_strat.mean_sampler.sample(5).shape == (5, 1)
        assert sam_strat.amplitude_sampler.sample(5).shape == (5, 3)
