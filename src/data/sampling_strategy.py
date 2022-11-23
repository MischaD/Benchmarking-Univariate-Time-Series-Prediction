from src.data.sampler import *
from dataclasses import dataclass
from abc import abstractmethod, ABC


@dataclass
class SamplingStrategy():
    """Abstract sampling strategy.

    A sampling strategy combines a mean sampler, an amplitude sampler and a noise sampler into
    a sampling strategy. """
    mean_sampler: MeanSampler
    amplitude_sampler: AmplitudeSampler
    noise_sampler: NoiseSampler

    @abstractmethod
    def get_descriptor(self) -> str:
        """Short identifier string that will be used for the filename to help identify sampling strategy from
         the filename. """
        pass


class DefaultSamplingStrategyPositiveAmplitudes(SamplingStrategy):
    def __init__(self, sequence_length, sigma):
        super().__init__(
            mean_sampler=MeanSamplerConstant(mean_value=72),
            amplitude_sampler=AmplitudeSamplerUniformInteger(
                ranges=np.array([[5, 61], [5, 61], [5, 61]])),
            noise_sampler=NoiseSamplerNormal(sigma, sequence_length),
        )

    def __str__(self):
        return f"Constant mean sampler, Amplitudes are sampled uniformly only positive, noise is i.i.d"

    def get_descriptor(self) -> str:
        return "default"


class DefaultSamplingStrategy(SamplingStrategy):
    """Sampling strategy with constant mean, uniform amplitdues and gaussion noise. Default sampling strategy
    for multiple initial experiments"""
    def __init__(self, sequence_length, sigma):
        super().__init__(
            mean_sampler=MeanSamplerConstant(mean_value=72),
            amplitude_sampler=AmplitudeSamplerUniformInteger(),
            noise_sampler=NoiseSamplerNormal(sigma, sequence_length),
        )

    def __str__(self):
        return f"Default Sampling strategy - Constant mean sampler, Amplitudes are sampled uniformly, noise is i.i.d"

    def get_descriptor(self) -> str:
        return "default"


class MeanDefaultSamplingStrategyGaussianAmplitudes(SamplingStrategy):
    """Default sampling strategy but with the added complexity of having a constant mean that is sampled

    Amplitudes are normally distributed.
    """
    def __init__(self, sequence_length, sigma):
        super().__init__(
            mean_sampler=MeanSamplerUniform(),
            amplitude_sampler=AmplitudeSamplerNormal(),
            noise_sampler=NoiseSamplerNormal(sigma,
                                             sequence_length,
                                             output_noise=False),
        )

    def __str__(self):
        return f"Default Sampling strategy with mean sampling - Amplitudes are sampled from mirrored gaussian distribution, noise is i.i.d"

    def get_descriptor(self) -> str:
        return "default_mean_normal"


class MeanDefaultSamplingStrategyUniform(SamplingStrategy):
    """Sampling strategy with uniform mean, uniform amplitudes and gaussian noise.
    """
    def __init__(self, sequence_length, sigma):
        super().__init__(
            mean_sampler=MeanSamplerUniform(),
            amplitude_sampler=AmplitudeSamplerUniformInteger(),
            noise_sampler=NoiseSamplerNormal(sigma,
                                             sequence_length,
                                             output_noise=False),
        )

    def __str__(self):
        return f"Default Sampling strategy with mean sampling - Amplitudes are sampled uniformly, noise is i.i.d"

    def get_descriptor(self) -> str:
        return "default_mean_uniform"


class ConstantAmplitudesSamplingStrategy(SamplingStrategy):
    """
    Sampling strategy that always returns the same amplitudes. Used for mainly for visualizations.
    """
    def __init__(self, sequence_length, sigma):
        super().__init__(
            mean_sampler=MeanSamplerConstant(mean_value=72),
            amplitude_sampler=AmplitudeSamplerConstant([60, 10, 40]),
            noise_sampler=NoiseSamplerNormal(sigma, sequence_length),
        )

    def __str__(self):
        return f"Constant amplitude - Constant mean sampler, noise is i.i.d"

    def get_descriptor(self) -> str:
        return "constant amplitudes"


class MissingAmplitudeStrategy(SamplingStrategy):
    """
    Sampling strategy with certain amplitudes missing from the range. deprecated and replaced by sampling from different distributions.
    """
    def __init__(self, sequence_length, sigma):
        super().__init__(
            mean_sampler=MeanSamplerConstant(mean_value=72),
            amplitude_sampler=AmplitudeSamplerUniformIntegerMissingAmplitudes(
            ),
            noise_sampler=NoiseSamplerNormal(sigma, sequence_length),
        )

    def __str__(self):
        return f"Constant mean, Uniform amplitudes but with missing amplitude values in between 40 and 50"

    def get_descriptor(self) -> str:
        return "missing_amplitude_interval"
