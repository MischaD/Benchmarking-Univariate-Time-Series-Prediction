import numpy as np


class Sampler:
    def sample(self, num):
        raise NotImplementedError()


class NoiseSampler:
    """Generic noise sampler. Has attribute sigma to set the impact of the noise and one call of sample
    create *sequence_length* - noise values. Sequence length is supposed to be part of this class to enable
    non i.i.d. noise if necessary.

    :param sigma: standard deviation of the noise
    :param sequence_length: length of the noise sequence to be sampled
    :param output_noise: Boolean indicating if the output should be noisy
    """
    def __init__(self, sigma, sequence_length, output_noise):
        self.sigma = sigma
        self.sequence_length = sequence_length
        self.output_noise = output_noise


class MeanSampler:
    """Generic mean sampler. Returns one value for each sample."""


class AmplitudeSampler:
    """Generic amplitude sampler. Returns three values for each sample corresponding to the first
    three amplitudes of the signal. """


class MeanSamplerConstant(MeanSampler):
    def __init__(self, mean_value=72):
        self.mean_value = mean_value

    def sample(self, num):
        return np.ones(shape=(num, 1)) * self.mean_value


class MeanSamplerUniform(MeanSampler):
    def __init__(self, mean_value=72):
        self.range = [47, 97]
        self.mean_value = mean_value

    def sample(self, num):
        return np.random.randint(low=self.range[0],
                                 high=self.range[1],
                                 size=num).reshape(-1, 1)


class AmplitudeSamplerUniformInteger(AmplitudeSampler):
    def __init__(self, ranges: list = None):
        # exclusive
        if ranges is not None:
            self.ranges = ranges
        else:
            self.ranges = np.array([(-60, 61), (-60, 61), (-60, 61)])

    def sample(self, num):
        amplitudes = [
            np.random.randint(low=l, high=h, size=num) for l, h in self.ranges
        ]
        return np.array(amplitudes).T


class AmplitudeSamplerNormal(AmplitudeSampler):
    def __init__(self, ranges: list = None):
        # exclusive
        if ranges is not None:
            self.ranges = ranges
        else:
            self.ranges = np.array([(-60, 61), (-60, 61), (-60, 61)])

    def sample(self, num):
        amplitudes = np.array((np.random.randn(num * 3) * 10 + 30),
                              dtype=np.int)
        amplitudes[amplitudes > 60] = 60
        signs = np.random.randint(0, 2, size=(num * 3)) * 2 - 1  # -1 or 1
        amplitudes *= signs
        return amplitudes.reshape(-1, 3)


class AmplitudeSamplerUniformIntegerMissingAmplitudes(AmplitudeSampler):
    """Samples amplitudes as uniform integer but leaves out a small interval"""
    def __init__(self,
                 ranges: list = None,
                 missing_amplitudes_range: tuple = (40, 50)):
        # exclusive
        self.missing_amplitudes_range = missing_amplitudes_range
        if ranges is not None:
            self.ranges = ranges
        else:
            self.ranges = np.array([(5, 61), (5, 61), (5, 61)])

    def sample(self, num):
        amplitudes = np.zeros((num, len(self.ranges)))
        for i in range(len(amplitudes)):
            for j, (l, h) in enumerate(self.ranges):
                while True:
                    rdm = np.random.randint(low=l, high=h)
                    if rdm < self.missing_amplitudes_range[
                            0] or rdm >= self.missing_amplitudes_range[1]:
                        break
                amplitudes[i, j] = rdm

        return amplitudes


class AmplitudeSamplerConstant(AmplitudeSampler):
    """Sampling always returns the same amplitudes. Used for debugging/visualization"""
    def __init__(self, amplitudes):
        self.amplitudes = np.array(amplitudes)

    def sample(self, num):
        amplitudes = [self.amplitudes for _ in range(num)]
        return np.array(amplitudes)


class NoiseSamplerNormal(NoiseSampler):
    """Noise is sampled i.i.d. gaussian distributed with standard deviation sigma.
    """
    def __init__(self, sigma, sequence_length, output_noise=False):
        super().__init__(sigma, sequence_length, output_noise)

    def sample(self, num):
        noise = np.zeros((num, self.sequence_length))
        if not self.output_noise:
            noise[:, :self.sequence_length //
                  2] += self.sigma * np.random.randn(num,
                                                     self.sequence_length // 2)
        else:
            noise += self.sigma * np.random.randn(num, self.sequence_length)
        return noise
