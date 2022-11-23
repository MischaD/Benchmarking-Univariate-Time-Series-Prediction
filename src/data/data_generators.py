import numpy as np
from abc import ABC
from abc import abstractmethod
import pickle
from src.data.sampler import *
from src.data.sampling_strategy import SamplingStrategy, DefaultSamplingStrategy
import torch


class AbstractDataGenerator(ABC):
    """Abstract base class of a data generator which is a class that can be used to sample new realizations
    of a stochastic process to be added to a dataset. They are restricted to one single characteristic."""
    @property
    def data(self):
        return self._data

    @abstractmethod
    def sample(self, num_samples):
        pass


class PiecewiseSinusoidal(AbstractDataGenerator):
    """Data Generator to create Piecewise sinusoidal functions

    """
    def __init__(
        self,
        sequence_length: int = 256,
        sample_sequence_length: int = 128,
        delay_length: int = 32,
        frequency: float = 1 / 8,
        sampling_strategy: SamplingStrategy = None,
    ):
        """
            :param sequence_length: length of padded sequence
            :param sample_sequence_length: length of sequence that is supposed to be predicted
            :param delay_length: length of the delay part of the sequence
            :param frequency: frequency of sinusoidal base function
            :param sampling_strategy: Sampling strategy from src.data.sampling_strategy
        """
        assert (
            sequence_length - sample_sequence_length
        ) % 2 == 0, "generated singals cannot be padded equally on both sides."
        self.sequence_length = sequence_length

        self._split_points = None
        self._mean_sampler = None
        self._amplitude_sampler = None
        self._noise_sampler = None
        self.sampling_strategy = None
        self.sample_sequence_length = sample_sequence_length
        self.set_sampling_strategy(sampling_strategy)

        self.f = frequency
        self.delay_length = delay_length
        self.sigma = self.sampling_strategy.noise_sampler.sigma
        self.one_side_pad_length = (self.sequence_length -
                                    self.sample_sequence_length) // 2

        # saving means and amplitudes of each sample
        self.means = None
        self.amplitudes = None

        # raw data without noise and without padding
        self.data_without_noise = None

        # data with noise and padding
        self._data = None

        # Vector that can be uses to see how the data was created
        self.data_creation_descriptor = None

    def __str__(self):
        return f"Piecewise Sinusoidal - sequence_length: {self.sequence_length}, non padded samples: {self.sample_sequence_length}, delay_length:{self.delay_length}, frequency: {self.f}, sigma: {self.sigma}"

    @property
    def _t(self):
        if self._split_points is None:
            self._split_points = self._calculate_split_points()
        return self._split_points

    def _calculate_split_points(self):
        """ Calculate the split points but for the part of the signal that is not padded. If the padded part of the
        sequence is of interest then this should be changed to

        :return: split points
        """
        assert self.sample_sequence_length % 4 == 0, "Sequence length has to be divisible by 4"
        t_3 = int(self.sample_sequence_length * 3 / 4)
        t_2 = self.sample_sequence_length // 2
        t_1 = t_2 - self.delay_length
        assert t_1 % 2 == 0, "The information part of the sequence is not divisible by 2. Please change the delay length"
        t_0 = t_1 // 2
        return [t_0, t_1, t_2, t_3]

    def _build_samples(self):
        x = np.arange(self.sample_sequence_length)
        a_0_part = np.s_[:self._t[0]]
        a_1_part = np.s_[self._t[0]:self._t[1]]
        a_2_part = np.s_[self._t[1]:self._t[2]]
        a_3_part = np.s_[self._t[2]:self._t[3]]
        a_3_part_out = np.s_[0:self.sample_sequence_length // 4]
        a_4_part = np.s_[self._t[3]:]
        a_4_part_out = np.s_[self.sample_sequence_length //
                             4:self.sample_sequence_length // 2]
        self.data_without_noise[:,
                                a_0_part] = self.amplitudes[:, 0:1] * np.sin(
                                    2 * np.pi * x[a_0_part] *
                                    self.f) + self.means
        self.data_without_noise[:,
                                a_1_part] = self.amplitudes[:, 1:2] * np.sin(
                                    2 * np.pi * x[a_1_part] *
                                    self.f) + self.means
        self.data_without_noise[:,
                                a_2_part] = self.amplitudes[:, 2:3] * np.sin(
                                    2 * np.pi * x[a_2_part] * self.f
                                ) + self.means  # informationless part
        self.data_without_noise[:,
                                a_3_part] = self.amplitudes[:, 3:4] * np.sin(
                                    2 * np.pi * x[a_3_part_out] * self.f /
                                    2) + self.means
        self.data_without_noise[:,
                                a_4_part] = self.amplitudes[:, 4:5] * np.sin(
                                    2 * np.pi * x[a_4_part_out] * self.f /
                                    2) + self.means

    def _noise(self):
        num_samples = self.data_without_noise.shape[0]
        return self._noise_sampler.sample(num=num_samples)

    def set_sampling_strategy(self, sampling_strategy):
        if sampling_strategy is None:
            sampling_strategy = DefaultSamplingStrategy(
                sigma=0, sequence_length=self.sample_sequence_length)
        self._mean_sampler = sampling_strategy.mean_sampler if sampling_strategy.mean_sampler is not None else self._mean_sampler
        self._amplitude_sampler = sampling_strategy.amplitude_sampler if sampling_strategy.amplitude_sampler is not None else self._amplitude_sampler
        self._noise_sampler = sampling_strategy.noise_sampler if sampling_strategy.noise_sampler is not None else self._noise_sampler
        self.sampling_strategy = sampling_strategy

    def sample(self, num_samples):
        # deterministic part
        self.means = self._mean_sampler.sample(num=num_samples)
        self.amplitudes = np.zeros(shape=(num_samples, 5))
        self.amplitudes[:, :3] = self._amplitude_sampler.sample(
            num=num_samples)
        self.amplitudes[:, 3] = np.max(self.amplitudes[:, :2], axis=1)
        self.amplitudes[:, 4] = np.min(self.amplitudes[:, :2], axis=1)

        # data without padding and without noise
        self.data_without_noise = np.zeros(shape=(num_samples,
                                                  self.sample_sequence_length))

        # data with padding and with noise
        self._data = np.zeros(shape=(num_samples, self.sequence_length))

        self._build_samples()  # sets self.data_without_noise
        noise = self._noise()

        # self._data is padded sequence where the actual sequence is in the middle
        if self.one_side_pad_length != 0:
            self._data[:, self.one_side_pad_length:-self.
                       one_side_pad_length] = self.data_without_noise + noise
        else:
            self._data = self.data_without_noise + noise

        # save descriptor of how each sample was created. Save as list because I am later going to shuffle sequences
        self.data_creation_descriptor = np.repeat([
            [
                self.f, self.delay_length, self.sample_sequence_length,
                self.sigma
            ],
        ],
                                                  num_samples,
                                                  axis=0)
        return self.data

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file=file)


class PiecewiseSinusoidalFixedLengthPartTwo(PiecewiseSinusoidal):
    """Piecewise sinusoidal with fixed length of the A_2 part. (unused, here for reference)
    """
    def __init__(self, part_two_length=16, *args, **kwargs):
        """
        :param part_two_length: length of part two  (A_2 amplitude)
        :param args: PiecewiseSinusoidal args
        :param kwargs: PiecewiseSinusoidal kwargs
        """
        super().__init__(*args, **kwargs)
        self.part_two_length = part_two_length

    def _calculate_split_points(self):
        # on longer has to be divisible by 4
        assert self.sample_sequence_length % 2 == 0, "Sequence length has to be divisible by 2"
        t_3 = int(self.sample_sequence_length * 3 / 4)
        t_2 = self.sample_sequence_length // 2
        t_1 = t_2 - self.delay_length
        t_0 = t_1 - self.part_two_length
        return [t_0, t_1, t_2, t_3]


class PiecewiseSinusoidalFixedRelativeLengthPartTwo(PiecewiseSinusoidal):
    """
    Piecewise sinusoidal with fixed relative length of the A_2 part.
    The second part of the synthetic signal has length that is independant of the dalay length.
    Fix the length by providing a value.

    :param part_two_length: (int, float), either the absolute length of the A_2 part or the length relative to the signal length.
    """
    def __init__(self, part_two_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if part_two_length > 1:
            # length argument interpreted as constant value
            self.part_two_length = part_two_length
        elif part_two_length > 0:
            # part two length interpreted as relative to the sequence length
            self.part_two_length = int(part_two_length *
                                       self.sample_sequence_length)

    def _calculate_split_points(self):
        # on longer has to be divisible by 4
        assert self.sample_sequence_length % 2 == 0, "Sequence length has to be divisible by 2"
        t_3 = int(self.sample_sequence_length * 3 / 4)
        t_2 = self.sample_sequence_length // 2
        t_1 = t_2 - self.delay_length
        t_0 = t_1 - self.part_two_length
        return [t_0, t_1, t_2, t_3]
