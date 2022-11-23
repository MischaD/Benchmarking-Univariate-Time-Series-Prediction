import pytest
from src.data.data_generators import PiecewiseSinusoidalFixedRelativeLengthPartTwo
from src.data.synthetic_dataset import SyntheticDataset
from src.data.sampling_strategy import SamplingStrategy, DefaultSamplingStrategy
import numpy as np


class TestSyntheticDataset:
    def test_ik(self):
        sequence_length, sigma, frequencies, delays, = 128, 0, [1 / 8, 1 / 32
                                                                ], [32, 0]
        sampling_strategy = DefaultSamplingStrategy(
            sequence_length=sequence_length, sigma=sigma)
        samples_per_generator = 10
        data_generators = []
        for f in frequencies:
            for d in delays:
                pws = PiecewiseSinusoidalFixedRelativeLengthPartTwo(
                    part_two_length=0.125,
                    sequence_length=sequence_length,
                    delay_length=d,
                    frequency=f,
                    sampling_strategy=sampling_strategy)
                pws.sample(samples_per_generator)
                data_generators.append(pws)

        dataset = SyntheticDataset(data_generators)

        assert len(
            dataset) == len(frequencies) * len(delays) * samples_per_generator
