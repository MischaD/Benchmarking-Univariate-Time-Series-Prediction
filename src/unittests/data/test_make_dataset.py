import numpy as np
import pytest
from copy import deepcopy
from src.data.make_dataset import make_mixed_dataset
from src.data.sampling_strategy import DefaultSamplingStrategyPositiveAmplitudes, DefaultSamplingStrategy
from src.unittests.data.utils import CharacteristicEstimator


class TestMakeMixedDataset:
    sample_size = 320
    sample_sequence_lengths = [64, 128]
    frequencies = [1 / 8, 1 / 32]
    delays = [32, 0]
    sigmas = [0, 5]

    @pytest.fixture
    def dataset(self):
        dataset = make_mixed_dataset(
            sample_size=self.sample_size,
            sampling_strategy_class=DefaultSamplingStrategy,
            frequencies=self.frequencies,
            delays=self.delays,
            sigmas=self.sigmas,
            sample_sequence_lengths=self.sample_sequence_lengths,
        )

        return dataset

    def test_make_dataset(self, dataset):
        assert dataset._data.shape == (self.sample_size,
                                       max(self.sample_sequence_lengths), 1)

    def test_make_dataset_characteristics(self, dataset):
        # verify that each characteristic is sample_size times in the dataset
        for sl in self.sample_sequence_lengths:
            for sigma in self.sigmas:
                for f in self.frequencies:
                    for d in self.delays:
                        assert sum([
                            np.array_equal(
                                dataset.get_characteristics(i)[0][0],
                                [f, d, sl, sigma]) for i in range(len(dataset))
                        ]) == self.sample_size // (len(self.frequencies) * len(
                            self.delays) * len(self.sigmas) * len(
                                self.sample_sequence_lengths))

    def test_get_multiple_characteristics(self, dataset):
        dataset.get_characteristics(np.array([0, 1]))

    def test_shuffle(self, dataset):
        dataset_copy = deepcopy(dataset)
        original_order = deepcopy(dataset_copy.characteristics_index_mapper)
        assert not dataset_copy.is_shuffled
        assert np.array_equal(original_order,
                              dataset_copy.characteristics_index_mapper)

        batch_size = 8
        dataset_copy.shuffle(batch_size)
        assert dataset_copy.is_shuffled
        assert not np.array_equal(original_order,
                                  dataset_copy.characteristics_index_mapper)

        # verify that each characteristic is still present in the dataset with equal count
        for sl in self.sample_sequence_lengths:
            for sigma in self.sigmas:
                for f in self.frequencies:
                    for d in self.delays:
                        assert sum([
                            np.array_equal(
                                dataset_copy.get_characteristics(i)[0][0],
                                [f, d, sl, sigma])
                            for i in range(len(dataset_copy))
                        ]) == self.sample_size // (len(self.frequencies) * len(
                            self.delays) * len(self.sigmas) * len(
                                self.sample_sequence_lengths))

        # Test that batch comes out correctly i.e. the sequence length within one batch is always the same
        for batch_num in range(len(dataset) // batch_size):
            sl_batch = dataset_copy.get_sample_sequence_length(batch_num *
                                                               batch_size)[0]
            for i in np.arange(batch_num * batch_size,
                               (batch_num + 1) * batch_size):
                x, y, chars = dataset_copy[i]
                sl_ = chars[0][0][2]
                assert sl_batch == sl_

    def test_shuffle_with_augmentation(self, dataset):
        """If the batchsize does not completely fit """
        # Test augmentation
        dataset_copy = deepcopy(dataset)
        batch_size = 33  # too large, one sample has to be used twice
        dataset_copy.shuffle(batch_size)

        # Test that batch comes out correctly i.e. the sequence length within one batch is always the same
        for batch_num in range(len(dataset_copy) // batch_size):
            sl_batch = dataset_copy.get_sample_sequence_length(batch_num *
                                                               batch_size)[0]
            for i in np.arange(batch_num * batch_size,
                               (batch_num + 1) * batch_size):
                x, y, chars = dataset_copy[i]
                sl_ = chars[0][0][2]
                assert sl_batch == sl_

        # check that dataset_copy is now augmented
        assert len(dataset) < len(dataset_copy)


class TestCharacteristics:
    """Verifies the correct implementation of shuffle and get_characteristics by using a very basic estimator that
    can turn the raw sequence into estimates for the characteristics.
    """
    sample_size = 3200
    sample_sequence_lengths = [128, 256]
    frequencies = [1 / 8, 1 / 16]
    delays = [32, 0]
    sigmas = [0]

    @pytest.fixture
    def dataset(self):
        dataset = make_mixed_dataset(
            sample_size=self.sample_size,
            sampling_strategy_class=DefaultSamplingStrategyPositiveAmplitudes,
            frequencies=self.frequencies,
            delays=self.delays,
            sigmas=self.sigmas,
            sample_sequence_lengths=self.sample_sequence_lengths,
        )
        assert 1 / min(self.frequencies) < min(
            self.sample_sequence_lengths
        ) // 4, "Characteristic estimator doesn't work with this setting"
        assert max(
            self.sigmas
        ) == 0, "Characteristic estimator doesn't work with this setting"

        return dataset

    def test_get_characteristics_vs_individual_characteristics(self, dataset):
        """Verify that all ways of accessing the characteristics return the same values after shuffling"""
        dataset_copy = deepcopy(dataset)
        dataset_copy.shuffle(8)

        for i in range(len(dataset)):
            descriptor, means, amplitudes = dataset.get_characteristics(i)
            descriptor = descriptor[0]
            assert means == dataset.get_means(i)
            assert (amplitudes == dataset.get_amplitudes(i)).all()
            assert (descriptor[0] == dataset.get_frequency(i)).all()
            assert (descriptor[1] == dataset.get_delay(i)).all()
            assert (
                descriptor[2] == dataset.get_sample_sequence_length(i)).all()
            assert (descriptor[3] == dataset.get_sigma(i)).all()

        for i, (x, y, chars) in enumerate(dataset):
            descriptor, means, amplitudes = chars
            descriptor = descriptor[0]
            assert means == dataset.get_means(i)
            assert (amplitudes == dataset.get_amplitudes(i)).all()
            assert (descriptor[0] == dataset.get_frequency(i)).all()
            assert (descriptor[1] == dataset.get_delay(i)).all()
            assert (
                descriptor[2] == dataset.get_sample_sequence_length(i)).all()
            assert (descriptor[3] == dataset.get_sigma(i)).all()

    def test_get_characteristics(self, dataset):
        """Tests whether the get_characteristics function works correctly by estimating the characteristics
        from the sequence and comparing this value to the actual value.

        Note that the estimator is not perfect for some characteristics but by keeping the sample size high enough
        this problem can be mitigated.
        Sigma cannot reliably be estimated (at least not as simply) because most estimates rely on the fact that the
        data has no noise in it.
        """
        for i in range(len(dataset)):
            x, y, chars = dataset[i]
            estimator = CharacteristicEstimator(np.concatenate((x, y))[:, 0])
            estimator.est_characteristics()

            descriptor, means, amplitudes = dataset.get_characteristics(i)

            assert pytest.approx(estimator.mean_est) == means
            assert estimator.seq_len_est == dataset.get_sample_sequence_length(
                i)
            assert estimator.f_est == dataset.get_frequency(i)

            # amplitudes will not be correctly estimated if two of them have the same value
            for amp_est in list(estimator.amplitudes_est):
                amplitude_correctly_detected = False
                for amp in amplitudes[0]:
                    #estimate may not be completely exact
                    if abs(amp - amp_est) <= 1e-4:
                        amplitude_correctly_detected = True
                assert amplitude_correctly_detected

            # delay length estimator only works reliable if three different amplitudes were used for data synthesis
            if not (amplitudes[0, 0] == amplitudes[0, 1]
                    or amplitudes[0, 1] == amplitudes[0, 2]
                    or amplitudes[0, 0] == amplitudes[0, 2]):
                assert bool(estimator.delay_len_est) == bool(
                    dataset.get_delay(
                        i))  # delay is not estimate only if one exists or not

    def test_get_characteristics_shuffle(self, dataset):
        dataset.shuffle(8)
        for i in range(len(dataset)):
            x, y, chars = dataset[i]
            estimator = CharacteristicEstimator(np.concatenate((x, y))[:, 0])
            estimator.est_characteristics()

            descriptor, means, amplitudes = dataset.get_characteristics(i)

            assert pytest.approx(estimator.mean_est) == means
            assert estimator.seq_len_est == dataset.get_sample_sequence_length(
                i)
            assert estimator.f_est == dataset.get_frequency(i)

            # amplitudes will not be correctly estimated if two of them have the same value
            for amp_est in list(estimator.amplitudes_est):
                amplitude_correctly_detected = False
                for amp in amplitudes[0]:
                    #estimate may not be completely exact
                    if abs(amp - amp_est) <= 1e-4:
                        amplitude_correctly_detected = True
                assert amplitude_correctly_detected

            # delay length estimator only works reliable if three different amplitudes were used for data synthesis
            if not (amplitudes[0, 0] == amplitudes[0, 1]
                    or amplitudes[0, 1] == amplitudes[0, 2]
                    or amplitudes[0, 0] == amplitudes[0, 2]):
                assert bool(estimator.delay_len_est) == bool(
                    dataset.get_delay(
                        i))  # delay is not estimate only if one exists or not
