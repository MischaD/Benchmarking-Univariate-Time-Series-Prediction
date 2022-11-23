import pickle
import os
from src.data.sampler import *
from src.data.data_generators import AbstractDataGenerator
from log import logger_data
import torch
from typing import List


class CustomMinMaxScaler():
    """
    Custom min max scaler that just shrinks the entire datasets down to the range of 0, 1 instead of to a range
    dependant on the feature/specific time series.
    """
    def __init__(self):
        self._min = None
        self._max = None

    def fit_transform(self, data):
        self._min = data.min()
        self._max = data.max()
        return self.transform(data)

    def transform(self, data):
        return (data - self._min) / (self._max - self._min)

    def inverse_transform(self, data):
        return data * (self._max - self._min) + self._min


class SyntheticDataset(torch.utils.data.Dataset):
    """
    Torch Dataset. Instantiated multiple smaller datasets that have different frequency and delay characteristics
    but the same sampling strategy and sequence length.
    """
    def __init__(self, data_generators: List[AbstractDataGenerator]):
        """
        :param data_generators: list of data generators that have potentially different characteristics (e.g. different delays)
        """
        self.sequence_length = data_generators[0].sequence_length
        self.sampling_strategy = data_generators[0].sampling_strategy
        self._check_generators_validity(data_generators)

        self._data = np.concatenate(
            [dataset.data for dataset in data_generators])

        # arrays that saves information about the characteristic sample-wise
        self._data_creation_descriptor = np.concatenate(
            [dataset.data_creation_descriptor for dataset in data_generators])
        self._amplitudes = np.concatenate(
            [dataset.amplitudes for dataset in data_generators])
        self._means = np.concatenate(
            [dataset.means for dataset in data_generators])

        self._data = np.expand_dims(self._data, axis=2)
        self.scaler = None  # for data normalization. Assigned in self.normalize()
        self.is_normalized = False
        self.is_shuffled = False

        # Use one array to keep track of the mapping of the shuffled index to the original index
        self.characteristics_index_mapper = np.arange(len(self._data))

    def __str__(self):
        return "len_" + str(
            self.sequence_length
        ) + "_strat_" + self.sampling_strategy.get_descriptor(
        ) + "_sig_" + str(self.sampling_strategy.noise_sampler.sigma)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (
            torch.FloatTensor(self._data[idx, :self.sequence_length // 2, :]),
            torch.FloatTensor(self._data[idx, self.sequence_length // 2:, :]),
            self.get_characteristics(idx),
        )

    def _check_generators_validity(self, data_generators):
        """
        Checks whether this dataset was properly initialized. Data Generators have to share their sequence lengths
        (only after padding) and all of them have to be Subcalses of AbstractDataGenerator.
        """
        assert all([
            dataset.sequence_length == self.sequence_length
            for dataset in data_generators
        ]), "Cannot Mix Datasets with different sequence lengths."
        assert all([
            isinstance(dataset, AbstractDataGenerator)
            for dataset in data_generators
        ])

    def normalize(self, scaler=None):
        if self.is_normalized:
            logger_data.info("Dataset already normalized")
            if scaler:
                logger_data.warn(
                    "Tried to normalize data using a Scaler but data was already normalized. Please undo "
                    "normalization before calling normalize with scaler")
            else:
                return
        self.is_normalized = True
        if scaler is None:
            logger_data.info("Initializing and fitting new scaler to dataset")
            self.scaler = CustomMinMaxScaler()
            self._data = self.scaler.fit_transform(self._data)
        else:
            logger_data.info("Using provided scaler to normalize dataset")
            self.scaler = scaler
            self._data = self.scaler.transform(self._data)

    def inverse_normalization(self, data=None):
        if not self.is_normalized:
            logger_data.warning(
                "trying to undo normalization on data that was never normalized"
            )
        else:
            return self.scaler.inverse_transform(data)

    def get_frequency(self, sample_num):
        return self.get_characteristics(sample_num)[0][:, 0]

    def get_delay(self, sample_num):
        return self.get_characteristics(sample_num)[0][:, 1]

    def get_sample_sequence_length(self, sample_num):
        return self.get_characteristics(sample_num)[0][:, 2]

    def get_sigma(self, sample_num):
        return self.get_characteristics(sample_num)[0][:, 3]

    def get_means(self, sample_num):
        return self.get_characteristics(sample_num)[1]

    def get_amplitudes(self, sample_num):
        return self.get_characteristics(sample_num)[2]

    def get_characteristics(self, sample_num):
        """ Get the characteristics of a sample from the data set by inverting the shuffling operation.

        :param sample_num: Sample number in the datasit
        :return: Tuple of Lists. The first lists describes frequency, delay length, sample sequence length and sigma, the second list are the means of the samples and the third list are the amplitudes used for generation
        """
        # data creation descriptor constist of [f, delay_length, sample_sequence_length, sigma]

        # data undo data shuffling to get the correct description
        if isinstance(sample_num, int) or isinstance(sample_num, np.int32):
            sample_num = np.array([
                sample_num,
            ])

        shuffled_index = []
        for sample in sample_num:
            shuffled_index.append(self.characteristics_index_mapper[sample])
        shuffled_index = np.array(shuffled_index)

        return self._data_creation_descriptor[shuffled_index], self._means[
            shuffled_index], self._amplitudes[shuffled_index]

    def shuffle(self, batch_size):
        """ Batchwise shuffling of the dataset.

        :param batch_size: batch_size that defines the amount of samples in a row that have the same sample
        """
        logger_data.info("Shuffling dataset")
        sample_seq_lengths = self._data_creation_descriptor[:, 2]

        # count number of occurrences of each seq_len and make sure they are all the same
        seq_len_counts = np.bincount(sample_seq_lengths.astype(np.int))
        different_seq_lens = np.argwhere(
            seq_len_counts != 0)  # 0 is not a sequence_len
        count_samples_per_len = np.max(seq_len_counts)
        # sequence occurs either 0 times or count_samples_per_len times
        assert np.count_nonzero(
            np.bincount(seq_len_counts)
        ) == 2, "Some sequence lengths occur more often than others"

        # make sure that sequences of each length can be equally distributed across batches
        if count_samples_per_len % batch_size != 0:
            # augmentation necessary to make this possible
            logger_data.info(
                f"Data augementation necessary for shuffling with batch size {batch_size} and {count_samples_per_len} number of samples for each sequence length"
            )
            num_aug = batch_size - count_samples_per_len % batch_size
            for seq_len in different_seq_lens:
                aug_samples = np.random.choice(
                    np.where(sample_seq_lengths == seq_len)[0], size=num_aug)
                self._data = np.concatenate(
                    (self._data, self._data[aug_samples]))
                self.characteristics_index_mapper = np.concatenate(
                    (self.characteristics_index_mapper,
                     self.characteristics_index_mapper[aug_samples]))
                sample_seq_lengths = np.concatenate(
                    (sample_seq_lengths, np.repeat(seq_len, num_aug)))

        # shuffle all sequences. Note that this achieves the same as doing this last
        permutation = np.random.permutation(len(self._data))
        self._data = self._data[permutation]
        self.characteristics_index_mapper = self.characteristics_index_mapper[
            permutation]
        sample_seq_lengths = sample_seq_lengths[permutation]

        # sort by seq_lengths
        sort_permutation = np.argsort(sample_seq_lengths)
        self._data = self._data[sort_permutation]
        self.characteristics_index_mapper = self.characteristics_index_mapper[
            sort_permutation]

        num_batches = len(
            self._data
        ) // batch_size  # no remainder because self._data = n * count_samples_per_len

        # shuffle data using batchwise permutation
        batchwise_permutation = np.random.permutation(num_batches)
        batchwise_permutation = np.repeat(
            batchwise_permutation, batch_size) * batch_size + np.concatenate(
                (np.arange(batch_size), ) * num_batches)
        self._data = self._data[batchwise_permutation]
        self.characteristics_index_mapper = self.characteristics_index_mapper[
            batchwise_permutation]

        self.is_shuffled = True

    def save(self, path_dir, filename=".pkl"):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        path = os.path.join(path_dir, filename)
        logger_data.info(f"Saving dataset to {path}")
        with open(path, 'wb') as file:
            pickle.dump(self, file=file)

    @staticmethod
    def load(file_path):
        logger_data.info(f"Loading dataset from {file_path}")
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        return obj
