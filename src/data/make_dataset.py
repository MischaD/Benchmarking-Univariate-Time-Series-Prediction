# -*- coding: utf-8 -*-
import click
import os
from log import logger_data
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data.data_generators import PiecewiseSinusoidalFixedRelativeLengthPartTwo
from src.data.synthetic_dataset import SyntheticDataset
from src.data.sampling_strategy import DefaultSamplingStrategy, ConstantAmplitudesSamplingStrategy, MissingAmplitudeStrategy, MeanDefaultSamplingStrategyGaussianAmplitudes, MeanDefaultSamplingStrategyUniform


def make_mixed_dataset(sample_size, sampling_strategy_class, frequencies,
                       delays, sigmas, sample_sequence_lengths):
    """ Helper function to create one dataset with different frequencies/delays

    :param sequence_length: Sequence length, twice the input length
    :param sigma: Noise level
    :param sample_size: Final sample size of the datasets. Samples will be equally distributed across different characteristics
    :param sampling_strategy_class: Class of type sampling strategy used dataset creation
    :param frequencies: frequencies that are supposed to be present in the final dataset
    :param delays: Delays or long vs short term dependencies that are supposed to be present in the final dataset
    :param sigmas: List of parameters for noise in the signal
    :param sample_sequence_lengths: List of sample sequence lengths that will end up in the datatset. Should smaller or equal sequence length. If smaller, the sequence will be padded with zeros.
    :return: Final dataset
    """
    sequence_length = max(sample_sequence_lengths)
    # amount of different dataset genreators within this dataset to evenly distribute across samples
    n_different_datasets = len(frequencies) * len(delays) * len(sigmas) * len(
        sample_sequence_lengths)
    sample_size_per_generator = sample_size // n_different_datasets
    if sample_size % n_different_datasets != 0:
        logger_data.warning(
            f"Sample size {sample_size} cannot be equally distributed into {n_different_datasets} different datasets"
        )

    data_generators = []
    for sample_sequence_length in sample_sequence_lengths:
        for sigma in sigmas:
            sampling_strategy = sampling_strategy_class(
                sequence_length=sample_sequence_length, sigma=sigma)
            for f in frequencies:
                for d in delays:
                    pws = PiecewiseSinusoidalFixedRelativeLengthPartTwo(
                        part_two_length=0.125,
                        sequence_length=sequence_length,
                        sample_sequence_length=sample_sequence_length,
                        delay_length=d,
                        frequency=f,
                        sampling_strategy=sampling_strategy)
                    pws.sample(sample_size_per_generator)
                    data_generators.append(pws)

    dataset = SyntheticDataset(data_generators)
    return dataset


@click.command()
@click.option(
    '--sample_sequence_lengths',
    type=click.STRING,
    default="128,256",
    help=
    'String with integers seperated by commas that indicate the different number of sampling points in each sequence'
)
@click.option(
    '--sigmas',
    type=click.STRING,
    default=0,
    help=
    'Parameter that influences the impact of the noise part of the signal. String of Integers seperated by a comma.'
)
@click.option(
    '--sampling_strategy',
    type=click.STRING,
    default='default',
    help=
    'Type of strategy to use to sample amplitude, noise and mean. Current strategies available: "default"'
)
@click.option(
    '--period_lengths',
    type=click.STRING,
    default="8, 32",
    help='Period lengths that will appear in the dataset. Default: "8, 32"')
@click.option(
    '--delay_lengths',
    type=click.STRING,
    default="0, 32",
    help='Delay lengths to switch between long vs. short term dependencies.')
@click.option('--sample_size',
              type=click.INT,
              default=10000,
              help='Amount of samples to be created')
@click.argument('output_filepath', type=click.Path())
def main(sample_sequence_lengths, sigmas, sampling_strategy, period_lengths,
         delay_lengths, sample_size, output_filepath):
    """ Data processing script that is used to create a synthetic dataset and save it to OUTPUT_FILEPATH.

    The created datasets has fixed frequencies and delays and the length of the second part is fixed to be 1/8 of the
    entire sequence independent of how long the sequence is. The length of the sequences in the dataset is determined
    by SAMPLE_SEQUENCE_LENGTHS and is equal to the largest provided sequence lengths (i.e. '64,128,256' would mean
    every sample in the dataset is 256 sampling points long, but many of them would be zero-padded symmetrically.
    """
    if sampling_strategy is None or sampling_strategy == "default":
        logger_data.info("using default sampling strategy")
        sampling_strategy_class = DefaultSamplingStrategy
    elif sampling_strategy == "constant":
        logger_data.info("Using constant sampling strategy")
        sampling_strategy_class = ConstantAmplitudesSamplingStrategy
    elif sampling_strategy == "missing_amplitudes":
        logger_data.info("Using missing amplitudes sampling strategy")
        sampling_strategy_class = MissingAmplitudeStrategy
    elif sampling_strategy == "default_mean_normal":
        logger_data.info(
            "Using default + mean sampling strategy with from mirrored normal distribution"
        )
        sampling_strategy_class = MeanDefaultSamplingStrategyGaussianAmplitudes
    elif sampling_strategy == "default_mean_uniform" or sampling_strategy == "default_mean":
        logger_data.info(
            "Using default + mean sampling strategy with uniform sampling")
        sampling_strategy_class = MeanDefaultSamplingStrategyUniform
    else:
        raise click.BadParameter(
            "Unkown sampling strategy for option --sampling_strategy")

    frequencies = [1 / int(T)
                   for T in period_lengths.split(",")]  # high vs. low dynamics
    delays = [int(d) for d in delay_lengths.split(",")
              ]  # long vs short term dependencies
    dataset = make_mixed_dataset(
        sample_size=sample_size,
        sampling_strategy_class=sampling_strategy_class,
        frequencies=frequencies,
        delays=delays,
        sigmas=[int(x) for x in sigmas.split(",")],
        sample_sequence_lengths=[
            int(x) for x in sample_sequence_lengths.split(",")
        ])

    logger_data.info(f'Synthesizing dataset: {dataset}')
    dataset.save(path_dir=os.path.dirname(output_filepath),
                 filename=os.path.basename(output_filepath))


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
