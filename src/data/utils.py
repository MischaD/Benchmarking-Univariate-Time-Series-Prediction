from log import logger_data
import torch


def get_sample_seq_len(characteristics):
    """Retrieves sample sequence length from characteristics. Note that shuffling made sure that the """
    if characteristics[0].ndim == 2:
        # if data was retrieved sample-wise
        sample_seq_lens = characteristics[0][:, 2]
    else:
        # if data was retrieved batch-wise
        sample_seq_lens = characteristics[0][:, 0, 2]
    if not torch.all(sample_seq_lens[0] == sample_seq_lens):
        logger_data.warn(
            "Different sample sequence lengths in a single batch detected. This may lead to unpredictable errors."
        )
    return sample_seq_lens[0]
