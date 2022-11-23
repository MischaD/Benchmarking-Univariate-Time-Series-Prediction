import numpy as np


class CharacteristicEstimator:
    """Estimator of sequence characteristics to verify correct shuffling.

    Fails if:
    Only works if we don't add noise.
    Frequency estimate fails if frequency is so low that one partial sinusoidal does not complete its entire period.
    Only positive amplitudes work.

    """
    def __init__(self, sequence):
        self.sequence = sequence

        self.seq_len_est = None
        self.mean_est = None
        self.f_est = None
        self.amplitudes_est = None
        self.delay_len_est = None

    def est_characteristics(self):
        self.est_sample_seq_len()
        self.est_mean()
        self.est_frequency()
        self.est_amplitudes()
        self.est_delay_len()

    def est_sample_seq_len(self):
        """Estimates sample seq len of sequence by counting when the first nonzero value occurs and shortens the sequence"""
        self.seq_len_est = len(
            self.sequence) - np.argwhere(self.sequence > 0).min() * 2
        one_sided_pad_length = (len(self.sequence) - self.seq_len_est) // 2
        if one_sided_pad_length > 0:
            self.sequence = self.sequence[
                one_sided_pad_length:-one_sided_pad_length]

    def est_mean(self):
        """Estimate and substract mean"""
        self.mean_est = np.mean(self.sequence)
        self.sequence -= self.mean_est

    def est_frequency(self):
        """Estimate frequency may fail """
        input_end = len(self.sequence) // 2
        argmin = self.sequence[:input_end].argmin()
        argmax = self.sequence[:input_end].argmax()
        self.f_est = 1 / (2 * np.abs((argmax - argmin)))

    def est_amplitudes(self):
        """Estimate amplitudes by saying every local maxima is amplitude"""
        amplitudes = set()
        for i in range(1, len(self.sequence) - 1):
            if self.sequence[i] > self.sequence[
                    i - 1] and self.sequence[i] > self.sequence[i + 1]:
                # local maxima
                amplitudes.add(self.sequence[i])

        self.amplitudes_est = amplitudes

    def est_delay_len(self):
        """Estimates delay in a lazy fashion by either predicting """
        if len(self.amplitudes_est) >= 3:
            self.delay_len_est = 1
        else:
            self.delay_len_est = 0
