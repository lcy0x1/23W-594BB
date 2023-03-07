import torchaudio
import torch

import random
from torchaudio import transforms


class AudioHandler:
    @staticmethod
    def open(file_path):
        # throw away sample rate since it is a constant
        signal, _ = torchaudio.load(file_path)
        return signal

    @staticmethod
    def pad(signal, audio_length):
        new_signal = torch.zeros((1, audio_length))
        new_signal[0, 48000 - len(signal):] = signal

        return new_signal

    @staticmethod
    def time_shift(signal, shift_limit):
        """ Data augmentation: shift forward or back by a bit. """
        signal_length = signal.shape[1]
        shift = int(random.random() * shift_limit * signal_length)
        return signal.roll(shift)

    @staticmethod
    def spectrogram(signal, n_mels=64, n_fft=2034, hop_len=None, mfcc=False, n_mfcc=32):
        top_db = 80

        if mfcc:
            melkwargs = {"n_fft": n_fft,
                         "hop_length": hop_len,
                         "n_mels": n_mels,
                         "center": False}
            spec = transforms.MFCC(sample_rate=48000,
                                   n_mfcc=n_mfcc,
                                   melkwargs=melkwargs)(signal)

        else:
            spec = transforms.MelSpectrogram(sample_rate=48000, n_fft=n_fft,
                                             hop_length=hop_len,
                                             n_mels=n_mels)(signal)
            spec = transforms.AmplitudeToDB(top_db=top_db)(torch.Tensor(spec))

        return spec

    @staticmethod
    def spectral_augmentation(spec, max_mask_ptc=0.1,
                              n_freq_masks=1,
                              n_time_masks=1):
        """ Augment the spectrogram data. """
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_ptc * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(
                freq_mask_param
            )(aug_spec, mask_value)

        time_mask_param = max_mask_ptc * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(
                time_mask_param
            )(aug_spec, mask_value)

        return aug_spec
