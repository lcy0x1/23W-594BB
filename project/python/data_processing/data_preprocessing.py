import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

from data_processing.datahandler import AudioHandler


class DataPreprocess:
    def __init__(self, datapath, visualization=False, spkgen=None, mfcc=True, augment=False):
        self.datapath = datapath
        self.audio_len = 48000
        self.shift_ptc = 0.4
        self.visualization = visualization
        self.spkgen = spkgen
        self.mfcc = mfcc
        self.augment = augment
        self.files = self._build_files()
        self.data = self._build_data()

    def _build_files(self):
        print('start building files!')
        files = {}
        index = 0
        for ii in range(1, 61):
            num = "0%d" % ii if ii < 10 else "%d" % ii
            for jj in range(50):
                for kk in range(10):
                    files[index] = [
                        self.datapath + num + "/%d_%s_%d.wav" % (
                            kk, num, jj),
                        kk
                    ]
                    index += 1

        return files

    def _build_data(self):
        print('start building data!')
        data = {}
        for i in tqdm(range(self.__len__())):
            signal = AudioHandler.open(self.files[i][0])[0]
            pad = AudioHandler.pad(signal, self.audio_len)
            # shift = AudioHandler.time_shift(pad, self.shift_ptc)
            shift = pad
            sgram = AudioHandler.spectrogram(
                shift, n_mels=64, n_fft=1024, hop_len=None, mfcc=self.mfcc
            )
            if self.augment:
                sgram = AudioHandler.spectral_augmentation(
                    sgram, max_mask_ptc=0.1, n_freq_masks=2, n_time_masks=2
                )

            sgram = sgram[0]
            padded = torch.sum(sgram, 0) > 1e-3
            sgram = ((sgram.T - torch.mean(sgram, 1)) / torch.std(sgram, 1)).T
            sgram = sgram * padded - 3 * (1 - 1 * padded)

            spkgen = self.spkgen
            spk = spkgen.transform(0, sgram.unsqueeze(0)).permute(1, 2, 0)[0]
            data[i] = (signal, pad, sgram, spk)

        return data

    def __len__(self):
        return 30000

    def __getitem__(self, idx):
        signal, pad, sgram, spk = self.data[idx]
        if self.visualization:
            self.plot(signal, pad, sgram, spk)

        return spk, self.files[idx][1]

    def plot(self, signal, pad, sgram, spk):
        fig, axs = plt.subplots(4, figsize=[20, 25])
        axs[0].plot(np.arange(len(signal)), np.array(signal))
        axs[0].set_title('Original Signal')
        axs[0].set_xlabel('time')

        axs[1].plot(np.arange(len(pad[0])), np.array(pad[0]))
        axs[1].set_title('Padded Signal')
        axs[1].set_xlabel('time')

        img = librosa.display.specshow(np.array(sgram), sr=48000, ax=axs[2])
        title3 = 'MFCC' if self.mfcc else 'Mel Spectrogram'
        axs[2].set_title(title3)
        axs[2].set_xlabel('timestep')
        fig.colorbar(img, ax=axs[2], format="%+2.f")

        img = librosa.display.specshow(np.array(spk), sr=48000, ax=axs[3])
        title3 = 'spikes'
        axs[3].set_title(title3)
        axs[3].set_xlabel('timestep')
        fig.colorbar(img, ax=axs[3], format="%+2.f")

    def preprocess(self):
        print('start storing spike files!')
        for i in tqdm(range(self.__len__())):
            file_name = self.files[i][0][len(self.datapath):-4]
            _, _, _, spk = self.data[i]
            output = np.array(spk, dtype=np.uint8)
            name_mfcc = 'MFCC' if self.mfcc else 'Mel_Scale'

            filepath = self.datapath[:-6] + 'spike/' + name_mfcc + '/' + file_name + '.bin'

            # check the directory does not exist
            if not (os.path.exists(filepath[:-11])):
                # create the directory you want to save to
                os.makedirs(filepath[:-11])

            output.tofile(filepath)

        return output.shape
