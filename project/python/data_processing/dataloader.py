import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from data_processing.datahandler import AudioHandler
from data_processing.byte_loader import compress, decompress


class AudioMNIST(Dataset):
    def __init__(self, datapath, visualization=False):
        self.datapath = datapath
        self.visualization = visualization
        self.data = self._rebuild_data()

    def _rebuild_data(self):
        print('rebuilding data!')
        data = {i: [] for i in range(10)}

        for i in tqdm(range(10)):
            sample = torch.load(self.datapath + str(i) + '_dataset.pth')
            data[i].append(decompress(sample))

        return data

    def __len__(self):
        return 30000

    def __getitem__(self, idx):
        if self.visualization:
            fig, ax = plt.subplots()
            img = librosa.display.specshow(self.data[idx], sr=48000, ax=ax)
            ax.set_title('spikes')
            ax.set_xlabel('timestep')
            fig.colorbar(img, ax=ax, format="%+2.f")

        return self.data[idx // 3000][idx % 3000], idx//3000


class DataParam:
    def __init__(self, ratio, batch_size, shuffle):
        """
        @param ratio: the ratio of the length of subset of data to the dataset
        @param batch_size: size of the batch
        @param shuffle: T/F
        """
        self.ratio = ratio
        self.batch_size = batch_size
        self.shuffle = shuffle


class LoaderCreator:
    def __init__(self, datapath, mfcc=True, num_workers=2):
        """
        @param datapath: the path to dataset
        @param num_workers:
        """
        self.num_workers = num_workers
        self.dataset = AudioMNIST(datapath)

    def create_loaders(self, train_param: DataParam, val_param: DataParam, test_param: DataParam):
        train_ds, val_ds, test_ds = random_split(self.dataset, [train_param.ratio, val_param.ratio, test_param.ratio])
        train_dl = DataLoader(train_ds,
                              batch_size=train_param.batch_size,
                              shuffle=train_param.shuffle,
                              num_workers=self.num_workers)
        val_dl = DataLoader(val_ds,
                            batch_size=val_param.batch_size,
                            shuffle=val_param.shuffle,
                            num_workers=self.num_workers)
        test_dl = DataLoader(test_ds,
                             batch_size=test_param.batch_size,
                             shuffle=test_param.shuffle,
                             num_workers=self.num_workers)
        return train_dl, val_dl, test_dl
