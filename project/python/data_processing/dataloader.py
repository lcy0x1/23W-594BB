import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from data_processing.datahandler import AudioHandler


class AudioMNIST(Dataset):
    def __init__(self, datapath, datashape, visualization=False):
        self.datapath = datapath
        self.datashape = datashape
        self.visualization = visualization
        self.files = self._build_files()
        self.data = self._rebuild_data()

    def _build_files(self):
        files = {}
        index = 0
        for ii in range(1, 61):
            num = "0%d" % ii if ii < 10 else "%d" % ii
            for jj in range(50):
                for kk in range(10):
                    files[index] = [
                        self.datapath + num + "/%d_%s_%d.bin" % (
                            kk, num, jj),
                        kk
                    ]
                    index += 1

        return files

    def _rebuild_data(self):
        print('rebuilding data!')
        data = {}
        for i in tqdm(range(self.__len__())):
            data[i] = np.fromfile(self.files[i][0], dtype=np.uint8).reshape(self.datashape)

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

        return torch.from_numpy(self.data[idx]), self.files[idx][1]


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
    def __init__(self, datapath, datashape, mfcc=True, num_workers=2):
        """
        @param datapath: the path to dataset
        @param num_workers:
        """
        self.num_workers = num_workers
        self.dataset = AudioMNIST(datapath, datashape)

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
