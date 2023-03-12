import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
from data_processing.datahandler import AudioHandler
from data_processing.byte_loader import compress, decompress


class AudioMNIST(Dataset):
    def __init__(self, datapath, visualization=False):
        self.datapath = datapath
        self.visualization = visualization
        self.data = self._rebuild_data()

    def _rebuild_data(self):
        print('rebuilding data!')
        data = []

        for i in tqdm(range(10)):
            sample = torch.load(self.datapath + str(i) + '_dataset.pth')
            data.append(decompress(sample))

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
    def __init__(self, datapath, mfcc=True, num_workers=4):
        """
        @param datapath: the path to dataset
        @param num_workers:
        """
        self.num_workers = num_workers
        self.dataset = AudioMNIST(datapath)


    def random_split(self, dataset, lengths,
                    generator=default_generator):
        r"""
        Randomly split a dataset into non-overlapping new datasets of given lengths.

        If a list of fractions that sum up to 1 is given,
        the lengths will be computed automatically as
        floor(frac * len(dataset)) for each fraction provided.

        After computing the lengths, if there are any remainders, 1 count will be
        distributed in round-robin fashion to the lengths
        until there are no remainders left.

        Optionally fix the generator for reproducible results, e.g.:

        >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
        ...   ).manual_seed(42))

        Args:
            dataset (Dataset): Dataset to be split
            lengths (sequence): lengths or fractions of splits to be produced
            generator (Generator): Generator used for the random permutation.
        """
        if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
            subset_lengths: List[int] = []
            for i, frac in enumerate(lengths):
                if frac < 0 or frac > 1:
                    raise ValueError(f"Fraction at index {i} is not between 0 and 1")
                n_items_in_split = int(
                    math.floor(len(dataset) * frac)  # type: ignore[arg-type]
                )
                subset_lengths.append(n_items_in_split)
            remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
            # add 1 to all the lengths in round-robin fashion until the remainder is 0
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths)
                subset_lengths[idx_to_add_at] += 1
            lengths = subset_lengths
            for i, length in enumerate(lengths):
                if length == 0:
                    warnings.warn(f"Length of split at index {i} is 0. "
                                f"This might result in an empty dataset.")

        # Cannot verify that dataset is Sized
        if sum(lengths) != len(dataset):    # type: ignore[arg-type]
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
        return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

    def create_loaders(self, train_param: DataParam, val_param: DataParam, test_param: DataParam):
        train_ds, val_ds, test_ds = self.random_split(self.dataset, [train_param.ratio, val_param.ratio, test_param.ratio])
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
