from torch.utils.data import DataLoader, Dataset, random_split
from data_processing.datahandler import AudioHandler


class AudioMNIST(Dataset):
    def __init__(self, datapath, load_entire_filetree=False):
        self.datapath = datapath
        self.files = self._build_files()
        self.audio_len = 48000
        self.shift_ptc = 0.4
        

    def _build_files(self):
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

    def __len__(self):
        return 30000

    def __getitem__(self, idx):
        signal = AudioHandler.open(self.files[idx][0])[0]
        pad = AudioHandler.pad(signal, self.audio_len)
        shift = AudioHandler.time_shift(pad, self.shift_ptc)
        sgram = AudioHandler.spectrogram(
            shift, n_mels=64, n_fft=1024, hop_len=None, mfcc=True
        )
        aug_sgram = AudioHandler.spectral_augmentation(
            sgram, max_mask_ptc=0.1, n_freq_masks=2, n_time_masks=2
        )

        return aug_sgram, self.files[idx][1]


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
    def __init__(self, datapath, num_workers=2):
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
