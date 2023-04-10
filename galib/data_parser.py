from sklearn.model_selection import train_test_split
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import pyedflib
import mne
from scipy import signal


class DataParser:
    def __init__(self):
        self.data = None
        self.channels_names = []
        self.SAMPLING_FREQ = 0

    def load_dataset(self):
        DATA_DIR_A = 'data/MDD/'
        
        def load_data(directory):
            data = []

            for fname in tqdm(sorted(os.listdir(directory))):
                img = mne.io.read_raw_edf(os.path.join(directory, fname), preload=True)

                for c in ['EEG A2-A1', 'EEG 23A-23R', 'EEG 24A-24R']:
                    if c not in img.ch_names: continue
                    img.drop_channels([c])

                data.append({
                    "filename": fname,
                    "img": img,
                    "path": os.path.join(directory, fname),
                    "label": 0 if "MDD" in fname else 1
                }
                )

            return data

        data = []
        data += load_data(DATA_DIR_A)

        print("Total len of data", len(data))

        channel_names = None
        for d in data:
            #print(d["img"].ch_names, d["label"])
            if channel_names is None:
                channel_names = d["img"].ch_names

            assert channel_names == d["img"].ch_names
        self.channel_names = channel_names
        self.data = data

    def parse_data(self):
        if not self.data:
            print(f"\033[91mLoading data\033[0m")
            self.load_dataset()

        def parse_data(data):
            data_x = []
            data_y = []
            data_id = []
            for did, d in enumerate(data):
                raw = d["img"].get_data()
                if not self.SAMPLING_FREQ:
                    self.SAMPLING_FREQ = d["img"].info["sfreq"]
                assert self.SAMPLING_FREQ == d["img"].info["sfreq"]
                #for i in range(0, raw.shape[1], WINDOW_SIZE_R):
                #    if i + WINDOW_SIZE_R > raw.shape[1]:
                #        continue
                #data_x.append(raw[:, i:i+WINDOW_SIZE_R])
                data_x.append(raw[:, :])
                data_y.append(d["label"])
                data_id.append(did)

                #print(d["img"].get_data().shape[1], d["img"].info["sfreq"])
            return data_x, data_y, data_id

        data_x, data_y, data_id = parse_data(self.data)

        print("MDD ration train {:.1%}".format(np.mean(data_y)))

        def run_filter(data_x, data_y):
            data_filt = []
            freqs = None
            for d in data_x:
                #print(d.shape)

                ff, filt = signal.welch(d, self.SAMPLING_FREQ, nperseg=1024, axis=1)
                filtdb = 10 * np.log10(filt)
                if freqs is None:
                    freqs = ff

                #print(d.shape, filtdb.shape)
                data_filt.append(filtdb)
                assert (freqs == ff).all()

            dt_x = np.array(data_filt)
            dt_y = np.array(data_y)
            return dt_x, dt_y, freqs

        X, y, freqs = run_filter(data_x, data_y)

        return dict(
            x=X,
            y=y,
            id=np.array(data_id),
            freqs=freqs,
            features=self.channel_names

        )

    # %%

    def get_data(self, force = False):
        # save model
        dir_name = f"data/datasets/"
        os.makedirs(dir_name, exist_ok=True)

        filename = os.path.join(dir_name, f"dataset.npz")

        if force or not os.path.exists(filename):
            print(f"\033[91mFile doesnt exists {filename}\033[0m")
            res = self.parse_data()
            np.savez(filename, **res)

        return np.load(filename)

# Data organization:
# X: 0 - samples, 1 - channels, 2 - freqs