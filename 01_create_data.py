"""
A script that loads EEG data
2) for each sample runs a Welch filter, transforms to dB
3) creates train and test data
4) stores train, test data and freqs to npz file

Output data organization
X - [samples, channels, freqs]
y - [samples] 1 = alcoholic; 0 = control

"""
# %%
from galib import DataParser


if __name__ == "__main__":
    d = DataParser()
    r = d.get_data(force=True)
    for k in r.keys():
        print(k, r[k].shape)


# %%
