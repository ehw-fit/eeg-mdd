# %% [markdown]
# # Technical background
# - A search of features for 
# - k-fold (10) evaluation with unmixed persons
# - the overall specificity and sensitivity is calculed as a __minimal__ value of 10 folds
# - NSGA-II search
# - selects the window size in the gene
# 
# # Issues and weaknesses
# - How to determine, if we found a specific feature, or some recording-specific noise? control and test dataset were captured using different device
# 
# 

# %%
#%pip install py-paretoarchive
import pickle, gzip, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from paretoarchive.pandas import pareto
import re
import os
import glob
from galib import DataParser
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
d = DataParser()
data = d.get_data()
# %%
viz = data["x"][0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

cm = plt.get_cmap("inferno")
for i in range(viz.shape[0]):
    ax1.plot(data["freqs"], viz[i, :], label=data["features"][i].replace("EEG ", ""), c = cm(i / viz.shape[0]))

ax1.legend(ncol=3)
ax1.set(
    xlim=(0, 128),
    ylim=(-180, -80),
    title="(a) PSD diagram",
    xlabel="Frequency [Hz]",
    ylabel="Power [dB]"
)
divider = make_axes_locatable(ax2)

im = ax2.imshow(viz, aspect="auto", interpolation="none", 
    extent=(0, 128, 19, 0), vmin=-180, vmax=-80, rasterized=True)

cax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(im, cax=cax, orientation='vertical')
cb.set_label("Power [dB]")

ax2.set_yticks(
    np.arange(viz.shape[0]) + 0.5,
    [x.replace("EEG ", "") for x in data["features"]]
)
ax2.set(
    title="(b) Matrix representation",
    xlabel="Frequency [Hz]",
    ylabel="Channel"
)

plt.tight_layout()
plt.savefig("plt/subject.pdf")
# %%
viz = data["x"][0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

cm = plt.get_cmap("inferno")
for i in range(viz.shape[0]):
    ax1.plot(data["freqs"], viz[i, :], label=data["features"][i].replace("EEG ", ""), c = cm(i / viz.shape[0]))

ax1.legend(ncol=2)
ax1.set(
    xlim=(0, 128),
    ylim=(-180, -80),
    title="(a) PSD diagram",
    xlabel="Frequency [Hz]",
    ylabel="Power [dB]"
)
divider = make_axes_locatable(ax2)

im = ax2.imshow(viz, aspect="auto", interpolation="none", 
    extent=(0, 128, 19, 0), vmin=-180, vmax=-80, rasterized=True)

cax = divider.append_axes('right', size='5%', pad=0.05)
cb = fig.colorbar(im, cax=cax, orientation='vertical')
cb.set_label("Power [dB]")

ax2.set_yticks(
    np.arange(viz.shape[0]) + 0.5,
    [x.replace("EEG ", "") for x in data["features"]]
)
ax2.set(
    title="(b) Matrix representation",
    xlabel="Frequency [Hz]",
    ylabel="Channel"
)

plt.tight_layout()
plt.savefig("plt/subject_line.pdf")
# %%
