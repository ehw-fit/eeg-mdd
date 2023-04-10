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
# %pip install py-paretoarchive
import random
from galib import ChromosomeChannels, GAops
import svgutils.compose as svc
from galib import EEGschema
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import gzip
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from paretoarchive.pandas import pareto
import re
import os
import glob

import parameters

# %%
alld = []
resdir = "res"
for fn in glob.glob(f"{resdir}/*.pkl.gz"):
    population = pickle.load(gzip.open(fn))
    log = open(fn.replace(".pkl.gz", ".log")).readlines()
    eltime = np.nan
    for l in log:
        if l.startswith("Run arguments: "):
            l = l.split("arguments: ")[1]
            params = json.loads(l.replace("'", '"'))
        if l.startswith("Elapsed time: "):
            eltime = float(l.split(":")[-1].strip())

    df = pd.DataFrame([{"chrom": str(p), "data": p, "elapsed": eltime,
                      "window_size": np.nan, **p.parameters, **params} for p in population]).sort_values(["features", "accuracy"])

    df["logname"] = os.path.basename(fn).replace(".pkl.gz", "")
    alld.append(df)

df = pd.concat(alld).reset_index().drop(columns=["index"])
df = df.query("classifier in ['kneighbors', 'svm']")
df


# %%
df["features_cnt"] = pd.cut(df["features"], bins=[3, 10, 20, 30, 40, 50, 60])
g = sns.displot(data=df,
                x="sensitivity",
                kind="hist",
                col="classifier",
                col_wrap=2,
                col_order=["svm", "kneighbors"],
                facet_kws=dict(sharex=False, sharey=False),
                bins=15,

                height=2,
                aspect=1.5,
                hue="features_cnt",
                multiple="stack",
                lw=0.2

                )
g.set(
    xlim=(0.85, 1),
    ylim=(0, None)
)
g._legend.set_title("# features")
g.set_titles("{col_name}")


for i, ax in enumerate(g.axes.ravel()):

    t = ["a)", "b)"][i]
    ax.set_title(t + " " + parameters.classifier_titles[ax.get_title()])


g.fig.tight_layout(rect=(0, 0, 0.8, 1))
g.savefig("plt/featurescnt.pdf")


# %%
# pareto optimal soulutions over runs
allp = []
for r, dfr in df.query("features > 0").groupby("classifier"):
    dfp = pareto(dfr, ["features", "accuracy", "sensitivity", "specificity"], minimizeObjective1=True,
                 minimizeObjective2=False, minimizeObjective3=False, minimizeObjective4=False)
    allp.append(dfp)
dfp = pd.concat(allp)
dfp

# %%


# %%
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8, 4))

axis = {
    # "dt": ax1,
    # "ridge": ax2,
    "svm": ax1,
    "kneighbors": ax2
}

allpar = []
df_best = []

for classifier, dfr in df.query("features > 0").groupby("classifier"):
    t = {"svm": "a)", "kneighbors": "b)"}[classifier]
    dfpr = pareto(dfr, ["features", "accuracy", "sensitivity", "specificity"], minimizeObjective1=True,
                  minimizeObjective2=False, minimizeObjective3=False, minimizeObjective4=False)
    allpar.append(dfpr)
    dfp_glob = dfpr.sort_values(["sensitivity", "specificity", "features"], ascending=[
                                False, False, True]).reset_index(drop=True)
    c = dfp_glob.loc[0]["data"]
    df_best.append(dfp_glob.loc[0])
    c.ops.channels = [x.replace("EEG ", "") for x in c.ops.channels]
    ax = axis[classifier]
    c.vizualize(ax)
    ax.set(xlim=(0, 50))
    ax.set_title(
        f"{t} {parameters.classifier_titles[classifier]} ({c.parameters['sensitivity']:.2%}/{c.parameters['specificity']:.2%}/{c.parameters['features']:d})")


#fig.suptitle("Best solutions (Pareto-optimal)")
#fig.tight_layout(rect=(0, 0, .9, 1))
fig.tight_layout()
# plt.show()

fig.savefig("plt/bestsolutions.pdf")

# %%%
pd.DataFrame(df_best).drop(columns=["data", "window_size", "fit_time", "generations", "log",
                                    "output_file", "logname", "features_cnt"]).to_excel("md_res/best_solutions.xlsx", index=False)

# %%
df.sort_values(["classifier", "sensitivity", "specificity", "features"], ascending=[False, False, False, True]).drop(columns=[
    "data", "window_size", "fit_time", "generations", "log", "output_file", "logname", "features_cnt"]).to_excel("md_res/all_solutions.xlsx", index=False)


# %% [markdown]
# # Results

# %%
stats = {}
for cl in df["classifier"].unique():
    stats[cl] = np.array([c.get_heat()
                         for c in df.query("classifier == @cl")["data"]])
    chrom1 = dfp["data"].iloc[0]
stats
# %%
# table how many channels are used for a given freq

fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8, 4))

axis = {
    # "dt": ax1,
    # "ridge": ax2,
    "svm": ax1,
    "kneighbors": ax2
}


for cl in df["classifier"].unique():
    # stat_freq = stats[cl].sum(axis=1).sum(axis=1).mean(axis=0)  # .any(axis=1) #.sum(axis=0)
    stat_freq = np.logical_or.reduce(
        np.logical_or.reduce(stats[cl], axis=1), axis=1).sum(axis=0)
    stat_freq.shape
    t = {"svm": "a)", "kneighbors": "b)"}[cl]

    ax = axis[cl]
    print(parameters.classifier_colors[cl])
    ax.bar(chrom1.ops.freqs[:chrom1.ops.freqs_max],
           stat_freq, color=parameters.classifier_colors[cl])
    # #plt.show()

    ax.set(
        xlabel="Frequency (Hz)",
        ylabel="# planes",
        title=t + " " + parameters.classifier_titles[cl],
        xlim=(0, 50),
        ylim=(0, 32)

    )
fig.tight_layout()

# stat_freq.shape
# #stat_channel.shape
# #pd.DataFrame(stat_channel, columns=chrom1.ops.channels, index=[chrom1.fn2str(i) for i in range(5)])
fig.savefig("plt/frequency.pdf")

# %%
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 7))

# axis = {
#     "dt": ax1,
#     "ridge": ax2,
#     "svm": ax3,
#     "kneighbors": ax4
# }


# for cl in df["classifier"].unique():

#     stat_channel = stats.mean(axis=0) #sum(axis=1)
#     max_freq = chrom1.ops.freqs[chrom1.ops.freqs_max]
#     stat_channel.shape

#     fig, axes = plt.subplots(3, 2, figsize=(16, 10))
#     axes = axes.ravel()

#     for i, ax in zip(range(chrom1.ops.fun_max), axes):
#         ax.set_title("fun " + chrom1.fn2str(i))

#         im = ax.imshow(stat_channel[i], aspect="auto", interpolation = "None", vmin=0,
#             vmax=stat_channel.max(), extent=[0, max_freq, 0,
#             chrom1.ops.channels_max])

#         ax.set_yticks(np.arange(chrom1.ops.channels_max) + 0.5, chrom1.ops.channels)

#         cax = plt.colorbar(im, ax=ax)
#         cax.set_label("proportion in found filters")
#     axes[-1].axis("off")

# plt.tight_layout()
# stat_channel.max()
# plt.show()


# %%


fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8, 4))

axis = {
    # "dt": ax1,
    # "ridge": ax2,
    "svm": ax1,
    "kneighbors": ax2
}


max_freq = 50
for cl in df["classifier"].unique():
    ax = axis[cl]
    t = {"svm": "a)", "kneighbors": "b)"}[cl]

    # stat_channel = stats[cl].mean(axis=0).sum(axis=0)  # sum(axis=1)
    stat_channel = np.logical_or.reduce(
        stats[cl], axis=1).sum(axis=0).astype("f")
    #stat_channel[stat_channel == 0] = np.nan
    # if cl == "svm":
    #     stat_channel[stat_channel < 3] = np.nan
    # if cl == "kneighbors":
    #     stat_channel[stat_channel < 5] = np.nan
    img = ax.imshow(stat_channel, interpolation="none", origin="lower",
                    aspect="auto", extent=[0, max_freq, 0, chrom1.ops.channels_max],
                    vmin=0, rasterized=True)

    ax.set_yticks(np.arange(chrom1.ops.channels_max) + 0.5,
                  [x.replace("EEG ", "") for x in chrom1.ops.channels])

    divider = make_axes_locatable(ax)

    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(img, cax=cax, orientation='vertical')

    #cax = plt.colorbar(img)
    cb.set_label("# planes")
    ax.set(
        xlabel="Frequency (Hz)",
        title=t + " " + parameters.classifier_titles[cl]
    )
stat_channel.shape

fig.tight_layout()

fig.savefig("plt/features.pdf")
fig.savefig("plt/features.png", dpi=300)

# %%%

max_freq = 50
vmax = 0.625
for cl in df["classifier"].unique():
    #    cl = "svm"
    t = {"svm": "a)", "kneighbors": "b)"}[cl]
    print("CL is ", cl)

    stat_chan = np.logical_or.reduce(np.logical_or.reduce(
        stats[cl], axis=1), axis=2).sum(axis=0).astype("f")
    print(stats[cl].shape)

    print("max = ", stat_chan.max())
    chmax = stat_chan.max()
    stat_chan /= stats[cl].shape[0]
    stat_chan /= stat_chan.max()

    sch = EEGschema()
    cm = plt.get_cmap("viridis")
    for ch, c in zip(chrom1.ops.channels, stat_chan):
        #print(ch, c)
        sch.set_channel_color(ch.replace(
            "EEG ", "").replace("-LE", "").lower(), cm(c))

    sch.savefig(f"usage_{cl}.svg")

    vmax = 1
    a = np.array([[0, chmax]])
    plt.figure(figsize=(4, 0.5))
    img = plt.imshow(a, cmap="viridis")
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    plt.colorbar(orientation="horizontal", cax=cax, label="# planes")
    plt.savefig(f"usage_bar_{cl}.svg")

sch
# %%%
vmax = 1
a = np.array([[0, vmax * 100]])
plt.figure(figsize=(9, 1.0))
img = plt.imshow(a, cmap="viridis")
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
plt.colorbar(orientation="horizontal", cax=cax, label="Occurance [%]")
plt.savefig("usage_bar.svg")

# %%%

# s = svc.Figure("310mm", "180mm",
#                svc.Panel(
#                    svc.Text("k-NN", 70, 6, size=6),
#                    svc.SVG("usage_kneighbors.svg").move(0, 8),
#                ), svc.Panel(
#                    svc.Text("SVM", 70, 6, size=6),
#                    svc.SVG("usage_svm.svg").move(0, 8)
#                ).move(150, 0),
#                svc.Panel(
#                    svc.SVG("usage_bar.svg", fix_mpl=True).scale(
#                        0.3).move(60, 150)
#                )

#                )

s = svc.Figure("300mm", "175mm", svc.Panel(
                   svc.Text("SVM", 69, 6, size=6),
                   svc.SVG("usage_svm.svg").move(0, 8),
                   svc.SVG("usage_bar_svm.svg", fix_mpl=True).scale(0.4).move(17, 145)
               ),
               svc.Panel(
                   svc.Text("k-NN", 69, 6, size=6),
                   svc.SVG("usage_kneighbors.svg").move(0, 8),
                   svc.SVG("usage_bar_kneighbors.svg", fix_mpl=True).scale(0.4).move(17, 145)
               ).move(150, 0),
            #    svc.Panel(
            #        svc.SVG("usage_bar.svg", fix_mpl=True).scale(
            #            0.3).move(60, 150)
            #    )

               )

s.save("usage_all.svg")

os.system("inkscape usage_all.svg  --export-filename=usage_all.pdf")
s



# %%

ops = GAops([f"CH{x+1:02d}" for x in range(8)], np.arange(0, 513) / 4)

fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=6)
ax2 = plt.subplot2grid((8, 1), (7, 0))

random.seed(42)
np.random.seed(60)
ch = ChromosomeChannels(ops).random(6)
ch.vizualize(ax=ax1)

chrom = str(ch)
chrom = chrom.replace(" ", "")


chrom = re.sub("(.{64}.*?,)", "\\1\n", chrom, 0, re.DOTALL)


ax2.text(0, 0, chrom, font="monospace")
ax2.axis("off")

#ax1.arrow(0.25, 0.2, 0.11, 0.22, transform=fig.transFigure, clip_on=False,    head_width=0.02, edgecolor="tab:red", fc="tab:red")

ax1.annotate("", xytext=(20, -1.5), xy=(40, 1.5), clip_on=False,
             c="tab:red", arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
                                          shrinkA=0, shrinkB=1, fc="tab:red", ec="tab:red"))

ax1.annotate("", xytext=(50, -1.5), xy=(6, 4.5), clip_on=False,
             c="tab:red", arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",
                                          shrinkA=0, shrinkB=1, fc="tab:red", ec="tab:red"))
str(ch)

plt.tight_layout()
plt.savefig("plt/chromosome.pdf")


# %%

dft = df.groupby("classifier").agg({
    "elapsed": "mean",
    "sensitivity": ("min", "mean", "max"),
    "specificity": ("min", "mean", "max"),
    "features": ("min", "mean", "max"),
}).reset_index()

dft["classifier"] = dft["classifier"].apply(
    lambda x: parameters.classifier_titles[x])
dft
# %%%
style = dft.style.format(
    "{:.1%}", subset=["sensitivity", "specificity"], escape="latex"
).format("{:.2f}", subset=[["features", "mean"]]
         ).format(precision=2, subset="elapsed"
                  ).hide(axis="index")

print(style.to_latex().replace("%", "\\%"))

# %%
df_dt = pd.read_pickle("test.pkl.gz")
df_dt

# %%%
# from galib import ChromosomeBase, ChromosomeChannels, FeaturesException, GAops, DataParser

# import sklearn.neighbors
# import sklearn.linear_model
# import sklearn.tree
# import sklearn.svm
# from sklearn.model_selection import GroupKFold, cross_validate, train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix

# def sas_score(estimator, X_test, y_test):
#     y_pred = estimator.predict(X_test)
#     conf_m = confusion_matrix(y_test, y_pred).ravel()
#     if len(conf_m) != 4:  # wrong test data
#         return {
#             "specificity": 0,
#             "accuracy": 0,
#             "sensitivity": 0,
#         }
#     tn, fp, fn, tp = conf_m

#     r = {
#         "specificity": 0 if tn == 0 else (tn) / (fp + tn),  # specificity"
#         "accuracy": (tp + tn) / (tp + fp + tn + fn),  # accuracy
#         "sensitivity": 0 if tp == 0 else (tp) / (tp + fn),  # sensitivity"
#     }
#     return r

# data_parser = DataParser()
# cv = GroupKFold(5)

# def evaluate_chromosome(chrom):
#     global cv
#     data = data_parser.get_data()
#     X = data["x"]
#     y = data["y"]
#     samples_id = data["id"]
#     freqs = data["freqs"]
#     channels = data["features"]

#     model = sklearn.tree.DecisionTreeClassifier(random_state=42)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

#     try:
#         cv = cross_validate(model, chrom.execute(X), y,
#                             groups=samples_id, cv=cv,
#                             scoring=sas_score, n_jobs=-1
#                             )

#         model.fit(chrom.execute(X_train), y_train)
#         print(model.score(chrom.execute(X_test), y_test))
#     except FeaturesException as e:
#         print("Features exception: {} for chromosome {}".format(e, chrom))
#         cv = {
#             "test_accuracy": np.array([0.0]),
#             "test_specificity": np.array([0.0]),
#             "test_sensitivity": np.array([0.0])
#         }

#     aggfunc = np.mean

#     score = {

#         "accuracy": aggfunc(cv["test_accuracy"]),
#         "sensitivity": aggfunc(cv["test_sensitivity"]),
#         "specificity": aggfunc(cv["test_specificity"]),
#         "features": chrom.features,
#     }
#     return score, model

# chrom = df_dt[3]
# chrom

# e, m = evaluate_chromosome(chrom)
# e, m
# # %%
# from sklearn.tree import plot_tree
# plot_tree(m)

# %%
