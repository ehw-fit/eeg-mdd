# %%
# Load all libraries
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_validate, train_test_split
import sklearn.svm
import sklearn.tree
import sklearn.linear_model
import sklearn.neighbors
from galib import ChromosomeBase, ChromosomeChannels, FeaturesException, GAops, DataParser
from galib import ChromosomeChannels, DataParser, GAops
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import seaborn as sns
from matplotlib import rcParams
rcParams['savefig.facecolor'] = 'white'
# %%
# Loading data
data_parser = DataParser()
data = data_parser.get_data()
ga_ops = GAops(channels=data["features"], freqs=data["freqs"])
# Limit the frequency to 50 Hz
ga_ops.limit_frequency(50)
chrom = ChromosomeChannels(ga_ops)

#%%%
chrom_1 = "(EEG F8-LE, 8.0, 12.0, DS8)(EEG F4-LE, 4.0, 8.0, DS8)(EEG P4-LE, 12.0, 20.0, DS2)(EEG T5-LE, 30.0, 49.75, AGG)"
chrom_2 = "(EEG T5-LE, 30.0, 49.75, AGG)(EEG F8-LE, 8.0, 12.0, DS8)(EEG Cz-LE, 4.0, 8.0, AGG)(EEG F8-LE, 8.0, 12.0, DS8)"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

fmt = lambda x: str(x).replace("EEG ", "")
ChromosomeChannels(ga_ops).from_str(chrom_1).vizualize(ax = ax1, channel_formatter=fmt)
ChromosomeChannels(ga_ops).from_str(chrom_2).vizualize(ax = ax2, channel_formatter=fmt)

settings = dict(
    xlim = (0, 50)
)
ax1.set(title="a) Solution #1 (SVM)",**settings)
ax2.set(title="b) Solution #2 (k-NN)", **settings)

fig.tight_layout()
fig.savefig("plt/selected.pdf")
# %%%
# Chromosome to work with
chrom = ChromosomeChannels(ga_ops)
#chrom.from_str('(EEG C3-LE, 30.0, 48.0, DS8)(EEG T5-LE, 20.0, 30.0, AGG)(EEG T3-LE, 8.0, 12.0, DS2)')
chrom.from_str('(EEG O2-LE, 30.0, 49.75, AGG)(EEG F8-LE, 8.0, 12.0, DS8)(EEG T6-LE, 8.0, 12.0, DS8)(EEG P4-LE, 12.0, 20.0, DS2)(EEG Fz-LE, 30.0, 49.75, AGG)(EEG P4-LE, 0.0, 4.0, DS8)')
chrom.vizualize()
print(chrom)

head = chrom.plot_head("tab:blue")
head.savefig("head.svg")
head



# %%%

# Analysis od dataset
x_data = chrom.execute(data["x"])
labels = data["y"]

print("# features: ", x_data.shape[1])

# label == 0 => MDD; label == 1 => CONTROL

print("# subjects: ", labels.shape)
print("# MDD subjects: ", (labels == 0).sum())
print("# Control subjects: ", (labels == 1).sum())

# %%

fig, ax = plt.subplots(figsize=(16, 4))

features = x_data.shape[1]


bp_mdd = ax.boxplot(x_data[labels == 0, :],
                    positions=np.arange(features) * 2 + 0.5,
                    patch_artist=True, )
bp_control = ax.boxplot(x_data[labels == 1, :],
                        positions=np.arange(features) * 2 + 1.5,
                        patch_artist=True, )

# round to tens
y_min = np.floor(x_data.min() / 10) * 10
y_max = np.ceil((x_data.max()) / 10) * 10

y_min -= 10  # place to annotations

ax.vlines(np.arange(features - 1) * 2 + 2, y_min,
          y_max, ls=":", color="gray", zorder=1)

prev = None
prev_start = 0


for i, g in enumerate(chrom.genes_features + ["last"]):
    if prev != g:
        if prev:
            # draw rectangle
            r = Rectangle((prev_start * 2, y_min), 2*(i-prev_start), 10,
                          ec="tab:blue", fc="w", zorder=100, lw=2)
            ax.add_patch(r)
            print(prev.replace("EEG", "").replace(" ", ""))
            ax.text((i-((i - prev_start) / 2)) * 2, y_min + 5,
                    prev.replace("EEG ", ""),
                    zorder=200, va="center", ha="center", font="monospace", fontsize=8)

        prev = g
        prev_start = i

ax.set(ylim=(y_min, y_max), ylabel="Power [dB]", xlabel="Feature")

ax.set_xticks(
    np.arange(features) * 2 + 1,
    [f"{x+1}" for x in np.arange(features)]
)

# set colors
for bplot, color in [(bp_mdd, "pink"), (bp_control, "lightgreen")]:
    for patch in bplot['boxes']:
        patch.set_facecolor(color)

ax.legend(
    handles=[Patch("pink", "pink"), Patch("lightgreen", "lightgreen")],
    labels=["MDD subjects", "Control subjects"]
)


fig.tight_layout()

# example to save the results
# fig.savefig("plt/some.pdf")
# fig.savefig("plt/some.png")
# fig.show()

# %%%


def sas_score(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    conf_m = confusion_matrix(y_test, y_pred).ravel()
    if len(conf_m) != 4:  # wrong test data
        return {
            "specificity": 0,
            "accuracy": 0,
            "sensitivity": 0,
        }
    tn, fp, fn, tp = conf_m

    r = {
        "specificity": 0 if tn == 0 else (tn) / (fp + tn),  # specificity"
        "accuracy": (tp + tn) / (tp + fp + tn + fn),  # accuracy
        "sensitivity": 0 if tp == 0 else (tp) / (tp + fn),  # sensitivity"
    }
    return r

 # %%

 # Evaluate the chromosome in the model
 # standard method
 # be aware: these methods depends oon the random state, they are not deterministic!!!


# target model
model = sklearn.neighbors.KNeighborsClassifier(2)
#model = sklearn.linear_model.RidgeClassifier(0.2)
#model = sklearn.svm.SVC()
#model = sklearn.tree.DecisionTreeClassifier(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    x_data, labels, train_size=0.7)

model.fit(X_train, y_train)

print("Score", sas_score(model, X_test, y_test))


# %%
# Evaluate the chromosome in the model using k-fold algorithm
# the methods are not deterministic as well

# target model
#model = sklearn.neighbors.KNeighborsClassifier(3)
#model = sklearn.linear_model.RidgeClassifier(0.2)
model = sklearn.svm.SVC()
# #model = sklearn.tree.DecisionTreeClassifier(random_state=42)

cv = cross_validate(model, x_data, labels,
                    groups=data["id"], cv=GroupKFold(5),
                    scoring=sas_score, n_jobs=-1
                    )
print("Accuracies:   ", cv["test_accuracy"], cv["test_accuracy"].mean())
print("Sensitivities:", cv["test_sensitivity"],cv["test_sensitivity"].mean())
print("Specificities:", cv["test_specificity"], cv["test_specificity"].mean())


print("\t".join([f"{x:.3f}" for x in np.concatenate([cv["test_accuracy"], cv["test_sensitivity"], cv["test_specificity"]])]))
# %%


# Evaluatioon 

# target model
model = sklearn.neighbors.KNeighborsClassifier(3)
#model = sklearn.linear_model.RidgeClassifier(0.2)
# model = sklearn.svm.SVC()
# #model = sklearn.tree.DecisionTreeClassifier(random_state=42)

cv = cross_validate(model, data["x"].reshape(61, 19*513), labels,
                    groups=data["id"], cv=GroupKFold(5),
                    scoring=sas_score, n_jobs=-1
                    )
print("Accuracies:   ", cv["test_accuracy"], cv["test_accuracy"].mean())
print("Sensitivities:", cv["test_sensitivity"],cv["test_sensitivity"].mean())
print("Specificities:", cv["test_specificity"], cv["test_specificity"].mean())

print("\t".join([f"{x:.3f}" for x in np.concatenate([cv["test_accuracy"], cv["test_sensitivity"], cv["test_specificity"]])]))
# %%


