# %%
# Load all libraries
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, KFold, cross_validate, train_test_split
from sklearn.utils import shuffle
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

ch1 = ChromosomeChannels(ga_ops).from_str(chrom_1)
ch2 = ChromosomeChannels(ga_ops).from_str(chrom_2)
chJoin = ChromosomeChannels(ga_ops).from_str(chrom_1 + chrom_2)

class chromAll:
    def execute(self, x):
        return x.reshape(61, 19*513)

chAll = chromAll()


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

#%%
##
# Note: works only if the data are not groupped, e.g. data["id"] contains unique elements
# otherwise change KFold to GroupKFold
assert len(np.unique(data["id"])) == len(data["id"])

# %%
alld = []
from tqdm.auto import tqdm
for run in tqdm(range(100)):
    for mname in ["KNN", "SVM"]:
        for cht, chrom in [("ch1", ch1), ("ch2", ch2), ("chJoin", chJoin), ("chAll", chAll)]:
                
            if mname == "KNN":
                model = sklearn.neighbors.KNeighborsClassifier(3)
            elif mname == "SVM":
                model = sklearn.svm.SVC()
            else:
                raise Exception("Unknown model")
            
            shuffled_idx = np.random.permutation(len(data["id"]))          

            x_data, labels, ids = shuffle(
                    chrom.execute(data["x"]),
                    data["y"],
                    data["id"]
                    )

            # target model
            cv = cross_validate(model, x_data, labels,
                                groups=ids, #cv=GroupKFold(5),
                                cv=KFold(5),
                                scoring=sas_score
                                )

            row = {
                "model": mname,
                "chrom": cht,
                "run": run,
                "accuracy": cv["test_accuracy"],
                "sensitivity": cv["test_sensitivity"],
                "specificity": cv["test_specificity"],
            }
            alld.append(row)

# %%
# alld = []
# for run in range(10):
#     for mname in ["KNN", "SVM"]:
#         for cht, chrom in [("ch1", ch1), ("ch2", ch2), ("chJoin", chJoin), ("chAll", chAll)]:
                
#             if mname == "KNN":
#                 model = sklearn.neighbors.KNeighborsClassifier(3)
#             elif mname == "SVM":
#                 model = sklearn.svm.SVC()
#             else:
#                 raise Exception("Unknown model")
            

#             X_train, X_test, y_train, y_test = train_test_split(
#                 chrom.execute(data["x"]),  data["y"], train_size=0.6)

#             model.fit(X_train, y_train)

#             print("Score", sas_score(model, X_test, y_test))

#             row = {
#                 "model": mname,
#                 "chrom": cht,
#                 "run": run,
#                 **sas_score(model, X_test, y_test)
#             }
#             alld.append(row)

# %%
df = pd.DataFrame(alld)
df
#%%%
# reuse the results
df = pd.read_pickle("res/stat.pkl.gz")

# %%

dfc = df.copy().drop(columns=["run"])
dfc["accuracy"] = dfc["accuracy"].apply(lambda x: x.mean())
dfc["sensitivity"] = dfc["sensitivity"].apply(lambda x: x.mean())
dfc["specificity"] = dfc["specificity"].apply(lambda x: x.mean())
dfc = dfc.groupby(["model", "chrom"]).agg(["min", "mean", "max"])


dfc2 = dfc["accuracy"].unstack(level=0)
dfc2 = dfc["specificity"].unstack(level=0)
# swap two levels of multilevel column index
dfc2 = dfc2.swaplevel(axis=1).sort_index(axis=1)
dfc2


# %%
from scipy.stats import ttest_rel

dfc = df.copy().drop(columns=["run"])
alld = {}
for (m1, c1), d1 in dfc.groupby(["model", "chrom"]):
    row = {}
    for (m2, c2), d2 in dfc.groupby(["model", "chrom"]):
        if m1 != m2:
            continue
        #row[c2] = ttest_rel(np.concatenate(list(d1["accuracy"])), np.concatenate(list(d2["accuracy"]))).pvalue
        #row[c2] = ttest_rel(d1["accuracy"], d2["accuracy"]).pvalue
        row[c2] = ttest_rel(d1["accuracy"].mean(), d2["accuracy"].mean()).pvalue

    alld[(m1, c1)] = row
pd.DataFrame(alld)

# %%
