"""
An example of usage of classifier for trained data

It uses only delta waves on all channels

Example filtering
```
X_train[:, :, (freqs > 1) & (freqs <= 4)] # freqs 1 < f <= 4 Hz
X_train[:, 0:7, :] # only channels 0 to 7 (channels[0:7])
```
"""

# %%
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn.neighbors
import sklearn.linear_model
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from galib import DataParser

# %%
data = DataParser().get_data()

X = data["x"]
y = data["y"]
samples_id = data["id"]
freqs = data["freqs"]
channels = data["features"]

# %%
def sas_score(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    r = {
        "specificity": (tn) / (fp + tn),  # specificity"
        "accuracy": (tp + tn) / (tp + fp + tn + fn),  # accuracy
        "sensitivity": (tp) / (tp + fn),  # sensitivity"
    }
    return r


model = sklearn.neighbors.KNeighborsClassifier(2)
#model = sklearn.linear_model.RidgeClassifier(0.2)


r = cross_validate(model, X[:, :, (freqs > 1) & (freqs <= 4)].reshape(61, -1), y,
               groups=samples_id, cv=GroupKFold(5), scoring=sas_score, n_jobs=-1
               )

print(r)
# %%
