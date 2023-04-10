#%%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from paretoarchive.pandas import pareto
from tqdm.auto import tqdm
import glob
rcParams['savefig.facecolor']='white'
# %%
import gzip
import json

def get_item(x):
    a, b = x.split("=")
    return a.strip(), b.strip()

def parse_log(logfile):
    data = []
    allpops = []
    runargs = {}
    with gzip.open(logfile, "rt") as f:
        for l in f.readlines():
            if l.startswith("runarg"):
                l = l.split("=")[1]
                aa = l.split("#") # bug in logs
                if len(aa) == 2:
                    l, rest = aa
                
                dkey = None
                for k in json.loads(l):
                    if k.startswith("--"):
                        dkey = k
                    else:
                        if dkey:
                            runargs[dkey.replace("--", "")] = k
                        dkey = None

                print(runargs)
                if len(aa) == 2:
                    l = rest
                else:
                    continue

            if not l.startswith("#"): continue
            l = l[1:].strip()
            row = dict([get_item(x) for x in l.split(", ", 3)])
            row

            row["parent_pop"] = json.loads(row["parent_pop"])
            row["gen"] = int(row["gen"])
            row["good"] = int(row["good"])
            row["limit_acc"] = float(row["limit_acc"])

            pop = pd.DataFrame(row["parent_pop"])
            pop["gen"] = int(row["gen"])
            pop["good"] = int(row["good"])
            pop["limit_acc"] = float(row["limit_acc"])

            allpops.append(pop)
            data.append(row)

    df_gens = pd.DataFrame(data)
    df_gens["classifier"] = runargs["classifier"]
    df_pops = pd.concat(allpops).reset_index()
    df_pops["classifier"] = runargs["classifier"]

    return df_gens, df_pops

all_gen = []
all_pops = []
for fn in tqdm(glob.glob("res/*.pkl.gz")):
    logfn = fn.replace(".pkl.gz", ".gz")
    df_gens, df_pops = parse_log(logfn)
    df_gens["logfn"] = logfn
    df_pops["logfn"] = logfn
    all_gen.append(df_gens)
    all_pops.append(df_pops)
#%%
df_gens = pd.concat(all_gen, ignore_index=True).query("classifier in ['kneighbors', 'svm']")
df_gens = df_gens.reset_index(drop=True)
df_gens.info()
#%%

df_pops = pd.concat(all_pops).query("classifier in ['kneighbors', 'svm']")
df_pops["logfn"] = df_pops["logfn"].astype("category")
df_pops["classifier"] = df_pops["classifier"].astype("category")
df_pops = df_pops.reset_index(drop=True)
df_pops.info()
#%%
df_gens.to_pickle("data/df_allgen.pkl.gz")
df_pops.to_pickle("data/df_pops.pkl.gz")
#%%
# %%
if False: # just for debuging because of speed
    df_tmp = df_pops.query("gen < 10")
    _, df_gens_tmp = list(df_gens.groupby("logfn"))[0]
else:
    df_tmp = df_pops
    df_gens_tmp = df_gens

def plot_minmax(ax=None, data=None, x=None, y=None, color=None, **kwargs):
    if not ax:
        ax = plt.gca()


    plot_df = data.groupby(x).agg({y: ("min", "max", "mean")}).reset_index()
    display(plot_df)

    ax.fill_between(
        x=plot_df[x],
        y1 = plot_df[(y, "min")],
        y2 = plot_df[(y, "max")],
        color = color,
        alpha=0.3
    )

    ax.plot(
        plot_df[x],
        plot_df[(y, "mean")],
        color = color, **kwargs
    )

    ax.set(xlabel=x, ylabel=y)


#fig, axes = plt.subplots(1, 2,  figsize=(8, 3))
#ax1, ax2 = axes.ravel()

fig = plt.figure(figsize=(8, 4)) #, layout="constrained")
spec = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(spec[:, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[1, 1])



minparams = dict(ls="-", lw="0.5")
sns.lineplot(data=df_tmp, x="gen", y="features", color="tab:blue", ax=ax1)

#df_t = df_tmp.groupby("gen").agg({"features":"min"}).reset_index()
#ax1.plot(df_t["gen"], df_t["features"], color="tab:blue", **minparams)



sns.lineplot(data=df_tmp, ax=ax2, x="gen", y="sensitivity", label="sensitivity", color="tab:red")
#xdf_t = df_tmp.groupby("gen").agg({"sensitivity":"max"}).reset_index()
#ax2.plot(df_t["gen"], df_t["sensitivity"], color="tab:red", label="best value", **minparams)
sns.lineplot(data=df_gens_tmp, x="gen", y="limit_acc", drawstyle="steps-post", lw=1, ls=":", label="limit", color="tab:gray", ax=ax2)



sns.lineplot(data=df_tmp, ax=ax3, x="gen", y="specificity", label="specificity", color="tab:orange")
#df_t = df_tmp.groupby("gen").agg({"specificity":"max"}).reset_index()
#ax3.plot(df_t["gen"], df_t["specificity"], color="tab:orange", label="best value", **minparams)
sns.lineplot(data=df_gens_tmp, x="gen", y="limit_acc", drawstyle="steps-post", lw=1, ls=":", label="limit", color="tab:gray", ax=ax3)



ax1.set(
    ylim = (0, None),
    ylabel="# features",
    xlabel="Generation", xlim=(0, 1000),
)

ax2.set(
    ylim=(0.5,1),
    xlabel="Generation", xlim=(0, 1000),
    ylabel="Sensitivity"
)


ax3.set(
    ylim=(0.5,1),
    xlabel="Generation", xlim=(0, 1000),
    ylabel="Specificity"
)

#ax3.set_ylim(0,1)

plt.tight_layout(pad=0)
fig.savefig("plt/search.pdf")

# %%

import parameters
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
ax1, ax2, ax3 = axes.ravel()

for classifier, df_lastgen in reversed(list(df_gens.query("gen == 999").groupby("classifier"))):
    df_t = df_lastgen.copy()
    df_lastgen = pd.concat([pd.DataFrame(x) for x in df_t["parent_pop"]], ignore_index=True)
    
    df_lastgen = df_lastgen.sort_values(["features", "accuracy"]).reset_index(drop=True)



    def plt_points(df, x, y, ax):

        print(df.columns)

        if "features" not in [x,y]:
            z = "features", "min"
        if "sensitivity" not in [x,y]:
            z = "sensitivity", "max"
        if "specificity" not in [x,y]:
            z = "specificity", "max"

        df = df.groupby([x, y]).agg({"chrom": "count", z[0]: z[1]}).reset_index()
        #print(df.columns)
        df_p = pareto(df, [x, y], minimizeObjective1 = x == "features", minimizeObjective2 = y == "features").sort_values(x)


        ax.scatter(df[x], df[y], label=parameters.classifier_titles[classifier], alpha=0.3, clip_on=False)
        ax.plot(df_p[x], df_p[y], marker="x")
        #display(df)

        print(classifier, x, "vs",  y, df_p.shape[0] )

        #print(df)
        df_pall = pareto(df, ["sensitivity", "specificity", "features"], minimizeObjective1 = False, minimizeObjective2 = False, minimizeObjective3=True)
        print(classifier, " vs. ".join(["sensitivity", "specificity", "features"]) , df_pall.shape[0] )

        pass

    plt_points(df_lastgen, "sensitivity", "specificity", ax1)
    plt_points(df_lastgen, "features", "specificity", ax2)
    plt_points(df_lastgen, "features", "sensitivity", ax3)


    #ax1.scatter(df_lastgen["sensitivity"], df_lastgen["specificity"], c=df_lastgen.index, alpha=0.2)
    #ax2.scatter(df_lastgen["features"], df_lastgen["specificity"], c=df_lastgen.index, alpha=0.2)
    #ax3.scatter(df_lastgen["features"], df_lastgen["sensitivity"], c=df_lastgen.index, alpha=0.2)


plt.figlegend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=4)
ax1.set(xlabel="Sensitivity", ylabel="Specificity", xlim=(0.85, 1), ylim=(0.85, 1))
ax2.set(xlabel="#features", ylabel="Specificity", xlim=(0, 80), ylim=(0.85, 1))
ax3.set(xlabel="#features", ylabel="Sensitivity", xlim=(0, 80), ylim=(0.85, 1))
plt.tight_layout(rect=(0, 0, 1, .9))

plt.savefig("plt/pareto.pdf")



# %%
