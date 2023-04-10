"""
This classifier selects the features


1) channel
2) starting frequency and length
3) function:
 0: use all data as feature
 1,2,3: downsample them by 2, 4, 8
 4: aggregate: min, max, avg, max-min
"""
# %%
from __future__ import annotations
import gzip
import pickle
import random
import numpy as np
from tqdm import tqdm
from paretoarchive import PyBspTreeArchive

from galib.logger import Logger
from galib import ChromosomeBase, ChromosomeChannels, FeaturesException, GAops, DataParser

import sklearn.neighbors
import sklearn.linear_model
import sklearn.tree
import sklearn.svm
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix

# configuration which type of chromosome use
Chromosome = ChromosomeChannels
#Chromosome = ChromosomeBase


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


def run_nsga(p_size=50, q_size=50, generations=1000, output_file=None, classifier=None, seed=None, log=None):
    if not output_file:
        raise ValueError("No output file was set!")
    # Data loading
    data_parser = DataParser()
    logger = Logger(log)

    # Evaluate the chromosome (simulate training and testing)

    def evaluate_chromosome(chrom: ChromosomeBase) -> float:
        if not chrom.parameters:
            data = data_parser.get_data()
            X = data["x"]
            y = data["y"]
            samples_id = data["id"]
            freqs = data["freqs"]
            channels = data["features"]

            if classifier == "kneighbors":
                model = sklearn.neighbors.KNeighborsClassifier(2)
            elif classifier == "ridge":
                model = sklearn.linear_model.RidgeClassifier(0.2)
            elif classifier == "svm":
                model = sklearn.svm.SVC()
            elif classifier == "dt":
                model = sklearn.tree.DecisionTreeClassifier(random_state=42)
            else:
                raise NotImplemented(f"Unknown classifier {classifier}")

            try:
                cv = cross_validate(model, chrom.execute(X), y,
                                    groups=samples_id, cv=GroupKFold(5),
                                    scoring=sas_score, n_jobs=-1
                                    )
            except FeaturesException as e:
                print("Features exception: {} for chromosome {}".format(e, chrom))
                cv = {
                    "test_accuracy": np.array([0.0]),
                    "test_specificity": np.array([0.0]),
                    "test_sensitivity": np.array([0.0])
                }

            aggfunc = np.mean

            score = {

                "accuracy": aggfunc(cv["test_accuracy"]),
                "sensitivity": aggfunc(cv["test_sensitivity"]),
                "specificity": aggfunc(cv["test_specificity"]),
                "features": chrom.features,
                **cv
            }

            chrom.parameters = score
        return chrom.parameters

    def evaluate_population(population):
        for chrom in population:
            evaluate_chromosome(chrom)

    def pareto_filter(population):
        if not population:
            return []
        # print(  [(p.parameters["features"], p.parameters["accuracy"], p.parameters["sensitivity"], p.parameters["specificity"]) for p in population
        #       ])

        try:
            return PyBspTreeArchive(3, minimizeObjective1=True, minimizeObjective2=False, minimizeObjective3=False).filter(
                [(p.parameters["features"], p.parameters["sensitivity"], p.parameters["specificity"]) for p in population

                 ], returnIds=True
            )
        except Exception as e:
            print()
            print([(i, p.parameters["features"], p.features, str(p))
                  for i, p in enumerate(population) if p.parameters["features"] is None])
            raise

    def crowding_distance(par, objs=["features", "sensitivity", "specificity"]):
        dst = np.zeros(len(par), dtype="f")
        for o in objs:
            values = np.array([p.parameters[o] for p in par])
            sort_id = np.argsort(values)
            sort_values = values[sort_id]

            distances = sort_values[1:] - sort_values[:-1]
            distances = np.concatenate(
                [[np.infty], distances[:-1], [np.infty]])

            vmin, vmax = values.max(), values.min()
            if vmax != vmin:
                distances /= vmax - vmin
            else:
                distances[~np.isinf(distances)] = 0

            #print(dst[sort_id], distances)
            dst[sort_id] += np.abs(distances)

        return par, distances

    def crowding_reduce(par, number):
        par = par
        while len(par) > number:
            _, dst = crowding_distance(par)
            to_delete = np.argmin(dst)
            par.pop(to_delete)
        return par

    if seed:
        random.seed(seed)

    data = data_parser.get_data()
    ga_ops = GAops(channels=data["features"], freqs=data["freqs"])

    # Limit the frequency to 50 Hz
    ga_ops.limit_frequency(50)

    parent_pop = [Chromosome(ga_ops).random() for _ in range(p_size)]
    evaluate_population(parent_pop)

    pbar = tqdm(range(generations))

    limit_acc = 0.1  # sens/spec limit

    for gen in pbar:

        offspring_pop = []
        for _ in range(q_size + p_size - len(parent_pop)):
            c = Chromosome(ga_ops).crossover(
                random.choice(parent_pop), random.choice(parent_pop)
            ).mutate(random.randint(0, 3))
            offspring_pop.append(c)
        evaluate_population(offspring_pop)

        new_population = offspring_pop + parent_pop

        #print("newpopsize", len(new_population))
        parent_pop = []


        while len(parent_pop) < p_size:
            pareto_id = pareto_filter(new_population)

            current_pareto = [new_population[i] for i in pareto_id]
            missing = p_size - len(parent_pop)

            if(len(current_pareto) <= missing):
                parent_pop += current_pareto
                #print(gen, "parent", len(parent_pop))
            else:  # distance crowding
                parent_pop += crowding_reduce(current_pareto, missing)

            for i in reversed(sorted(pareto_id)):
                new_population.pop(i)

        # filter out parameters with low sensitivity / specificity

        np_sensitivity = np.array(
            [p.parameters["sensitivity"] for p in parent_pop])
        np_specificity = np.array(
            [p.parameters["specificity"] for p in parent_pop])
        np_features = np.array([p.parameters["features"] for p in parent_pop])

        good = (np_sensitivity >= limit_acc) & (np_specificity >= limit_acc)
        good2 = (np_sensitivity >= (limit_acc + 0.05)
                 ) & (np_specificity >= (limit_acc + 0.05))

        #print(limit_acc, good.sum(), good2.sum())

        # remove
        # print(len(parent_pop))
        for i in reversed(list(np.arange(p_size)[~good])):
            #    print("remove: ", i)
            parent_pop.pop(i)
        # print(len(parent_pop))


        logger.log_generation(
            gen,
            good.sum(),
            limit_acc,
            parent_pop
        )

        # increase the limit if lot of candidates satisfies the condition
        if good2.sum() > 0.3 * p_size:
            limit_acc += 0.05


        # remove equals
        parent_pop = list(set(parent_pop))

        pbar.set_description(
            f"Best {np_sensitivity[good].max():.2%}/{np_specificity[good].max():.2%}/{np_features[good].min():d} / of {len(parent_pop)} lim {limit_acc:.2%} (b:{good2.sum()})"
        )

        #print("gen", gen, len(parent_pop))

    print("Elapsed time:", pbar.format_dict["elapsed"])
    if output_file:
        print(pickle.dump(parent_pop, gzip.open(output_file, "w")))


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Run NSGA-II search for the features.')
    parser.add_argument('--p_size', default=30, type=int,
                        help='size of parent population')
    parser.add_argument('--q_size', default=30, type=int,
                        help='size of offspring population')
    parser.add_argument('--generations', default=100000,
                        type=int, help='total number of generations')

    parser.add_argument('--log', default=None,
                        type=str, help='log file (*.gz)')
    parser.add_argument("--classifier", type=str, help="Selected classifier")
    parser.add_argument('output_file', type=str, help='output file (*.pkl.gz)')
    args = parser.parse_args()

    print("Run arguments: ", vars(args))
    run_nsga(**vars(args))


if __name__ == "__main__":
    main()

# %%
