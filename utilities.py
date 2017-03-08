from __future__ import print_function, division

import numpy as np
import pandas as pd

from sklearn import base, pipeline, preprocessing
from sklearn import svm, linear_model, tree, ensemble, neighbors

import matplotlib.pyplot as plt


def load_data():
    df_file = pd.read_csv("california_housing.csv")
    Xtrain = df_file[df_file["sample_id"] == 0].copy()
    ytrain = np.asarray(Xtrain["y"]).copy()
    del(Xtrain["sample_id"], Xtrain["y"])
    Xtest = df_file[df_file["sample_id"] == 1].copy()
    ytest = np.asarray(Xtest["y"]).copy()
    del(Xtest["sample_id"], Xtest["y"])
    return Xtrain, ytrain, Xtest, ytest


def evaluate(prediction, truth):
    if isinstance(prediction, np.ndarray):
        p = plt.hist(prediction, bins=50, color="g", label='Vorhersage')
    else:
        p = plt.bar(prediction, 250, width=0.125, color="g", label='Vorhersage')
    t = plt.hist(truth, bins=50, color="b", label='Wahrheit')
    plt.ylabel("Anzahl")
    plt.xlabel("logarithmierter Hauspreis.")
    plt.legend()
    print("Mittlere absolute Abweichung: {}".format(np.mean(np.abs(prediction - truth))))
    print("Mittlere quadratische Abweichung: {}".format(np.mean(np.square(prediction - truth))))


def visualize(geo_prediction):
    Xtrain, ytrain, Xtest, ytest = load_data()

    x1 = np.asarray(Xtest['Longitude'])
    x2 = np.asarray(Xtest['Latitude'])

    from matplotlib import colors
    cm = plt.cm.get_cmap('RdYlBu')

    sc = plt.scatter(x1, x2, c=geo_prediction, s=20, vmin=0, vmax=5)
    plt.colorbar(sc)
    plt.show()


class RegressionOnSubset(base.BaseEstimator):
    """Sklearn style meta-estimator that allows to train
    on a subset of the data and adds the prediction gained
    from this training to the DataFrame to be used as a feature.

    :param subest: subestimator used for the training
    :type subest: sklearn estimator

    :param columns: column identifier(s) used for slicing the DataFrame
    :type columns: list of str
    """
    def __init__(self, subest, columns):
        self.columns = columns
        self.subest = subest

    def fit(self, X, y):
        Xtemp = X[self.columns] # Slice of the columns the subest should fit on
        self.subest.fit(Xtemp, y) # Fit the subestimator!
        return self

    def transform(self, X):
        Xtemp = X[self.columns] # Slice of the columns the subest uses to predict
        X['knearest'] = self.subest.predict(Xtemp) # Our new feature is the prediction of our subest
        for col in self.columns: # We delete the old features that are not needed anymore
            del X[col]
        return X
