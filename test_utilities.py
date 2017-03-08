from __future__ import print_function, division

import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import base, pipeline, preprocessing
from sklearn import svm, linear_model, tree, ensemble, neighbors

import utilities


class TestUtilities(unittest.TestCase):

    def test_load_data(self):
        Xtrain, ytrain, Xtest, ytest = utilities.load_data()
        self.assertTrue(len(Xtrain) == len(ytrain))
        self.assertTrue(len(Xtest) == len(ytest))
        self.assertTrue(Xtrain.shape[1] == Xtest.shape[1])

    def test_evaluate(self):
        np.random.seed(123)
        truth = np.random.poisson(2.0, 100)
        prediction = np.ones(100) * 2.0
        plt.clf()
        utilities.evaluate(prediction, truth)

    def test_visualize(self):
        plt.clf()
        utilities.visualize(utilities.load_data()[-1])


    def test_RegressionOnSubset(self):
        Xtrain, ytrain, Xtest, ytest = utilities.load_data()
        columns = ['Longitude', 'Latitude']
        est = ensemble.RandomForestRegressor()
        est.fit(Xtrain, ytrain)
        predict_est = est.predict(Xtest)
        mad_est = np.mean(np.abs(predict_est - ytest))
        msd_est = np.mean(np.square(predict_est - ytest))

        meta_est = utilities.RegressionOnSubset(est, columns)
        pipe = pipeline.Pipeline([('RegressionOnSubest', meta_est),
                                  ('Regression', ensemble.RandomForestRegressor())])
        pipe.fit(Xtrain, ytrain)
        predict_pipe = pipe.predict(Xtest)
        mad_pipe = np.mean(np.abs(predict_pipe - ytest))
        msd_pipe = np.mean(np.square(predict_pipe - ytest))
        self.assertTrue(mad_pipe < mad_est)
        self.assertTrue(msd_pipe < msd_est)




if __name__ == "__main__":
    unittest.main()
