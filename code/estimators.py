# Estimators

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import sklearn
from model_nn import get_model
from tensorflow.contrib import keras
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Test classifier
class TestClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_, self.y_ = X, y
         # Return the classifier
        return self

    def predict(self, X):

        # Input validation
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


# NN classifier
class NN(BaseEstimator, ClassifierMixin):

    def __init__(self, epochs=1000, batch_size=64, extra_layer=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.extra_layer = extra_layer

    def fit(self, X, y):
        # Check that X and y have correct shape.
        X, y = check_X_y(X, y)
        # Store the classes seen during fit.
        self.classes_ = unique_labels(y)

        self.X_, self.y_ = X, y
        
        raise NotImplementedError("Haven't implemented NN model.")

        return self

    def predict_proba(self, X):

        # Input validation
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        # Convert X (list of matrices) to a matrix before sending to model.predict().
        predictions_class_1 = self.model.predict(X.reshape([-1, img_x, img_y, n_channels]))
        predictions_class_1_tranpose = predictions_class_1.reshape([-1, 1])
        preda = np.hstack([1-predictions_class_1_tranpose, predictions_class_1_tranpose])
        return preda


# XGBoost classifier
class XGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 eval_metric="auc", 
                 n_folds=None,
                 stratify=False,
                 params=None,
                 fit_params=None,
                 **kwargs):
        """
        @param n_folds: Number of folds on which to cross-validate. None to train with xgb.train instead of xgb.cv.
        @param stratify: True to preserve positive/negative ratio within folds.
        @param kwargs: Additional parameters. Currently unused, except to absorb parameter 'scoring'.
        """
        if params is None:
            self.params = {}
        else:
            self.params = params
        if fit_params is None:
            self.fit_params = {}
        else:
            self.fit_params = fit_params
        self.n_folds = n_folds
        self.stratify = stratify
        self.model, self.X_, self.y_, self.classes_ = None, None, None, None

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_, self.y_ = X, y

        xgtrain = xgb.DMatrix(self.X_, label=self.y_)

        kwargs = dict(params=self.params, 
                      dtrain=xgtrain, 
                      evals=[(xgtrain,'train')],
                      **self.fit_params)
        if self.n_folds is None:
            self.model = xgb.train(**kwargs)
        else:
            if stratify:
                stratifiedKFolds = stratifiedKFolds(self.y_, n_folds=self.n_folds)
                self.model = xgb.cv(folds=stratifiedKFolds, **kwargs)
            else:
                self.model = xgb.cv(nfold=self.n_folds, **kwargs)

        return self

    def predict_proba(self, X):

        # Input validation
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        xgtest = xgb.DMatrix(X)
        predictions_class_1 = self.model.predict(xgtest,ntree_limit=self.model.best_ntree_limit)
        predictions_class_1_tranpose = predictions_class_1.reshape([-1, 1])
        preda = np.hstack([1-predictions_class_1_tranpose, predictions_class_1_tranpose])
        return preda


def FoldsEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """An ensemble meta-estimator that fits base classifiers on specified
    subsets of the original dataset and then aggregating their individual
    predictions by averaging. Trains one base classifier per fold.
    """
    
    def __init__(self,
                 base_estimator,
                 folds):
        """
        @param base_estimator Should already have been initialised.
        @param folds Folds. (Not method that creates folds.)
        """
        self.base_estimator = base_estimator
        self.folds = folds

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_, self.y_ = X, y

        # TODO Check that can map like this in Python.
        # TODO Check that estimators have copy method.
        self._fitted_base_estimators = map(self.folds,
                                           self.base_estimator.copy().fit)

    def predict_proba(self, X):
        # Input validation
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        # Apply each fitted base estimator to X, and average results.
        # TODO Check that can use reduce like this. Probably different verb.
        return mean(reduce(X, lambda x: self._fitted_base_estimators(x)[:, 1]))

