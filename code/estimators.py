# Estimators

import numpy as np
import pandas as pd
import toolz
from functools import reduce
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import sklearn
from model_nn import get_model
import tensorflow.contrib.keras as keras
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

    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size
 
    def fit(self, X, y):

        # Check that X and y have correct shape.
        X, y = check_X_y(X, y)
        # Store the classes seen during fit.
        self.classes_ = unique_labels(y)
        self.X_, self.y_ = X, y

        # NN object.
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(128, activation='relu', input_dim=self.X_.shape[1]))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        
        # Construct NN.
       # Need targets to be binary matrix of shape (samples, classes).
        self.model.fit(self.X_, self.y_, epochs=self.epochs, batch_size=self.batch_size)
        return self

    def predict_proba(self, X):
        # Input validation
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        preda_pos = self.model.predict(X, batch_size=128)
        # predict_proba needs to return predictions for both negative and positive.
        preda = np.hstack([(1 - preda_pos), preda_pos])
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


class StratifiedBaggingClassifier(BaseEstimator, ClassifierMixin):
    """An ensemble meta-estimator that fits base classifiers to
    subsets of the original dataset, with proportions in each label
    class preserved, and then aggregating their individual
    predictions by averaging. Trains one base classifier per subset.
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 fit_params=None):
        """
        @param base_estimator Should already have been initialised.
        @param n_estimators How many base estimators to use.
        @param fit_params Dictionary of parameters to be passed to base estimator fit method.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        if fit_params is None:
            self.fit_params = {}
        else:
            self.fit_params = fit_params

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_, self.y_ = X, y

        # Take samples of X and y, maintaining proportions in y.
        n_classes = len(self.classes_)
        indices_of_classes = [np.where(y == z)[0] for z in self.classes_] # Need [0] because np.where returns a tuple.
        # TODO This fails without list when it gets to the np.random.choice line, but seems like it should be possible without list.
        n_in_each_class = list(map(len, indices_of_classes))
        # Take subsets of the data,
        # maintaining the proportion of y classes,
        (np.random.choice(a, b) for a, b in zip(indices_of_classes, n_in_each_class))
        indices_for_estimators = [list(toolz.itertoolz.concat(np.random.choice(a, b) for a, b in zip(indices_of_classes, n_in_each_class))) for _ in range(self.n_estimators)]
        # Fit base estimators.
        self._fitted_base_estimators = [sklearn.base.clone(self.base_estimator).fit(self.X_[sample_indices, :], self.y_[sample_indices], **self.fit_params)
                                           for sample_indices in indices_for_estimators]

    def predict_proba(self, X):
        # Input validation
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        # Apply each fitted base estimator to X, and average results.
        # TODO Check that sum is applied over the right axis.
        proba =  reduce(np.add, (z.predict_proba(X) for z in self._fitted_base_estimators)) / self.n_estimators
        return proba
