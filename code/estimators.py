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
                 min_child_weight=15, 
                 max_depth=4, 
                 gamma=1,
                 n_folds=None,
                 stratify=False,
                 **kwargs):
        """
        @param n_folds: Number of folds on which to cross-validate. None to train with xgb.train instead of xgb.cv.
        @param stratify: True to preserve positive/negative ratio within folds.
        @param kwargs: Additional parameters. Currently unused, except to absorb parameter 'scoring'.
        """
        self.params = {
            'eta':0.05,
            'silent':1,
            'verbose_eval':True,
            'verbose':False,
            'seed':4,
            'objective':'binary:logistic',
            'eval_metric':eval_metric,
            'min_child_weight':min_child_weight,
            'cosample_bytree':0.8,
            'cosample_bylevel':0.9,
            'max_depth':max_depth,
            'subsample':0.9,
            'max_delta_step':10,
            'gamma':gamma,
            'alpha':0,
            'lambda':1
        }
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
                      num_boost_round=5000, 
                      evals=[(xgtrain,'train')],
                      early_stopping_rounds=25, 
                      verbose_eval=50)
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
