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
from sklearn import cross_validation
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

        self.X_ = X
        self.y_ = y
         # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
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

        self.X_ = X
        self.y_ = y
        # self.X_ is an n x (128*128*3) matrices.
        # For CNN, we want an nx128x128x3 matrix.
        self.X_4Dmatrix = self.X_.reshape([-1, img_x, img_y, n_channels])

        datagen.fit(self.X_4Dmatrix)

        self.model = get_model(self.extra_layer)

        # Do the fit.
        steps_per_epoch = len(self.X_) / self.batch_size
        self.model.fit_generator(datagen.flow(self.X_4Dmatrix, self.y_, batch_size=self.batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            verbose=2)
        # Return the classifier
        return self

    def predict_proba(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
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
                 scoring=None):
        self.eval_metric = eval_metric
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.gamma = gamma
        self.model, self.X_, self.y_, self.classes_ = None, None, None, None
        
    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Center and scale.
        self.scaler = StandardScaler().fit(self.X_)
        X_scaled = self.scaler.transform(self.X_)
        # Return the classifier
        xgtrain = xgb.DMatrix(X_scaled, label=self.y_)

        params = {
            'eta': 0.05, #0.03
            'silent': 1,
            'verbose_eval': True,
            'verbose': False,
            'seed': 4
        }
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = self.eval_metric
        params['min_child_weight'] = self.min_child_weight
        params['cosample_bytree'] = 0.8
        params['cosample_bylevel'] = 0.9
        params['max_depth'] = self.max_depth
        params['subsample'] = 0.9
        params['max_delta_step'] = 10
        params['gamma'] = self.gamma
        params['alpha'] = 0
        params['lambda'] = 1

        watchlist = [(xgtrain,'train')]
        self.model = xgb.train(list(params.items()), xgtrain, 5000, watchlist,
                        early_stopping_rounds=25, verbose_eval = 50)

        return self

    def predict_proba(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        # Center and scale (same transformation as for train features).
        X_scaled = self.scaler.transform(X)
        xgtest = xgb.DMatrix(X_scaled_PCA)
        predictions_class_1 = self.model.predict(xgtest,ntree_limit=self.model.best_ntree_limit)
        predictions_class_1_tranpose = predictions_class_1.reshape([-1, 1])
        preda = np.hstack([1-predictions_class_1_tranpose, predictions_class_1_tranpose])
        return preda
