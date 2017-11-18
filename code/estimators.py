# Estimators

import sys
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

    def __init__(self, v=1):
        self.v = v

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

    def predict_proba(self, X):

        # Input validation
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        pred_one = [1 - self.v, self.v]
        y_pred = np.repeat(np.array([pred_one]), len(X), axis=0)
        return y_pred


# NN classifier
class NN(BaseEstimator, ClassifierMixin):

    def __init__(self, epochs, batch_size, hidden_layers, dropout):
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        if dropout < 0:
            self.dropout = None
        else:
            self.dropout = dropout
 
    def fit(self, X, y):

        # Check that X and y have correct shape.
        X, y = check_X_y(X, y)
        # Store the classes seen during fit.
        self.classes_ = unique_labels(y)
        self.X_, self.y_ = X, y

        upsample = True # Always do for now.
        if upsample:
            # Upsample the positives.
            upsample_indices = np.random.choice(np.where(y == 1)[0], # [0] is necessary because where returns a tuple in this case.
                                                size=len(y) - sum(y == 0),
                                                replace=True)
            extra_X = self.X_[upsample_indices, :]
            extra_y = self.y_[upsample_indices]
            X_for_training = np.vstack([self.X_, extra_X])
            y_for_training = np.hstack([self.y_, self.y_[upsample_indices]])
        else:
            X_for_training = self.X_
            y_for_training = self.y_
            
        # NN object.
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.hidden_layers[0], activation='relu', input_dim=self.X_.shape[1]))
        self.model.add(keras.layers.BatchNormalization())
        if not self.dropout is None:
            self.model.add(keras.layers.Dropout(self.dropout))
        for layer in self.hidden_layers[1:]:
            self.model.add(keras.layers.Dense(layer, activation='relu'))
            self.model.add(keras.layers.BatchNormalization())
            if not self.dropout is None:
                self.model.add(keras.layers.Dropout(self.dropout))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])
        
        # Construct NN.
       # Need targets to be binary matrix of shape (samples, classes).
        self.model.fit(X_for_training, y_for_training, epochs=self.epochs, batch_size=self.batch_size)
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
class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 eval_metric="auc", 
                 tree_method='auto',
                 learning_rate=0.3,
                 min_child_weight=1,
                 max_depth=6,
                 max_leaf_nodes=4,
                 gamma=0,
                 max_delta_step=0,
                 subsample=1,
                 colsample_bytree=1,
                 colsample_bylevel=1,
                 reg_alpha=0,
                 reg_lambda=1):
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.gamma = gamma
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.model, self.X_, self.y_, self.classes_ = None, None, None, None

    def fit(self, X, y, **kwargs):
        """
            @kwargs: Other arguments to be passed to xgb.train().
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_, self.y_ = X, y

        xgtrain = xgb.DMatrix(self.X_, label=self.y_)

        params = {  'eval_metric':self.eval_metric,
                    'tree_method':self.tree_method,
                    'eta':self.learning_rate,
                    'min_child_weight':self.min_child_weight,
                    'max_depth':self.max_depth,
                    'max_leaf_nodes':self.max_leaf_nodes,
                    'gamma':self.gamma,
                    'max_delta_step':self.max_delta_step,
                    'subsample':self.subsample,
                    'colsample_bytree':self.colsample_bytree,
                    'colsample_bylevel':self.colsample_bylevel,
                    'alpha':self.reg_alpha,
                    'lambda':self.reg_lambda}
                               
        self.model = xgb.train(params=params,
                               dtrain=xgtrain,
                               evals=[(xgtrain,'train')],
                               **kwargs)
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
