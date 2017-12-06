"""Wrapper for loading data, training a model (possibly on specified folds),
and writing out predictions.
"""

import os, sys, yaml, pickle, argparse
import pandas as pd
import numpy as np
import toolz
from sklearn import svm
from scipy.stats import randint, uniform
from utils import datetime_for_filename, eval_gini
from xgboost import XGBClassifier
from estimators import NN, XGBoostWrapper, TestClassifier, StratifiedBaggingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

float_format = '%.8f'

def gini_scoring_fn(estimator, scoring_X, scoring_y):
    preds = estimator.predict_proba(scoring_X)[:, 1]
    return eval_gini(y_true=scoring_y, y_prob=preds)

with open('config.yaml', 'r') as f:
    config = yaml.load(f)
with open(config['hyperparams_file'], 'r') as f:
    hyperparams = yaml.load(f)

# Load data.
print('Loading data...')
train_df = pd.read_pickle(config['train'])
test_df = pd.read_pickle(config['test'])

# Define model.
print('Define model...')
model = XGBClassifier(**hyperparams['xgb']['constructor'])
X = train_df.drop(['target', 'fold'], axis=1)
y = train_df.loc[:, 'target']
fit_params = hyperparams['xgb']['fit']
n_splits = 3
kf = StratifiedKFold(n_splits=n_splits)
# CV with and without early stopping.
for early_stopping, early_stopping_string in zip([True, False], ['early stopping', 'no early stopping']):
    scores = np.empty(n_splits)
    for i_fold, (fold_train_index, fold_test_index) in enumerate(kf.split(X, y)):
        print("Training for " + early_stopping_string + " fold " + str(i_fold + 1) + "/" + str(n_splits))
        # Split train_index into train set and eval set for early stopping.
        fold_X = X.loc[fold_train_index, :]
        fold_y = y[fold_train_index]
        X_train, X_test, y_train, y_test = train_test_split(fold_X, fold_y, test_size=0.2, stratify=fold_y) 
        # eval_set: A list of (X, y) pairs to use as a validation set for early stopping 
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], **fit_params)
        if early_stopping:
            # Get best iteration based on eval_set.
            sorted_iteration_scores = np.argsort(model.evals_result()['validation_0']['auc'])
            best_round = sorted_iteration_scores[-1]
            # Evaluate on test_index.
            proba = model.predict_proba(X.loc[fold_test_index, :], ntree_limit=best_round)[:, 1]
        else:
            proba = model.predict_proba(X.loc[fold_test_index, :])[:, 1]
        y_true = y[fold_test_index]
        scores[i_fold] = eval_gini(y_true, proba)
    # Report error.
    print('For ' + early_stopping_string + ', Gini score mean (standard deviation): ' + str(np.mean(scores)) + ' (' +  str(np.sqrt(np.var(scores))) + ')')

