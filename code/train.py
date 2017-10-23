#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import os, sys, yaml, pickle
import pandas as pd
import numpy as np
from sklearn import cross_validation, svm
from sklearn.ensemble import BaggingClassifier
from utils import datetime_for_filename
from estimators import NN, XGBoost, TestClassifier
#import warnings
#warnings.filterwarnings('ignore')

def main(config_file, i_model, fold):
    print('config_file: ' + str(config_file))
    print('i_model: ' + str(i_model))
    print('fold: ' + str(fold))

    with open(config_file, 'r') as f:
        config = yaml.load(f)
    with open(config['hyperparams_file'], 'r') as f:
        hyperparams = yaml.load(f)

    # Load file of training image names and correct labels.
    train_set = pd.read_csv(config['train_set'])
    train_labels = train_set['invasive'].values
    # Load file of test image names and dummy labels.
    test_set = pd.read_csv(config['test_set'])

    print('Loading features...')
    if i_model == 'nn':
        print('Using NN model.')
        model = NN(**hyperparams['nn'])
        with open(config['train_features_nn'], 'rb') as f:
            train_features = pickle.load(f)
        with open(config['test_features_nn'], 'rb') as f:
            test_features = pickle.load(f)
    elif i_model == 'xgboost':
        print('Using XGBoost model.')
        model = XGBoost(**hyperparams['gbt'])
        train_features = pd.read_csv(config['train_features_gbt'], header=None)
        test_features = pd.read_csv(config['test_features_gbt'], header=None)
    elif i_model == 'svm':
        print('Using SVM model.')
        model = svm.SVC(probability=True, **hyperparams['svm'])
        train_features = pd.read_csv(config['train_features_gbt'], header=None)
        test_features = pd.read_csv(config['test_features_gbt'], header=None)
    else:
        model = TestClassifier()
        train_features = pd.read_csv(config['train_features_gbt'], header=None)
        test_features = pd.read_csv(config['test_features_gbt'], header=None)

    if fold != None:
        mask_fold_val = np.array(train_set['fold'] == fold)
        mask_fold_train = ~mask_fold_val
    else:
        mask_fold_train = np.ones(len(train_labels), dtype=bool)

    print('Fitting...')
    n_estimators = config['n_bagging_estimators']
    max_samples = 1.0 * (n_estimators - 1) / n_estimators
    clf = BaggingClassifier(model, n_estimators=n_estimators, max_samples=max_samples)
    #clf = model
    clf.fit(X=train_features[mask_fold_train], y=train_labels[mask_fold_train])

    print("Predicting...")
    model_col_name = 'M' + str(i_model)
    # If training on a fold, add predictions for this fold only to train CSV.
    if fold != None:
        # Get predictions for probability of class 1 membership.
        predictions = clf.predict_proba(train_features[mask_fold_val])[:,1]
        # If there isn't yet a column for the model's predictions, add one.
        # Warns about setting on copy of a slice, but seems to work.
        if not model_col_name in train_set:
            train_set[model_col_name] = np.nan
        train_set[model_col_name].loc[mask_fold_val] = predictions
        train_set.to_csv(config['train_set'], index=None)
        print('Added predictions for model ' + str(i_model) + ', fold ' + str(fold) + ' to column ' + model_col_name + ' of ' + config['train_set'])
    else:
        # If training on whole training set, add predictions for whole test set to test CSV.
        predictions = clf.predict_proba(test_features)[:,1]
        # If there isn't yet a column for the model's predictions, add one.
        # Warns about setting on copy of a slice, but seems to work.
        if not model_col_name in test_set:
            test_set[model_col_name] = np.nan
        test_set[model_col_name].loc[:] = predictions
        test_set.to_csv(config['test_set'], index=None)
        print('Added predictions for model ' + str(i_model) + ' to column ' + model_col_name + ' of ' + config['test_set'])

if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], int(sys.argv[2]), fold=int(sys.argv[3]))
    else:
        main(sys.argv[1], int(sys.argv[2]), fold=None)
