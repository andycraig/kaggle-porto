"""Wrapper for loading data, training a model (possibly on specified folds),
and writing out predictions.
"""

import os, sys, yaml, pickle, argparse
import pandas as pd
import numpy as np
import toolz
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from utils import datetime_for_filename
from estimators import NN, XGBoost, TestClassifier
from sklearn.model_selection import GridSearchCV

# The model names and their definitions.
model_dict = {'nn':NN, 
              'xgb':toolz.partial(XGBoost, stratify=False),
              'xgbStratified':toolz.partial(XGBoost, stratify=True),
              'svm':toolz.partial(svm.SVC, probability=True)}

tuning_hyperparams = {'xgb':{'min_child_weight': [3, 5, 7], 'max_depth': [5,  6,  7], 'gamma': [1.5, 2, 2.5] },
                    'svm':{'gamma': [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'C': [00.1, 0.1, 1, 10, 100, 1000]}}


def main(config_file, model_name, fit_hyperparams, fold):
    print('Config file: ' + config_file)
    print('Model: ' + model_name)
    print('Fit hyperparams: ' + str(fit_hyperparams))
    print('Fold for which predictions will be added: ' + str(fold))

    with open(config_file, 'r') as f:
        config = yaml.load(f)
    with open(config['hyperparams_file'], 'r') as f:
        hyperparams = yaml.load(f)
        
    # Load data.
    print('Loading data...')
    train_df = pd.read_csv('../generated-files/train.csv')
    test_df = pd.read_csv('../generated-files/test.csv')

    if fit_hyperparams:
        # Define model.
        print('Define model...')
        # Define model with only non-tuning parameters, as tuning parameters will
        # be adjusted in GridSearchCV.
        all_hyperparams = hyperparams[model_name]
        non_tuning_hyperparams = {x:all_hyperparams[x] for x in all_hyperparams if not x in non_tuning_hyperparams}
        model = model_dict[model_name](**non_tuning_hyperparams)

        print('Finding hyperparameters...')
        clf = GridSearchCV(model, tuning_hyperparams, cv=5,
                    scoring=hyperparams[model_name]['scoring'])
        clf.fit(train_features, train_labels)
        print('Found best hyperparams:')
        print(clf.best_params_)

        # Put grid search best params in hyperparams dict.
        for key in clf.best_params_:
            hyperparams[model_name] = clf.best_params_[key]
        # Save hyperparams.
        with open(config['hyperparams_file'], 'w') as f:
            yaml.dump(hyperparams, f)
        print('Wrote best params to ' + str(config['hyperparams_file']))
    else:
        # Define model.
        print('Define model...')
        model = model_dict[model_name](**hyperparams[model_name])
        model_col_name = 'model_' + model_name
        if fold != -1: # Fit for a specific fold.
            print('Fitting...')
            model.fit(X=train_df.loc[train_df['fold'] != fold, [x for x in train_df.columns if x != 'target']], 
                    y=train_df.loc[train_df['fold'] != fold, 'target'])
            # Add predictions for fold.
            print("Predicting...")
            if not model_col_name in train_df:
                train_df = train_df.assign(model_col_name=np.nan)
            train_df.loc[train_df['fold'] == fold, model_col_name] = model.predict_proba(train_df.loc[train_df['fold'] == fold, :])[:,1]
            train_df.to_csv(config['train_set'], index=None)
            print('Added predictions for model ' + model_name + ', fold ' + str(fold) + ' to column ' + model_col_name + ' of ../generated-files/train.csv.')
        else: # Ignore folds and fit all data.
            print('Fitting...')
            model.fit(X=train_df.loc[:, [x for x in train_df.columns if x != 'target']], 
                    y=train_df.loc[:, 'target'])
            # Add predictions for whole test set to test CSV.
            print("Predicting...")
            test_df.assign(model_col_name=model.predict_proba(test_df)[:,1])
            test_set.to_csv('../generated-files/test.csv', index=None)
            print('Added predictions for model ' + model_name + ' to column ' + model_col_name + ' of ../generated-files/test.csv.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit model.')
    parser.add_argument('config', help='name of config file')
    parser.add_argument('model', choices=['nn', 'xgb', 'xgbStratified', 'svm'], help='model to fit')
    parser.add_argument('--hyperparams', action='store_true', help='fit hyperparameters instead of training model')
    parser.add_argument('--fold', default=-1, type=int, help='fold for which values will be predicted and added')
    args = parser.parse_args()
    if args.hyperparams and args.fold != -1:
        print("Can't specify both --hyperparams and --fold.")
    else:
        main(config_file=args.config, model_name=args.model, fit_hyperparams=args.hyperparams, fold=args.fold)
