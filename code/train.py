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
from estimators import NN, XGBoost, TestClassifier, StratifiedBaggingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

float_format = '%.8f'

def gini_scoring_fn(estimator, scoring_X, scoring_y):
    preds = estimator.predict_proba(scoring_X)[:, 1]
    return eval_gini(y_true=scoring_y, y_prob=preds)

def main(config_file, model_name, fit_hyperparams, fold, submission, cv):
    print('Config file: ' + config_file)
    print('Model: ' + model_name)
    print('Fit hyperparams? ' + str(fit_hyperparams))
    print('Fold for which predictions will be added: ' + str(fold))
    print('Generate submission file? ' + str(submission))
    print('Cross-validate? ' + str(cv))

    with open(config_file, 'r') as f:
        config = yaml.load(f)
    with open(config['hyperparams_file'], 'r') as f:
        hyperparams = yaml.load(f)
        
    # Load data.
    print('Loading data...')
    train_df = pd.read_pickle(config['train'])
    test_df = pd.read_pickle(config['test'])

    # The model names and their definitions.
    model_dict = {'nn':NN, 
                  'nnBagged':toolz.partial(StratifiedBaggingClassifier,
                                           base_estimator=NN(**hyperparams['nn']['constructor']),
                                           fit_params=hyperparams['nn']['fit']),
                  'xgbBagged':toolz.partial(StratifiedBaggingClassifier,
                                            base_estimator=XGBClassifier(**hyperparams['xgb']['constructor']),
                                            fit_params=hyperparams['xgb']['fit']),
                  'xgb':toolz.partial(XGBClassifier),
                  'xgbHist':toolz.partial(XGBoost),
                  'xgbStratified':toolz.partial(XGBoost, stratify=True),
                  'svm':toolz.partial(svm.SVC, probability=True),
                  'randomForest':toolz.partial(RandomForestClassifier),
                  'logisticRegression':toolz.partial(LogisticRegression, class_weight='balanced'),
                  'logisticRegressionBagged':toolz.partial(StratifiedBaggingClassifier,
                                            base_estimator=LogisticRegression(**hyperparams['logisticRegression']['constructor']),
                                            fit_params=hyperparams['logisticRegression']['fit'])}

    if fit_hyperparams:
        # Define model.
        print('Define model...')
        # Define model with only non-tuning parameters, as tuning parameters will
        # be adjusted in GridSearchCV.
        constructor_hyperparams = hyperparams[model_name]['constructor']
        tuning_hyperparams = hyperparams[model_name]['tuning_hyperparams']
        non_tuning_hyperparams = {x:constructor_hyperparams[x] for x in constructor_hyperparams if not x in tuning_hyperparams}
        model = model_dict[model_name](**non_tuning_hyperparams)

        print('Finding hyperparameters...')
        n_splits = 2
        # Construct distributions from tuning_hyperparams.
        param_dists = {}
        for param in tuning_hyperparams:
            vals = tuning_hyperparams[param]['vals']
            min = np.min(vals)
            max = np.max(vals)
            print("For param " + param + ", fitting over range: " + str(min) + ", " + str(max))
            if tuning_hyperparams[param]['type'] == 'int':
                param_dists[param] = randint(min, max + 1) # randint is like [min, max).
            elif tuning_hyperparams[param]['type'] == 'float':
                param_dists[param] = uniform(loc=min, scale=(max - min))
            else:
                raise ValueError("Unexpected tuning parameter type: " + str(tuning_hyperparams[param]['type']))
        clf = RandomizedSearchCV(model, param_distributions=param_dists,
                                    n_iter=10, scoring='roc_auc', verbose=5, n_jobs=3, cv=n_splits)
        X = train_df.drop(['target', 'fold'], axis=1)
        y = train_df.loc[:, 'target']
        clf.fit(X=X, y=y, **hyperparams[model_name]['fit'])
        print('Found best hyperparams:')
        print(clf.best_params_)
        print('With AUC score:')
        print(clf.best_score_)

        # Put grid search best params in hyperparams dict.
        # Floats are in numpy format, and trying to write them as-is to file
        # causes it to be filled with junk, so convert to normal float first if
        # necessary.
        for param, value in clf.best_params_.items():
            try:
                sanitised_value = value.item() # Gets number from numpy class.
            except AttributeError as e: # Was plain number anyway.
                sanitised_value = value
            hyperparams[model_name]['constructor'][param] = sanitised_value
        # Save hyperparams.
        with open(config['hyperparams_file'], 'w') as f:
            yaml.dump(hyperparams, f, default_flow_style=False, indent=2)
        print('Wrote best params to ' + str(config['hyperparams_file']))

    if cv: # Cross-validate model to estimate accuracy.
            # Define model.
            print('Define model...')
            model = model_dict[model_name](**hyperparams[model_name]['constructor'])
            X = train_df.drop(['target', 'fold'], axis=1)
            y = train_df.loc[:, 'target']
            n_splits = 3
            fit_params = hyperparams[model_name]['fit']
            print("Estimating scores using cross-validation...")
            scores = cross_val_score(estimator=model, X=X, y=y, cv=n_splits, verbose=5, fit_params=fit_params, scoring=gini_scoring_fn, n_jobs=1)
            # Report error.
            print('Gini score mean (standard deviation): ' + str(np.mean(scores)) + ' (' +  str(np.sqrt(np.var(scores))) + ')')

    if submission: # Train and produce submission file.
        # Define model.
        print('Define model...')
        model = model_dict[model_name](**hyperparams[model_name]['constructor'])
        print('Fitting...')
        model.fit(X=train_df.drop(['target', 'fold'], axis=1),
                  y=train_df.loc[:, 'target'])
        # Create submission file with predictions.
        print("Predicting...")
        submit_file = config['submit_prefix'] + '_' + model_name + '_' + datetime_for_filename() + '.csv'
        (test_df
         .assign(target=model.predict_proba(test_df.drop('id', axis=1))[:,1])
         .loc[:, ['id', 'target']]
         .to_csv(submit_file, float_format=float_format))
        print("Saved submit file to " + submit_file)
    elif not fold is None: # Train with folds, for stacking.
        # Define model.
        print('Define model...')
        model = model_dict[model_name](**hyperparams[model_name]['constructor'])
        model_col_name = 'model_' + model_name
        if fold != -1: # Fit for a specific fold.
            print('Fitting...')
            model.fit(X=train_df.loc[train_df['fold'] != fold, [x for x in train_df.columns if x != 'target']], 
                      y=train_df.loc[train_df['fold'] != fold, 'target'],
                      **(hyperparams[model_name]['fit']))
            # Add predictions for fold.
            print("Predicting...")
            if not model_col_name in train_df:
                train_df = train_df.assign(model_col_name=np.nan)
            train_df.loc[train_df['fold'] == fold, model_col_name] = model.predict_proba(train_df.loc[train_df['fold'] == fold, :])[:,1]
            train_df.to_pickle(config['train'])
            print('Added predictions for model ' + model_name + ', fold ' + str(fold) + ' to column ' + model_col_name + ' of ' +  config['train'])
        else: # Ignore folds and fit all data.
            print('Fitting...')
            model.fit(X=train_df.drop('target', axis=1), 
                      y=train_df.loc[:, 'target'])
            # Add predictions for whole test set to test CSV.
            print("Predicting...")
            test_file = config['test']
            (test_df
             .assign(model_col_name=model.predict_proba(test_df)[:,1])
             .to_pickle(test_file))
            print('Added predictions for model ' + model_name + ' to column ' + model_col_name + ' of ' + test_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit model.')
    parser.add_argument('config', help='name of config file')
    parser.add_argument('model', choices=['nn', 'nnBagged', 'xgb', 'xgbStratified', 'xgbBagged', 'svm', 'logisticRegression', 'logisticRegressionBagged', 'xgbHist', 'randomForest'], help='model to fit')
    parser.add_argument('--hyperparams', action='store_true', help='fit hyperparameters instead of training model')
    parser.add_argument('--cv', action='store_true', help='cross-validate file and estimate accuracy')
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument('--fold', default=None, type=int, help='fold for which values will be predicted and added. Set to negative to train on all folds and add to test')
    g.add_argument('--sub', action='store_true', help='fit model and produce submission file')
    args = parser.parse_args()
    if (not args.hyperparams) and (not args.cv) and (not args.sub) and (args.fold is None):
        print("At least one of --hyperparams, --cv, --fold or --sub must be specified")
    else:
        main(config_file=args.config, model_name=args.model, fit_hyperparams=args.hyperparams, fold=args.fold, submission=args.sub, cv=args.cv)
