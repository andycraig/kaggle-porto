#!/usr/bin/env python3

import os, sys, yaml, pickle
import pandas as pd
import numpy as np
from sklearn import cross_validation, svm
from sklearn.ensemble import BaggingClassifier
from utils import datetime_for_filename
from estimators import NN, XGBoost, TestClassifier

model_dict = {'nn':NN, 'xgboost':XGBoost, 'svm':(lambda x: svm.SVC(x, probability=True))}

def main(config_file, model_name, fold):
    print('config file: ' + config_file)
    print('model_name: ' + model_name)
    print('fold: ' + str(fold))

    with open(config_file, 'r') as f:
        config = yaml.load(f)
    with open(config['hyperparams_file'], 'r') as f:
        hyperparams = yaml.load(f)

    # Load data.
    print('Loading data...')
    train_df = pd.read_csv('../generated-files/train.csv')
    test_df = pd.read_csv('../generated-files/test.csv')

    model = model_dict[model_name](**hyperparams[model_name])
    model_col_name = 'model_' + model_name
    
    if fold != None:
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
    else:
        print('Fitting...')
        model.fit(X=train_df.loc[:, [x for x in train_df.columns if x != 'target']], 
                  y=train_df.loc[:, 'target'])
        # Add predictions for whole test set to test CSV.
        print("Predicting...")
        test_df.assign(model_col_name=model.predict_proba(test_df)[:,1])
        test_set.to_csv('../generated-files/test.csv', index=None)
        print('Added predictions for model ' + model_name + ' to column ' + model_col_name + ' of ../generated-files/test.csv.')

if __name__ == "__main__":
    if len(sys.argv) == 4:
        main(sys.argv[1], model_name=sys.argv[2], fold=int(sys.argv[3]))
    elif len(sys.argv) == 3:
        main(sys.argv[1], model_name=sys.argv[2], fold=None)
    else:
        print("First argument: config file; second argument: model name; third argument (optional): fold number.")
