#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import os, sys, yaml
import pandas as pd
import numpy as np
from utils import datetime_for_filename
from sklearn.linear_model import LogisticRegression

def main(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    # Load file of training image names and correct labels.
    train_set = pd.read_csv(config['train_set'])
    train_labels = train_set['invasive'].values
    # Load file of test image names and dummy labels.
    test_set = pd.read_csv(config['test_set'])

    # Fit stacking model predicting 'invasive' from model columns.
    # Model predictions are those in columns beginning with 'M'.
    model_cols = [x for x in test_set.columns if x[0:6] == 'model_']
    S = LogisticRegression()
    S.fit(X=train_set.loc[:,model_cols], y=train_labels)
    # Make predictions based on model columns of test set.
    predictions = S.predict_proba(X=test_set.loc[:,model_cols])[:,1]
    test_set['invasive'] = predictions
    # Write these predictions to submit file.
    submit_file = config['submit_prefix'] + '_' + datetime_for_filename() + '.csv'
    test_set[['name', 'invasive']].to_csv(submit_file, header=True, index=None)
    # Some reporting.
    print("Stacking model parameters:")
    print("Intercept:" + str(S.intercept_) + " Cofficients: " + str(S.coef_))
    print("Saved submit file to " + submit_file)

if __name__ == "__main__":
    main(sys.argv[1])
