import os, sys, yaml
import pandas as pd
import numpy as np
from utils import datetime_for_filename
from sklearn.linear_model import LogisticRegression

def main(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    # Load file of training image names and correct labels.
    train_df = pd.read_pickle(config['train'])
    # Load file of test image names and dummy labels.
    test_df = pd.read_pickle(config['test'])

    # Fit stacking model from model columns.
    model_cols = [x for x in test_df.columns if x.startswith('model_')]
    S = LogisticRegression()
    S.fit(X=train_df.loc[:,model_cols], y=train_df['target'].values)
    # Make predictions based on model columns of test set.
    test_df.loc[:, 'invasive'] = S.predict_proba(X=test_df.loc[:,model_cols])[:,1]
    # Write these predictions to submit file.
    submit_file = config['submit_prefix'] + '_stack_' + datetime_for_filename() + '.csv'
    test_df[['id', 'target']].to_csv(submit_file)
    # Some reporting.
    print("Stacking model parameters:")
    print("Intercept:" + str(S.intercept_) + " Cofficients: " + str(S.coef_))
    print("Saved submit file to " + submit_file)

if __name__ == "__main__":
    main(sys.argv[1])
