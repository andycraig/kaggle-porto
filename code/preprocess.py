"""Add folds, scale features.
"""

import shutil, itertools, yaml, sys
import toolz
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import target_encode


with open('config.yaml', 'r') as f:
    config = yaml.load(f)

# --dummy command line option: produce dummy data only.
dummy_only = False
try:
    if sys.argv[1] == '--dummy':
        dummy_only = True
        print("Only produce dummy data.")
except IndexError as e:
    pass

if not dummy_only:

    np.random.seed(1031) # Month, date on which file was created.

    print("Loading train...")
    train = (pd.read_csv(config['train_original'])
            .drop('id', axis=1)) # Get rid of id so it doesn't get used accidentally for training.
    n_rows = len(train) 

    print("Loading test...")
    test = pd.read_csv(config['test_original'])

    # fold: close-to-equal numbers of each fold, in random order.
    fold_vals = np.random.permutation(list(toolz.take(n_rows, itertools.cycle(range(config['n_folds'])))))

    # Replace categorical variables with target_encode versions.
    print("Replacing categorical variables with target_encode versions...")
    cat_cols = [x for x in train.columns if '_cat' in x]
    for f in cat_cols:
        new_train_col, new_test_col = target_encode(trn_series=train[f], tst_series=test[f], target=train['target'],
                                                    min_samples_leaf=200, smoothing=10, noise_level=0)
        train = train.assign(**{f + '_avg': new_train_col})
        test = test.assign(**{f + '_avg': new_test_col})
    train = train.drop(cat_cols, axis=1)
    test = test.drop(cat_cols, axis=1)

    # Drop 'calc' variables, as they seem almost completely uncorrelated with anything.
    print("Dropping calc variables...")
    calc_cols = [x for x in train.columns if '_calc' in x]
    train = train.drop(calc_cols, axis=1)
    test = test.drop(calc_cols, axis=1)

    # Fit scaler before adding folds.
    print("Fitting scaler...")
    scaler = StandardScaler().fit(train.drop('target', axis=1))

    # scaler.transform returns a numpy array, so create a wrapper to return a data frame.
    scale_df = lambda x: pd.DataFrame(data=scaler.transform(x), columns=x.columns)

    print("Scaling train...")
    targets = train.loc[:, 'target']
    (train
    .drop('target', axis=1)
    .pipe(scale_df)
    .assign(fold=fold_vals, target=targets) # Add stacking folds to train, and re-attach unscaled targets.
    .to_csv(config['train'], index=False))

    print("Scaling test...")
    ids = test.loc[:, 'id'] # Need these for creating submission files.
    (test.drop('id', axis=1)
    .pipe(scale_df)
    .assign(id=ids)
    .to_csv(config['test'], index=False))

# Create dummy files for testing.
with open('test_config.yaml', 'r') as f:
   test_config = yaml.load(f)

# Train: Take a sample of the rows from each fold.
print("Preparing dummy train data...")
(pd.read_csv(config['train'])
 .groupby(['fold', 'target'])
 .apply(lambda x: x.sample(n=10))
 .to_csv(test_config['train'], index=False))

# Test: Take the first few rows.
print("Preparing dummy test data...")
(pd.read_csv(config['test'])
 .loc[1:20, :]
 .to_csv(test_config['test'], index=False))

print("Done.")
