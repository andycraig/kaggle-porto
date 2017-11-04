"""Add folds, scale features.
"""

import shutil, itertools, yaml
import toolz
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(1031) # Month, date on which file was created.

with open('config.yaml', 'r') as f:
    config = yaml.load(f)

print("Loading train...")
train = (pd.read_csv(config['train_original'])
         .drop('id', axis=1)) # Get rid of id so it doesn't get used accidentally for training.

n_rows = len(train) 

# fold: close-to-equal numbers of each fold, in random order.
fold_vals = np.random.permutation(list(toolz.take(n_rows, itertools.cycle(range(config['n_folds'])))))

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
test = pd.read_csv(config['test_original'])
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
 .groupby('fold')
 .apply(lambda x: x.sample(n=10))
 .to_csv(test_config['train'], index=False))

# Test: Take the first few rows.
print("Preparing dummy test data...")
(pd.read_csv(config['test'])
 .loc[1:20, :]
 .to_csv(test_config['test'], index=False))

print("Done.")
