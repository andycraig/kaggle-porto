"""Add folds, scale features.
"""

import shutil, itertools
import toolz
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(1031) # Month, date on which file was created.

n_folds = 5

print("Loading train...")
train = (pd.read_csv("../data/train.csv")
         .drop('id', axis=1)) # Get rid of id so it doesn't get used accidentally for training.

n_rows = len(train) 

# fold: close-to-equal numbers of each fold, in random order.
fold_vals = np.random.permutation(list(toolz.take(n_rows, itertools.cycle(range(n_folds)))))

# Fit scaler before adding folds.
print("Fitting scaler...")
scaler = StandardScaler().fit(train)

# Scale and save.
scale_df = lambda x: pd.DataFrame(data=scaler.transform(x), columns=x.columns)

print("Scaling train...")
(train
 .pipe(scale_df)
 .assign(fold=fold_vals) # Add stacking folds to train.
 .to_csv("../generated-files/train.csv", index=False))

print("Scaling test...")
(pd.read_csv("../data/test.csv")
 .pipe(scale_df)
 .to_csv("../generated-files/test.csv", index=False))
 
print("Done.")
