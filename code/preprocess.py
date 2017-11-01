"""Add folds, scale features.
"""

import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

np.random.seed(1031) # Month, date on which file was created.

n_folds = 5

train = pd.read_csv("../data/train.csv")

n_rows = len(train) 

# fold: close-to-equal numbers of each fold, in random order.
fold_vals = np.random.permutation(np.arange(0, n_folds).repeat(np.ceil(n_rows / n_folds)))[0:n_rows]

train_with_folds = (train.assign(fold=fold_vals)
                    .drop('id', axis=1)) # Get rid of this so it doesn't get used accidentally for training.

# Scale and save.
scale_and_save = lambda x, f: (StandardScaler()
                                 .fit(x)
                                 .transform(x)
                                 .to_csv(f, index=False))
scale_and_save(train_with_folds, "../generated-files/train.csv")
scale_and_save(pd.read_csv("../data/test.csv"), "../generated-files/test.csv")
