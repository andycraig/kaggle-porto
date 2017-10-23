```
Rscript set_up_stacking_folds.R
```

```
python fit_hyperparams.py xgboost
python fit_hyperparams.py nn
```

Once for each fold.
```
python train.py xgboost 1
...
python train.py xgboost 5
python train.py nn 1
...
python train.py nn 5
```

```
python stack.py
```
