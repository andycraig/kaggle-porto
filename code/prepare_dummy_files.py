import pandas as pd
import yaml

with open('test_config.yaml', 'r') as f:
   config = yaml.load(f)

# Train: Take a sample of the rows from each fold.
print("Preparing dummy train data...")
(pd.read_csv(config['train_original'])
 .groupby('fold')
 .apply(lambda x: x.sample(n=10))
 .to_csv(config['train'], index=False))

# Test: Take the first few rows.
print("Preparing dummy test data...")
(pd.read_csv(config['test_original'])
 .loc[1:20, :]
 .to_csv(config['test'], index=False))

print("Done.")
