from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np

CONTINUOUS_COLUMNS = ['I{}'.format(idx) for idx in range(1, 14)]
CATEGORICAL_COLUMNS = ['C{}'.format(idx) for idx in range(1, 27)]

print("loading original data, ......")
df = pd.read_csv("dataset/criteo/train_eval.tsv", sep='\t', header=None)
# filling NaN
mode = df.mode(axis = 0)
df.fillna(mode.iloc[0], inplace = True)
df.columns = ['label'] + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
print("original data[{}] loaded Nan value filled".format(df.shape))

print("\n============ preprocessing numeric features, ......")
# clip upper and clip lower
upper_bound = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
df.loc[:, CONTINUOUS_COLUMNS].clip(upper = upper_bound, axis=1, inplace=True)
df['I2'] = (df['I2'] + 1).clip(lower = 0)

# normalize numeric columns
df.loc[:, CONTINUOUS_COLUMNS] = np.log1p(df.loc[:, CONTINUOUS_COLUMNS])
col_min = df.loc[:, CONTINUOUS_COLUMNS].min()
col_max = df.loc[:, CONTINUOUS_COLUMNS].max()
df.loc[:, CONTINUOUS_COLUMNS] = (df.loc[:, CONTINUOUS_COLUMNS] - col_min) / (col_max - col_min)
print("\n============ numeric features preprocessed")


print("\n============ preprocessing categorical features, ......")
min_occur = 200
vocab = {}
for c in CATEGORICAL_COLUMNS:
    cat_counts = df[c].value_counts()
    valid_catcounts = cat_counts.loc[cat_counts >= min_occur]
    vocab[c] = valid_catcounts.index

tag2idx = {}
for i in range(1, 27):
    tag2idx['C{}'.format(i)] = {}
for i in range(1, 27):
    c = 'C{}'.format(i)
    for idx, tag in enumerate(vocab[c], start = 1):
        tag2idx[c][tag] = idx
    df[c] = df[c].map(tag2idx[c]).fillna(0).astype(int)
   

    
print("\n============ categorical features preprocessed")

test_ratio = 0.2
sample_ratio = 1
prefix = "whole"
df = df.sample(frac=sample_ratio)
train_df, test_df = train_test_split(df, test_size=test_ratio)

outfname = 'dataset/criteo/{}_train.tsv'.format(prefix)
train_df.to_csv(outfname, sep='\t', index=False)
print("data[{}] saved to '{}'".format(train_df.shape, outfname))

outfname = 'dataset/criteo/{}_test.tsv'.format(prefix)
test_df.to_csv(outfname, sep='\t', index=False)
print("data[{}] saved to '{}'".format(test_df.shape, outfname))

