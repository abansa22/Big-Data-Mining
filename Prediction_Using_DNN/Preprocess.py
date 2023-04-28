import os.path

import pandas as pd

import Main
from Main import *

# allDataCSV path

# IMPORTANT: This program MUST ONLY BE RUN ONCE PER MODEL!
# It randomly selects the set of data to use as validation,
# if it is run multiple times for the same model the model will see parts of the validation set during training.
# This is very bad. If you want to use an old model and add new data you must manually add it to the correct csv

csv = 'train.csv'

savePath = ''
# size of the validation set (as a ratio of the total size)
val = .15

csv = pd.read_csv(csv, na_values=['NA'])
csv = csv.iloc[:,1:]
csv = csv.fillna(value = -1)
tcsv = pd.read_csv('test.csv', na_values=['NA']).iloc[:,1:]
tcsv = tcsv.fillna(value = -1)
super = pd.concat([csv,tcsv], axis=0, ignore_index=True).iloc[:,1:]
# print(super)

for col in csv.columns:
    if not all(pd.Series(csv[col]).astype(str).str.isnumeric()):
        if col in tcsv.columns:
            cdict = pd.Series(super[col].unique()).to_dict()
            cdict = {v: k for k, v in cdict.items()}
            csv[col] = [cdict[i] for i in csv[col]]
            tcsv[col] = [cdict[i] for i in tcsv[col]]

test = csv.sample(frac = .1)
train = csv[~csv.isin(test)]
train = train.dropna()
test.to_csv(os.path.join(savePath, 'testset.csv'))
train.to_csv(os.path.join(savePath, 'trainset.csv'))
tcsv.to_csv(os.path.join(savePath, 'submission.csv'))

print(csv)

