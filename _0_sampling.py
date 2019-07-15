import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
train_all = pd.read_csv('input/train.csv')

#分层采样  train_sample.csv训练样本数样本数2021449
split = StratifiedShuffleSplit(n_splits=1, train_size=0.05, random_state=42)
for train_index, test_index in split.split(train_all, train_all["click"]):
    strat_train_set = train_all.loc[train_index]
    strat_train_set.to_csv("input/train_sample.csv", header = True)