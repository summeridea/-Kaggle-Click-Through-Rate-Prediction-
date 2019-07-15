import csv
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

site_id_users = defaultdict(set)
app_id_users = defaultdict(set)
tr_path = 'input/train_FE.csv'
with open(tr_path, 'r') as csv_file:
    for i, row in enumerate(csv.DictReader(csv_file), start=1):
        site_id_users[row['C_site_id']].add(row['device_ip'])
        app_id_users[row['C_app_id']].add(row['device_ip'])



ts_path = 'data/train_FE.csv'
with open(ts_path, 'r') as csv_file:
    for i, row in enumerate(csv.DictReader(csv_file), start=1):
        site_id_users[row['C_site_id']].add(row['device_ip'])  #广告对应设备id
        app_id_users[row['C_app_id']].add(row['device_ip'])    #app对应设备id




app_id_dict = pd.Series()
site_id_dict = pd.Series()

for item in app_id_users:
    app_id_dict[item] = int(np.log10(len(app_id_users[item])))


for item in site_id_users:
    site_id_dict[item] = int(np.log10(len(site_id_users[item])))

app_id_dict = app_id_dict.sort_values(ascending=False)
site_id_dict = site_id_dict.sort_values(ascending=False)

app_id_users = app_id_dict.to_dict()
site_id_users = site_id_dict.to_dict()


ts_csv_path = 'data/test_FE.csv'
tr_csv_path = 'data/train_FE.csv'

test  = pd.read_csv(ts_csv_path, dtype={'id': 'U'}, index_col='id')
train = pd.read_csv(tr_csv_path, dtype={'id': 'U'}, index_col='id')
tr_ts = pd.concat([test, train], copy=False)


tr_ts['app_id_users'] = tr_ts.C_app_id.apply(lambda x: app_id_users[str(x)] if str(x) in app_id_users else 0)
tr_ts['site_id_users'] = tr_ts.C_site_id.apply(lambda x: site_id_users[str(x)] if str(x) in site_id_users else 0)

scaler = StandardScaler()
age_scale_param = scaler.fit(tr_ts[['C14','C18','C19','C20','C21']])
tr_ts[['C14','C18','C19','C20','C21']] = age_scale_param.transform(tr_ts[['C14','C18','C19','C20','C21']])


tr_ts.iloc[:test.shape[0],].to_csv('data/ts_FE.csv')
tr_ts.iloc[test.shape[0]:,].to_csv('data/tr_FE.csv')








