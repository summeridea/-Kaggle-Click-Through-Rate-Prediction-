import json
import datetime
import pandas as pd
from sklearn import preprocessing

tr_csv_path = 'input/train_sample.csv'
ts_csv_path = 'input/test.csv'

data_type = {'id': 'U', 'hour': 'U', 'device_type':'U', 'C1':'U', 'C15':'U', 'C16':'U'}

train = pd.read_csv(tr_csv_path, dtype=data_type, index_col='id')
test  = pd.read_csv(ts_csv_path, dtype=data_type, index_col='id')
test.insert(0, 'click', 0)

tr_ts = pd.concat([test, train], copy=False)


tr_ts['day'] = tr_ts['hour'].apply(lambda x: x[-4:-2])
tr_ts['hour'] = tr_ts['hour'].apply(lambda x: x[-2:])

tr_ts['is_device'] = tr_ts['device_id'].apply(lambda x: 0 if x=='a99f214a' else 1)  #详见探索性数据分析部分


app_id_C_type_dict = json.load(open("output/app_id_C_type_dict.json", "r"))
site_id_C_type_dict = json.load(open("output/site_id_C_type_dict.json", "r"))
site_domain_C_type_dict = json.load(open("output/site_domain_C_type_dict.json", "r"))
device_model_C_type_dict = json.load(open("output/device_model_C_type_dict.json", "r"))

tr_ts['C_app_id'] = tr_ts["app_id"].apply(lambda x: x if app_id_C_type_dict.get(x)==0 else "other_app_id")
tr_ts['C_site_id'] = tr_ts['site_id'].apply(lambda x: x if site_id_C_type_dict.get(x)==0 else "other_site_id")
tr_ts['C_site_domain'] = tr_ts['site_domain'].apply(lambda x: x if site_domain_C_type_dict.get(x)==0 else "other_site_domain")
tr_ts['C_device_model'] = tr_ts['device_model'].apply(lambda x: x if device_model_C_type_dict.get(x)==0 else "other_device_model")

tr_ts["C_pix"] = tr_ts["C15"] + '&' + tr_ts["C16"]
tr_ts["C_device_type_1"] = tr_ts["device_type"] + '&' + tr_ts["C1"]

tr_ts.drop(['device_id', "device_type", 'app_id', 'site_id', 'site_domain', 'device_model',"C1", "C17", 'C15', 'C16'], axis=1, inplace=True)




lenc = preprocessing.LabelEncoder()
C_fields = [ 'hour', 'banner_pos', 'site_category', 'app_domain', 'app_category',
            'device_conn_type', 'C14', 'C18', 'C19', 'C20','C21', 'is_device', 'C_app_id', 'C_site_id',
            'C_site_domain', 'C_device_model', 'C_pix', 'C_device_type_1']
for f, column in enumerate(C_fields):
    print("convert " + column + "...")
    tr_ts[column] = lenc.fit_transform(tr_ts[column])


dummies_site_category = pd.get_dummies(tr_ts['site_category'], prefix = 'site_category')
dummies_app_category = pd.get_dummies(tr_ts['app_category'], prefix = 'app_category')



tr_ts_new = pd.concat([tr_ts, dummies_site_category, dummies_app_category], axis=1)
tr_ts_new.drop(['site_category', 'app_category'], axis = 1, inplace=True)


tr_ts_new.iloc[:test.shape[0],].to_csv('output/test_FE.csv')
tr_ts_new.iloc[test.shape[0]:,].to_csv('output/train_FE.csv')