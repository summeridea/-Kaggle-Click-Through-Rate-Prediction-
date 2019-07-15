import json
import pandas as pd

tr_csv_path = 'input/train_sample.csv'
ts_csv_path = 'input/test.csv'

data_type = {'id': 'U', 'hour': 'U', 'device_type':'U', 'C1':'U', 'C15':'U', 'C16':'U'}

train = pd.read_csv(tr_csv_path, dtype=data_type, index_col='id')
test  = pd.read_csv(ts_csv_path, dtype=data_type, index_col='id')
test.insert(0, 'click', 0)

tr_ts = pd.concat([test, train], copy=False)




site_id_count = tr_ts.site_id.value_counts()
site_id_category={}
site_id_category[0] = site_id_count.loc[site_id_count>20].index.values
site_id_category[1] = site_id_count.loc[site_id_count<=20].index.values

site_id_C_type_dict = {}
for key, values in site_id_category.items():
    for item in values:
        site_id_C_type_dict[str(item)] = key

json.dump(site_id_C_type_dict, open("output/site_id_C_type_dict.json", "w"))



site_domain_count = tr_ts.site_domain.value_counts()
site_domain_category={}
site_domain_category[0] = site_domain_count.loc[site_domain_count>20].index.values
site_domain_category[1] = site_domain_count.loc[site_domain_count<=20].index.values

site_domain_C_type_dict = {}
for key, values in site_domain_category.items():
    for item in values:
        site_domain_C_type_dict[str(item)] = key

json.dump(site_domain_C_type_dict, open("output/site_domain_C_type_dict.json", "w"))



app_id_count = tr_ts.app_id.value_counts()
app_id_category={}
app_id_category[0] = app_id_count.loc[app_id_count>20].index.values
app_id_category[1] = app_id_count.loc[app_id_count<=20].index.values

app_id_C_type_dict = {}
for key, values in app_id_category.items():
    for item in values:
        app_id_C_type_dict[str(item)] = key

json.dump(app_id_C_type_dict, open("output/app_id_C_type_dict.json", "w"))



device_model_count = tr_ts.device_model.value_counts()
device_model_category={}
device_model_category[0] = device_model_count.loc[device_model_count>200].index.values
device_model_category[1] = device_model_count.loc[device_model_count<=200].index.values

device_model_C_type_dict = {}
for key, values in device_model_category.items():
    for item in values:
        device_model_C_type_dict[str(item)] = key

json.dump(device_model_C_type_dict, open("output/device_model_C_type_dict.json", "w"))
