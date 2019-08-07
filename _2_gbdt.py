import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #分类器
from sklearn.metrics import accuracy_score #评估指标
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV     #网格搜索

import matplotlib.pylab as plt

features = pd.read_csv('feature.csv')
x_columns = features.head(30)['feature'].tolist() #特征重要性排序中，选择前30的特征



train = pd.read_csv('data/tr_FE.csv')


y_train = train.click
X_train = train[x_columns]


X = pd.read_csv('data/ts_FE.csv', dtype = {'id':'U'})
id_ = X.id
X_ = X[x_columns]


gbm = GradientBoostingClassifier(max_depth=15, learning_rate=0.02, n_estimators=1200, min_samples_leaf=60,
               min_samples_split =1200, max_features=4, subsample=1.0, random_state=10)

gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_)
y_predprob = gbm.predict_proba(X_)[:,1]

y_out = pd.DataFrame({'id': list(id_), 'click': y_predprob})
y_out.to_csv('y_out1.csv', index = False, sep = ',')  #保存预测值
