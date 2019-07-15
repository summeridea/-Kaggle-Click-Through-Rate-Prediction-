import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score #评估指标
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

import matplotlib.pyplot as plt

train = pd.read_csv("input/tr_FE.csv")
#test = pd.read_csv("input/tr_FE.csv")

features = pd.read_csv('feature.csv')
y_train = train.click
X_train = train.drop(['click'], axis = 1)





def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params() #参数
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values) #训练集数据与标签
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])


    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("Model Report")
    #print("Accuracy : %.4g" % accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    #print("AUC Score (Train): %f" % roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

model = xgb.XGBRegressor(n_estimators=350, max_depth=10, objective='binary:logistic', min_child_weight=50,
                         subsample=0.8, gamma=0, learning_rate=0.2, colsample_bytree=0.5, seed=27)

model.fit(X_train, y_train)
# y_test = model.predict(X_test)
plot_importance(model, importance_type="gain")

features = X_train.columns
feature_importance_values = model.feature_importances_

feature_importances = pd.DataFrame({'feature': list(features), 'importance': feature_importance_values})

feature_importances.sort_values('importance', inplace=True, ascending=False)
print(feature_importances)

# print(model.get_booster().get_fscore())

print(model.get_booster().get_score(importance_type="gain"))


feature_importances.to_csv('feature.csv')

