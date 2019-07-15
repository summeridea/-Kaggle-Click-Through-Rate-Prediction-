from sklearn.ensemble import GradientBoostingClassifier  # 分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np


train = pd.read_csv('data/tr_FE.csv')
test = pd.read_csv('data/te_FE.csv')
features = pd.read_csv('feature.csv')
x_columns = features.head(30)['feature'].tolist() #特征重要性排序中，选择前30的特征
id_ = test.id
X = train[x_columns]
y = train.click
X_submission = test[x_columns]


INITIAL_PARAMS = {
    'RF:': {
        'n_estimators': 400, 'n_jobs': -1, 'criterion': 'entropy',
        'min_samples_leaf': 3, 'bootstrap': False,
        'max_depth': 12, 'min_samples_split': 6, 'max_features': 0.14357
    },

    'AdaBoost:': {
        'n_estimators': 500, 'learning_rate': 0.01
    },

    'GBDT:': {
        'max_depth': 15, 'learning_rate': .02, 'n_estimators': 1200,
        'min_samples_leaf': 60, 'min_samples_split': 1200,
        'max_features': 4, 'subsample': 1.0, 'random_state': 10
    },

}

MODEL_NAME = 'blend_ensemble'

if __name__ == '__main__':

    NFOLDS = 5  # set folds for out-of-fold prediction
    SEED = 0  # for reproducibility
    LM_CV_NUM = 100

    skf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)

    clfs = [
        RandomForestClassifier().set_params(**INITIAL_PARAMS.get("RF:", {})),
        AdaBoostClassifier().set_params(**INITIAL_PARAMS.get("AdaBoost:", {})),
        GradientBoostingClassifier().set_params(**INITIAL_PARAMS.get("GBDT:", {})),
    ]

    print("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], NFOLDS))
        for i, (train_index, test_index) in enumerate(skf.split(X)):
            # 训练集 与 验证集
            print("Fold", i)
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]

            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]

            dataset_blend_train[test_index, j] = y_submission  # 第j个基学习器的预测值
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print("Blending.")

    clf = Ridge(alpha=0.5)

    clf.fit(dataset_blend_train, y)

    y_submission = clf.predict(dataset_blend_test)  # for RidgeCV

    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    y_out = pd.DataFrame({'id': list(id_), 'click': y_submission})
    y_out.to_csv('out_1.csv', index=False, sep=',')  # 保存预测值
