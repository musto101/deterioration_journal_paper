import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_validate
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
pre = datetime.now()

dat = pd.read_csv('data/mci_progress.csv') # 0 is cn_mci
mcPerf = []
mcRep = 1

X = dat.drop(['Unnamed: 0', 'last_DX'], axis=1)

y = dat['last_DX']

reg = LogisticRegression()

param_grid = [
  {'C': np.linspace(0,1,10)},
 ]

for j in range(0,mcRep):
    print(j)
    nrfolds = round(dat.shape[0]/3)

    outer_cv = KFold(n_splits=nrfolds, shuffle=False)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=3335)

    outer_cv.get_n_splits(X)


    totalNewPrediction = []
    totalProbabilities = []

    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        imputer = KNNImputer(n_neighbors=5)
        X_model = imputer.fit(X_train)
        X_train = pd.DataFrame(X_model.transform(X_train), columns=X_train.columns)
        X_test_model = imputer.fit(X)
        X_test = pd.DataFrame(X_test_model.transform(X_test))
        X_test.columns = X_train.columns

        scaler_train = preprocessing.StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler_train.transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler_train.transform(X_test), columns=X_test.columns)

        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)


        clf = RandomizedSearchCV(reg, param_grid, cv=inner_cv, n_iter=20, scoring='roc_auc',
                           n_jobs=-1, random_state=4444, refit=True)

        search = clf.fit(X_train, y_train)

        totalNewPrediction.append(clf.predict(X_test))
        totalProbabilities.append(clf.predict_proba(X_test))

    pd.DataFrame(totalProbabilities).shape
    pd.DataFrame(totalNewPrediction).shape
    dat.shape






