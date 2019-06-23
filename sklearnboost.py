import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams["figure.figsize"] = 12, 4

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

"""
Data can be downloaded from https://datahack.analyticsvidhya.com/contest/data-hackathon-3x/
"""

train = pd.read_csv("train_modified.csv")
target = "Disbursed"
IDcol = "ID"

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data.
    alg.fit(dtrain[predictors], dtrain["Disbursed"])
    #Predict training set.
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    #Perform cross-validation.
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain["Disburse"], cv=cv_folds, scoring="roc_auc")