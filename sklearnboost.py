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
