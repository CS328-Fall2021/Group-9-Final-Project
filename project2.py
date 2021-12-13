import os

from matplotlib import cm
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import joblib

# matplotlib.style.use('ggplot')

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.tree import DecisionTreeRegressor


# hadi_walk = pd.read_csv("data2/hadi_walk.csv")
# plt.plot(hadi_walk["Sample Count"],hadi_walk["Heart Rate (bpm)"])
# plt.title('Hadi Walk')
# plt.xlabel('Time')
# plt.ylabel('BPM')
# plt.show()

# hadi_rest = pd.read_csv("data2/hadi_rest.csv")
# plt.plot(hadi_rest["Sample Count"],hadi_rest["Heart Rate (bpm)"])
# plt.title('Hadi Rest')
# plt.xlabel('Time')
# plt.ylabel('BPM')
# plt.show()

# mike_rest = pd.read_csv("data2/mike_rest.csv")
# plt.plot(mike_rest["Sample Count"],mike_rest["Heart Rate (bpm)"])
# plt.title('Mike Rest')
# plt.xlabel('Time')
# plt.ylabel('BPM')
# plt.show()

# mike_walk = pd.read_csv("data2/mike_walk.csv")
# plt.plot(mike_walk["Sample Count"],mike_walk["Heart Rate (bpm)"])
# plt.title('Mike Walk')
# plt.xlabel('Time')
# plt.ylabel('BPM')
# plt.show()


#___________________________________________ Tryna use pandas here


activity_data = pd.read_csv("data2/activity_data2.csv")

feature_cols = ['Acceleration X (g)', 'Acceleration Y (g)', 'Acceleration Z (g)']
# feature_cols = ["Heart Rate (bpm)"]
target_col = ['Activity']

X = activity_data[feature_cols] 
y = activity_data[target_col] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

tree.plot_tree(clf)
plt.show()

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

ccp_alphas = clf.cost_complexity_pruning_path(X_train, y_train)["ccp_alphas"]

# print(ccp_alphas)

ccp_alpha_grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    scoring=make_scorer(accuracy_score),
    param_grid=ParameterGrid({"ccp_alpha": [[alpha] for alpha in ccp_alphas]}),
)

ccp_alpha_grid_search.fit(X_train, y_train)

# print(ccp_alpha_grid_search.best_params_)

best_ccp_alpha_tree = ccp_alpha_grid_search.best_estimator_

print(classification_report(y_test, clf.predict(X_test)))
print(classification_report(y_test, best_ccp_alpha_tree.predict(X_test)))


tree.plot_tree(best_ccp_alpha_tree)
plt.show()

print(best_ccp_alpha_tree.get_depth())