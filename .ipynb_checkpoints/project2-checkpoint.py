import matplotlib
from matplotlib import cm
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

matplotlib.style.use('ggplot')

import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn import tree
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

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

tree.plot_tree(clf)
plt.show()