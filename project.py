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


# def pull_data(dir_name, file_name):
#     f = open(dir_name + '/' + file_name + '.csv')
#     x = []
#     timestamps = []
#     for line in f:
#         value = line.split(',')
#         if len(value) > 1:
#             timestamps.append(float(value[-2]))
#             p = float(value[-1])
#             x.append(p)
#     c = timestamps[0]
#     timestamps[:] = [(y - c)/1000 for y in timestamps]
#     return np.array(x), np.array(timestamps)


# signal, timestamps = pull_data('data', 'running2')
# sampling_rate = len(timestamps)/max(timestamps)
# plt.figure(figsize=(10,5))
# plt.plot(timestamps, signal, 'y-',label='PPG')
# plt.title("Heart Rate while Running")
# pl.grid()
# pl.show()

# signal, timestamps = pull_data('data', 'walking2')
# sampling_rate = len(timestamps)/max(timestamps)
# plt.figure(figsize=(10,5))
# plt.plot(timestamps, signal, 'y-',label='PPG')
# plt.title("Heart Rate while Walking")
# pl.grid()
# pl.show()

# signal, timestamps = pull_data('data', 'hiking2')
# sampling_rate = len(timestamps)/max(timestamps)
# plt.figure(figsize=(10,5))
# plt.plot(timestamps, signal, 'y-',label='PPG')
# plt.title("Heart Rate while Hiking")
# pl.grid()
# pl.show()

# signal, timestamps = pull_data('data', 'biking2')
# sampling_rate = len(timestamps)/max(timestamps)
# plt.figure(figsize=(10,5))
# plt.plot(timestamps, signal, 'y-',label='PPG')
# plt.title("Heart Rate while Biking")
# pl.grid()
# pl.show()

# signal, timestamps = pull_data('data', 'rest2')
# sampling_rate = len(timestamps)/max(timestamps)
# plt.figure(figsize=(10,5))
# plt.plot(timestamps, signal, 'y-',label='PPG')
# plt.title("Heart Rate while Resting")
# pl.grid()
# pl.show()

#___________________________________________ Tryna use pandas here


activity_data = pd.read_csv("data/activity_data.csv")

feature_cols = ['Acceleration_X', 'Acceleration_Y', 'Acceleration_Z', 'BPM']
target_col = ['Activity']

X = activity_data[feature_cols] # Features
y = activity_data[target_col] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# regr = DecisionTreeRegressor(max_depth=3, random_state=1234)
# model = regr.fit(X, y)

# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(regr, feature_names=feature_cols, filled=True)