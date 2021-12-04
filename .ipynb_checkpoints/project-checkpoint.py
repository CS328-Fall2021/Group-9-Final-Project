import matplotlib
from matplotlib import cm
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

matplotlib.style.use('ggplot')

def pull_data(dir_name, file_name):
    f = open(dir_name + '/' + file_name + '.csv')
    x = []
    timestamps = []
    for line in f:
        value = line.split(',')
        if len(value) > 1:
            timestamps.append(float(value[-2]))
            p = float(value[-1])
            x.append(p)
    c = timestamps[0]
    timestamps[:] = [(y - c)/1000 for y in timestamps]
    return np.array(x), np.array(timestamps)


signal, timestamps = pull_data('data', 'running2')
sampling_rate = len(timestamps)/max(timestamps)
plt.figure(figsize=(10,5))
plt.plot(timestamps, signal, 'y-',label='PPG')
plt.title("Heart Rate while Running")
pl.grid()
pl.show()

signal, timestamps = pull_data('data', 'walking2')
sampling_rate = len(timestamps)/max(timestamps)
plt.figure(figsize=(10,5))
plt.plot(timestamps, signal, 'y-',label='PPG')
plt.title("Heart Rate while Walking")
pl.grid()
pl.show()

signal, timestamps = pull_data('data', 'hiking2')
sampling_rate = len(timestamps)/max(timestamps)
plt.figure(figsize=(10,5))
plt.plot(timestamps, signal, 'y-',label='PPG')
plt.title("Heart Rate while Hiking")
pl.grid()
pl.show()

signal, timestamps = pull_data('data', 'biking2')
sampling_rate = len(timestamps)/max(timestamps)
plt.figure(figsize=(10,5))
plt.plot(timestamps, signal, 'y-',label='PPG')
plt.title("Heart Rate while Biking")
pl.grid()
pl.show()

signal, timestamps = pull_data('data', 'rest2')
sampling_rate = len(timestamps)/max(timestamps)
plt.figure(figsize=(10,5))
plt.plot(timestamps, signal, 'y-',label='PPG')
plt.title("Heart Rate while Resting")
pl.grid()
pl.show()