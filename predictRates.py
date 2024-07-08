import csv 
import pickle
import tensorflow as tf 
import numpy as np 
from dataclasses import dataclass 
import matplotlib.pyplot as plt
from absl import logging 
logging.set_verbosity(logging.ERROR)

EUROUSD_CSV = 'EURUSD=X.csv'

with open(EUROUSD_CSV, 'r') as csvfile:
    print(f"Header looks like this:\n\n{csvfile.readline()}")    
    print(f"First data point looks like this:\n\n{csvfile.readline()}")
    print(f"Second data point looks like this:\n\n{csvfile.readline()}")

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time (in days)")
    plt.ylabel("Value (Exchange Rate EUR/USD)")
    plt.grid(True)

def parse_data_from_file(filename):
    
    times = []
    prices = []

    with open(filename) as csvfile:
        
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        i = 0
        
        for row in reader:
            prices.append(float(row[1]))
            times.append(i)
            i+=1
            
    return times, prices

@dataclass
class G:
    EUROUSD_CSV = 'EURUSD=X.csv'
    times, prices = parse_data_from_file(EUROUSD_CSV)
    TIME = np.array(times)
    SERIES = np.array(prices)
    SPLIT_TIME = 4270
    WINDOW_SIZE = 150
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 5346

# Displaying the Yahoo Finance Data on EUR/USD 
# plt.figure(figsize=(10, 6))
# plot_series(G.TIME, G.SERIES)
# plt.show()

def train_val_split(time, series, time_step=G.SPLIT_TIME):

    time_train = time[:time_step]
    series_train = series[:time_step]
    time_valid = time[time_step:]
    series_valid = series[time_step:]

    return time_train, series_train, time_valid, series_valid

# Split the dataset
time_train, series_train, time_valid, series_valid = train_val_split(G.TIME, G.SERIES)

def windowed_dataset(series, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

# Apply the transformation to the training set
train_set = windowed_dataset(series_train, window_size=G.WINDOW_SIZE, batch_size=G.BATCH_SIZE, shuffle_buffer=G.SHUFFLE_BUFFER_SIZE)
# Repeat the dataset indefinitely
train_set = train_set.repeat()  

def create_uncompiled_model():

    model = tf.keras.models.Sequential([ 
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                      strides=1,
                      activation="relu",
                      padding='causal',
                      input_shape=[None, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1),
        
    ]) 

    return model

def adjust_learning_rate(dataset):
    
    model = create_uncompiled_model()
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
    
    # Selecting SGD optimizer 
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
    
    # Compiling the model using Huber loss 
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer, 
                  metrics=["mae"]) 

    # Providing steps_per_epoch when calling model.fit() if dataset is repeated indefinitely
    steps_per_epoch = G.SPLIT_TIME // G.BATCH_SIZE
    
    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule], steps_per_epoch=steps_per_epoch)
    
    return history

lr_history = adjust_learning_rate(train_set)

# Plot shows the best learning rates are between 1.3e-4 and 3.0 with the best maes within 0.0089 and 0.0631
plt.semilogx(lr_history.history["learning_rate"], lr_history.history["loss"])
plt.axis([1e-4, 10, 0, 10])
plt.show()