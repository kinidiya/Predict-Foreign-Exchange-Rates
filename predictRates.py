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
        
        ### START CODE HERE
        
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        i = 0
        
        for row in reader:
            prices.append(float(row[1]))
            times.append(i)
            i+=1
            
        
        ### END CODE HERE
            
    return times, prices

@dataclass
class G:
    EUROUSD_CSV = 'EURUSD=X.csv'
    times, prices = parse_data_from_file(EUROUSD_CSV)
    TIME = np.array(times)
    SERIES = np.array(prices)
    SPLIT_TIME = 2500
    WINDOW_SIZE = 64
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000


plt.figure(figsize=(10, 6))
plot_series(G.TIME, G.SERIES)
plt.show()