import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request
import json
import os
import numpy as np
import tensorflow as tf  # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from config.connectMongo import db

# Load in sensorData
dataSetRaw = db.trainDataNew36.find({}, {'_id': False, 'orginalReadingId': False})
# Make panda dataframe
dataFrame = pd.DataFrame(dataSetRaw)
# Change to a float

# DATA INSIGHT
print("DATA INSIGHT")
print(dataFrame.shape)
print(dataFrame.head())

featuresPredicting = ['temp', 'soil_temperature', 'soil_moisture_10']

features = dataFrame[featuresPredicting]
features.index = dataFrame['timeReading']
print(features.head())

features.plot(subplots=True)
