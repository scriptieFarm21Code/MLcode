# THIS IS FILE EXIST WITH THE LOOSE SCRIPTS TO CHANGE/CONTROL THINGS
# FILE IS MADE TO TRAIN MODELS
from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import sys
import gc
import matplotlib.pyplot as pyplot
import numpy as num
import pandas as pd
import seaborn
import tensorboard
import tensorflow as tf
import seaborn as sns
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorflow import keras
from config.connectMongo import db


# Get the whole dataset
dataSetRaw = db.trainData.find({}, {'_id': False})
dataFrame = pd.DataFrame(dataSetRaw)  # Change data into a dataframe

# Looking the dataset
print('DIT IS EEN OVERVIEW VAN DE DATA')
print(dataFrame)
# Check if there is missing data
print("THIS DATA IS MISSING")
print(dataFrame.isna().sum())
# Remove that data
dataFrame = dataFrame.dropna()

# Move data to a training set and testing set
trainDataset = dataFrame.sample(frac=0.8, random_state=0)
testDataset = dataFrame.drop(trainDataset.index)

# sns.pairplot(trainDataset[['airPressure', 'temp', 'windSpeed', 'rain', 'soil_moisture_10', 'soil_temperature']], diag_kind='kde')
# pyplot.show()  # Show graph

# In general statistics
trainStats = trainDataset.describe()
trainStats.pop('soil_moisture_10')
trainStats = trainStats.transpose()

# STATS OF DATA
print("STATS OF DATA")
print(trainStats)

trainLabels = trainDataset.pop('soil_moisture_10')
testLabels = testDataset.pop('soil_moisture_10')


def normalizedData(data):
    return (data - trainStats['mean']) / trainStats['std']
    pass


normalizedTrainData = normalizedData(trainDataset)
normalizedTestData = normalizedData(testDataset)


def buildModel():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[len(trainDataset.keys())]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse']
    )

    return model
    pass


model = buildModel()

print("EXAMPLE MODEL")
print(model.summary())
# EXAMPLE CODE
# example = normalizedTrainData[:10]
# result = model.predict(example)
# print(result)

# Training model
EPOCHS = 1000

# Seeing the history of training
history = model.fit(
    normalizedTrainData, trainLabels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[tfdocs.modeling.EpochDots()]
)

print("HISTORY")
hist = pd.DataFrame(history.history)
hist['epochs'] = history.epoch
print(hist)


print(model.to_json())
model.save('testin', save_format='tf')
print(keras.models.load_model('testin.h5').get_weights())
# Plotting data
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# Mean Absulite Error Graph plotting
pyplot.figure('Mean Absulite Error')
plotter.plot({'Basic': history}, metric="mae")
pyplot.ylim([0, 20])
pyplot.ylabel('MAE [MPG]')
pyplot.show()

# Mean Squared Errors
pyplot.figure('Mean Squared Errors')
plotter.plot({'Basic': history}, metric="mse")
pyplot.ylim([0, 20])
pyplot.ylabel('MSE [MPG^2]')
pyplot.show()

# Make early stop if training isnt going well
model = buildModel()

# The patience parameter is the amount of epochs to check for improvement
earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

earlyHistory = model.fit(normalizedTrainData, trainLabels,
                         epochs=EPOCHS, validation_split=0.2, verbose=0,
                         callbacks=[earlyStop, tfdocs.modeling.EpochDots()])

pyplot.figure("Early Stopping")
plotter.plot({'Early Stopping': earlyHistory}, metric="mae")
pyplot.ylim([0, 10])
pyplot.ylabel('MAE [MPG]')
pyplot.show()

loss, mae, mse = model.evaluate(normalizedTestData, testLabels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


testPredictions = model.predict(normalizedTestData).flatten()

pyplot.figure("Predictions")
a = pyplot.axes(aspect="equal")
pyplot.scatter(testLabels, testPredictions)
pyplot.xlabel('True values [SOIL_MOISTURE]')
pyplot.ylabel('Predictions [SOIL_MOISTURE]')
lims = [0, 50]
pyplot.xlim(lims)
pyplot.ylim(lims)
pyplot.plot(lims, lims)
pyplot.show()

print(testPredictions)

pyplot.figure('Error')
error = testPredictions - testLabels
pyplot.hist(error, bins=25)
pyplot.xlabel("Prediction Error [MPG]")
pyplot.ylabel("Count")
pyplot.show()
