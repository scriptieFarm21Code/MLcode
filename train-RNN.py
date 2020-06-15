# THIS IS FILE EXIST WITH THE LOOSE SCRIPTS TO CHANGE/CONTROL THINGS
# FILE IS MADE TO TRAIN MODELS
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# Imports for connections
from config.connectMongo import db
from config.connectSql import farm21DB

# Uncommented when to use the other set
dataSetRaw = db.trainDataNew36.find({}, {'_id': False})
df = pd.DataFrame(dataSetRaw)

# print(df)

# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
# csv_path, _ = os.path.splitext(zip_path)

# df = pd.read_csv(csv_path)

# TODO: FIND OUT WHAT THIS DOES AND REWRITE FOR MYSELF


def normalizeData(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # RESHAPE THE DATA
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])

    return np.array(data), np.array(labels)


train_split = 6500

tf.random.set_seed(13)

uni_data = df['soil_moisture_10']
uni_data.index = df['timeReading']
uni_data.head()

plt.figure('Overview')
uni_data.plot(subplots=True)
plt.show()

uni_data = uni_data.values

uniTrainMean = uni_data[:train_split].mean()
uniTrainStd = uni_data[:train_split].std()

uni_data = (uni_data - uniTrainMean) / uniTrainMean

pastHistory = 20
futureTarget = 0

x_train_uni, y_train_uni = normalizeData(uni_data, 0, train_split,
                                         pastHistory, futureTarget)
x_val_uni, y_val_uni = normalizeData(uni_data, train_split, None,
                                     pastHistory, futureTarget)

# Print extra data
print('Single window of past history')
print(x_train_uni[0])
print('\n Target temperature to predict')
print(y_train_uni[0])


def createTimeSteps(lenght):
    return list(range(-lenght, 0))


def showPlot(plotData, delta, title):
    labels = ['History', 'True Future', 'Model Predictinos']
    marker = ['.-', 'rx', 'go']
    timeSteps = createTimeSteps(plotData[0].shape[0])

    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)

    for i, testing in enumerate(plotData):
        if i:
            plt.plot(future, plotData[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(timeSteps, plotData[i], marker[i], label=labels[i])

    plt.legend()
    plt.xlim([timeSteps[0], (future+5)*2])
    plt.xlabel('Time-Step')

    return plt.show()


# Plot for example
showPlot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')


def baseline(history):
    return np.mean(history)


#  Plot for testing
showPlot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,  'Baseline Prediction Example')


batchSize = 256
bufferSize = 10000

trainUnivariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
trainUnivariate = trainUnivariate.cache().shuffle(bufferSize).batch(batchSize).repeat()

valUnivariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
valUnivariate = valUnivariate.batch(batchSize).repeat()

lstmModel = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

lstmModel.compile(optimizer='adam', loss='mae')

for x, y in valUnivariate.take(1):
    print(lstmModel.predict(x).shape)

interval = 200
epochs = 10

lstmModel.fit(trainUnivariate, epochs=epochs, steps_per_epoch=interval, validation_data=valUnivariate, validation_steps=50)
# Uncomment to see first plot
for x, y in trainUnivariate.take(3):
    plot = showPlot([x[0].numpy(), y[0].numpy(), lstmModel.predict(x)[0]], 0, 'Simple LSTM model')
    pass

# ADDING MORE DATAPOINTS SINGLE STEP
features_considered = ['temp', 'airPressure', 'windSpeed', 'rain']
features = df[features_considered]
features.index = df['timeReading']
    features.head()

features.plot(subplots=True)

dataset = features.values
data_mean = dataset[:train_split].mean(axis=0)
data_std = dataset[:train_split].std(axis=0)

dataset = (dataset-data_mean)/data_std


def multiStepData(dataset, target, startIndex, endIndex, historySize, targetSize, step, singleStep=False):
    data = []
    labels = []

    startIndex = startIndex + historySize
    if endIndex is None:
        endIndex = len(dataset) - targetSize

    for i in range(startIndex, endIndex):
        indices = range(i-historySize, i, step)
        data.append(dataset[indices])

        if singleStep:
            labels.append(target[i+targetSize])
        else:
            labels.append(target[i:i+targetSize])

    return np.array(data), np.array(labels)


pastHistory = 720
futureTarget = 72
step = 6

x_train_single, y_train_single = multiStepData(dataset, dataset[:, 1], 0, train_split, pastHistory, futureTarget, step, singleStep=True)
x_val_single, y_val_single = multiStepData(dataset, dataset[:, 1], train_split, None, pastHistory, futureTarget, step, singleStep=True)

print('Single window of past history : {}'.format(x_train_single[0].shape))

trainSingleData = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
trainSingleData = trainSingleData.cache().shuffle(bufferSize).batch(batchSize).repeat()

valDataSingle = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
valDataSingle = valDataSingle.batch(batchSize).repeat()
# Setting up the singlestep model
singleStepModel = tf.keras.models.Sequential()
singleStepModel.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
singleStepModel.add(tf.keras.layers.Dense(1))
singleStepModel.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

for x, y in valDataSingle.take(1):
    print(singleStepModel.predict(x).shape)

# Train model
singleStepHistory = singleStepModel.fit(trainSingleData, epochs=epochs, steps_per_epoch=interval, validation_data=valDataSingle, validation_steps=50)


def plotTrainHistory(history, title):
    loss = history.history['loss']
    valLoss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, valLoss, 'r', label='Validation Loss')
    plt.title(title)
    plt.legend()

    plt.show()


plotTrainHistory(singleStepHistory, 'Single Step Training and Validation Loss')

# Predict one step in the future
for x, y in valDataSingle.take(3):
    plot = showPlot([x[0][:, 1].numpy(), y[0].numpy(),
                     singleStepModel.predict(x)[0]], 12,
                    'Single Step Prediction')

# MULTE STEP
futureTarget = 72

x_train_multi, y_train_multi = multiStepData(dataset, dataset[:, 1], 0, train_split, pastHistory, futureTarget, step)
x_val_multi, y_val_multi = multiStepData(dataset, dataset[:, 1], train_split, None, pastHistory, futureTarget, step)

print('Single window of past history : {}'.format(x_train_multi[0].shape))
print('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

trainMultiData = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
trainMultiData = trainMultiData.cache().shuffle(bufferSize).batch(batchSize).repeat()

valDataMulti = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
valDataMulti = valDataMulti.batch(batchSize).repeat()


def multiStepPlot(history, trueFuture, predicition):
    plt.figure(figsize=(12, 6))
    num_in = createTimeSteps(len(history))
    num_out = len(trueFuture)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out) / step, np.array(trueFuture), 'bo',
             label='True Future')

    if predicition.any():
        plt.plot(np.arange(num_out)/step, np.array(predicition), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


for x, y in trainMultiData.take(1):
    multiStepPlot(x[0], y[0], np.array([0]))


multiStepModel = tf.keras.models.Sequential()
multiStepModel.add(tf.keras.layers.LSTM(32,  return_sequences=True, input_shape=x_train_multi.shape[-2:]))
multiStepModel.add(tf.keras.layers.LSTM(16, activation='relu'))
multiStepModel.add(tf.keras.layers.Dense(72))

multiStepModel.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

for x, y in valDataMulti.take(1):
    print(multiStepModel.predict(x).shape)
    pass

multiStepHistory = multiStepModel.fit(trainMultiData, epochs=epochs, steps_per_epoch=interval, validation_data=valDataMulti, validation_steps=50)

plotTrainHistory(multiStepHistory, 'Multo step history and loss')

for x, y in valDataMulti.take(3):
    multiStepPlot(x[0], y[0], multiStepModel.predict(x)[0])
    pass
