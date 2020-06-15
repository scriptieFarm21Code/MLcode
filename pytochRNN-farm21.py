# Testing with real data
import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from config.connectMongo import db

# Load in sensorData
dataSetRaw = db.readingDataRaw.find({}, {'_id': False, 'orginalReadingId': False})
# Make panda dataframe
sensorData = pd.DataFrame(dataSetRaw)

print(sensorData.head())  # Print first charactar
print(sensorData.shape)  # Print form

# Code to check if gpu is ready
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')

# Plotting overall grapg
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

plt.title('All Soil Moisture')
plt.ylabel('Percentage')
plt.xlabel('All soil moisture points')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(sensorData['soil_moisture_10'], label="soil moisture 10")
plt.plot(sensorData['soil_moisture_20'], label="soil moisture 20")
plt.plot(sensorData['soil_moisture_30'], label="soil moisture 30")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

print("we running bois")

# Show all Columns
print(sensorData.columns)

# Change passenger to floats
# allData = sensorData['soil_moisture_10'].values.astype(float)
# allData = sensorData['soil_moisture_10'].values.astype(float)
# del sensorData['timeReading']  # Not used to train the model
allData = sensorData['soil_moisture_10'].values.astype(float)
# allData = sensorData['passengers'].values.astype(float)
# allData = sensorData['passengers'].values.astype(float)
# allData = sensorData['passengers'].values.astype(float)

print(allData)

# Data prepping by splitting it into test
# Var to change to select the last year to validate
# Can be changed in a int
splitPercentage = 0.9

# Make training data and test data
testData = allData[:-int((len(allData) * splitPercentage))]
trainData = allData[-int((len(allData) * splitPercentage)):]

# Print len
print(f"Length of TrainData {len(trainData)}")
print(f"Length of testData {len(testData)}")

# Scale data to let training to be more easy between 0,1
scaler = MinMaxScaler(feature_range=(-1, 1))
trainData = scaler.fit_transform(trainData .reshape(-1, 1))

# Print data to show if its done correctly
print(trainData[:5])
print(trainData[-5:])

# Convert into tensors for pytorch
trainDataTensor = torch.FloatTensor(trainData).view(-1)

# Train window is the time how much forward we are going to predict/ is in between the data points
trainWindow = 24


def create_inout_sequences(input_data, tw):  # Create tuple section
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        trainSeg = input_data[i:i+tw]
        trainLabel = input_data[i+tw:i+tw+1]
        inout_seq.append((trainSeg, trainLabel))
    return inout_seq


trainInoutSeq = create_inout_sequences(trainDataTensor, trainWindow)


class LSTM(nn.Module):  # Making LSTM model
    def __init__(self, inputSize=1, hiddenLayerSize=150, outputSize=1, batchSize=256, numberLayers=0):
        super().__init__()
        self.hiddenLayerSize = hiddenLayerSize
        self.hiddenCell = (torch.zeros(1, 1, self.hiddenLayerSize), torch.zeros(1, 1, self.hiddenLayerSize))

        # self.numberLayers = numberLayers
        self.batchSize = batchSize
        # LSTM
        self.lstm = nn.LSTM(inputSize, hiddenLayerSize)
        # Output of later
        self.linear = nn.Linear(hiddenLayerSize, outputSize)

    def forward(self, inputSeq):
        lstmOut, self.hiddenCell = self.lstm(inputSeq.view(len(inputSeq), 1, -1), self.hiddenCell)
        lstmOut, self.hiddenCell = self.lstm(inputSeq.view(len(inputSeq), 1, -1), self.hiddenCell)

        predictions = self.linear(lstmOut.view(len(inputSeq), -1))

        return predictions[-1]


model = LSTM()
lossFunction = nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)

print(model)

# Training part
Epochs = 10
batchSize = 0
for i in range(Epochs):
    for seg, labels in trainInoutSeq:
        optimizer.zero_grad()
        model.hiddenCell = (torch.zeros(1, 1, model.hiddenLayerSize),
                            torch.zeros(1, 1, model.hiddenLayerSize))

        yPrediction = model(seg)

        # Calculate loss
        singleLoss = lossFunction(yPrediction, labels)
        singleLoss.backward()

        optimizer.step()

    if i % 2 == 1:
        print(f'epoch: {i:3} loss: {singleLoss.item():10.8f}')

print(f'epoch: {i:3} loss: {singleLoss.item():10.10f}')

# future var
future = 168

testInputs = trainDataTensor[-trainWindow:].tolist()
print(testInputs)

# Make the prediction
model.eval()

for i in range(future):
    seq = torch.FloatTensor(testInputs[-trainWindow:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hiddenLayerSize),
                        torch.zeros(1, 1, model.hiddenLayerSize))
        testInputs.append(model(seq).item())

print(testInputs[future:])

# Change things to normal values again
actualPredictions = scaler.inverse_transform(np.array(testInputs[trainWindow:]).reshape(-1, 1))
print(actualPredictions)

# Print agains true future
# The extra months
x = np.arange(len(trainData), len(trainData) + future, 1)
print(x)

plt.title('Soil Moisture prediction')
plt.ylabel('Soil Moisture percentage')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(sensorData['soil_moisture_10'][:len(trainData) + future], label="Truth")
plt.plot(x, actualPredictions, label="Predictions")
plt.show()

plt.title('Soil Moisture Prediction')
plt.ylabel('Soil Moisture Percentage')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(sensorData['soil_moisture_10'][x], label="Truth")
plt.plot(x, actualPredictions, label="Predictions")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
plt.show()
