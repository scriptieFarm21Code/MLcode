import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Load in Flight Data
flightData = sns.load_dataset('flights')  # type = DataFrame

print(type(flightData))
print(flightData.head())  # Print first charactar
print(flightData.shape)  # Print form

# Plotting overall grapg
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.xlabel('Months')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flightData['passengers'])
# plt.show()

print("we running bois")

# Show all Columns
print(flightData.columns)

# Change passenger to floats
allData = flightData['passengers'].values.astype(float)

print(allData)

# Data prepping by splitting it into test
# Var to change to select the last year to validate
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
trainDataNormalized = torch.FloatTensor(trainData).view(-1)

# Train window is the time how much forward we are going to predict/ is in between the data points
trainWindow = 12


def create_inout_sequences(input_data, tw):  # Create tuple section
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        trainSeg = input_data[i:i+tw]
        trainLabel = input_data[i+tw:i+tw+1]
        inout_seq.append((trainSeg, trainLabel))
    return inout_seq


trainInoutSeq = create_inout_sequences(trainDataNormalized, trainWindow)


class LSTM(nn.Module):  # Making LSTM model
    def __init__(self, inputSize=1, hiddenLayerSize=100, outputSize=1):
        super().__init__()
        self.hiddenLayerSize = hiddenLayerSize
        self.lstm = nn.LSTM(inputSize, hiddenLayerSize)
        self.linear = nn.Linear(hiddenLayerSize, outputSize)
        self.hiddenCell = (torch.zeros(1, 1, self.hiddenLayerSize), torch.zeros(1, 1, self.hiddenLayerSize))

    def forward(self, inputSeq):
        lstmOut, self.hiddenCell = self.lstm(inputSeq.view(len(inputSeq), 1, -1), self.hiddenCell)

        predictions = self.linear(lstmOut.view(len(inputSeq), -1))

        return predictions[-1]


model = LSTM()
lossFunction = nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)

print(model)


# Training part
Epochs = 150
for i in range(Epochs):
    for seg, labels in trainInoutSeq:
        optimizer.zero_grad()
        model.hiddenCell = (torch.zeros(1, 1, model.hiddenLayerSize),
                            torch.zeros(1, 1, model.hiddenLayerSize))

        yPrediction = model(seg)

        singleLoss = lossFunction(yPrediction, labels)
        singleLoss.backward()

        # try to calculate accuracy
        output = model(yPrediction)

        optimizer.step()

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {singleLoss.item():10.8f}')

print(f'epoch: {i:3} loss: {singleLoss.item():10.10f}')

# Making the prediction
# future var
future = 12

testInputs = trainDataNormalized[-trainWindow:].tolist()
print(testInputs)

# Make the prediction
model.eval()

for i in range(future):
    seq = torch.FloatTensor(testInputs[-trainWindow:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hiddenLayerSize),
                        torch.zeros(1, 1, model.hiddenLayerSize))
        testInputs.append(model(seq).item())


# Change things to normal values again
actualPredictions = scaler.inverse_transform(np.array(testInputs[trainWindow:]).reshape(-1, 1))
# print("ACTUAL PREDICTIONS")
# print(actualPredictions)


# Print agains true future
# The extra months
x = np.arange(132, 144, 1)
# print(x)

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(flightData['passengers'])
plt.plot(x, actualPredictions)
plt.show()

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(flightData['passengers'][-trainWindow:])
plt.plot(x, actualPredictions)
plt.show()
