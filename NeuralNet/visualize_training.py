import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CNN_with_test import create_input_samples, CNN
from FCNN import FCNN


######################################
# Test models on training data    #
######################################

# hyperparameters
BATCH_SIZE = 128


# Load the test data
data = pd.read_csv('Final_dataframe_train.csv')
test_input = data.drop(["RUL"], axis=1)
test_label = data.filter(["unit_number","RUL"], axis=1)
#testplot = test_label.filter(["RUL"], axis=1)
#plt.plot(testplot[0:200])
# Load the model from .pt file
model_type = "CNN"  # "FCNN" or "CNN"
if model_type == "FCNN":
    model = FCNN(test_input.shape[1])  # Instantiate your model
else:
    model = CNN()  # Instantiate your model

# specify model file name
model.load_state_dict(torch.load("CNN2.pt"))
model.eval()

# preprocess the data
if model_type == "CNN":
    test_input_sampled, test_label_sampled = create_input_samples(test_input, test_label)
    input_samples_swapped = test_input_sampled.transpose(0,2,1)

    # Compute the loss
    criterion = nn.MSELoss()
    total_loss = 0
    predictions = []
    num_batches = len(test_input_sampled) // BATCH_SIZE

    for batch in range(num_batches):
        # Get the batch indices
        start_idx = batch * BATCH_SIZE
        end_idx = (batch + 1) * BATCH_SIZE

        # Get the batch input and target tensors
        batch_input = torch.tensor(input_samples_swapped[start_idx:end_idx], dtype=torch.float32)
        batch_target = torch.tensor(test_label_sampled[start_idx:end_idx], dtype=torch.float32)
        
        # Forward pass
        output = model(batch_input)
        predictions.append(output)
        loss = criterion(output, batch_target)
        loss = torch.sqrt(loss)
        total_loss += loss.item()

# model is FCNN
else:

    # Compute the loss
    criterion = nn.MSELoss()

    # Calculate the total number of batches
    num_batches = test_input.shape[0] // BATCH_SIZE
    total_loss = 0
    predictions = []

    for batch in range(num_batches):
        # Get the batch indices
        start_idx = batch * BATCH_SIZE
        end_idx = (batch + 1) * BATCH_SIZE

        # Get the batch input and target tensors
        batch_input = test_input.iloc[start_idx:end_idx].values
        test_label_sampled = test_label.filter(["RUL"], axis=1)
        batch_target = test_label_sampled.iloc[start_idx:end_idx].values

        # Convert the batch input to a tensor
        input_tensor = torch.tensor(batch_input, dtype=torch.float32)

        # Forward pass
        RUL_predicted = model(input_tensor)
        predictions.append(RUL_predicted)

        # Convert the batch target to a tensor
        target_tensor = torch.tensor(batch_target, dtype=torch.float32)

        # Compute loss
        #print("target_tensor: {}".format(target_tensor[0]))
        #print("RUL_predicted: {}".format(RUL_predicted[0]))
        loss = criterion(target_tensor, RUL_predicted)
        # Compute RMSE
        loss = torch.sqrt(loss)     # paper says RMSE
        total_loss += loss.item()

predictions = torch.cat(predictions, dim=0).detach().numpy()
#print("len(predictions): {}".format(len(predictions)))

for unit_num in range(1,100):
    ######################################
    # plot predictions for a specific unit
    #unit_num = 30
    ######################################
    plt.figure()
    # compute lengths of each unit
    unit_lengths_FCNN = []
    unit_lengths_CNN = []
    for i in range(1,100):
        unit_lengths_FCNN.append(len(test_label[test_label["unit_number"] == i]))
        unit_lengths_CNN.append(len(test_label[test_label["unit_number"] == i])-29)

    if model_type == "CNN":
        # visualize predictions for a single unit
        unit_RUL = test_label[test_label["unit_number"] == unit_num]
        unit_RUL = unit_RUL.drop(["unit_number"], axis=1)
        unit_RUL.reset_index(drop=True, inplace=True)
        batch1_RUL = test_label.drop(["unit_number"], axis=1)
        batch1_RUL = batch1_RUL.iloc[0:BATCH_SIZE]

        plt.plot(unit_RUL , label="True RUL [test dataset]")
        start_idx = sum(unit_lengths_CNN[0:unit_num-1])
        end_idx = sum(unit_lengths_CNN[0:unit_num-1])+unit_lengths_CNN[unit_num-1]
        # plot prediction index shifted by 29
        plt.plot(unit_RUL.index[29:], predictions[start_idx:end_idx], label="Predicted RUL [test dataset]")
    else:
        # visualize predictions for a single unit
        unit_RUL = test_label[test_label["unit_number"] == unit_num]
        unit_RUL = unit_RUL.drop(["unit_number"], axis=1)
        unit_RUL.reset_index(drop=True, inplace=True)
        batch1_RUL = test_label.drop(["unit_number"], axis=1)
        batch1_RUL = batch1_RUL.iloc[0:BATCH_SIZE]

        plt.plot(unit_RUL , label="True RUL")
        start_idx = sum(unit_lengths_FCNN[0:unit_num-1])
        end_idx = sum(unit_lengths_FCNN[0:unit_num-1])+unit_lengths_FCNN[unit_num-1]
        plt.plot(predictions[start_idx:end_idx], label="Predicted RUL")

    plt.title("Unit {}".format(unit_num))
    plt.xlabel("Time Cycles")
    plt.ylabel("RUL")
    plt.legend()
    plt.savefig("plots/reconstruction_training/CNN_Unit_{}.png".format(unit_num))
    plt.close()
avg_loss = total_loss / num_batches
print("Average loss: {}".format(avg_loss))
