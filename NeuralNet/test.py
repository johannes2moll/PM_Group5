import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CNN import create_input_samples, CNN
from FCNN import FCNN


######################################
# Test models on unseen test data    #
######################################

# hyperparameters
BATCH_SIZE = 128


# Load the test data
data = pd.read_csv('Final_dataframe_test.csv')
test_input = data.drop(["RUL"], axis=1)
test_label = data.filter(["unit_number","RUL"], axis=1)

# Load the model from .pt file
model_type = "FCNN"  # "FCNN" or "CNN"
if model_type == "FCNN":
    model = FCNN(test_input.shape[1])  # Instantiate your model
else:
    model = CNN()  # Instantiate your model

# specify model file name
model.load_state_dict(torch.load("FCNN.pt"))
model.eval()

# preprocess the data
if model_type == "CNN":
    test_input,test_label = create_input_samples(test_input, test_label)
    input_samples_swapped = test_input.transpose(0,2,1)

    # Compute the loss
    criterion = nn.MSELoss()
    total_loss = 0
    num_batches = len(test_input) // BATCH_SIZE

    for batch in range(num_batches):
        # Get the batch indices
        start_idx = batch * BATCH_SIZE
        end_idx = (batch + 1) * BATCH_SIZE

        # Get the batch input and target tensors
        batch_input = torch.tensor(input_samples_swapped[start_idx:end_idx], dtype=torch.float32)
        batch_target = torch.tensor(test_label[start_idx:end_idx], dtype=torch.float32)
        
        # Forward pass
        output = model(batch_input)

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
    for batch in range(num_batches):
        # Get the batch indices
        start_idx = batch * BATCH_SIZE
        end_idx = (batch + 1) * BATCH_SIZE

        # Get the batch input and target tensors
        batch_input = test_input.iloc[start_idx:end_idx].values
        test_label = test_label.filter(["RUL"], axis=1)
        batch_target = test_label.iloc[start_idx:end_idx].values

        # Convert the batch input to a tensor
        input_tensor = torch.tensor(batch_input, dtype=torch.float32)

        # Forward pass
        RUL_predicted = model(input_tensor)

        # Convert the batch target to a tensor
        target_tensor = torch.tensor(batch_target, dtype=torch.float32)

        # Compute loss
        loss = criterion(target_tensor, RUL_predicted)
        # Compute RMSE
        loss = torch.sqrt(loss)     # paper says RMSE
        total_loss += loss.item()
        
avg_loss = total_loss / len(test_input)
print("Average loss: {}".format(avg_loss))