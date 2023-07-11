import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CNN import create_input_samples, CNN


######################################
# Test models on unseen test data    #
######################################

# hyperparameters
BATCH_SIZE = 128

# Load the model from .pt file
model = CNN()  # Instantiate your model
model.load_state_dict(torch.load("CNN1.pt"))
model.eval()
model_type = "CNN"  # "FCNN" or "CNN"

# Load the test data
data = pd.read_csv('Final_dataframe_test.csv')
test_input = data.drop(["RUL"], axis=1)
test_label = data.filter(["unit_number","RUL"], axis=1)

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

    avg_loss = total_loss / len(test_input)
    print("Average loss: {}".format(avg_loss))


