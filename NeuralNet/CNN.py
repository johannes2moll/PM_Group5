import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
def create_input_samples(df_input, RUL_target):
    # find number of unique values in 
    num_units = df_input["unit_number"].nunique()
    print("Number of unique units: {}".format(num_units))
    input_samples = []
    target_samples = []

    for unit_id in range(1, num_units+1):
        # find number of time cycles for current unit
        num_time_cycles = df_input[df_input["unit_number"] == unit_id].shape[0]
        unit_data = df_input[df_input["unit_number"] == unit_id].iloc[:, 2:]
        print("target: {}".format(RUL_target))
        unit_target = RUL_target[RUL_target["unit_number"] == unit_id]

        unit_target = unit_target.filter(["RUL"], axis=1)
        print("unit_target: {}".format(unit_target))
        num_samples = unit_data.shape[0] - num_time_cycles + 1

        for i in range(num_samples):
            input_sample = unit_data[i:i+num_time_cycles]
            target_sample = unit_target[i+num_time_cycles-1]

            input_samples.append(input_sample)
            target_samples.append(target_sample)

    return np.array(input_samples), np.array(target_samples)

def train_model(data_path="training_input.csv"):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    NUM_EPOCHS = 4000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01

    # Load the data
    df = pd.read_csv(data_path, sep=",")
    
    RUL_target = df.filter(["unit_number","RUL"], axis=1)
    df_input = df.drop(["RUL"], axis=1)
    # Preprocess the data (reshape for CNN input)

    # Preprocess the data and create input samples
    input_samples, target_samples = create_input_samples(df_input, RUL_target)

    # Create the CNN model
    model = CNN()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    # Training loop
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # Shuffle the data
        indices = np.random.permutation(len(input_samples))
        input_samples_shuffled = input_samples[indices]
        target_samples_shuffled = target_samples[indices]

        # Calculate the total number of batches
        num_batches = len(input_samples_shuffled) // BATCH_SIZE

        # Track total loss for the epoch
        total_loss = 0


        for batch in range(num_batches):
            # Get the batch indices
            start_idx = batch * BATCH_SIZE
            end_idx = (batch + 1) * BATCH_SIZE

            # Get the batch input and target tensors
            batch_input = torch.tensor(input_samples_shuffled[start_idx:end_idx], dtype=torch.float32)
            batch_target = torch.tensor(target_samples_shuffled[start_idx:end_idx], dtype=torch.float32)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(batch_input)

            # Compute loss
            loss = criterion(output, batch_target)
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

        # Print progress
        if epoch % 100 == 0:
            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
            print("Epoch: %d, Loss: %.7f" % (epoch, avg_loss))

    torch.save(model.state_dict(), "CNN.pt")

    # plot the loss value over the epochs
    plt.tight_layout()
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("../plots/loss_FCNN.png")


def main():
    train_model()


if __name__ == "__main__":
    main()


    