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

        ### MY MODEL ###
        # input samples have shape 30x13 (30: time cycles, 13: features)
        # output samples have shape 1x1 (1: RUL)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=10, kernel_size=5, stride=1, padding=2)
        #self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        #self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=30, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(30*30, 100)    # TODO
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        #print("x.shape: {}".format(x.shape))
        #x = self.pool(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        #x = self.pool(x)
        #print("x.shape: {}".format(x.shape))
        x = self.conv3(x)
        x = torch.tanh(x)
        #print("x.shape: {}".format(x.shape))
        #x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
    
def create_input_samples(df_input, RUL_target):
    # find number of unique values in 
    num_units = df_input["unit_number"].nunique()
    # print("Number of unique units: {}".format(num_units))
    input_samples = []
    target_samples = []

    for unit_id in range(1, num_units+1):
        # find number of time cycles for current unit
        num_time_cycles = 30
        unit_data = df_input[df_input["unit_number"] == unit_id].iloc[:, 2:]
        unit_data = unit_data.filter(["unit_number","time_cycle","sensor_2",
                                      "Degradation Classification"], axis=1)
        unit_target = RUL_target[RUL_target["unit_number"] == unit_id]

        unit_target = unit_target.filter(["RUL"], axis=1).to_numpy()
        #print("unit_target: {}".format(unit_target.shape))
        num_samples = unit_data.shape[0] - num_time_cycles + 1
        #print("num_samples: {}".format(num_samples))
        #print("num_time_cycles: {}".format(num_time_cycles))
        for i in range(num_samples):
            input_sample = unit_data[i:i+num_time_cycles]
            target_sample = unit_target[i+num_time_cycles-1]

            input_samples.append(input_sample)
            target_samples.append(target_sample)
    return np.array(input_samples), np.array(target_samples)

def loss_function(model, output, target):
    # implement RMSE with ridge regression
    criterion = nn.MSELoss()
    # get all weights
    weights = torch.cat([x.view(-1) for x in model.parameters()])
    # calculate loss
    loss = torch.sqrt(criterion(output, target)) + 0.1 * torch.norm(weights, 2)

    return loss



def train_model(data_path="Final_dataframe_train.csv"):
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    NUM_EPOCHS = 100    #50
    BATCH_SIZE = 128    #512
    LEARNING_RATE = 0.01

    # Load the data
    df = pd.read_csv(data_path, sep=",")
    
    RUL_target = df.filter(["unit_number","RUL"], axis=1)
    df_input = df.drop(["RUL"], axis=1)
    # Preprocess the data (reshape for CNN input)

    # Preprocess the data and create input samples
    input_samples, target_samples = create_input_samples(df_input, RUL_target)
    input_samples_swapped = input_samples.transpose(0,2,1)

    # Create the CNN model
    model = CNN()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # try different optimizer
    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    #optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    #optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE, lr_decay=0.01)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    losses = []
    test_losses = []
    predictions = []
    epochs = []
    # Training loop
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # Shuffle the data
        indices = np.random.permutation(len(input_samples_swapped))
        input_samples_shuffled = input_samples_swapped[indices]
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
            #loss = torch.sqrt(criterion(output, batch_target))
            loss = loss_function(model, output, batch_target)
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
            
            # print progress
            #if batch % 100 == 0:
            #    print("Epoch: %d, Batch: %d / %d" % (epoch, batch, num_batches))

        # Adjust the learning rate
        scheduler.step()
        # Compute the average loss for the epoch
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        ####################################
        ### Compute loss on test set     ###
        ####################################
        # Load the test data
        data = pd.read_csv('Final_dataframe_test.csv')
        test_input = data.drop(["RUL"], axis=1)
        test_input = test_input.filter(["unit_number","time_cycle","sensor_2",
                                        "Degradation Classification"], axis=1)
        test_label = data.filter(["unit_number","RUL"], axis=1)
        test_input_sampled, test_label_sampled = create_input_samples(test_input, test_label)
        test_input_samples_swapped = test_input_sampled.transpose(0,2,1)

        # Compute the loss
        criterion = nn.MSELoss()
        total_testloss = 0
        num_batches = len(test_input_sampled) // BATCH_SIZE

        for batch in range(num_batches):
            # Get the batch indices
            start_idx = batch * BATCH_SIZE
            end_idx = (batch + 1) * BATCH_SIZE

            # Get the batch input and target tensors
            batch_input = torch.tensor(test_input_samples_swapped[start_idx:end_idx], dtype=torch.float32)
            batch_target = torch.tensor(test_label_sampled[start_idx:end_idx], dtype=torch.float32)
            
            # Forward pass
            output = model(batch_input)
            predictions.append(output)
            #test_loss = criterion(output, batch_target)
            #test_loss = torch.sqrt(test_loss)
            test_loss = loss_function(model, output, batch_target)
            total_testloss += test_loss.item()

        # Compute the average test loss
        avg_testloss = total_testloss / num_batches
        test_losses.append(avg_testloss)
        # Print progress
        if epoch % 10 == 0:
            print("Epoch: %d, Training Loss: %.7f, Test Loss: %.7f" % (epoch, avg_loss, avg_testloss))
        epochs.append(epoch)

    torch.save(model.state_dict(), "CNN3.pt")

    # plot the loss value over the epochs
    plt.tight_layout()
    plt.figure()
    plt.semilogy(epochs, losses, label="Training Loss")
    plt.semilogy(epochs, test_losses, label="Test Loss")
    plt.title("Loss")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("plots\lena.png")


def main():
    train_model()


if __name__ == "__main__":
    main()
