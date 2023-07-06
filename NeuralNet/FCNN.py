###########################################################################
#                    AI in Production Engineering                         #
#                             SS 2023                                     #
#                                                                         #
#                 Predictive Maintenance Group 5                          #
#                                                                         #
#                                                                         #
#                                                                         #
#                                                                         #
###########################################################################
"""
TODO:
    - define loss function
    - define optimizer
    - include normalization
    - include health indicator
    - include test and train split
    - ...
"""
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd

from utils import preprocessing

class FCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 20)     # TODO: change input dimension to dimension of input feature array -> done?
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc_end = nn.Linear(20, 1)

        # add Xavier initialization
        weights = [self.fc1.weight, self.fc2.weight, self.fc3.weight, self.fc_end.weight]
        biases = [self.fc1.bias, self.fc2.bias, self.fc3.bias, self.fc_end.bias]
        for w in weights:
            nn.init.xavier_uniform_(w)
        for b in biases:
            nn.init.zeros_(b)
        


    def forward(self, x):
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc_end(x)

        return x
    
def lossfunction(RUL_target, RUL_predicted):
    loss = 0
    ### TODO
    # define loss function
    # e.g. MSE
    # curently not used
    loss = MSELoss(RUL_target - RUL_predicted)
    ###

    return loss
    
def train_model(data_path="merged_dataframe.csv"):

    # setting a seed for pytorch as well as one for numpy
    torch.manual_seed(2)
    np.random.seed(2)

    # hyperparameters
    NUM_EPOCHS = 400
    LEARNING_RATE = 0.005 
    BATCH_SIZE = 512


    # loading the data
    # TODO: data is currently loaded without preprocessing and then being preprocessed in utils.py
    # Future implementation should reference preprocessed data directly via data_path
    #df = pd.read_csv(data_path,sep=" ",header=None)
    #df_input = preprocessing(data=df)
    df_input = pd.read_csv(data_path, sep=",")
    print("input_shape: ",df_input.shape)
    # load real RUL
    # filter column with label "RUL"
    RUL_target = df_input.filter(["RUL"], axis=1)
    # creating an instance of our neural network class
    imput_dim = df_input.shape[1]
    print("input_dim: ",imput_dim)
    model = FCNN(imput_dim)
    

    # creating a list for predictions and an array for our loss function values
    predictions = []
    losses = np.zeros(NUM_EPOCHS)

    # Create an instance of the mean squared error (MSE) loss function
    loss_function = nn.MSELoss()        

    # creating an instance of the Adam optimizer with the specified learning rate
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # The scheduler will reduce the learning rate of the optimizer by a factor of 0.5 after 1000 epochs
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    for epoch in range(1, NUM_EPOCHS + 1):
        # use each sample once 
        # TODO: so far no Stochastic GD, or batchwise learning -> done
        
        # TODO: implement random shuffling of data -> done
        # TODO: implement batchwise learning (batchsize 512) -> done
        # TODO: add RUL to data input or find workaround. Right now, I have to find out which RUL belongs to which sample. And when one sample ends -> done

        # Shuffle the data
        df_input_shuffled = df_input.sample(frac=1).reset_index(drop=True)
        RUL_target_shuffled = RUL_target.sample(frac=1).reset_index(drop=True)

        #RUL_target_shuffled = torch.tensor(RUL_target_shuffled.values, dtype=torch.float32)

        # Calculate the total number of batches
        num_batches = df_input.shape[0] // BATCH_SIZE

        for batch in range(num_batches):
            # Get the batch indices
            start_idx = batch * BATCH_SIZE
            end_idx = (batch + 1) * BATCH_SIZE

            # Get the batch input and target tensors
            batch_input = df_input_shuffled.iloc[start_idx:end_idx].values
            batch_target = RUL_target_shuffled.iloc[start_idx:end_idx].values

            # Convert the batch input to a tensor
            input_tensor = torch.tensor(batch_input, dtype=torch.float32)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            RUL_predicted = model(input_tensor)

            # Convert the batch target to a tensor
            target_tensor = torch.tensor(batch_target, dtype=torch.float32)

            # Compute loss
            loss = loss_function(target_tensor, RUL_predicted)
            # Compute RMSE
            loss = torch.sqrt(loss)     # paper says RMSE
            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

        # updating learning rate
        scheduler.step()
        losses[epoch - 1] = loss.detach().numpy()

        # Save predictions every 100 epochs for plotting later
        if epoch % 100 == 0:
            print("Epoch: %d, Loss: %.7f" % (epoch, losses[epoch - 1]))


    torch.save(model.state_dict(), "FCNN.pt")

    # plot the loss value over the epochs
    plt.tight_layout()
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("loss_FCNN.png")


def main():
    train_model()


if __name__ == "__main__":
    main()


    






