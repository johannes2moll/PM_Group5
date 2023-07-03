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


    def forward(self, x):
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc_end(x)

        return x
    
def lossfunction(RLU_real, RLU_predicted):
    loss = 0
    ### TODO
    # define loss function
    # e.g. MSE
    # curently not used
    loss = MSELoss(RLU_real - RLU_predicted)
    ###

    return loss
    
def train_model(data_path="../../CMAPSSdata/train_FD001.txt"):

    # setting a seed for pytorch as well as one for numpy
    torch.manual_seed(2)
    np.random.seed(2)

    # hyperparameters
    NUM_EPOCHS = 4_000 
    LEARNING_RATE = 0.01 

    # loading the data
    # TODO: data is currently loaded without preprocessing and then being preprocessed in utils.py
    # Future implementation should reference preprocessed data directly via data_path
    df = pd.read_csv(data_path,sep=" ",header=None)
    df_input = preprocessing(data=df)
    print("input_shape: ",df_input.shape)
    # load real RLU
    RLU_real = pd.read_csv("../../CMAPSSdata/RUL_FD001.txt", header=None)
    # creating an instance of our neural network class
    imput_dim = df_input.shape[1]
    print("input_dim: ",imput_dim)
    model = FCNN(imput_dim)

    # creating a list for predictions and an array for our loss function values
    predictions = []
    losses = np.zeros(NUM_EPOCHS)

    # Create an instance of the mean squared error (MSE) loss function
    loss_function = nn.MSELoss()        # TODO: own loss function could be implemented
    # creating an instance of the Adam optimizer with the specified learning rate
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # The scheduler will reduce the learning rate of the optimizer by a factor of 0.5 after 1000 epochs
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    for epoch in range(1, NUM_EPOCHS + 1):
        # use each sample once 
        # TODO: so far no Stochastic GD, or batchwise learning
        
        # TODO: implement random shuffling of data
        # TODO: implement batchwise learning
        # TODO: add RUL to data input or find workaround. Right now, I have to find out which RUL belongs to which sample. And when one sample ends
        counter = 0
        for index in range (0,df_input.shape[0] -1):
            # predict RLU
            selected_row = df_input.iloc[index].values
            input_tensor = torch.tensor(selected_row, dtype=torch.float32)
            RLU_predicted = model(input_tensor)

            # reset gradients
            optimizer.zero_grad()

            # compute loss
            # TODO: implement computation of "real" RLU value
            ##############################
            RLU_real_computed = RLU_real.iloc[counter].values # TODO: linear decrease from start may be wrong due to plateau in the beginning


            ##############################
            RLU_real_computed = torch.tensor(RLU_real_computed, dtype=torch.float32)

            loss = loss_function(RLU_real_computed, RLU_predicted)
            # propagating backward
            loss.backward()

            # updating parameters
            optimizer.step()

            # decide if we moved on to next sample
            # I do this here as we havent implemented the computation of the real RLU value yet. So far, there is only one value for all time_cycles of a sample
            # Therefore, we must detect, when we move to the next sample
            if df_input.iloc[index+1].values[1] == 1: # time_cycle equals 1  TODO: please test if this is correct, it is already late and my heads not working anymore
                counter += 1
        # updating learning rate
        scheduler.step()
        losses[epoch - 1] = loss.detach().numpy()

        # Save predictions every 100 epochs for plotting later
        if epoch % 100 == 0:
            print("Epoch: %d, Loss: %.7f" % (epoch, losses[epoch - 1]))
            #with torch.no_grad():
            #    df_input = torch.tensor(df_input, dtype=torch.float32)
            #    RLU_pred = model(df_input)
            #    RLU_pred = RLU_pred.detach().numpy()
            #    predictions.append([RLU_pred, epoch])

    torch.save(model.state_dict(), "FCNN.pt")

    # plot the loss value over the epochs
    plt.tight_layout()
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')


def main():
    train_model()


if __name__ == "__main__":
    main()


    






