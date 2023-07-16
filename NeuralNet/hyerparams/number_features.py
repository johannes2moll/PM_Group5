from matplotlib import pyplot as plt
import numpy as np
# plot loss after 100 epochs for different number of features

number_features = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
training_loss = [37.2318097, 36.4703135, 34.7959147, 34.2612454, 33.5669311, 32.5489311,32.4849557,
                 31.9115235,31.4209600,31.3211210]
test_loss = [38.8370608, 43.4023569, 38.8924672, 39.9109809, 42.5677, 34.5563668,41.6561397,
             54.6614739,57.1007286,51.2144063]
width = 0.35
x_positions1 = np.arange(len(number_features)) 
x_positions2 = x_positions1 + width

# plot loss after 100 epochs for different optimizers
plt.figure(figsize=(10, 5))
plt.bar(x_positions1 +2, training_loss, width=width, label="Training loss")
plt.bar(x_positions2+2, test_loss, width=width, label="Test loss")
plt.title("Loss after 30 epochs")
plt.xlabel("Number of input features")
plt.ylabel("Loss")
plt.legend()
plt.savefig("NeuralNet/hyerparams/features.png")
