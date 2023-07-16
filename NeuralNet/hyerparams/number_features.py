from matplotlib import pyplot as plt
# plot loss after 100 epochs for different number of features

number_features = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
training_loss = [37.2318097, 36.4703135, 34.7959147, 34.2612454,32.5489311]
test_loss = [38.8370608, 43.4023569, 38.8924672, 39.9109809,34.5563668]

# plot loss after 100 epochs for different optimizers
plt.figure(figsize=(10, 5))
plt.plot(number_features, training_loss, label="Training Loss")
plt.plot(number_features, test_loss, label="Test Loss")
plt.title("Loss after 30 epochs")
plt.xlabel("Number of input features")
plt.ylabel("Loss")
plt.legend()
plt.savefig("features.png")