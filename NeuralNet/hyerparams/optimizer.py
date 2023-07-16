from matplotlib import pyplot as plt
# plot loss after 100 epochs for different optimizers

optimizer = ["Adam", "SGD", "RMSprop", "Adagrad"]
training_loss = [23.1959769,0,76.4283372,]
test_loss = [58.2739934,0,60.2523492,]

# plot loss after 100 epochs for different optimizers
plt.figure(figsize=(10, 5))
plt.plot(optimizer, training_loss, label="Training Loss")
plt.plot(optimizer, test_loss, label="Test Loss")
plt.title("Loss after 100 epochs")
plt.xlabel("Optimizer")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Optim_Loss_after_100_epochs.png")