from matplotlib import pyplot as plt
# plot loss after 100 epochs for different optimizers

batch_size = ["128", "256", "512"]
training_loss = [23.1959769, 29.4311306, 28.8268240]
test_loss = [44.4540882, 46.5880180, 49.2974036]

# plot loss after 100 epochs for different optimizers
plt.figure(figsize=(10, 5))
plt.plot(batch_size, training_loss, label="Training Loss")
plt.plot(batch_size, test_loss, label="Test Loss")
plt.title("Loss after 100 epochs")
plt.xlabel("batch_size")
plt.ylabel("Loss")
plt.legend()
plt.savefig("batch_size_Loss_after_100_epochs.png")