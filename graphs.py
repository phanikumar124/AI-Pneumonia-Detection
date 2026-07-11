import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv("model/history.csv")

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("model/accuracy.png")
plt.show()

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("model/loss.png")
plt.show()