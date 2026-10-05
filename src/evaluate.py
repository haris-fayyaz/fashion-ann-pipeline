import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    data = np.load("data/processed/fashion_mnist.npz")
    x_test, y_test = data["x_test"], data["y_test"]

    model = keras.models.load_model("models/model.h5")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("reports", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, xticks_rotation=45, colorbar=False
    )
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=120)

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)

    print(f"test loss {loss:.4f}, test accuracy {acc:.4f}")


if __name__ == "__main__":
    main()