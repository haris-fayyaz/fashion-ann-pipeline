import csv
import os

import numpy as np
import yaml
from tensorflow import keras


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    data = np.load("data/processed/fashion_mnist.npz")

    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(p["dense_units"], activation="relu"),
        keras.layers.Dropout(p["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist = model.fit(
        data["x_train"], data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        verbose=2,
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")

    keys = list(hist.history)
    with open("models/history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(p["epochs"]):
            writer.writerow([i + 1] + [hist.history[k][i] for k in keys])


if __name__ == "__main__":
    main()