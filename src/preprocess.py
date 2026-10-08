from locale import normalize
import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


MEAN, STD = 0.286, 0.353


def normalize(x):
    # scale to [0, 1], then standardize with Fashion-MNIST mean and std
    return (x.astype("float32") / 255.0 - MEAN) / STD

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    raw = np.load("data/raw/fashion_mnist.npz")
    y_train_full = raw["y_train"]

    x_train_full = normalize(raw["x_train"])
    x_test = normalize(raw["x_test"])

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=params["val_size"],
        random_state=params["seed"],
        stratify=y_train_full,
    )

    os.makedirs("data/processed", exist_ok=True)
    np.savez(
        "data/processed/fashion_mnist.npz",
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=raw["y_test"],
    )
    print(f"train {x_train.shape}, val {x_val.shape}, test {x_test.shape}")
    print(f"pixel range: {x_train.min():.1f} to {x_train.max():.1f}")


if __name__ == "__main__":
    main()