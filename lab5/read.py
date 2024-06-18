import pandas as pd
import numpy as np


def read_mnist(path: str) -> (np.ndarray, np.ndarray):
    df = pd.read_csv(path, header=None)
    images = df.drop(columns=[0]).to_numpy().reshape(-1, 28, 28)
    labels = df[[0]].to_numpy()
    labels_oh = np.eye(10)[labels.flatten()]

    return images, labels.flatten()
