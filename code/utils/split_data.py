""" Split numpy data into torch TensorDatasets for training and validation
Validation is not used except for thesis sec. 8.3
"""

import math

import torch

from lib.ml_utilities import h

def split_train_val(X, device, val_prop):
    """
    Return train/val split

    :param X: ndarray
    :param device: torch.device to use for return
    :return: tuple of 2 torch datasets on device
    """
    total_len = X.shape[0]
    X = torch.tensor(X, dtype=torch.float32, device=device)
    val_len = math.ceil(val_prop * total_len)
    train_len = total_len - val_len

    X_train = X[:train_len]
    X_val = X[train_len: train_len + val_len]

    X_train = torch.utils.data.TensorDataset(X_train)
    X_val = torch.utils.data.TensorDataset(X_val)
    return X_train, X_val
