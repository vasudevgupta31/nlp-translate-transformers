import numpy as np


def simple_train_val_split(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=None):
    """
    numpy implementation of train/validation split.
    """
    X = np.array(X)
    y = np.array(y)

    n_samples = len(X)
    if isinstance(test_size, float):
        val_size = int(n_samples * test_size)
    else:
        val_size = test_size

    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
