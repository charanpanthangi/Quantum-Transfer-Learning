import numpy as np
from app import dataset


def test_generate_source_and_target_shapes():
    Xs, ys = dataset.generate_source_dataset(n_samples=20)
    Xt, yt = dataset.generate_target_dataset(n_samples=20)
    assert Xs.shape[1] == 2
    assert Xt.shape[1] == 2
    assert len(ys) == len(Xs)
    assert len(yt) == len(Xt)


def test_train_test_split():
    X, y = dataset.generate_source_dataset(n_samples=40)
    X_train, X_test, y_train, y_test = dataset.train_test_split(X, y, test_size=0.25, random_state=0)
    assert len(X_train) == 30
    assert len(X_test) == 10
    assert len(y_train) == 30
    assert len(y_test) == 10
