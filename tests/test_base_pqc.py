import numpy as np

from app import dataset
from app import base_pqc


def test_base_forward_pass_runs():
    X, y = dataset.generate_source_dataset(n_samples=5)
    params = base_pqc.init_base_pqc_params(n_layers=1, seed=1)
    probs = base_pqc.predict_proba(X, params)
    assert probs.shape == (5,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_train_base_pqc_returns_history():
    X, y = dataset.generate_source_dataset(n_samples=10)
    params, history = base_pqc.train_base_pqc(X, y, n_epochs=2, lr=0.1, n_layers=1, seed=1)
    assert params.shape[0] == 1
    assert len(history) == 2
