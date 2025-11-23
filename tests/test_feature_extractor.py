import numpy as np

from app import dataset
from app import base_pqc
from app.feature_extractor import extract_features


def test_extract_features_shape():
    X, y = dataset.generate_source_dataset(n_samples=4)
    params = base_pqc.init_base_pqc_params(n_layers=1, seed=0)
    features = extract_features(X, params, n_observables=3)
    assert features.shape == (4, 3)
