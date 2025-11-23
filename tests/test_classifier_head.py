import numpy as np

from app.classifier_head import init_head, train_head


def test_head_predictions():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(6, 3)).astype(np.float32)
    head = init_head(n_features=3, seed=1)
    logits = head.predict_logits(features)
    probs = head.predict_proba(features)
    assert logits.shape == (6,)
    assert probs.shape == (6,)


def test_train_head_runs():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(8, 3)).astype(np.float32)
    y = (rng.random(8) > 0.5).astype(int)
    head, history = train_head(features, y, lr=0.1, n_epochs=3, seed=2)
    assert len(history) == 3
    assert head.weights.shape == (3,)
