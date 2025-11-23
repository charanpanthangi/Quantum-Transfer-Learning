from app import dataset
from app.transfer_pipeline import run_all_experiments


def test_run_all_experiments_executes():
    Xs, ys = dataset.generate_source_dataset(n_samples=20)
    Xt, yt = dataset.generate_target_dataset(n_samples=20)
    results = run_all_experiments(Xs, ys, Xt, yt, n_base_epochs=2, n_target_epochs=2)
    assert "frozen" in results
    assert "finetune" in results
    assert 0 <= results["frozen"]["accuracy"] <= 1
    assert 0 <= results["finetune"]["accuracy"] <= 1
