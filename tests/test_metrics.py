from icr.evaluation.metrics import compute_binary_metrics

def test_compute_binary_metrics_keys():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    y_prob = [0.1, 0.9, 0.2, 0.4]

    out = compute_binary_metrics(y_true, y_pred, y_prob)
    assert set(out.keys()) == {
        "roc_auc",
        "pr_auc",
        "f1",
        "balanced_accuracy",
        "brier",
    }
