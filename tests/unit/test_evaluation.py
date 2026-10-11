import pytest

from inference_security.evaluation import decision_metrics


def test_known_confusion_matrix():
    result = decision_metrics([0, 0, 1, 1], [False, True, False, True])
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert result["undefined_metrics"] == []


def test_no_predicted_positive_is_explicit():
    result = decision_metrics([0, 1], [False, False])
    assert result["precision"] is None
    assert result["recall"] == 0
    assert result["f1"] == 0
    assert result["undefined_metrics"] == ["precision"]


def test_single_class_metrics_are_explicit():
    result = decision_metrics([0, 0], [False, False])
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert result["false_positive_rate"] == 0


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [
        ([], []),
        ([0], []),
        ([True], [False]),
        ([2], [False]),
        ([0], [0]),
        ([1], ["true"]),
    ],
)
def test_invalid_metric_inputs_are_rejected(labels, predictions):
    with pytest.raises(ValueError):
        decision_metrics(labels, predictions)
