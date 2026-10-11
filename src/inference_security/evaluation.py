"""Binary decision metrics with explicit undefined values."""


def decision_metrics(labels, predictions):
    labels = tuple(labels)
    predictions = tuple(predictions)

    if not labels or len(labels) != len(predictions):
        raise ValueError("Nonempty, equally sized inputs required.")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("Labels must be binary integers.")
    if any(type(prediction) is not bool for prediction in predictions):
        raise ValueError("Predictions must be booleans.")

    tn = fp = fn = tp = 0
    for label, prediction in zip(labels, predictions, strict=True):
        if label == 0:
            if prediction:
                fp += 1
            else:
                tn += 1
        elif prediction:
            tp += 1
        else:
            fn += 1

    undefined = []

    def divide(name, numerator, denominator):
        if denominator == 0:
            undefined.append(name)
            return None
        return numerator / denominator

    return {
        "records": len(labels),
        "positive_label": 1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision": divide("precision", tp, tp + fp),
        "recall": divide("recall", tp, tp + fn),
        "f1": divide("f1", 2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": divide("false_positive_rate", fp, fp + tn),
        "undefined_metrics": undefined,
    }
