"""Precision / Recall / F1 / Jaccard for sets of nodes or edges."""


def _safe_div(a, b):
    return a / b if b else 0.0


def _from_counts(tp, fp, fn):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    jaccard = _safe_div(tp, tp + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calculate_metrics(expected, predicted):
    """expected / predicted: sets of items (node names or (src, dst) tuples)."""
    expected, predicted = set(expected), set(predicted)

    # Nothing expected and nothing predicted = perfect agreement.
    if not expected and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0,
                "tp": 0, "fp": 0, "fn": 0}

    return _from_counts(
        len(expected & predicted),
        len(predicted - expected),
        len(expected - predicted),
    )


def macro_average(metric_dicts):
    """Mean of per-sample scores (every sample counts equally)."""
    if not metric_dicts:
        return {k: 0.0 for k in ("precision", "recall", "f1", "jaccard")}
    return {
        k: sum(m[k] for m in metric_dicts) / len(metric_dicts)
        for k in ("precision", "recall", "f1", "jaccard")
    }


def micro_average(metric_dicts):
    """Pool all TP/FP/FN first (big diagrams count more)."""
    return _from_counts(
        sum(m["tp"] for m in metric_dicts),
        sum(m["fp"] for m in metric_dicts),
        sum(m["fn"] for m in metric_dicts),
    )