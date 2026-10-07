def calculate_metrics(expected, predicted):
    """
    Calculate Precision, Recall, F1 Score and Jaccard similarity.

    expected  : set of ground-truth items
    predicted : set of model-predicted items
    """

    true_positive = len(expected & predicted)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)

    # Precision
    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (
            true_positive + false_positive
        )

    # Recall
    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (
            true_positive + false_negative
        )

    # F1 Score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (
            precision + recall
        )

    # Jaccard Similarity
    union = expected | predicted

    if len(union) == 0:
        jaccard = 1.0
    else:
        jaccard = len(expected & predicted) / len(union)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard
    }