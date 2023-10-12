"""
This file contains ways to compute the Exact Match and F1 scores for SQuAD.
We could have directly adopt the library like huggingface.co/spaces/evaluate-metric/squad,
but it reuires the predictions to be in a specific format (with ids and starting locations
attached to the answers).
In this simplified implementation, we only requires:
- predictions: a list of strings, each string is the predicted answer to the corresponding question
- references: a list of list of strings, each list of strings is the list of possible answers to the
  corresponding question

It is based on the the following squad metric implementations:
https://huggingface.co/spaces/evaluate-metric/squad/blob/main/compute_score.py
https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/evaluation/qa_utils.py
"""
import re
import string
from collections import Counter

import numpy as np


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Computes F1 score given the normalized prediction and ground truth strings."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Checks if the normalized prediction string exactly matches the normalized ground truth string."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Computes maximum score for a predicted answer with all reference answers."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_score(predictions, references):
    """Compute Exact Match and F1 scores for SQuAD.

    Args:
        predictions: a list of strings, each string is the predicted answer to the corresponding question
        references: a list of list of strings, each list of strings is the list of possible answers to the

    Returns:
        a dictionary with keys "em" and "f1" for Exact Match and F1 scores respectively.
    """
    if len(predictions) != len(references):
        raise ValueError("The number of predictions and references should be the same.")
    em = np.mean(
        [metric_max_over_ground_truths(exact_match_score, p, r)
         for p, r in zip(predictions, references)]
    )
    f1 = np.mean(
        [metric_max_over_ground_truths(f1_score, p, r)
         for p, r in zip(predictions, references)]
    )
    em *= 100
    f1 *= 100
    return{
        "em": em,
        "f1": f1,
    }
