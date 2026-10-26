"""Reproducible row-level splitting using the pipeline configuration."""

import math

import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(dataframe, config):
    """Return all rows in original order with configured train/validation/test codes."""
    processing = config.processing
    if processing.split_unit != "metadata_row":
        raise ValueError("Only metadata_row splitting is supported")
    fractions = [
        processing.train_fraction,
        processing.validation_fraction,
        processing.test_fraction,
    ]
    if any(not math.isfinite(x) or x < 0 for x in fractions) or not math.isclose(
        sum(fractions), 1
    ):
        raise ValueError("Split fractions must be nonnegative and sum to 1")
    labels = [
        config.dataset_split.labels[name] for name in ("train", "validation", "test")
    ]
    if len(set(labels)) != 3:
        raise ValueError("Split labels must be distinct")
    column = config.dataset_split.column
    if column in dataframe:
        raise ValueError(f"Split column already exists: {column}")
    if dataframe.empty:
        raise ValueError("Cannot split empty metadata")
    result = dataframe.copy()
    assignments = np.empty(len(result), dtype=object)
    remaining = np.arange(len(result))
    active = [
        (fraction, label) for fraction, label in zip(fractions, labels) if fraction > 0
    ]
    total = 1.0
    for fraction, label in active[:-1]:
        selected, remaining = train_test_split(
            remaining, train_size=fraction / total, random_state=processing.random_state
        )
        assignments[selected] = label
        total -= fraction
    assignments[remaining] = active[-1][1]
    result[column] = assignments
    return result
