"""Shared data preparation helpers."""

from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def split_and_convert(
    logits: Iterable[Iterable[float]],
    labels: Sequence[str],
    correct: Sequence[bool],
    test_size: float = 0.7,
    random_state: int = 1,
    label_map_func: Callable[[Sequence[str]], Sequence[int]] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Split logits and labels into calibration/test sets and convert to tensors."""
    (
        cal_logits,
        test_logits,
        cal_labels,
        test_labels,
        cal_correct,
        test_correct,
    ) = train_test_split(
        logits,
        labels,
        correct,
        test_size=test_size,
        random_state=random_state,
    )

    cal_logits_tensor = torch.tensor(cal_logits)
    test_logits_tensor = torch.tensor(test_logits)

    if label_map_func:
        cal_labels_tensor = torch.tensor(label_map_func(cal_labels))
        test_labels_tensor = torch.tensor(label_map_func(test_labels))
    else:
        cal_labels_tensor = torch.tensor(cal_labels)
        test_labels_tensor = torch.tensor(test_labels)

    return (
        cal_logits_tensor,
        test_logits_tensor,
        cal_labels_tensor,
        test_labels_tensor,
        np.array(cal_correct),
        np.array(test_correct),
    )
