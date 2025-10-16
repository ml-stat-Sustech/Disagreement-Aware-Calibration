"""Shared evaluation helpers."""

from typing import Iterable

import numpy as np
import torch
from netcal.metrics import ECE


def return_conf(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """Return model confidences after optional temperature scaling."""
    probs = torch.softmax(logits / temperature, dim=-1)
    confidences, _ = torch.max(probs, dim=-1)
    return confidences.detach().cpu().numpy()


def compute_ece(confidence: Iterable[float], correctness: Iterable[bool], bins: int = 10) -> float:
    """Compute the Expected Calibration Error (ECE)."""
    ece_score = ECE(bins)
    return ece_score.measure(np.array(confidence), np.array(correctness))
