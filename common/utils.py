"""Common helper utilities shared across modules."""

from typing import List, Sequence


CHOICE_LABELS: List[str] = ["A", "B", "C", "D"]


def label_map(labels: Sequence[str]) -> Sequence[int]:
    """Convert label characters to integer indices."""
    mapping = {label: index for index, label in enumerate(CHOICE_LABELS)}
    try:
        return [mapping[label] for label in labels]
    except KeyError as exc:
        raise ValueError(f"Unsupported label encountered: {exc.args[0]}") from exc
