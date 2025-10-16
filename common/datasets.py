"""Dataset helpers for text generation."""

from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset

from .utils import CHOICE_LABELS


def format_example(example: Dict[str, Iterable[str]]) -> Tuple[str, str]:
    """Construct a prompt for a multiple-choice question and return the answer."""
    prompt = (
        "The following are multi choice questions. Give ONLY the correct option, "
        "no other words or explanation:\n"
    )
    question = example["question"]
    label = example["label"]
    answer = example["answer"]
    choices = example["choices"]

    prompt += f"Question: {question}\n"
    for idx, text in enumerate(choices):
        prompt += f"{label[idx]}: {text}\n"
    prompt += "Answer: "

    return prompt, answer


def load_calibration_dataset(
    dataset_path: str,
    split: str = "validation",
) -> List[Dict[str, object]]:
    """Load and normalize the multiple-choice dataset used for calibration."""
    raw_dataset = load_dataset(dataset_path)[split]

    def _normalize(sample: Dict[str, object]) -> Dict[str, object]:
        choices = [sample["opa"], sample["opb"], sample["opc"], sample["opd"]]
        return {
            "question": sample["question"],
            "choices": choices,
            "answer": CHOICE_LABELS[sample["cop"]],
            "label": CHOICE_LABELS,
        }

    return [_normalize(example) for example in raw_dataset]
