"""End-to-end generation pipeline."""

from __future__ import annotations

import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from common import (
    CHOICE_LABELS,
    create_sampling_params,
    format_example,
    load_calibration_dataset,
    load_model,
)


def generate(
    model_path: str,
    dataset_path: str,
    split: str = "validation",
    output_path: Optional[str] = None,
) -> Dict[str, List]:
    """Generate logits and predictions for the specified model and dataset."""
    model, tokenizer = load_model(model_path=model_path)
    sampling_params = create_sampling_params(CHOICE_LABELS)
    dataset = load_calibration_dataset(dataset_path=dataset_path, split=split)

    results = {
        "question": [],
        "predicted_answer": [],
        "correct_answer": [],
        "logits": [],
        "is_correct": [],
    }

    for example in tqdm(dataset):
        prompt, label = format_example(example)

        outputs = model.generate(
            prompts=prompt,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        logprob_dict = outputs[0].outputs[0].logprobs[0]
        token_ids = {
            token: next(key for key, value in logprob_dict.items() if value.decoded_token == token)
            for token in CHOICE_LABELS
        }
        logits = [logprob_dict[token_id].logprob for token_id in token_ids.values()]
        preds = CHOICE_LABELS[np.argmax(logits)]

        results["question"].append(prompt)
        results["is_correct"].append(np.array(preds == label))
        results["logits"].append(logits)
        results["predicted_answer"].append(preds)
        results["correct_answer"].append(label)

    if output_path:
        with open(output_path, "wb") as handle:
            pickle.dump(results, handle)

    torch.cuda.empty_cache()
    del model
    del tokenizer

    return results
