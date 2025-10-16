"""End-to-end calibration pipeline."""

from __future__ import annotations

from typing import Dict, Sequence

from common import compute_ece, label_map, return_conf, split_and_convert

from .optimization import optimize_temperature


def daca(
    results_pre: Dict[str, Sequence],
    results_post: Dict[str, Sequence],
    *,
    temperature_epochs: int = 400,
    temperature_batch_size: int = 256,
    temperature_lr: float = 0.1,
    ece_bins: int = 10,
) -> Dict[str, float]:
    """Run Disagreement-Aware Calibration on pre- and post-processed model outputs."""
    logits_pre = results_pre["logits"]
    labels_pre = results_pre["correct_answer"]
    correct_pre = results_pre["is_correct"]

    logits_post = results_post["logits"]
    labels_post = results_post["correct_answer"]
    correct_post = results_post["is_correct"]

    pre_data = split_and_convert(logits_pre, labels_pre, correct_pre, label_map_func=label_map)
    post_data = split_and_convert(logits_post, labels_post, correct_post, label_map_func=label_map)

    cal_logits_pre, *_ = pre_data
    cal_logits_post, test_logits_post, _, _, _, test_correct_post = post_data

    temperature = optimize_temperature(
        cal_logits_pre,
        cal_logits_post,
        epochs=temperature_epochs,
        batch_size=temperature_batch_size,
        learning_rate=temperature_lr,
    )

    conf_test = return_conf(test_logits_post)
    conf_test_scaled = return_conf(test_logits_post, temperature)

    ece = compute_ece(conf_test, test_correct_post, bins=ece_bins)
    ece_scaled = compute_ece(conf_test_scaled, test_correct_post, bins=ece_bins)

    print("=" * 40 + "ECE" + "=" * 40)
    print(ece)
    print("=" * 40 + "ECE Scaled" + "=" * 40)
    print(ece_scaled)

    return {
        "temperature": temperature,
        "ece": ece,
        "ece_scaled": ece_scaled,
    }
