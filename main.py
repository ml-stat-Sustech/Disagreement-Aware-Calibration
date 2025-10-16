import argparse
from typing import Optional

from calibration import daca
from generation import generate

DEFAULT_DATASET_SPLIT = "validation"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the calibration pipeline."""
    parser = argparse.ArgumentParser(description="Run disagreement-aware calibration.")

    parser.add_argument(
        "--pre-model-path",
        required=True,
        help="Path or identifier of the pre-calibration model.",
    )
    parser.add_argument(
        "--post-model-path",
        required=True,
        help="Path or identifier of the post-calibration model.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Dataset path or Hugging Face identifier used for calibration.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point orchestrating generation and calibration."""
    args = parse_args(argv)

    # Obtain logits for the pre- and post-calibrated models.
    results_pre = generate(
        model_path=args.pre_model_path,
        dataset_path=args.dataset_path,
        split=DEFAULT_DATASET_SPLIT,
    )
    results_post = generate(
        model_path=args.post_model_path,
        dataset_path=args.dataset_path,
        split=DEFAULT_DATASET_SPLIT,
    )

    # Run calibration.
    metrics = daca(results_pre, results_post)

    print("Calibration summary:")
    print(f"  Temperature: {metrics['temperature']:.4f}")
    print(f"  ECE (raw): {metrics['ece']:.6f}")
    print(f"  ECE (scaled): {metrics['ece_scaled']:.6f}")


if __name__ == "__main__":
    main()
