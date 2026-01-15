# Disagreement-Aware Calibration

This is the official implementation for [Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator](https://openreview.net/pdf?id=I4PJYZvfW5) at NeurIPS 2025.

This repository provides a small, modular pipeline for evaluating disagreement-aware calibration on multiple-choice question answering models. The code is structured around three packages:

- `generation/` – logic for running models and collecting logits.
- `calibration/` – temperature-optimization workflow and calibration metrics.
- `common/` – shared utilities for datasets, modeling, tensor preparation, and metrics.

The entry point `main.py` ties these pieces together and exposes a simple CLI.

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (required by `vllm`)
- Recommended Python packages:
  - `vllm`
  - `torch`
  - `datasets`
  - `netcal`
  - `tqdm`
  - `numpy`
  - `scikit-learn`

Install them with:

```bash
pip install vllm torch datasets netcal tqdm numpy scikit-learn
```

> **Note:** `vllm` may require additional setup (e.g., CUDA toolkit) depending on your environment. Refer to the [vLLM documentation](https://github.com/vllm-project/vllm) for details.

## Project Layout

```
.
├── calibration/
│   ├── __init__.py
│   └── pipeline.py
├── common/
│   ├── __init__.py
│   ├── data.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── modeling.py
│   └── utils.py
├── generation/
│   ├── __init__.py
│   └── pipeline.py
└── main.py
```

- `common/` factors out reusable helpers such as dataset normalization, label mapping, model loading, and metric computation.
- `generation/pipeline.py` handles prompt construction, vLLM inference, and persistence of raw logits.
- `calibration/pipeline.py` performs temperature scaling using disagreement-aware optimization and reports ECE metrics.

## Usage

Run the full pipeline from the repository root:

```bash
python main.py \
  --pre-model-path Qwen/Qwen2.5-7B \
  --post-model-path Qwen/Qwen2.5-7B-Instruct \
  --dataset-path openlifescienceai/medmcqa
```

Required arguments:

- `--pre-model-path` – model identifier or local path for the baseline model.
- `--post-model-path` – model identifier or local path for the post-training / instruction-tuned model.
- `--dataset-path` – Hugging Face dataset identifier or local dataset path.

The script uses the dataset split `validation` by default (adjust inside `main.py` if you need a different split). It prints calibration metrics after completion. Generation results can be serialized by extending the CLI or calling `generation.generate(..., output_path="...")` directly.

## Custom Integration

Both pipelines are exposed as importable functions:

- `generation.generate(model_path, dataset_path, split="validation", output_path=None)`
- `calibration.daca(results_pre, results_post, temperature_epochs=400, temperature_batch_size=256, temperature_lr=0.1, ece_bins=10)`

This makes it straightforward to:

- Experiment with different model checkpoints or datasets programmatically.
- Swap in alternative calibration strategies.
- Aggregate metrics across multiple runs.

# Citation

If you find this useful in your research, please consider citing:

```
@misc{luo2025pretrainedllmsecretlyunsupervised,
      title={Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator}, 
      author={Beier Luo and Shuoyuan Wang and Sharon Li and Hongxin Wei},
      year={2025},
      eprint={2505.16690},
}
```

