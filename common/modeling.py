"""Model loading and sampling configuration utilities."""

from typing import Sequence, Tuple

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


def load_model(model_path: str) -> Tuple[LLM, object]:
    """Instantiate the requested model and return it along with its tokenizer."""
    model = LLM(
        model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        guided_decoding_backend="xgrammar",
    )
    tokenizer = model.get_tokenizer()
    return model, tokenizer


def create_sampling_params(labels: Sequence[str]) -> SamplingParams:
    """Create sampling parameters constrained to the provided labels."""
    guided_decoding_params = GuidedDecodingParams(choice=list(labels))
    return SamplingParams(
        guided_decoding=guided_decoding_params,
        logprobs=len(labels),
        max_tokens=1,
    )
