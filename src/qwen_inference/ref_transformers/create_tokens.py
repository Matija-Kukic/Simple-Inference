import torch
from transformers import AutoTokenizer, BatchEncoding


def create_tokens(prompt: str) -> BatchEncoding:
    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {
            "role": "user",
            "content": "Give me a short introduction to large language models.",
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
    )

    return model_inputs


class ModelInputIOError(Exception):
    pass


def save_model_inputs(model_inputs: BatchEncoding, path: str) -> None:
    try:
        tensors = {
            key: value.cpu()
            for key, value in model_inputs.items()
            if isinstance(value, torch.Tensor)
        }

        torch.save(tensors, path)

    except (OSError, RuntimeError) as exc:
        raise ModelInputIOError(f"Failed to save model inputs to {path}") from exc
