import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, BatchEncoding


def create_tokens(prompt: str) -> BatchEncoding:
    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    return tokenizer(
        [text],
        return_tensors="pt",
    )


class ModelInputIOError(Exception):
    pass


def save_model_inputs(model_inputs: BatchEncoding, path: str) -> None:
    try:
        data = {
            key: value.cpu().tolist()
            for key, value in model_inputs.items()
            if isinstance(value, torch.Tensor)
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as file:
            json.dump(data, file, indent=2)

    except (OSError, TypeError) as exc:
        raise ModelInputIOError(f"Failed to save model inputs to {path}") from exc
