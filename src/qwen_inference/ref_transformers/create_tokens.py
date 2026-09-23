import argparse
import json
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer, BatchEncoding

from qwen_inference.config import DATA_DIR


def tokenize_prompt(prompt: str) -> BatchEncoding:
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


def save_model_inputs(model_inputs: BatchEncoding, path: Path) -> None:
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


def create_tokens() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")

    args = parser.parse_args()

    path = DATA_DIR / "input_ids.json"

    print("Creating tokens.")

    try:
        model_inputs = tokenize_prompt(args.prompt)
        save_model_inputs(model_inputs, path)

        print("Tokens saved successfully.")

        with open(path) as file:
            data = json.load(file)

        print("input_ids:")
        print(data["input_ids"])

        print("attention_mask:")
        print(data["attention_mask"])

    except Exception as exc:  # noqa: BLE001
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
