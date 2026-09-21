import argparse
import traceback

import torch


def main() -> None:
    print("Hello from inference!")


def test() -> None:
    print("Hello from inference testing!")


def reference_inference() -> None:
    from .ref_transformers.inference import inference

    print("Starting reference inference")
    output = inference()
    print(output)


def create_tokens() -> None:
    from .ref_transformers.create_tokens import create_tokens, save_model_inputs

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")

    args = parser.parse_args()
    prompt = args.prompt
    path = "./src/qwen_inference/data/input_ids.pt"

    print("Creating tokens.")
    try:
        model_inputs = create_tokens(prompt)
        save_model_inputs(model_inputs, path)
        print("Tokens saved successfully.")

    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    data = torch.load(path)

    print(type(data))
    print(data)
