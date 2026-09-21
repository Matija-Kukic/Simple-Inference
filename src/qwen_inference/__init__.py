import argparse
import json
import traceback



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

    from .ref_transformers.create_tokens import (
        create_tokens as tokenize_prompt,
    )
    from .ref_transformers.create_tokens import (
        save_model_inputs,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")

    args = parser.parse_args()

    path = "./src/qwen_inference/data/input_ids.json"

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
