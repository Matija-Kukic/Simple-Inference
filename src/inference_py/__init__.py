import argparse

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
    #from .ref_transformers.create_tokens import create_tokens
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")

    args = parser.parse_args()

    print(args.prompt)
    
