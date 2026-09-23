def main() -> None:
    print("Hello from inference!")


def test() -> None:
    print("Hello from inference testing!")


def reference_inference() -> None:
    from .ref_transformers.inference import inference_full

    print("Starting reference inference.")
    output = inference_full()
    print(output)


def create_tokens() -> None:
    from .ref_transformers.create_tokens import create_tokens as create_tokens_call

    create_tokens_call()


def create_sym() -> None:
    from .config import set_symlink

    set_symlink()
