import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM


def inference_full() -> list[int]:
    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

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
    ).to(model.device)

    generated_ids = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
        **model_inputs,
        max_new_tokens=32768,
    )

    output_ids = generated_ids[0, model_inputs.input_ids.shape[1] :].tolist()

    return output_ids
