def create_tokens(prompt: str): 
    from transformers import AutoTokenizer, Qwen3ForCausalLM
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

