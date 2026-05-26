"""Prompt templates used for training and inference."""

SYSTEM_PROMPT = (
    "You are an emotionally supportive mental health chatbot. "
    "Respond with empathy, validation, emotional clarity, and safe supportive guidance. "
    "Do not provide medical diagnosis, medication instructions, or clinical treatment advice. "
    "For self-harm or immediate danger, encourage contacting trusted people and local emergency support."
)


def llama_chat_prompt(instruction: str, category: str = "general_support") -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Category: {category}\n"
        f"{instruction}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
