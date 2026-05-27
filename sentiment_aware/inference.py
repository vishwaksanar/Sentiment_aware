"""Inference helpers for base and DPO-aligned Llama models."""

from __future__ import annotations

from dataclasses import dataclass

from .prompts import SYSTEM_PROMPT


@dataclass(slots=True)
class InferenceConfig:
    model_name: str = "unsloth/Llama-3.2-3B-Instruct"
    adapter_path: str | None = None
    max_seq_length: int = 2048
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    load_in_4bit: bool = True


def load_llama_model(config: InferenceConfig, use_adapter: bool = False):
    """Load a Llama model with optional LoRA adapter for inference."""

    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Inference requires unsloth and transformers.") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
    )
    if use_adapter:
        if not config.adapter_path:
            raise ValueError("adapter_path is required when use_adapter=True")
        model.load_adapter(config.adapter_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    user_text: str,
    category: str,
    config: InferenceConfig | None = None,
) -> str:
    """Generate a supportive response for one user message."""

    config = config or InferenceConfig()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Category: {category}\nUser: {user_text}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs.shape[-1] :], skip_special_tokens=True)
    return response.strip()


def compare_base_and_adapter(user_text: str, category: str, config: InferenceConfig):
    """Generate base-model and DPO-adapter responses for comparison."""

    base_model, tokenizer = load_llama_model(config, use_adapter=False)
    base_response = generate_response(base_model, tokenizer, user_text, category, config)

    dpo_model, dpo_tokenizer = load_llama_model(config, use_adapter=True)
    dpo_response = generate_response(dpo_model, dpo_tokenizer, user_text, category, config)
    return base_response, dpo_response
