"""Optional Unsloth/TRL training entrypoint for Colab or GPU environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DPOTrainingConfig:
    base_model: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    output_dir: str = "outputs/llama3_2_3b_dpo_adapter"
    max_seq_length: int = 512
    max_prompt_length: int = 128
    max_length: int = 256
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: int = 1
    learning_rate: float = 3e-6
    beta: float = 0.1


def train_dpo(dataset, config: DPOTrainingConfig | None = None):
    """Train a QLoRA DPO adapter.

    Heavy dependencies are imported inside this function so the rest of the
    project remains usable on a normal laptop.
    """

    config = config or DPOTrainingConfig()

    try:
        import torch
        from trl import DPOConfig, DPOTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "DPO training requires unsloth, trl, torch, peft, accelerate, and bitsandbytes."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
    )

    args = DPOConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=0.05,
        beta=config.beta,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        max_prompt_length=config.max_prompt_length,
        max_length=config.max_length,
    )
    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return trainer
