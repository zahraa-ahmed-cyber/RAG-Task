"""
app/models/finetune.py
──────────────────────
Fine-tunes a small causal-LM (default: TinyLlama-1.1B-Chat) on the
Bitext Q&A pairs using LoRA (PEFT) + SFTTrainer from TRL.

Usage:
    python -m app.models.finetune \
        --data     data/processed/finetune_pairs.jsonl \
        --output   data/finetuned_model \
        --base     TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --epochs   2 \
        --max_samples 500

The fine-tuned adapter is saved to --output and can be loaded via
app/models/inference.py.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

from loguru import logger


# ── chat template ─────────────────────────────────────────────────────────────

def _format_row(row: dict) -> str:
    """Format a Q&A pair as a chat-style string."""
    return (
        f"<|system|>\nYou are a helpful customer support assistant.\n"
        f"<|user|>\n{row['instruction']}\n"
        f"<|assistant|>\n{row['response']}"
    )


# ── data loader ───────────────────────────────────────────────────────────────

def _load_jsonl(path: str, max_samples: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_samples and len(rows) > max_samples:
        import random; random.seed(42)
        rows = random.sample(rows, max_samples)
    return rows


# ── training ──────────────────────────────────────────────────────────────────

def train(
    data_path: str,
    output_dir: str,
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    epochs: int = 2,
    max_samples: int = 500,
    batch_size: int = 4,
    lr: float = 2e-4,
) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        raise SystemExit(f"Missing dependency: {e}.  Run: pip install peft trl transformers datasets")

    logger.info(f"Base model : {base_model}")
    logger.info(f"Data       : {data_path}")
    logger.info(f"Output     : {output_dir}")

    # Load data
    rows = _load_jsonl(data_path, max_samples)
    logger.info(f"Training samples: {len(rows)}")
    texts = [_format_row(r) for r in rows]
    dataset = Dataset.from_dict({"text": texts})

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        trust_remote_code=True,
    )

    # LoRA config
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Training args (compatible with multiple TRL versions).
    use_cuda = torch.cuda.is_available()
    sft_cfg_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        # Keep precision settings conservative for broad HW compatibility.
        fp16=use_cuda,
        bf16=False,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
    )
    if "max_seq_length" in inspect.signature(SFTConfig.__init__).parameters:
        sft_cfg_kwargs["max_seq_length"] = 512
    if "dataset_text_field" in inspect.signature(SFTConfig.__init__).parameters:
        sft_cfg_kwargs["dataset_text_field"] = "text"
    training_args = SFTConfig(**sft_cfg_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        # Newer TRL renamed tokenizer -> processing_class.
        trainer_kwargs["processing_class"] = tokenizer
    if "max_seq_length" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["max_seq_length"] = 512
    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting fine-tuning …")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.success(f"Fine-tuned model saved → {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Windows: TRL reads bundled .jinja templates as UTF-8; default cp1252 breaks imports.
    # UTF-8 mode must be enabled at interpreter startup (PEP 540).
    if sys.platform == "win32" and not sys.flags.utf8_mode:
        os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.orig_argv[1:]])

    parser = argparse.ArgumentParser(description="Fine-tune with LoRA on Bitext dataset")
    parser.add_argument("--data",        default="data/processed/finetune_pairs.jsonl")
    parser.add_argument("--output",      default="data/finetuned_model")
    parser.add_argument("--base",        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--epochs",      type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size",  type=int, default=4)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        base_model=args.base,
        epochs=args.epochs,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
