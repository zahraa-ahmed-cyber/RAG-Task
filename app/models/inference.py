"""app/models/inference.py — load fine-tuned model and run inference."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from loguru import logger

from app.config import get_settings


@lru_cache(maxsize=1)
def _load_pipeline():
    """Load the fine-tuned model pipeline (cached after first call)."""
    cfg = get_settings()
    model_path = cfg.finetuned_model_path

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {model_path}. "
            "Run `python -m app.models.finetune` first."
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from peft import PeftModel, PeftConfig

        logger.info(f"Loading fine-tuned model from {model_path} …")

        # Detect if this is a PEFT/LoRA adapter
        peft_config_path = Path(model_path) / "adapter_config.json"
        if peft_config_path.exists():
            peft_cfg = PeftConfig.from_pretrained(model_path)
            base_name = peft_cfg.base_model_name_or_path
            logger.info(f"LoRA adapter detected — loading base: {base_name}")
            tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_name,
                device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.success("Fine-tuned model loaded ✓")
        return pipe

    except Exception as exc:
        logger.error(f"Failed to load fine-tuned model: {exc}")
        raise


def generate(prompt: str, context: str | None = None) -> str:
    """Run inference with the fine-tuned model.

    If context is provided, it is injected into the user message so the
    fine-tuned model can behave in a lightweight hybrid (RAG + fine-tuned)
    mode.
    """
    pipe = _load_pipeline()
    user_content = prompt
    if context and context.strip():
        user_content = (
            "Answer using the context below. If the answer is not present, "
            "say you don't have enough information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{prompt}"
        )
    formatted = (
        f"<|system|>\nYou are a helpful customer support assistant.\n"
        f"<|user|>\n{user_content}\n"
        f"<|assistant|>\n"
    )
    result = pipe(formatted)[0]["generated_text"]
    # Strip the prompt prefix
    answer = result[len(formatted):].strip()
    return answer
