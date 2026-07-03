import argparse
import inspect
import json
from pathlib import Path


def supported_kwargs(callable_obj, kwargs):
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def unsupported_keys(callable_obj, kwargs):
    signature = inspect.signature(callable_obj)
    return [key for key in kwargs if key not in signature.parameters]


def load_pairs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    return [
        {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }
        for item in pairs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA DPO for execution-guided Text-to-SQL preference pairs.")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--base_adapter_path", default=None, help="Optional SFT LoRA adapter to continue with DPO.")
    parser.add_argument("--dpo_pairs", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1536)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--api_check", action="store_true", help="Print detected TRL DPOTrainer API and exit.")
    args = parser.parse_args()

    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer

    trainer_signature = inspect.signature(DPOTrainer.__init__)
    if args.api_check:
        print(trainer_signature)
        print("uses_dpo_config:", "beta" not in trainer_signature.parameters)
        print("uses_processing_class:", "processing_class" in trainer_signature.parameters)
        return

    if not args.model_name_or_path or not args.dpo_pairs or not args.output_dir:
        raise ValueError("--model_name_or_path, --dpo_pairs, and --output_dir are required for training.")

    rows = load_pairs(args.dpo_pairs)
    if not rows:
        raise ValueError("No DPO pairs found.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if args.base_adapter_path:
        print(f"Loading trainable base adapter: {args.base_adapter_path}")
        model = PeftModel.from_pretrained(model, args.base_adapter_path, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    common_training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_steps=100,
        bf16=args.bf16,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": Dataset.from_list(rows),
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    if "beta" in trainer_signature.parameters:
        training_args = TrainingArguments(**common_training_kwargs)
        trainer_kwargs.update(
            {
                "args": training_args,
                "beta": args.beta,
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
            }
        )
    else:
        from trl import DPOConfig

        dpo_config_kwargs = {
            **common_training_kwargs,
            "beta": args.beta,
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
        }
        skipped = unsupported_keys(DPOConfig.__init__, dpo_config_kwargs)
        if skipped:
            print(f"Skipping unsupported DPOConfig args for this TRL version: {skipped}")
        training_args = DPOConfig(
            **supported_kwargs(DPOConfig.__init__, dpo_config_kwargs)
        )
        trainer_kwargs["args"] = training_args

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(Path(args.output_dir) / "adapter")


if __name__ == "__main__":
    main()
