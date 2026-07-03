import argparse
import json
from pathlib import Path


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dpo_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_sft_rows(args):
    rows = []
    if args.dpo_pairs:
        for item in load_dpo_json(args.dpo_pairs):
            rows.append({"prompt": item["prompt"], "response": item["chosen"]})

    if args.trajectories:
        for traj in load_jsonl(args.trajectories):
            if traj.get("status") == "bad_gold":
                continue
            prompt = traj.get("prompt")
            gold_sql = traj.get("gold_sql")
            if prompt and gold_sql:
                rows.append({"prompt": prompt, "response": "SQL: " + gold_sql})
    return rows


def build_completion_only_features(tokenizer, rows, max_seq_length: int):
    features = []
    eos = tokenizer.eos_token or ""
    for row in rows:
        prompt_text = row["prompt"].rstrip() + "\n"
        response_text = row["response"].strip() + eos
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        if len(input_ids) > max_seq_length:
            overflow = len(input_ids) - max_seq_length
            if overflow >= len(prompt_ids):
                input_ids = input_ids[-max_seq_length:]
                labels = labels[-max_seq_length:]
            else:
                input_ids = input_ids[overflow:]
                labels = labels[overflow:]

        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return features


class CompletionOnlyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [feature["labels"] for feature in features]
        model_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(label + [-100] * pad_len)

        import torch

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for Text-to-SQL prompts and SQL completions.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dpo_pairs", default=None, help="Optional dpo_pairs.json; chosen completions become SFT targets.")
    parser.add_argument("--trajectories", default=None, help="Optional trajectories.jsonl; gold SQL becomes SFT target.")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    if not args.dpo_pairs and not args.trajectories:
        raise ValueError("Pass --dpo_pairs or --trajectories.")

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    rows = build_sft_rows(args)
    if not rows:
        raise ValueError("No SFT rows were built from the provided files.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    tokenized_dataset = Dataset.from_list(
        build_completion_only_features(tokenizer, rows, args.max_seq_length)
    )
    data_collator = CompletionOnlyDataCollator(tokenizer)

    training_args = TrainingArguments(
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(Path(args.output_dir) / "adapter")


if __name__ == "__main__":
    main()
