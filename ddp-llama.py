import os
import torch
import wandb
import json
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import deepspeed

@dataclass
class LoRAArguments:
    """LoRA configuration parameters"""
    lora_r: int = 16  # LoRA attention dimension
    lora_alpha: int = 32  # LoRA scaling factor
    lora_dropout: float = 0.1  # LoRA dropout rate
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])  # LLaMA target modules

def setup_wandb(project_name: str, run_name: str):
    """Initialize Weights & Biases for logging"""
    wandb.init(project=project_name, name=run_name, config={"architecture": "LLaMA + LoRA + DeepSpeed", "dataset": "imdb"})

def create_deepspeed_config():
    """Create DeepSpeed configuration for Stage 2 with AdamW optimizer"""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 1.0,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 2e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 'auto'}
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 2e-4, "warmup_num_steps": 100}
        }
    }

def load_and_prepare_dataset(tokenizer, max_length: int = 512):
    """Load and preprocess IMDB dataset"""
    dataset = load_dataset("imdb", split="train[:2000]")  # Use small subset for faster training
    def preprocess_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding=True, max_length=max_length)
        tokenized["labels"] = examples["label"]
        return tokenized
    return dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

def setup_lora_model(model, lora_args: LoRAArguments):
    """Configure LoRA for LLaMA model"""
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    return model

def train_model(model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", output_dir: str = "./results/llama_lora"):
    """Main function to train LLaMA with LoRA and DeepSpeed on IMDB dataset"""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize wandb
    setup_wandb("llama-lora-training", "llama_deepspeed")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        dtype=torch.float16,
        use_cache=False  # Disable cache to avoid gradient checkpointing warning
    )
    model.config.pad_token_id = tokenizer.pad_token_id  # Align model config with tokenizer

    # Setup LoRA
    model = setup_lora_model(model, LoRAArguments())

    # Load dataset
    train_dataset = load_and_prepare_dataset(tokenizer)

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Save DeepSpeed config
    deepspeed_config = create_deepspeed_config()
    config_path = os.path.join(output_dir, "deepspeed_config.json")
    with open(config_path, "w") as f:
        json.dump(deepspeed_config, f, indent=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=500,
        save_strategy="steps",
        eval_strategy="no",
        report_to="wandb",
        deepspeed=config_path,
        fp16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer  # Use processing_class to avoid deprecation warning
    )

    # Train and save model
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(f"{output_dir}/lora_weights")
    wandb.finish()

if __name__ == "__main__":
    try:
        train_model()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()