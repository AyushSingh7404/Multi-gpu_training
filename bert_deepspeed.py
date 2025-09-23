# train_bert.py - BERT training with all DeepSpeed stages
import os
import torch
import wandb
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import json

def train_bert_with_deepspeed(stage=2, offload_optimizer=False, offload_params=False):
    """Train BERT with specified DeepSpeed configuration"""
    
    # Model configuration
    model_name = "bert-base-uncased"
    output_dir = f"./results/bert_stage{stage}"
    if offload_optimizer:
        output_dir += "_opt_offload"
    if offload_params:
        output_dir += "_param_offload"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="bert-deepspeed-lora",
        name=f"bert_stage{stage}_offload_{offload_optimizer}_{offload_params}",
        config={
            "model": model_name,
            "deepspeed_stage": stage,
            "offload_optimizer": offload_optimizer,
            "offload_params": offload_params
        }
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.float16
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value", "key", "dense"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load IMDB dataset
    dataset = load_dataset("imdb", split="train[:2000]")
    
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 2e-4,
                "warmup_num_steps": 100
            }
        },
        "fp16": {"enabled": True},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 200000000,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 200000000,
            "contiguous_gradients": True
        }
    }
    
    if stage >= 2 and offload_optimizer:
        ds_config["zero_optimization"]["cpu_offload"] = True
    
    if stage == 3 and offload_params:
        ds_config["zero_optimization"].update({
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "stage3_prefetch_bucket_size": 50000000,
            "stage3_param_persistence_threshold": 100000
        })
    
    # Save config
    config_path = os.path.join(output_dir, "ds_config.json")
    with open(config_path, "w") as f:
        json.dump(ds_config, f, indent=2)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=500,
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
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(f"{output_dir}/lora_weights")
    
    wandb.finish()
    print(f"BERT training completed! Saved to {output_dir}")

if __name__ == "__main__":
    # Train BERT with different DeepSpeed stages
    train_bert_with_deepspeed(stage=1, offload_optimizer=False, offload_params=False)
    # train_bert_with_deepspeed(stage=2, offload_optimizer=True, offload_params=False)
    # train_bert_with_deepspeed(stage=3, offload_optimizer=True, offload_params=True)