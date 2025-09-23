import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv

# Step 1: Load W&B API key from .env
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY not found in .env file!")
os.environ["WANDB_API_KEY"] = wandb_api_key

# Step 2: Initialize Accelerator (handles multi-GPU)
accelerator = Accelerator()

# Step 3: Load data (IMDB sentiment) - Full dataset as per your log
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
train_dataset = tokenized_dataset["train"]  # Full train (25k)
eval_dataset = tokenized_dataset["test"]    # Full test (25k)

# Step 4: Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 5: Training args with W&B logging
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    deepspeed="ds_config.json",  # ZeRO-2 config
    dataloader_num_workers=2,
    report_to="wandb",  # Enable W&B logging
    run_name="distilbert-zerotwo-2gpu",  # W&B run name
)

# Step 6: Initialize W&B - Only on main process
if accelerator.is_main_process:
    wandb.init(project="distilbert-imdb", name="zerotwo-2gpu-run", config=training_args.to_dict())

# Step 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Step 8: Train
trainer.train()

# Step 9: Save model locally and as W&B artifact - Only on main process
if accelerator.is_main_process:
    trainer.save_model("./results/final_model")
    # Create W&B artifact
    artifact = wandb.Artifact("stage2-distilbert-base-uncased.pt", type="model")
    artifact.add_dir("./results/final_model")
    wandb.log_artifact(artifact)

# Step 10: Finish W&B - Only on main process
if accelerator.is_main_process:
    wandb.finish()