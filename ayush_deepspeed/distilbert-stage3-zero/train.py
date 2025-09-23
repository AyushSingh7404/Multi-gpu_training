import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get W&B API key from .env
wandb_api_key = os.getenv("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY not found in .env file!")

# Set W&B environment
os.environ["WANDB_API_KEY"] = wandb_api_key

# Initialize Accelerator
accelerator = Accelerator()

# Initialize W&B only on main process
if accelerator.is_main_process:
    wandb.init(
        project="bert-deepspeed-training", 
        name="bert-base-stage3-2gpu",
        config={
            "model": "bert-base-uncased",
            "epochs": 1,
            "batch_size": 8,
            "gpus": 2
        }
    )

print(f"Using {accelerator.num_processes} GPUs")

# Load dataset (IMDB - small subset)
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Fix tokenization function
def tokenize_function(examples):
    # Make sure we return the correct format
    tokenized = tokenizer(
        examples["text"], 
        truncation=True, 
        padding=True, 
        max_length=256,
        return_tensors=None  # Don't return tensors here, let the trainer handle it
    )
    return tokenized

print("Tokenizing datasets...")
# Tokenize datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Use small subset for quick training
train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))  # 1K samples
eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(200))     # 200 samples

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments for ZeRO Stage 3 with CPU Offloading
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    bf16=True,  # Use bf16 for T4 GPUs
    logging_steps=20,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=100,
    deepspeed="deepspeed_config.json",  # DeepSpeed config with CPU offloading
    report_to="wandb" if accelerator.is_main_process else None,
    run_name="bert-base-stage3-offload",
    save_strategy="steps",
    save_total_limit=2,
    dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
    remove_unused_columns=False,
    label_names=["labels"],  # Explicitly specify label column name
    # Additional settings for CPU offloading
    dataloader_pin_memory=False,  # Disable pin memory when offloading to CPU
    gradient_checkpointing=True   # Enable gradient checkpointing for more memory savings
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer  # Use tokenizer instead of processing_class
)

# Start training
print("=== Starting Training ===")
trainer.train()

# Save model
print("=== Saving Model ===")
trainer.save_model("./bert_final_model")

# Save model weights in .pt format
if accelerator.is_main_process:
    torch.save(model.state_dict(), "bert_model.pt")
    print("✓ Model saved as bert_model.pt")

# Finish W&B
if accelerator.is_main_process:
    wandb.finish()
    
print("🎉 Training completed!")