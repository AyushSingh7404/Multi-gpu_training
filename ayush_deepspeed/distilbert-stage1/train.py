import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv
import logging
import glob
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="debug.log")
logger = logging.getLogger(__name__)

# Step 1: Load W&B API key from .env
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY not found in .env file!")
os.environ["WANDB_API_KEY"] = wandb_api_key
os.environ["WANDB_MODE"] = "offline"  # Offline mode to avoid upload hangs

# Step 2: Initialize Accelerator
accelerator = Accelerator()

# Step 3: Debug GPU and system usage
if accelerator.is_main_process:
    logger.info(f"Num GPUs: {accelerator.num_processes}, Rank: {accelerator.process_index}")
    disk_usage = psutil.disk_usage('.')
    logger.info(f"Disk: {disk_usage.free / (1024**3):.2f} GB free")

# Step 4: Load data (IMDB sentiment)
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
train_dataset = tokenized_dataset["train"]  # Full 25k
eval_dataset = tokenized_dataset["test"]   # Full 25k

# Step 5: Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 6: Training args with optimized checkpointing
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=10,
    save_steps=1000,  # Save only at end to minimize I/O
    eval_strategy="steps",
    eval_steps=100,
    deepspeed="ds_config.json",
    dataloader_num_workers=2,
    report_to="wandb",
    run_name="distilbert-zeroone-2gpu",
    save_strategy="steps",
    save_total_limit=1
)

# Step 7: Check for checkpoints
resume_from_checkpoint = False
checkpoint_dir = max(glob.glob(os.path.join("./results", "checkpoint-*")), key=os.path.getmtime, default=None)
if checkpoint_dir:
    logger.info(f"Found checkpoint: {checkpoint_dir}")
    resume_from_checkpoint = checkpoint_dir
else:
    logger.info("No checkpoints found in ./results, starting fresh")

# Step 8: Initialize W&B
if accelerator.is_main_process:
    logger.info("Initializing W&B")
    start_time = time.time()
    wandb.init(project="distilbert-imdb", name="zeroone-2gpu-run", config=training_args.to_dict())
    logger.info(f"W&B init took {time.time() - start_time:.2f} seconds")

# Step 9: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer
)

# Step 10: Train
logger.info("Starting training")
start_time = time.time()
trainer.train(resume_from_checkpoint=resume_from_checkpoint)
logger.info(f"Training took {time.time() - start_time:.2f} seconds")

# Step 11: Save model and W&B artifact
if accelerator.is_main_process:
    logger.info("Saving model locally")
    start_time = time.time()
    trainer.save_model("./results/final_model")
    logger.info(f"Model save took {time.time() - start_time:.2f} seconds")

    logger.info("Creating and saving W&B artifact")
    start_time = time.time()
    artifact = wandb.Artifact("stage1-distilbert-base-uncased.pt", type="model")
    artifact.add_dir("./results/final_model")
    try:
        artifact.upload_timeout = 300  # 5-minute timeout
        wandb.log_artifact(artifact, aliases=["latest"])
        logger.info(f"W&B artifact save took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"W&B artifact save failed: {e}")
        artifact.save()  # Save locally if fails

# Step 12: Finish W&B
if accelerator.is_main_process:
    logger.info("Finishing W&B")
    start_time = time.time()
    wandb.finish()
    logger.info(f"W&B finish took {time.time() - start_time:.2f} seconds")