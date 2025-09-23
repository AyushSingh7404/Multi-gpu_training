2
# # trying to test with bert-base-uncased
import datasets
import torch
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from accelerate import PartialState
import os
import wandb

# Initialize WandB without requiring API key (assumes wandb login has been done)
os.environ["WANDB_PROJECT"] = "bert-sft"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB__SERVICE_WAIT"] = "300"

# Device map for DDP
device_map = "DDP"
if device_map == "DDP":
    device_string = PartialState().process_index
    device_map = {"": device_string}

# Dataset
dataset_name = "timdettmers/openassistant-guanaco"
dataset = datasets.load_dataset(dataset_name, split="train")

# Model + tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # Adjust based on your task
    device_map=device_map,
)

# Preprocess dataset
def preprocess_function(examples):
    result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    result["labels"] = [0] * len(examples["text"])  # Replace with actual labels
    return result

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["query", "key", "value"],
    modules_to_save=["classifier"],
)

# Apply LoRA
model = get_peft_model(model, peft_config)

# Training config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=50,
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
    max_grad_norm=1.0,
    num_train_epochs=1,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    logging_first_step=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("./saved_bert_model")
tokenizer.save_pretrained("./saved_bert_model")

# Cleanup DDP
import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()