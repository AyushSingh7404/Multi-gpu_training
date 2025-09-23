# Based on a script from: https://github.com/huggingface/trl/issues/1303
# Run this with DDP with "accelerate launch test_scripts/test_ddp.py"


from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from accelerate import PartialState
import os

# WandB env vars - no login popup
os.environ["WANDB_PROJECT"] = "tinyllama-sft"
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
dataset = load_dataset(dataset_name, split="train")  # Use a subset for testing

# Model + tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=False,
    device_map=device_map,
)

# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,                   # lower dropout for small dataset
    r=16,                                # lower rank for stability
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
    modules_to_save=["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"],
)

# Training config - safer for 4-bit TinyLlama
training_args = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=4,       # smaller batch for stability
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    save_strategy="steps",
    save_steps=50,
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=5e-5,                 # lower LR for 4-bit + LoRA
    fp16=True,                           # use FP16 instead of BF16 on T4
    max_grad_norm=1.0,
    num_train_epochs=1,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="linear",          # more stable LR decay
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb",
    dataset_text_field="text",
    max_length=512,
    logging_first_step=True,
)


# Trainer - we do NOT need WandbCallback explicitly
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

# Train full epoch
trainer.train()

trainer.save_model("./saved_tinyllama_model")
tokenizer.save_pretrained("./saved_tinyllama_model")


