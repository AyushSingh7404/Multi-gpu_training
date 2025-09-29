import os
os.environ["DISABLE_APEX"] = "1"
import time
import torch
import torch.nn.functional as F
import argparse
import json
import pandas as pd
from datasets import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import wandb
from dotenv import load_dotenv
import logging
import glob
import psutil
from PIL import Image
from huggingface_hub import login

# Import diffusers components with error handling
try:
    from diffusers import DPMSolverMultistepScheduler, AutoencoderKL, SD3Transformer2DModel
    from diffusers.optimization import get_scheduler
except ImportError as e:
    print(f"Error importing diffusers: {e}")
    print("Trying to import with XFORMERS_MORE_DETAILS=1 for debugging")
    os.environ["XFORMERS_MORE_DETAILS"] = "1"
    try:
        from diffusers import DPMSolverMultistepScheduler, AutoencoderKL, SD3Transformer2DModel
        from diffusers.optimization import get_scheduler
    except ImportError as e2:
        print(f"Still failing after XFORMERS debug: {e2}")
        # Try to disable xformers entirely
        os.environ["DIFFUSERS_USE_XFORMERS"] = "0"
        from diffusers import DPMSolverMultistepScheduler, AutoencoderKL, SD3Transformer2DModel
        from diffusers.optimization import get_scheduler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="debug.log")
logger = logging.getLogger(__name__)

# Argument parser for dynamic configuration with environment variable fallbacks
parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion 3.5 Medium with LoRA")
parser.add_argument("--model_id", type=str, default=os.getenv("MODEL_ID", "stabilityai/stable-diffusion-3.5-medium"), help="Pretrained model ID")
parser.add_argument("--dataset_path", type=str, default=os.getenv("DATASET_PATH", "/workspace/cc_data"), help="Path to unzipped dataset directory")
parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR", "./results"), help="Output directory")
parser.add_argument("--num_train_epochs", type=int, default=int(os.getenv("NUM_EPOCHS", 2)), help="Number of training epochs")
parser.add_argument("--per_device_train_batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 2)), help="Batch size per device")
parser.add_argument("--gradient_accumulation_steps", type=int, default=int(os.getenv("GRAD_ACCUM_STEPS", 2)), help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=float(os.getenv("LEARNING_RATE", 5e-5)), help="Learning rate")
parser.add_argument("--resolution", type=int, default=int(os.getenv("RESOLUTION", 512)), help="Image resolution")
parser.add_argument("--logging_steps", type=int, default=int(os.getenv("LOGGING_STEPS", 20)), help="Logging steps")
parser.add_argument("--save_steps", type=int, default=int(os.getenv("SAVE_STEPS", 500)), help="Save steps")
parser.add_argument("--lora_rank", type=int, default=int(os.getenv("LORA_RANK", 16)), help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=int(os.getenv("LORA_ALPHA", 32)), help="LoRA alpha")
parser.add_argument("--run_name", type=str, default=os.getenv("RUN_NAME", "sd35-lora-custom-run"), help="W&B run name")
args = parser.parse_args()

# Step 1: Load W&B API key and HF token from env
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if not wandb_api_key:
    raise ValueError("WANDB_API_KEY not found in .env file!")
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file!")
os.environ["WANDB_API_KEY"] = wandb_api_key
os.environ["WANDB_MODE"] = "offline"  # Offline mode to avoid upload hangs

# Hugging Face login
login(hf_token)

# Step 2: Initialize Accelerator for DDP
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision="fp16")

# Step 3: Debug GPU and system usage
if accelerator.is_main_process:
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Num GPUs: {accelerator.num_processes}, Rank: {accelerator.process_index}")
    disk_usage = psutil.disk_usage('.')
    logger.info(f"Disk: {disk_usage.free / (1024**3):.2f} GB free")

# Step 4: Load dataset from JSONL
def load_custom_dataset(dataset_path):
    jsonl_path = os.path.join(dataset_path, "dataset.jsonl")
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convert to Hugging Face Dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    # Limit to first 1000 samples for quick testing
    dataset = dataset.select(range(min(1000, len(dataset))))
    
    return dataset

dataset = load_custom_dataset(args.dataset_path)

# Preprocessing transform - moved outside to avoid repeated creation
transform = transforms.Compose([
    transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(args.resolution),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def preprocess(examples):
    """Memory-efficient preprocessing function"""
    images = []
    texts = []
    
    for i, img_path in enumerate(examples["image_path"]):
        try:
            # Handle both possible path structures
            full_path = os.path.join(args.dataset_path, img_path)
            if not os.path.exists(full_path):
                # Try without the nested cc_data folder
                img_path_cleaned = img_path.replace("cc_data/", "") if img_path.startswith("cc_data/") else img_path
                full_path = os.path.join(args.dataset_path, img_path_cleaned)
            
            if os.path.exists(full_path):
                # Load and transform image
                with Image.open(full_path) as img:
                    img_rgb = img.convert("RGB")
                    transformed_img = transform(img_rgb)
                    images.append(transformed_img)
                    texts.append(examples["text"][i])
            else:
                logger.warning(f"Image not found: {full_path}")
                continue
                
        except Exception as e:
            logger.warning(f"Error processing image {img_path}: {e}")
            continue
    
    return {"images": images, "text": texts}

# Apply preprocessing with reduced batch size and no multiprocessing to avoid OOM
logger.info("Starting dataset preprocessing...")
dataset = dataset.map(
    preprocess, 
    batched=True, 
    batch_size=50,  # Reduced from 1000 to 50
    num_proc=1,     # Reduced from 4 to 1 to avoid conflicts
    remove_columns=dataset.column_names  # Remove original columns to save memory
)

# Filter out empty batches
# dataset = dataset.filter(lambda example: len(example["images"]) > 0)

logger.info(f"Dataset preprocessing complete. Final dataset size: {len(dataset)}")
dataset.set_format(type="torch", columns=["images", "text"])

# Custom collate function to handle variable batch sizes
def collate_fn(examples):
    images = []
    texts = []
    
    for example in examples:
        if isinstance(example["images"], list):
            images.extend(example["images"])
            if isinstance(example["text"], list):
                texts.extend(example["text"])
            else:
                texts.extend([example["text"]] * len(example["images"]))
        else:
            images.append(example["images"])
            texts.append(example["text"])
    
    return {"images": torch.stack(images), "text": texts}

# DataLoader with custom collate function
train_dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=args.per_device_train_batch_size, 
    shuffle=True, 
    num_workers=0,  # Set to 0 to avoid multiprocessing issues
    collate_fn=collate_fn,
    pin_memory=True
)

# Step 5: Load model components
logger.info("Loading model components...")
noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
transformer = SD3Transformer2DModel.from_pretrained(args.model_id, subfolder="transformer")
# torch.cuda.empty_cache()  # Clear GPU memory

# Enable gradient checkpointing to save memory
transformer.gradient_checkpointing = True
text_encoder.gradient_checkpointing = True

# Enable xformers if available (with error handling)
try:
    if hasattr(transformer, 'enable_xformers_memory_efficient_attention'):
        transformer.enable_xformers_memory_efficient_attention()
        logger.info("xformers memory efficient attention enabled")
    else:
        logger.info("xformers not available or not supported by this model")
except Exception as e:
    logger.warning(f"xformers not enabled: {e}")

# Apply LoRA
lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.0,
    bias="none",
)
transformer = get_peft_model(transformer, lora_config)

# Optimizer with weight decay for better training stability
optimizer = torch.optim.AdamW(
    transformer.parameters(), 
    lr=args.learning_rate,
    weight_decay=0.01,
    eps=1e-8
)

# LR scheduler
num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
num_train_steps = args.num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    name="constant",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps,
)

# Prepare with accelerator
transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    transformer, optimizer, train_dataloader, lr_scheduler
)
vae.to(accelerator.device)
text_encoder.to(accelerator.device)

# Enable memory efficient attention for VAE and text encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Step 6: Check for checkpoints
resume_from_checkpoint = False
checkpoint_dir = max(glob.glob(os.path.join(args.output_dir, "checkpoint-*")), key=os.path.getmtime, default=None)
if checkpoint_dir:
    logger.info(f"Found checkpoint: {checkpoint_dir}")
    accelerator.load_state(checkpoint_dir)
    resume_from_checkpoint = True
else:
    logger.info("No checkpoints found, starting fresh")

# Step 7: Initialize W&B
if accelerator.is_main_process:
    logger.info("Initializing W&B")
    start_time = time.time()
    wandb.init(project="sd35-finetune", name=args.run_name, config=vars(args))
    logger.info(f"W&B init took {time.time() - start_time:.2f} seconds")

# Step 8: Training loop
logger.info("Starting training")
start_time = time.time()
global_step = 0 if not resume_from_checkpoint else int(checkpoint_dir.split("-")[-1])
transformer.train()

for epoch in range(0 if not resume_from_checkpoint else 0, args.num_train_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(transformer):
            # Encode text with error handling
            try:
                text_input = tokenizer(
                    batch["text"], 
                    padding="max_length", 
                    max_length=tokenizer.model_max_length, 
                    truncation=True, 
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_input)[0]
            except Exception as e:
                logger.error(f"Error encoding text: {e}")
                continue

            # Encode images to latents with error handling
            try:
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
            except Exception as e:
                logger.error(f"Error encoding images: {e}")
                continue

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), 
                device=latents.device
            )

            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            try:
                model_pred = transformer(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=encoder_hidden_states
                ).sample
            except Exception as e:
                logger.error(f"Error in transformer forward pass: {e}")
                continue

            # Loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch")
                continue

            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

        global_step += 1

        # Logging
        if global_step % args.logging_steps == 0 and accelerator.is_main_process:
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
            wandb.log({
                "loss": loss.item(), 
                "avg_loss": avg_loss,
                "global_step": global_step,
                "epoch": epoch
            })

        # Save checkpoint
        if global_step % args.save_steps == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Memory cleanup
        if global_step % 100 == 0:
            torch.cuda.empty_cache()

    # End of epoch logging
    if accelerator.is_main_process and num_batches > 0:
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch})

logger.info(f"Training took {time.time() - start_time:.2f} seconds")

# Step 9: Save model
if accelerator.is_main_process:
    logger.info("Saving model locally")
    start_time = time.time()
    final_model_path = os.path.join(args.output_dir, "final_model")
    transformer.save_pretrained(final_model_path)
    logger.info(f"Model save took {time.time() - start_time:.2f} seconds")

    logger.info("Creating and saving W&B artifact")
    start_time = time.time()
    artifact = wandb.Artifact("stage1-sd35-medium-lora.pt", type="model")
    artifact.add_dir(final_model_path)
    try:
        artifact.upload_timeout = 300  # 5-minute timeout
        wandb.log_artifact(artifact, aliases=["latest"])
        logger.info(f"W&B artifact save took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"W&B artifact save failed: {e}")
        artifact.save()  # Save locally if fails

# Step 10: Finish W&B
if accelerator.is_main_process:
    logger.info("Finishing W&B")
    start_time = time.time()
    wandb.finish()
    logger.info(f"W&B finish took {time.time() - start_time:.2f} seconds")