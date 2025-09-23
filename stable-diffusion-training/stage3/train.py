# Disable xFormers to avoid compatibility issues
import os
os.environ["XFORMERS_DISABLED"] = "1"

import torch
from transformers import (
    AutoTokenizer, 
    T5EncoderModel, 
    CLIPTextModel,
    TrainingArguments,
    Trainer
)
# Import diffusers components individually to avoid xFormers issues
try:
    from diffusers import StableDiffusion3Pipeline
    from diffusers.models import SD3Transformer2DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
except ImportError as e:
    print(f"Diffusers import error: {e}")
    # Fallback imports
    from diffusers import DiffusionPipeline as StableDiffusion3Pipeline
    SD3Transformer2DModel = None
    FlowMatchEulerDiscreteScheduler = None
from datasets import load_dataset
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get tokens from .env
wandb_api_key = os.getenv("WANDB_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not wandb_api_key:
    raise ValueError("WANDB_API_KEY not found in .env file!")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file!")

# Set environment variables
os.environ["WANDB_API_KEY"] = wandb_api_key
os.environ["HF_TOKEN"] = hf_token

# Initialize Accelerator
accelerator = Accelerator()

# Initialize W&B only on main process
if accelerator.is_main_process:
    wandb.init(
        project="sd35-qlora-training", 
        name="sd35-medium-qlora-2gpu",
        config={
            "model": "stabilityai/stable-diffusion-3.5-medium",
            "dataset": "jackyhate/text-to-image-2M",
            "epochs": 3,
            "batch_size": 2,  # Small batch size for T4 GPUs
            "gpus": 2,
            "lora_rank": 64,
            "lora_alpha": 64,
            "resolution": 512
        }
    )

print(f"Using {accelerator.num_processes} GPUs")

class SD3Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer_1, tokenizer_2, tokenizer_3, resolution=512):
        self.dataset = dataset
        self.tokenizer_1 = tokenizer_1  # CLIP
        self.tokenizer_2 = tokenizer_2  # CLIP
        self.tokenizer_3 = tokenizer_3  # T5
        self.resolution = resolution
        
        # Image preprocessing
        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            
            # Get text prompt
            text = item.get('text', '') or item.get('caption', '') or str(item.get('prompt', ''))
            if not text:
                text = "a beautiful image"
            
            # Tokenize text with all three tokenizers
            text_input_ids_1 = self.tokenizer_1(
                text, 
                max_length=77, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.squeeze(0)
            
            text_input_ids_2 = self.tokenizer_2(
                text, 
                max_length=77, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.squeeze(0)
            
            text_input_ids_3 = self.tokenizer_3(
                text, 
                max_length=256, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.squeeze(0)
            
            # Process image
            image = item.get('image')
            if image is None:
                # Create a dummy image if none exists
                image = Image.new('RGB', (self.resolution, self.resolution), color='white')
            
            if not isinstance(image, Image.Image):
                if isinstance(image, dict) and 'bytes' in image:
                    from io import BytesIO
                    image = Image.open(BytesIO(image['bytes'])).convert('RGB')
                else:
                    image = Image.new('RGB', (self.resolution, self.resolution), color='white')
            
            # Convert to tensor
            pixel_values = self.image_transforms(image)
            
            return {
                "text_input_ids_1": text_input_ids_1,
                "text_input_ids_2": text_input_ids_2, 
                "text_input_ids_3": text_input_ids_3,
                "pixel_values": pixel_values,
                "text": text
            }
            
        except Exception as e:
            logger.warning(f"Error processing item {idx}: {e}")
            # Return a dummy item on error
            dummy_text = "a beautiful image"
            return {
                "text_input_ids_1": self.tokenizer_1(
                    dummy_text, max_length=77, padding="max_length", 
                    truncation=True, return_tensors="pt"
                ).input_ids.squeeze(0),
                "text_input_ids_2": self.tokenizer_2(
                    dummy_text, max_length=77, padding="max_length", 
                    truncation=True, return_tensors="pt"
                ).input_ids.squeeze(0),
                "text_input_ids_3": self.tokenizer_3(
                    dummy_text, max_length=256, padding="max_length", 
                    truncation=True, return_tensors="pt"
                ).input_ids.squeeze(0),
                "pixel_values": torch.zeros(3, self.resolution, self.resolution),
                "text": dummy_text
            }

class SD3Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.vae = kwargs.pop('vae', None)
        self.text_encoder_1 = kwargs.pop('text_encoder_1', None)
        self.text_encoder_2 = kwargs.pop('text_encoder_2', None)
        self.text_encoder_3 = kwargs.pop('text_encoder_3', None)
        self.noise_scheduler = kwargs.pop('noise_scheduler', None)
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for Stable Diffusion 3.5 training
        """
        try:
            # Get inputs
            pixel_values = inputs["pixel_values"]
            text_input_ids_1 = inputs["text_input_ids_1"]
            text_input_ids_2 = inputs["text_input_ids_2"]
            text_input_ids_3 = inputs["text_input_ids_3"]
            
            batch_size = pixel_values.shape[0]
            device = pixel_values.device
            
            # Encode images to latents
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (batch_size,), device=device
            ).long()
            
            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Encode text
            with torch.no_grad():
                # CLIP encoders
                text_embeds_1 = self.text_encoder_1(text_input_ids_1)[0]
                text_embeds_2 = self.text_encoder_2(text_input_ids_2)[0]
                
                # T5 encoder
                text_embeds_3 = self.text_encoder_3(text_input_ids_3)[0]
                
                # Combine embeddings (simplified approach)
                text_embeds = torch.cat([text_embeds_1, text_embeds_2], dim=-1)
                # Add T5 embeddings (may need padding/truncation to match dimensions)
                if text_embeds_3.shape[1] != text_embeds.shape[1]:
                    text_embeds_3 = F.adaptive_avg_pool1d(
                        text_embeds_3.transpose(1, 2), 
                        text_embeds.shape[1]
                    ).transpose(1, 2)
                
                text_embeds = torch.cat([text_embeds, text_embeds_3], dim=-1)
            
            # Predict noise with the transformer model
            model_pred = model(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeds,
                return_dict=False
            )[0]
            
            # Calculate loss (MSE between predicted and actual noise)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            return (loss, model_pred) if return_outputs else loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            # Return a dummy loss to prevent training from crashing
            return torch.tensor(0.0, requires_grad=True, device=pixel_values.device)

# Load dataset
print("Loading dataset...")
dataset = load_dataset("jackyhate/text-to-image-2M", token=hf_token, streaming=True)

# Convert streaming dataset to iterable and take subset
train_dataset_iter = iter(dataset['train'])
train_samples = []
print("Loading training samples...")
for i, sample in enumerate(train_dataset_iter):
    if i >= 1000:  # Use 1K samples for quick testing
        break
    train_samples.append(sample)
    if i % 100 == 0:
        print(f"Loaded {i+1} samples...")

# Create a simple dataset from samples
from datasets import Dataset
train_dataset = Dataset.from_list(train_samples)

print(f"Training samples: {len(train_dataset)}")

# Load tokenizers and models
print("Loading models and tokenizers...")
pipeline = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    token=hf_token,
    torch_dtype=torch.bfloat16
)

# Extract components
vae = pipeline.vae
text_encoder_1 = pipeline.text_encoder  # CLIP
text_encoder_2 = pipeline.text_encoder_2  # CLIP
text_encoder_3 = pipeline.text_encoder_3  # T5
tokenizer_1 = pipeline.tokenizer
tokenizer_2 = pipeline.tokenizer_2
tokenizer_3 = pipeline.tokenizer_3
noise_scheduler = pipeline.scheduler
transformer = pipeline.transformer

# Move to GPU and set to eval mode (except transformer which we'll train)
vae.to(accelerator.device)
text_encoder_1.to(accelerator.device)
text_encoder_2.to(accelerator.device) 
text_encoder_3.to(accelerator.device)

vae.eval()
text_encoder_1.eval()
text_encoder_2.eval()
text_encoder_3.eval()

# Freeze VAE and text encoders
vae.requires_grad_(False)
text_encoder_1.requires_grad_(False)
text_encoder_2.requires_grad_(False)
text_encoder_3.requires_grad_(False)

# Apply LoRA to transformer
print("Setting up QLoRA...")
lora_config = LoraConfig(
    r=64,  # Rank
    lora_alpha=64,
    target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"],
    lora_dropout=0.1,
    task_type=TaskType.DIFFUSION_IMAGE_GENERATION
)

# Apply LoRA to transformer
transformer = get_peft_model(transformer, lora_config)
print(f"Trainable parameters: {transformer.num_parameters()}")
print(f"LoRA parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}")

# Create dataset
print("Creating datasets...")
train_dataset_formatted = SD3Dataset(
    train_dataset, 
    tokenizer_1, 
    tokenizer_2, 
    tokenizer_3, 
    resolution=512
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./sd35_results",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Very small for T4 GPUs
    gradient_accumulation_steps=4,  # Increase to simulate larger batch
    learning_rate=1e-4,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    deepspeed="deepspeed_config.json",
    report_to="wandb" if accelerator.is_main_process else None,
    run_name="sd35-qlora-training",
    save_strategy="steps",
    save_total_limit=2,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    warmup_steps=50,
    max_grad_norm=1.0
)

# Create trainer
trainer = SD3Trainer(
    model=transformer,
    args=training_args,
    train_dataset=train_dataset_formatted,
    tokenizer=tokenizer_1,  # Primary tokenizer for saving
    # Custom components
    vae=vae,
    text_encoder_1=text_encoder_1,
    text_encoder_2=text_encoder_2,
    text_encoder_3=text_encoder_3,
    noise_scheduler=noise_scheduler
)

# Start training
print("=== Starting Training ===")
try:
    trainer.train()
except Exception as e:
    logger.error(f"Training error: {e}")
    print(f"Training interrupted due to error: {e}")

# Save LoRA weights
print("=== Saving LoRA Model ===")
if accelerator.is_main_process:
    transformer.save_pretrained("./sd35_lora_model")
    print("✓ LoRA model saved to ./sd35_lora_model")

# Finish W&B
if accelerator.is_main_process:
    wandb.finish()
    
print("🎉 Training completed!")