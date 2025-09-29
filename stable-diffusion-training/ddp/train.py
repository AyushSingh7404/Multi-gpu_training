#!/usr/bin/env python3
"""
Stable Diffusion 3.5 Medium QLoRA Training with DDP
Optimized for dual Tesla T4 GPUs
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from accelerate import Accelerator
import logging
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToImageDataset(torch.utils.data.Dataset):
    """Custom dataset for text-to-image training"""
    
    def __init__(self, dataset, tokenizer, resolution=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get text prompt
        text = item['text']
        
        # Tokenize text
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        # Process image
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        image_tensor = self.transform(image)
        
        return {
            'input_ids': text_inputs.input_ids.squeeze(),
            'attention_mask': text_inputs.attention_mask.squeeze(),
            'pixel_values': image_tensor
        }

def setup_ddp(rank, world_size):
    """Initialize DDP process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up DDP"""
    dist.destroy_process_group()

def setup_qlora_config():
    """Configure QLoRA parameters"""
    return LoraConfig(
        r=16,                    # Low-rank dimension
        lora_alpha=32,           # LoRA scaling parameter
        target_modules=[         # Target modules for LoRA adaptation
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out"
        ],
        lora_dropout=0.1,        # LoRA dropout
        bias="none",             # No bias adaptation
        task_type=TaskType.DIFFUSION  # Task type
    )

class StableDiffusionTrainer:
    """Main trainer class for Stable Diffusion 3.5 with QLoRA"""
    
    def __init__(self, rank, world_size, args):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        
        # Setup DDP
        setup_ddp(rank, world_size)
        
        # Initialize accelerator for mixed precision training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="fp16",  # Use FP16 for T4 compatibility
        )
        
        self.setup_models()
        self.setup_dataset()
        self.setup_optimizer()
    
    def setup_models(self):
        """Load and configure models"""
        logger.info("Loading Stable Diffusion 3.5 Medium...")
        
        # Load the pipeline
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # Extract components
        self.unet = self.pipeline.transformer  # SD3 uses transformer instead of UNet
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.scheduler = self.pipeline.scheduler
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Apply QLoRA to the transformer
        lora_config = setup_qlora_config()
        self.unet = get_peft_model(self.unet, lora_config)
        
        logger.info(f"LoRA trainable parameters: {self.unet.num_parameters(only_trainable=True):,}")
        
        # Move models to GPU
        self.vae = self.vae.to(f'cuda:{self.rank}')
        self.text_encoder = self.text_encoder.to(f'cuda:{self.rank}')
        self.unet = self.unet.to(f'cuda:{self.rank}')
        
        # Wrap UNet with DDP
        self.unet = DDP(self.unet, device_ids=[self.rank])
    
    def setup_dataset(self):
        """Setup dataset and dataloader"""
        logger.info("Loading dataset...")
        
        # Load dataset from HuggingFace
        dataset = load_dataset(
            "jackyhate/text-to-image-2M",
            split=f"train[:{self.args.num_samples}]"  # Use only specified number of samples
        )
        
        # Create custom dataset
        self.train_dataset = TextToImageDataset(
            dataset, 
            self.tokenizer,
            resolution=self.args.resolution
        )
        
        # Setup distributed sampler
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Dataset size: {len(self.train_dataset)} samples")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Only optimize LoRA parameters
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.num_epochs * len(self.train_dataloader),
            eta_min=1e-6
        )
    
    def encode_images(self, images):
        """Encode images to latent space"""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def encode_prompts(self, input_ids, attention_mask):
        """Encode text prompts"""
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]
        return encoder_hidden_states
    
    def train_step(self, batch):
        """Single training step"""
        # Move batch to device
        pixel_values = batch['pixel_values'].to(f'cuda:{self.rank}')
        input_ids = batch['input_ids'].to(f'cuda:{self.rank}')
        attention_mask = batch['attention_mask'].to(f'cuda:{self.rank}')
        
        # Encode images to latents
        latents = self.encode_images(pixel_values)
        
        # Encode text prompts
        encoder_hidden_states = self.encode_prompts(input_ids, attention_mask)
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=latents.device, dtype=torch.long
        )
        
        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Compute loss
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        return loss
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training on GPU {self.rank}")
        
        self.unet.train()
        global_step = 0
        
        for epoch in range(self.args.num_epochs):
            self.train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.args.num_epochs}",
                disable=self.rank != 0  # Only show progress on rank 0
            )
            
            for step, batch in enumerate(progress_bar):
                # Forward pass
                loss = self.train_step(batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.unet.parameters() if p.requires_grad], 
                        1.0
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                
                epoch_loss += loss.item()
                
                # Update progress bar
                if self.rank == 0:
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
                    })
                
                # Save checkpoint
                if global_step % self.args.save_steps == 0 and self.rank == 0:
                    self.save_checkpoint(global_step)
            
            avg_loss = epoch_loss / len(self.train_dataloader)
            if self.rank == 0:
                logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save final model
        if self.rank == 0:
            self.save_final_model()
    
    def save_checkpoint(self, step):
        """Save training checkpoint"""
        checkpoint_dir = f"{self.args.output_dir}/checkpoint-{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save LoRA weights
        self.unet.module.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved at step {step}")
    
    def save_final_model(self):
        """Save final trained model"""
        output_dir = f"{self.args.output_dir}/final_model"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LoRA weights
        self.unet.module.save_pretrained(output_dir)
        logger.info(f"Final model saved to {output_dir}")

def main(rank, world_size, args):
    """Main training function"""
    try:
        trainer = StableDiffusionTrainer(rank, world_size, args)
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed on rank {rank}: {e}")
        raise
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion 3.5 QLoRA Training")
    
    # Training arguments
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Set world size (number of GPUs)
    world_size = 2
    
    # Launch DDP training
    import torch.multiprocessing as mp
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)