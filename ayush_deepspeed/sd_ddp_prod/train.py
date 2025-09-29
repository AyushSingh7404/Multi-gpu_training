#!/usr/bin/env python3
"""
Production-Ready Stable Diffusion 3.5 Medium Fine-tuning with DDP
==================================================================

This script fine-tunes Stable Diffusion 3.5 Medium using LoRA with 
Distributed Data Parallel (DDP) for multi-GPU training without DeepSpeed.

Features:
- Environment variable configuration (no argparse)
- Native PyTorch DDP via Accelerate
- LoRA for parameter-efficient fine-tuning
- Comprehensive error handling and logging
- W&B integration for experiment tracking
- Memory optimization with gradient checkpointing
- Production-ready checkpoint management

Author: AI Training Pipeline
Version: 2.0 (DDP)
"""

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Disable APEX to avoid conflicts
os.environ["DISABLE_APEX"] = "1"

import torch
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import wandb
from dotenv import load_dotenv
import psutil
from PIL import Image
from huggingface_hub import login

# Import diffusers components with comprehensive error handling
try:
    from diffusers import DPMSolverMultistepScheduler, AutoencoderKL, SD3Transformer2DModel
    from diffusers.optimization import get_scheduler
except ImportError as e:
    print(f"Error importing diffusers: {e}")
    print("Attempting to resolve xformers compatibility issues...")
    
    # Try with xformers debugging enabled
    os.environ["XFORMERS_MORE_DETAILS"] = "1"
    try:
        from diffusers import DPMSolverMultistepScheduler, AutoencoderKL, SD3Transformer2DModel
        from diffusers.optimization import get_scheduler
    except ImportError as e2:
        print(f"Still failing with xformers debug: {e2}")
        
        # Disable xformers entirely as fallback
        os.environ["DIFFUSERS_USE_XFORMERS"] = "0"
        from diffusers import DPMSolverMultistepScheduler, AutoencoderKL, SD3Transformer2DModel
        from diffusers.optimization import get_scheduler


@dataclass
class TrainingConfig:
    """Configuration class for training parameters with type hints and defaults."""
    
    # Model and data paths
    model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    dataset_path: str = "/workspace/cc_data"
    output_dir: str = "./results"
    
    # Training hyperparameters
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    resolution: int = 512
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    
    # Optimization settings
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduler
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 0
    
    # Logging and saving
    logging_steps: int = 20
    save_steps: int = 500
    run_name: str = "sd35-lora-ddp-run"
    
    # Memory optimization
    gradient_checkpointing: bool = True
    enable_xformers: bool = True
    mixed_precision: str = "fp16"
    
    # Dataset processing
    max_train_samples: int = 1000  # Limit for quick testing
    preprocessing_num_workers: int = 1
    dataloader_num_workers: int = 0


def setup_logging(output_dir: str, verbose: bool = True) -> logging.Logger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        output_dir: Directory to save log files
        verbose: Whether to enable verbose logging
    
    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = logging.INFO if verbose else logging.WARNING
    
    # Set up file and console handlers
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_environment_config() -> TrainingConfig:
    """
    Load configuration from environment variables with fallbacks to defaults.
    
    Returns:
        TrainingConfig object with resolved values
    """
    # Load .env file if it exists
    load_dotenv()
    
    def get_env(key: str, default: Any, type_func: callable = str) -> Any:
        """Helper function to get environment variable with type conversion."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return type_func(value)
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid value for {key}='{value}', using default {default}")
            return default
    
    def str_to_bool(value: str) -> bool:
        """Convert string to boolean."""
        return value.lower() in ('true', '1', 'yes', 'on')
    
    # Create configuration from environment variables
    config = TrainingConfig(
        # Model and data paths
        model_id=get_env("MODEL_ID", TrainingConfig.model_id),
        dataset_path=get_env("DATASET_PATH", TrainingConfig.dataset_path),
        output_dir=get_env("OUTPUT_DIR", TrainingConfig.output_dir),
        
        # Training hyperparameters
        num_train_epochs=get_env("NUM_EPOCHS", TrainingConfig.num_train_epochs, int),
        per_device_train_batch_size=get_env("BATCH_SIZE", TrainingConfig.per_device_train_batch_size, int),
        gradient_accumulation_steps=get_env("GRAD_ACCUM_STEPS", TrainingConfig.gradient_accumulation_steps, int),
        learning_rate=get_env("LEARNING_RATE", TrainingConfig.learning_rate, float),
        resolution=get_env("RESOLUTION", TrainingConfig.resolution, int),
        
        # LoRA configuration
        lora_rank=get_env("LORA_RANK", TrainingConfig.lora_rank, int),
        lora_alpha=get_env("LORA_ALPHA", TrainingConfig.lora_alpha, int),
        lora_dropout=get_env("LORA_DROPOUT", TrainingConfig.lora_dropout, float),
        
        # Optimization settings
        weight_decay=get_env("WEIGHT_DECAY", TrainingConfig.weight_decay, float),
        adam_beta1=get_env("ADAM_BETA1", TrainingConfig.adam_beta1, float),
        adam_beta2=get_env("ADAM_BETA2", TrainingConfig.adam_beta2, float),
        adam_epsilon=get_env("ADAM_EPSILON", TrainingConfig.adam_epsilon, float),
        max_grad_norm=get_env("MAX_GRAD_NORM", TrainingConfig.max_grad_norm, float),
        
        # Learning rate scheduler
        lr_scheduler_type=get_env("LR_SCHEDULER_TYPE", TrainingConfig.lr_scheduler_type),
        warmup_steps=get_env("WARMUP_STEPS", TrainingConfig.warmup_steps, int),
        
        # Logging and saving
        logging_steps=get_env("LOGGING_STEPS", TrainingConfig.logging_steps, int),
        save_steps=get_env("SAVE_STEPS", TrainingConfig.save_steps, int),
        run_name=get_env("RUN_NAME", TrainingConfig.run_name),
        
        # Memory optimization
        gradient_checkpointing=get_env("GRADIENT_CHECKPOINTING", TrainingConfig.gradient_checkpointing, str_to_bool),
        enable_xformers=get_env("ENABLE_XFORMERS", TrainingConfig.enable_xformers, str_to_bool),
        mixed_precision=get_env("MIXED_PRECISION", TrainingConfig.mixed_precision),
        
        # Dataset processing
        max_train_samples=get_env("MAX_TRAIN_SAMPLES", TrainingConfig.max_train_samples, int),
        preprocessing_num_workers=get_env("PREPROCESSING_NUM_WORKERS", TrainingConfig.preprocessing_num_workers, int),
        dataloader_num_workers=get_env("DATALOADER_NUM_WORKERS", TrainingConfig.dataloader_num_workers, int),
    )
    
    return config


def validate_environment() -> Tuple[str, str]:
    """
    Validate required environment variables and authentication tokens.
    
    Returns:
        Tuple of (wandb_api_key, hf_token)
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Load environment variables
    wandb_api_key = os.getenv("WANDB_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    
    # Validate required tokens
    if not wandb_api_key:
        raise ValueError(
            "WANDB_API_KEY not found in environment variables! "
            "Please set it in your environment or .env file."
        )
    
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in environment variables! "
            "Please set it in your environment or .env file."
        )
    
    return wandb_api_key, hf_token


def log_system_info(logger: logging.Logger, accelerator: Accelerator) -> None:
    """
    Log comprehensive system information for debugging and monitoring.
    
    Args:
        logger: Logger instance
        accelerator: Accelerator instance for distributed info
    """
    if accelerator.is_main_process:
        logger.info("=== System Information ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {accelerator.num_processes}")
        logger.info(f"Current GPU rank: {accelerator.process_index}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        
        # Memory information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}): {gpu_memory:.2f} GB")
        
        # System resources
        memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('.')
        logger.info(f"System RAM: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
        logger.info(f"Disk space: {disk_usage.free / (1024**3):.2f} GB free")
        logger.info("===========================")


def load_and_prepare_dataset(config: TrainingConfig, logger: logging.Logger) -> Dataset:
    """
    Load and preprocess the training dataset from JSONL format.
    
    Args:
        config: Training configuration
        logger: Logger instance
    
    Returns:
        Preprocessed Hugging Face Dataset
    """
    logger.info(f"Loading dataset from {config.dataset_path}")
    
    # Load dataset from JSONL file
    jsonl_path = os.path.join(config.dataset_path, "dataset.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
    
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    if not data:
        raise ValueError(f"No valid data found in {jsonl_path}")
    
    logger.info(f"Loaded {len(data)} samples from JSONL")
    
    # Convert to Hugging Face Dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    # Limit samples for testing/debugging
    if config.max_train_samples > 0:
        dataset_size = min(config.max_train_samples, len(dataset))
        dataset = dataset.select(range(dataset_size))
        logger.info(f"Limited dataset to {dataset_size} samples for training")
    
    return dataset


def create_preprocessing_transform(resolution: int) -> transforms.Compose:
    """
    Create image preprocessing transformation pipeline.
    
    Args:
        resolution: Target image resolution
    
    Returns:
        Composed transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1] range
    ])


def preprocess_batch(examples: Dict, config: TrainingConfig, transform: transforms.Compose, logger: logging.Logger) -> Dict:
    """
    Memory-efficient batch preprocessing function.
    
    Args:
        examples: Batch of examples from dataset
        config: Training configuration
        transform: Image transformation pipeline
        logger: Logger instance
    
    Returns:
        Dictionary with preprocessed images and texts
    """
    images = []
    texts = []
    
    for i, img_path in enumerate(examples["image_path"]):
        try:
            # Handle flexible path structures
            full_path = os.path.join(config.dataset_path, img_path)
            if not os.path.exists(full_path):
                # Try without nested folder structure
                img_path_cleaned = img_path.replace("cc_data/", "") if img_path.startswith("cc_data/") else img_path
                full_path = os.path.join(config.dataset_path, img_path_cleaned)
            
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


def custom_collate_fn(examples: List[Dict]) -> Dict:
    """
    Custom collate function to handle variable batch sizes and nested lists.
    
    Args:
        examples: List of example dictionaries
    
    Returns:
        Collated batch dictionary
    """
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
    
    if not images:
        return {"images": torch.empty(0), "text": []}
    
    return {"images": torch.stack(images), "text": texts}


def load_model_components(config: TrainingConfig, logger: logging.Logger) -> Tuple[Any, ...]:
    """
    Load all required model components for Stable Diffusion 3.5.
    
    Args:
        config: Training configuration
        logger: Logger instance
    
    Returns:
        Tuple of loaded model components
    """
    logger.info("Loading model components...")
    
    try:
        # Load scheduler
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
            config.model_id, 
            subfolder="scheduler"
        )
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            config.model_id, 
            subfolder="vae"
        )
        
        # Load text encoder and tokenizer
        text_encoder = CLIPTextModel.from_pretrained(
            config.model_id, 
            subfolder="text_encoder"
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            config.model_id, 
            subfolder="tokenizer"
        )
        
        # Load transformer
        transformer = SD3Transformer2DModel.from_pretrained(
            config.model_id, 
            subfolder="transformer"
        )
        
        logger.info("All model components loaded successfully")
        
        return noise_scheduler, vae, text_encoder, tokenizer, transformer
        
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        logger.error(traceback.format_exc())
        raise


def setup_model_optimizations(transformer: SD3Transformer2DModel, text_encoder: CLIPTextModel, 
                            config: TrainingConfig, logger: logging.Logger) -> None:
    """
    Apply memory and performance optimizations to model components.
    
    Args:
        transformer: Transformer model
        text_encoder: Text encoder model
        config: Training configuration
        logger: Logger instance
    """
    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        transformer.gradient_checkpointing = True
        text_encoder.gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")
    
    # Enable xformers memory efficient attention if available
    if config.enable_xformers:
        try:
            if hasattr(transformer, 'enable_xformers_memory_efficient_attention'):
                transformer.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory efficient attention enabled for transformer")
            else:
                logger.info("xformers not available for transformer")
        except Exception as e:
            logger.warning(f"Failed to enable xformers for transformer: {e}")


def setup_lora(transformer: SD3Transformer2DModel, config: TrainingConfig, logger: logging.Logger) -> SD3Transformer2DModel:
    """
    Set up LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    
    Args:
        transformer: Transformer model
        config: Training configuration
        logger: Logger instance
    
    Returns:
        LoRA-enabled transformer model
    """
    logger.info(f"Setting up LoRA with rank={config.lora_rank}, alpha={config.lora_alpha}")
    
    # Configure LoRA parameters
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    
    # Apply LoRA to transformer
    transformer_lora = get_peft_model(transformer, lora_config)
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in transformer_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transformer_lora.parameters())
    logger.info(f"LoRA setup complete. Trainable parameters: {trainable_params:,} / {total_params:,} "
               f"({100 * trainable_params / total_params:.2f}%)")
    
    return transformer_lora


def setup_optimizer_and_scheduler(transformer: SD3Transformer2DModel, config: TrainingConfig, 
                                num_training_steps: int, logger: logging.Logger) -> Tuple[Any, Any]:
    """
    Set up optimizer and learning rate scheduler.
    
    Args:
        transformer: Model to optimize
        config: Training configuration
        num_training_steps: Total number of training steps
        logger: Logger instance
    
    Returns:
        Tuple of (optimizer, lr_scheduler)
    """
    logger.info("Setting up optimizer and scheduler")
    
    # Create AdamW optimizer with specified hyperparameters
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    logger.info(f"Optimizer: AdamW with lr={config.learning_rate}, weight_decay={config.weight_decay}")
    logger.info(f"Scheduler: {config.lr_scheduler_type} with {config.warmup_steps} warmup steps")
    
    return optimizer, lr_scheduler


def training_step(batch: Dict, transformer: SD3Transformer2DModel, vae: AutoencoderKL, 
                 text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, 
                 noise_scheduler: DPMSolverMultistepScheduler, accelerator: Accelerator, 
                 logger: logging.Logger) -> Optional[torch.Tensor]:
    """
    Perform a single training step.
    
    Args:
        batch: Training batch
        transformer: Transformer model
        vae: VAE model
        text_encoder: Text encoder
        tokenizer: Tokenizer
        noise_scheduler: Noise scheduler
        accelerator: Accelerator instance
        logger: Logger instance
    
    Returns:
        Loss value or None if error
    """
    try:
        # Encode text
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
        return None
    
    try:
        # Encode images to latents
        with torch.no_grad():
            latents = vae.encode(batch["images"].to(accelerator.device)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
    except Exception as e:
        logger.error(f"Error encoding images: {e}")
        return None
    
    # Sample noise and timesteps
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=latents.device
    )
    
    # Add noise to latents
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Predict noise with transformer
    try:
        model_pred = transformer(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Calculate MSE loss
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        # Check for NaN loss
        if torch.isnan(loss):
            logger.warning("NaN loss detected, skipping batch")
            return None
        
        return loss
        
    except Exception as e:
        logger.error(f"Error in transformer forward pass: {e}")
        return None


def main():
    """Main training function."""
    try:
        # Load configuration from environment variables
        config = load_environment_config()
        
        # Set up logging
        logger = setup_logging(config.output_dir, verbose=True)
        logger.info("Starting Stable Diffusion 3.5 Medium fine-tuning with DDP")
        
        # Log configuration
        logger.info("Training Configuration:")
        for key, value in vars(config).items():
            logger.info(f"  {key}: {value}")
        
        # Validate environment and get tokens
        wandb_api_key, hf_token = validate_environment()
        
        # Set up environment variables
        os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_MODE"] = "offline"  # Offline mode to avoid upload issues
        
        # Login to Hugging Face
        login(hf_token)
        logger.info("Hugging Face authentication successful")
        
        # Initialize Accelerator for DDP (no DeepSpeed)
        accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision
        )
        logger.info(f"Accelerator initialized for DDP with mixed_precision={config.mixed_precision}")
        
        # Log system information
        log_system_info(logger, accelerator)
        
        # Load and prepare dataset
        dataset = load_and_prepare_dataset(config, logger)
        
        # Create preprocessing pipeline
        transform = create_preprocessing_transform(config.resolution)
        
        # Apply preprocessing
        logger.info("Starting dataset preprocessing...")
        dataset = dataset.map(
            lambda examples: preprocess_batch(examples, config, transform, logger),
            batched=True,
            batch_size=50,  # Small batch size for memory efficiency
            num_proc=config.preprocessing_num_workers,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"Dataset preprocessing complete. Dataset size: {len(dataset)}")
        dataset.set_format(type="torch", columns=["images", "text"])
        
        # Create DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=True  # Ensure consistent batch sizes
        )
        
        # Load model components
        noise_scheduler, vae, text_encoder, tokenizer, transformer = load_model_components(config, logger)
        
        # Apply model optimizations
        setup_model_optimizations(transformer, text_encoder, config, logger)
        
        # Set up LoRA
        transformer = setup_lora(transformer, config, logger)
        
        # Calculate training steps
        num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        num_training_steps = config.num_train_epochs * num_update_steps_per_epoch
        
        # Set up optimizer and scheduler
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(
            transformer, config, num_training_steps, logger
        )
        
        # Prepare with accelerator (DDP)
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
        
        # Move VAE and text encoder to device (they don't get trained)
        vae.to(accelerator.device)
        text_encoder.to(accelerator.device)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # Check for existing checkpoints
        resume_from_checkpoint = False
        checkpoint_files = list(Path(config.output_dir).glob("checkpoint-*"))
        if checkpoint_files:
            # Find the latest checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split("-")[1]))
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            try:
                accelerator.load_state(str(latest_checkpoint))
                resume_from_checkpoint = True
                global_step = int(latest_checkpoint.name.split("-")[1])
                logger.info(f"Resumed from checkpoint at global step {global_step}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                global_step = 0
        else:
            logger.info("No existing checkpoints found, starting from scratch")
            global_step = 0
        
        # Initialize W&B for experiment tracking
        if accelerator.is_main_process:
            logger.info("Initializing Weights & Biases")
            try:
                wandb.init(
                    project="sd35-finetune-ddp",
                    name=config.run_name,
                    config=vars(config),
                    resume="auto" if resume_from_checkpoint else None
                )
                logger.info("W&B initialization successful")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}")
        
        # Training loop
        logger.info("Starting training loop")
        start_time = time.time()
        transformer.train()
        
        # Training metrics tracking
        total_loss = 0.0
        num_steps = 0
        
        for epoch in range(config.num_train_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0
            
            logger.info(f"Starting epoch {epoch + 1}/{config.num_train_epochs}")
            
            for step, batch in enumerate(train_dataloader):
                # Skip empty batches
                if len(batch["images"]) == 0:
                    logger.warning(f"Skipping empty batch at step {step}")
                    continue
                
                with accelerator.accumulate(transformer):
                    # Perform training step
                    loss = training_step(
                        batch, transformer, vae, text_encoder, tokenizer,
                        noise_scheduler, accelerator, logger
                    )
                    
                    # Skip if loss calculation failed
                    if loss is None:
                        logger.warning(f"Skipping step {step} due to training error")
                        continue
                    
                    # Backward pass
                    accelerator.backward(loss)
                    
                    # Gradient clipping
                    if accelerator.sync_gradients and config.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(transformer.parameters(), config.max_grad_norm)
                    
                    # Optimizer step
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update metrics
                    total_loss += loss.item()
                    epoch_loss += loss.item()
                    num_steps += 1
                    epoch_steps += 1
                
                global_step += 1
                
                # Logging
                if global_step % config.logging_steps == 0 and accelerator.is_main_process:
                    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item():.4f}, "
                               f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                    
                    # Log to W&B
                    if wandb.run is not None:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/avg_loss": avg_loss,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                            "train/learning_rate": current_lr,
                        })
                
                # Save checkpoint
                if global_step % config.save_steps == 0 and accelerator.is_main_process:
                    checkpoint_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    try:
                        accelerator.save_state(checkpoint_path)
                        logger.info(f"Checkpoint saved at step {global_step}: {checkpoint_path}")
                    except Exception as e:
                        logger.error(f"Failed to save checkpoint: {e}")
                
                # Periodic memory cleanup
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()
            
            # End of epoch logging
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. "
                           f"Average loss: {avg_epoch_loss:.4f}")
                
                if wandb.run is not None:
                    wandb.log({
                        "train/epoch_avg_loss": avg_epoch_loss,
                        "train/epoch_time": epoch_time,
                        "train/completed_epoch": epoch + 1,
                    })
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        
        # Save final model
        if accelerator.is_main_process:
            logger.info("Saving final model")
            final_model_path = os.path.join(config.output_dir, "final_model")
            os.makedirs(final_model_path, exist_ok=True)
            
            try:
                # Save the LoRA weights
                transformer.save_pretrained(final_model_path)
                logger.info(f"Final model saved to {final_model_path}")
                
                # Create W&B artifact
                if wandb.run is not None:
                    logger.info("Creating W&B artifact")
                    artifact = wandb.Artifact(
                        name=f"{config.run_name}-final-model",
                        type="model",
                        description=f"Final LoRA weights for {config.model_id}"
                    )
                    artifact.add_dir(final_model_path)
                    
                    try:
                        artifact.upload_timeout = 300  # 5-minute timeout
                        wandb.log_artifact(artifact, aliases=["latest", "final"])
                        logger.info("W&B artifact created successfully")
                    except Exception as e:
                        logger.error(f"W&B artifact upload failed: {e}")
                    
            except Exception as e:
                logger.error(f"Error saving final model: {e}")
        
        # Finish W&B run
        if accelerator.is_main_process and wandb.run is not None:
            logger.info("Finishing W&B run")
            wandb.finish()
        
        logger.info("Training script completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(traceback.format_exc())
        
        # Clean up W&B if needed
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        
        raise


if __name__ == "__main__":
    main()