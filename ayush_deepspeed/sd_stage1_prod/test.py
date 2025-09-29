#!/usr/bin/env python3
"""
Production-Ready Test Script for Fine-tuned Stable Diffusion 3.5 Medium
=======================================================================

This script tests the fine-tuned LoRA model by generating images with various prompts.
It supports both interactive and batch generation modes.

Features:
- Environment variable configuration
- Robust error handling
- Multiple prompt testing
- Image quality assessment
- Comprehensive logging

Author: AI Training Pipeline
Version: 2.0
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv
from PIL import Image
import json


@dataclass
class TestConfig:
    """Configuration class for testing parameters."""
    
    # Model paths and identifiers
    model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    lora_path: str = "./results/final_model"
    
    # Output settings
    output_dir: str = "./test_images"
    
    # Generation parameters
    resolution: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    
    # Device settings
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16
    
    # Safety and performance
    enable_cpu_offload: bool = False
    enable_xformers: bool = True
    
    # Testing options
    interactive_mode: bool = True
    save_metadata: bool = True


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Set up logging configuration for the test script.
    
    Args:
        output_dir: Directory to save log files
    
    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "test.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Test logging initialized. Log file: {log_file}")
    return logger


def load_test_config() -> TestConfig:
    """
    Load test configuration from environment variables.
    
    Returns:
        TestConfig object with resolved values
    """
    load_dotenv()
    
    def get_env(key: str, default: Any, type_func: callable = str) -> Any:
        """Helper to get environment variable with type conversion."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return type_func(value)
        except (ValueError, TypeError):
            print(f"Warning: Invalid value for {key}='{value}', using default {default}")
            return default
    
    def str_to_bool(value: str) -> bool:
        """Convert string to boolean."""
        return value.lower() in ('true', '1', 'yes', 'on')
    
    # Map torch dtype strings
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    dtype_str = get_env("TORCH_DTYPE", "float16")
    torch_dtype = dtype_map.get(dtype_str, torch.float16)
    
    config = TestConfig(
        # Model paths
        model_id=get_env("MODEL_ID", TestConfig.model_id),
        lora_path=get_env("LORA_PATH", TestConfig.lora_path),
        
        # Output settings
        output_dir=get_env("TEST_OUTPUT_DIR", TestConfig.output_dir),
        
        # Generation parameters
        resolution=get_env("RESOLUTION", TestConfig.resolution, int),
        num_inference_steps=get_env("NUM_INFERENCE_STEPS", TestConfig.num_inference_steps, int),
        guidance_scale=get_env("GUIDANCE_SCALE", TestConfig.guidance_scale, float),
        num_images_per_prompt=get_env("NUM_IMAGES_PER_PROMPT", TestConfig.num_images_per_prompt, int),
        
        # Device settings
        device=get_env("DEVICE", TestConfig.device),
        torch_dtype=torch_dtype,
        
        # Performance settings
        enable_cpu_offload=get_env("ENABLE_CPU_OFFLOAD", TestConfig.enable_cpu_offload, str_to_bool),
        enable_xformers=get_env("ENABLE_XFORMERS", TestConfig.enable_xformers, str_to_bool),
        
        # Testing options
        interactive_mode=get_env("INTERACTIVE_MODE", TestConfig.interactive_mode, str_to_bool),
        save_metadata=get_env("SAVE_METADATA", TestConfig.save_metadata, str_to_bool),
    )
    
    return config


def validate_test_environment() -> str:
    """
    Validate test environment and get HuggingFace token.
    
    Returns:
        HuggingFace token
        
    Raises:
        ValueError: If HF_TOKEN is missing
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in environment variables! "
            "Please set it in your .env file or environment."
        )
    return hf_token


def get_default_prompts() -> List[str]:
    """
    Get default test prompts for image generation.
    
    Returns:
        List of test prompts
    """
    return [
        "A serene mountain landscape at sunset with vibrant orange and pink clouds",
        "A futuristic city skyline at night with neon lights and flying cars",
        "A cozy coffee shop interior with warm lighting, books, and steam rising from a cup",
        "A magical forest with glowing mushrooms and ethereal light beams",
        "A vintage steam locomotive crossing a stone bridge over a rushing river",
        "A peaceful zen garden with raked sand, stones, and cherry blossoms",
        "An astronaut floating in space with Earth visible in the background",
        "A bustling market street in an ancient Mediterranean town",
    ]


def load_pipeline(config: TestConfig, logger: logging.Logger) -> StableDiffusion3Pipeline:
    """
    Load the Stable Diffusion pipeline with fine-tuned LoRA weights.
    
    Args:
        config: Test configuration
        logger: Logger instance
    
    Returns:
        Loaded pipeline
        
    Raises:
        Exception: If pipeline loading fails
    """
    logger.info(f"Loading base Stable Diffusion pipeline: {config.model_id}")
    
    try:
        # Load base pipeline
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            config.model_id,
            torch_dtype=config.torch_dtype,
            use_auth_token=True,
            variant="fp16" if config.torch_dtype == torch.float16 else None
        )
        
        # Load LoRA weights if path exists
        if os.path.exists(config.lora_path):
            logger.info(f"Loading LoRA weights from: {config.lora_path}")
            
            # Load LoRA model
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer, 
                config.lora_path, 
                is_trainable=False
            )
            logger.info("LoRA weights loaded successfully")
        else:
            logger.warning(f"LoRA path not found: {config.lora_path}. Using base model only.")
        
        # Move to device
        pipeline = pipeline.to(config.device)
        
        # Enable CPU offload if requested
        if config.enable_cpu_offload:
            pipeline.enable_model_cpu_offload()
            logger.info("CPU offload enabled")
        
        # Enable xformers if available
        if config.enable_xformers:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")
        
        logger.info("Pipeline loaded successfully")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        logger.error(traceback.format_exc())
        raise


def generate_image(pipeline: StableDiffusion3Pipeline, prompt: str, config: TestConfig, 
                  logger: logging.Logger) -> List[Image.Image]:
    """
    Generate image(s) from a text prompt.
    
    Args:
        pipeline: Loaded diffusion pipeline
        prompt: Text prompt for generation
        config: Test configuration
        logger: Logger instance
    
    Returns:
        List of generated images
    """
    logger.info(f"Generating image for prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    
    try:
        # Generate image(s)
        start_time = time.time()
        
        result = pipeline(
            prompt=prompt,
            height=config.resolution,
            width=config.resolution,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            num_images_per_prompt=config.num_images_per_prompt,
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Image generation completed in {generation_time:.2f} seconds")
        
        return result.images
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        logger.error(traceback.format_exc())
        return []


def save_image_with_metadata(image: Image.Image, filepath: str, prompt: str, 
                           config: TestConfig, generation_time: float = 0.0) -> None:
    """
    Save image with metadata to disk.
    
    Args:
        image: PIL Image to save
        filepath: Output file path
        prompt: Text prompt used
        config: Test configuration
        generation_time: Time taken for generation
    """
    # Save image
    image.save(filepath, quality=95)
    
    # Save metadata if requested
    if config.save_metadata:
        metadata = {
            "prompt": prompt,
            "model_id": config.model_id,
            "lora_path": config.lora_path,
            "resolution": config.resolution,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": config.guidance_scale,
            "torch_dtype": str(config.torch_dtype),
            "generation_time": generation_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        metadata_path = filepath.rsplit('.', 1)[0] + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def interactive_generation(pipeline: StableDiffusion3Pipeline, config: TestConfig, logger: logging.Logger) -> None:
    """
    Run interactive image generation mode.
    
    Args:
        pipeline: Loaded diffusion pipeline
        config: Test configuration
        logger: Logger instance
    """
    print("\n" + "="*60)
    print("Interactive Image Generation Mode")
    print("="*60)
    print("Enter prompts to generate images. Type 'quit' to exit.")
    print("Type 'examples' to see example prompts.")
    print("Type 'config' to see current settings.")
    print("-"*60)
    
    image_counter = 1
    
    while True:
        try:
            user_input = input("\nEnter prompt: ").strip()
            
            if not user_input:
                print("Please enter a prompt or 'quit' to exit.")
                continue
            
            if user_input.lower() == 'quit':
                print("Exiting interactive mode...")
                break
            
            if user_input.lower() == 'examples':
                print("\nExample prompts:")
                for i, prompt in enumerate(get_default_prompts()[:5], 1):
                    print(f"{i}. {prompt}")
                continue
            
            if user_input.lower() == 'config':
                print(f"\nCurrent configuration:")
                print(f"  Resolution: {config.resolution}x{config.resolution}")
                print(f"  Inference steps: {config.num_inference_steps}")
                print(f"  Guidance scale: {config.guidance_scale}")
                print(f"  Output directory: {config.output_dir}")
                continue
            
            # Generate image
            start_time = time.time()
            images = generate_image(pipeline, user_input, config, logger)
            generation_time = time.time() - start_time
            
            if images:
                for i, image in enumerate(images):
                    filename = f"interactive_{image_counter:03d}_{i+1}.png"
                    filepath = os.path.join(config.output_dir, filename)
                    save_image_with_metadata(image, filepath, user_input, config, generation_time)
                    print(f"Image saved: {filepath}")
                
                image_counter += 1
            else:
                print("Failed to generate image. Check logs for details.")
                
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except EOFError:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_generation(pipeline: StableDiffusion3Pipeline, prompts: List[str], 
                    config: TestConfig, logger: logging.Logger) -> None:
    """
    Run batch image generation mode.
    
    Args:
        pipeline: Loaded diffusion pipeline
        prompts: List of prompts to generate
        config: Test configuration
        logger: Logger instance
    """
    logger.info(f"Starting batch generation for {len(prompts)} prompts")
    
    total_start_time = time.time()
    successful_generations = 0
    
    for i, prompt in enumerate(prompts, 1):
        logger.info(f"Processing prompt {i}/{len(prompts)}")
        
        start_time = time.time()
        images = generate_image(pipeline, prompt, config, logger)
        generation_time = time.time() - start_time
        
        if images:
            for j, image in enumerate(images):
                filename = f"batch_{i:03d}_{j+1}.png"
                filepath = os.path.join(config.output_dir, filename)
                save_image_with_metadata(image, filepath, prompt, config, generation_time)
                logger.info(f"Saved: {filepath}")
            
            successful_generations += 1
        else:
            logger.error(f"Failed to generate image for prompt {i}")
    
    total_time = time.time() - total_start_time
    logger.info(f"Batch generation completed: {successful_generations}/{len(prompts)} successful "
               f"in {total_time:.2f} seconds")


def log_system_info(config: TestConfig, logger: logging.Logger) -> None:
    """
    Log system and configuration information.
    
    Args:
        config: Test configuration
        logger: Logger instance
    """
    logger.info("=== Test Configuration ===")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")
    
    logger.info("=== System Information ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name}, {gpu_memory:.2f} GB")
    
    logger.info("===========================")


def main():
    """Main test function."""
    try:
        # Load configuration
        config = load_test_config()
        
        # Set up logging
        logger = setup_logging(config.output_dir)
        logger.info("Starting Stable Diffusion 3.5 Medium test script")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Validate environment
        hf_token = validate_test_environment()
        
        # Login to Hugging Face
        login(hf_token)
        logger.info("Hugging Face authentication successful")
        
        # Log system information
        log_system_info(config, logger)
        
        # Load pipeline
        pipeline = load_pipeline(config, logger)
        
        # Determine test mode
        if config.interactive_mode and sys.stdin.isatty():
            # Interactive mode
            interactive_generation(pipeline, config, logger)
        else:
            # Batch mode with default prompts
            logger.info("Running in batch mode with default prompts")
            prompts = get_default_prompts()
            batch_generation(pipeline, prompts, config, logger)
        
        logger.info("Test script completed successfully!")
        
    except Exception as e:
        logger.error(f"Test script failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()