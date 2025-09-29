import os
import torch
import argparse
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="test.log")
logger = logging.getLogger(__name__)

# Argument parser
parser = argparse.ArgumentParser(description="Test fine-tuned Stable Diffusion 3.5 Medium model")
parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium", help="Pretrained model ID")
parser.add_argument("--lora_path", type=str, default="./results/final_model", help="Path to fine-tuned LoRA weights")
parser.add_argument("--output_dir", type=str, default="./test_images", help="Directory to save generated images")
parser.add_argument("--resolution", type=int, default=512, help="Resolution for generated images")
args = parser.parse_args()

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in .env file!")

# Hugging Face login
login(hf_token)

# Predefined prompts for fallback
default_prompts = [
    "A serene mountain landscape at sunset with vibrant colors",
    "A futuristic city skyline with flying cars at night",
    "A cozy coffee shop interior with warm lighting and books"
]

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load the base model
logger.info("Loading base Stable Diffusion model")
pipeline = StableDiffusionPipeline.from_pretrained(
    args.model_id,
    torch_dtype=torch.float16,
    use_auth_token=hf_token
).to("cuda")

# Load fine-tuned LoRA weights
logger.info(f"Loading LoRA weights from {args.lora_path}")
transformer = PeftModel.from_pretrained(pipeline.transformer, args.lora_path, is_trainable=False)
pipeline.transformer = transformer

# Enable xformers if available
try:
    pipeline.transformer.enable_xformers_memory_efficient_attention()
except Exception as e:
    logger.warning(f"xformers not enabled: {e}")

# Function to generate and save image
def generate_image(prompt, output_path, index):
    logger.info(f"Generating image for prompt: {prompt}")
    image = pipeline(
        prompt=prompt,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    image.save(output_path)
    logger.info(f"Saved image to {output_path}")

# Try interactive prompt
try:
    print("Enter a prompt for image generation (or press Enter to use default prompts):")
    user_prompt = input().strip()
    if user_prompt:
        output_path = os.path.join(args.output_dir, f"generated_image_user.png")
        generate_image(user_prompt, output_path, 0)
    else:
        raise ValueError("No prompt provided, using defaults")
except (EOFError, KeyboardInterrupt, ValueError):
    logger.info("No user input or non-interactive mode, using default prompts")
    for i, prompt in enumerate(default_prompts):
        output_path = os.path.join(args.output_dir, f"generated_image_{i+1}.png")
        generate_image(prompt, output_path, i+1)

logger.info("Image generation complete")