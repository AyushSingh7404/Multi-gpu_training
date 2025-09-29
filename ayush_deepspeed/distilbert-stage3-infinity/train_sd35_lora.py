#!/usr/bin/env python3
# SD3.5 LoRA trainer (hardcoded-friendly)
# - S3 dataset zip → extract → auto-find JSONL paths
# - Same training/metrics/logging behavior as your original
# - Adds status API calls (TRAINING_RECEIVED…COMPLETED/FAILED)
# - Robust presigned S3 uploads (single PUT + multipart) for final artifacts
# - MinIO/AWS compatible, no env editing required on your side

import os, json, math, random, shutil, time, zipfile, hashlib, mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silence fork warnings

import requests
import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline
from PIL import Image

# ==========================
# ENV helpers
# ==========================
def resolve_env(name, default, cast=str):
    val = os.getenv(name)
    used_default = val is None
    return (cast(val) if val is not None else cast(default)), used_default

def print_config(config_rows):
    width = max(len(k) for k, *_ in config_rows) + 2
    print("\n=== Resolved Configuration ===")
    for key, val, is_default in config_rows:
        tag = "(default)" if is_default else "(env)"
        print(f"{key.ljust(width)} {val}  {tag}")
    print("================================\n")

# ================= Environment / Hyperparams =================
MODEL_ID, _              = resolve_env("MODEL_ID", "./models/sd35")
OUTPUT_DIR, _            = resolve_env("OUTPUT_DIR", "./outputs")
IMAGE_SIZE, _            = resolve_env("IMAGE_SIZE", "512", int)
BATCH_SIZE, _            = resolve_env("BATCH_SIZE", "4", int)
EPOCHS, _                = resolve_env("EPOCHS", "1", int)
LR, _                    = resolve_env("LR", "1e-4", float)
RANK, _                  = resolve_env("LORA_RANK", "16", int)
LORA_ALPHA, _            = resolve_env("LORA_ALPHA", "32", int)
MIXED_PRECISION, _       = resolve_env("MIXED_PRECISION", "bf16")
VAL_SPLIT, _             = resolve_env("VAL_SPLIT", "0.1", float)
SPLIT_SEED, _            = resolve_env("SPLIT_SEED", "2025", int)
GRAD_ACCUM, _            = resolve_env("GRAD_ACCUM", "1", int)
NUM_WORKERS, _           = resolve_env("NUM_WORKERS", "4", int)
LOG_EVERY, _             = resolve_env("LOG_EVERY", "20", int)
SEED, _                  = resolve_env("SEED", "42", int)

WEIGHT_DECAY, _          = resolve_env("WEIGHT_DECAY", "1e-2", float)
CLIP_GRAD_NORM, _        = resolve_env("CLIP_GRAD_NORM", "0.0", float)
MAX_TRAIN_STEPS, _       = resolve_env("MAX_TRAIN_STEPS", "0", int)
LR_SCHEDULER, _          = resolve_env("LR_SCHEDULER", "none")
WARMUP_STEPS, _          = resolve_env("WARMUP_STEPS", "0", int)
WARMUP_RATIO, _          = resolve_env("WARMUP_RATIO", "0.0", float)

SAVE_EVERY_STEPS, _      = resolve_env("SAVE_EVERY_STEPS", "0", int)
SAVE_EVERY_EPOCHS, _     = resolve_env("SAVE_EVERY_EPOCHS", "0", int)
SAVE_LIMIT, _            = resolve_env("SAVE_LIMIT", "0", int)

VAL_GUIDANCE, _          = resolve_env("VAL_GUIDANCE", "7.0", float)
VAL_STEPS, _             = resolve_env("VAL_STEPS", "20", int)
VAL_PROMPTS, _           = resolve_env("VAL_PROMPTS", "")
VAL_EVERY_STEPS, _       = resolve_env("VAL_EVERY_STEPS", "0", int)
VAL_NUM_IMAGES, _        = resolve_env("VAL_NUM_IMAGES", "2", int)

GRADIENT_CHECKPOINTING, _= resolve_env("GRADIENT_CHECKPOINTING", "false")
GRADIENT_CHECKPOINTING   = GRADIENT_CHECKPOINTING.lower() == "true"

USE_XFORMERS, _          = resolve_env("USE_XFORMERS", "false")
USE_XFORMERS             = USE_XFORMERS.lower() == "true"

ALLOW_TF32, _            = resolve_env("ALLOW_TF32", "true")
ALLOW_TF32               = ALLOW_TF32.lower() == "true"

DETERMINISTIC, _         = resolve_env("DETERMINISTIC", "false")
DETERMINISTIC            = DETERMINISTIC.lower() == "true"

RESUME_FROM, _           = resolve_env("RESUME_FROM", "")
RUN_NAME, _              = resolve_env("RUN_NAME", "")

# --- S3 (dataset + artifacts) ---
S3_ENDPOINT, _           = resolve_env("S3_ENDPOINT", "http://172.17.128.4:9010")
S3_ACCESS_KEY, _         = resolve_env("S3_ACCESS_KEY", "")
S3_SECRET_KEY, _         = resolve_env("S3_SECRET_KEY", "")
S3_BUCKET, _             = resolve_env("S3_BUCKET", "")           # dataset bucket (zip)
S3_KEY, _                = resolve_env("S3_KEY", "")              # dataset key (zip)
DATA_DIR, _              = resolve_env("DATA_DIR", "./cc_data")
ORG_ID, _                = resolve_env("ORG_ID", "default_org")
PROJECT_ID, _            = resolve_env("PROJECT_ID", "default_proj")
WEIGHTS_PREFIX           = f"s3://buinternalbkt01/UserData/{ORG_ID}/Trained_Weights/{PROJECT_ID}"  # dest (bucket inferred)

# --- Status API (hardcoded-friendly) ---
TRAINING_ID, _           = resolve_env("PROJECT_ID", "0")  # <-- per your request, left unchanged
STATUS_API_URL, _        = resolve_env("STATUS_API_URL", "https://devqouiapi.asmadiya.net/Infer/api/model-training/update-status")

# --- Logging / W&B (optional) ---
WANDB_ENABLE, _          = resolve_env("WANDB_ENABLE", "false")
WANDB_ENABLE             = WANDB_ENABLE.lower() == "true"
WANDB_PROJECT, _         = resolve_env("WANDB_PROJECT", "sd35-lora")
WANDB_ENTITY, _          = resolve_env("WANDB_ENTITY", "")
WANDB_RUN_NAME, _        = resolve_env("WANDB_RUN_NAME", RUN_NAME or "sd35-lora-run")
WANDB_MODE, _            = resolve_env("WANDB_MODE", "online")

VERBOSE_LOGS, _          = resolve_env("VERBOSE_LOGS", "true")
VERBOSE_LOGS             = VERBOSE_LOGS.lower() == "true"
LOG_GPU_MEMORY, _        = resolve_env("LOG_GPU_MEMORY", "true"); LOG_GPU_MEMORY = LOG_GPU_MEMORY.lower() == "true"
LOG_THROUGHPUT, _        = resolve_env("LOG_THROUGHPUT", "true"); LOG_THROUGHPUT = LOG_THROUGHPUT.lower() == "true"
LOG_GRAD_NORM, _         = resolve_env("LOG_GRAD_NORM", "true"); LOG_GRAD_NORM = LOG_GRAD_NORM.lower() == "true" and (CLIP_GRAD_NORM == 0.0)
LOG_LR_EVERY             = int(resolve_env("LOG_LR_EVERY", str(max(1, LOG_EVERY)))[0])

# --- Metrics (optional) ---
M_VAL_LOSS_ENABLE, _     = resolve_env("METRICS_VAL_LOSS", "true"); M_VAL_LOSS_ENABLE = M_VAL_LOSS_ENABLE.lower() == "true"
M_VAL_LOSS_MAX_SAMP, _   = resolve_env("METRICS_VAL_LOSS_MAX_SAMPLES", "64", int)
M_CLIP_ENABLE, _         = resolve_env("METRICS_CLIP", "true"); M_CLIP_ENABLE = M_CLIP_ENABLE.lower() == "true"
M_CLIP_MODEL_ID, _       = resolve_env("METRICS_CLIP_MODEL", "openai/clip-vit-base-patch32")
M_FID_ENABLE, _          = resolve_env("METRICS_FID", "false"); M_FID_ENABLE = M_FID_ENABLE.lower() == "true"
M_KID_ENABLE, _          = resolve_env("METRICS_KID", "false"); M_KID_ENABLE = M_KID_ENABLE.lower() == "true"
M_FID_KID_SAMPLES, _     = resolve_env("METRICS_FID_KID_SAMPLES", "64", int)
M_METRICS_EVERY_STEPS, _ = resolve_env("METRICS_EVERY_STEPS", "0", int)

print_config([
    ("MODEL_ID", MODEL_ID, os.getenv("MODEL_ID") is None),
    ("OUTPUT_DIR", OUTPUT_DIR, os.getenv("OUTPUT_DIR") is None),
    ("IMAGE_SIZE", IMAGE_SIZE, os.getenv("IMAGE_SIZE") is None),
    ("BATCH_SIZE", BATCH_SIZE, os.getenv("BATCH_SIZE") is None),
    ("EPOCHS", EPOCHS, os.getenv("EPOCHS") is None),
    ("LR", LR, os.getenv("LR") is None),
    ("RANK", RANK, os.getenv("LORA_RANK") is None),
    ("LORA_ALPHA", LORA_ALPHA, os.getenv("LORA_ALPHA") is None),
    ("MIXED_PRECISION", MIXED_PRECISION, os.getenv("MIXED_PRECISION") is None),
    ("VAL_SPLIT", VAL_SPLIT, os.getenv("VAL_SPLIT") is None),
    ("S3_ENDPOINT", S3_ENDPOINT, os.getenv("S3_ENDPOINT") is None),
    ("S3_BUCKET(DATA)", S3_BUCKET, os.getenv("S3_BUCKET") is None),
    ("S3_KEY(DATA)", S3_KEY, os.getenv("S3_KEY") is None),
    ("DATA_DIR", DATA_DIR, os.getenv("DATA_DIR") is None),
    ("WEIGHTS_PREFIX", WEIGHTS_PREFIX, False),
    ("TRAINING_ID", TRAINING_ID, os.getenv("TRAINING_ID") is None),
    ("STATUS_API_URL", STATUS_API_URL, os.getenv("STATUS_API_URL") is None),
])

# ==========================
# Simple status updates
# ==========================
_STATUS_TIMEOUT   = 5
_STATUS_RETRIES   = 2

def post_status(status: str):
    try_id = int(str(TRAINING_ID)) if str(TRAINING_ID).isdigit() else 0
    payload = {"id": try_id, "status": status}
    last_err = None
    for _ in range(1 + _STATUS_RETRIES):
        try:
            r = requests.post(STATUS_API_URL, json=payload, timeout=_STATUS_TIMEOUT)
            if 200 <= r.status_code < 300:
                print(f"[STATUS] {status} (id={try_id})")
                return True
            last_err = f"HTTP {r.status_code} {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.4)
    print(f"[STATUS] Failed to post {status}: {last_err}")
    return False

post_status("TRAINING_RECEIVED")

# ==================== S3 Client ====================
def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY or None,
        aws_secret_access_key=S3_SECRET_KEY or None,
        region_name="us-east-1",
        config=BotoConfig(signature_version="s3v4"),
    )

s3 = s3_client()

# Torch perf/determinism
torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
torch.backends.cudnn.allow_tf32 = ALLOW_TF32
if DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random.seed(SEED)
torch.manual_seed(SEED)

# Inject RUN_NAME into OUTPUT_DIR
if RUN_NAME:
    OUTPUT_DIR = str(Path(OUTPUT_DIR) / RUN_NAME)

# Optional W&B import
try:
    import wandb
except Exception:
    wandb = None

# Optional torchmetrics for FID/KID
tm_available = True
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
except Exception:
    tm_available = False

# ============ Small helpers ============
def _world_size():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1

def _get_lr(optimizer):
    return optimizer.param_groups[0].get("lr", None)

def _grad_global_norm(parameters) -> Optional[float]:
    total = 0.0
    found = False
    for p in parameters:
        if p.grad is not None:
            found = True
            with torch.no_grad():
                total += (p.grad.detach().data.float().norm(2) ** 2).item()
    return (total ** 0.5) if found else None

def _gpu_mem_mb():
    if not torch.cuda.is_available(): return None
    try:
        return torch.cuda.max_memory_allocated() / (1024*1024)
    except Exception:
        return None

############### fetch and extract dataset dynamically #######################
def fetch_and_extract_from_s3(bucket, key, out_dir: str):
    if not bucket or not key:
        raise ValueError("S3_BUCKET and S3_KEY are required to fetch dataset.")
    post_status("DOWNLOADING_DATA")
    import tempfile
    tmp_zip = Path(tempfile.gettempdir()) / Path(key).name
    s3.download_file(bucket, key, str(tmp_zip))
    print(f"[INFO] Downloaded s3://{bucket}/{key} → {tmp_zip}")

    with zipfile.ZipFile(tmp_zip, "r") as zf:
        names = zf.namelist()
        top_level = names[0].split("/")[0] if "/" in names[0] else ""
        for member in names:
            target_path = Path(out_dir) / Path(member)
            if top_level and member.startswith(top_level + "/"):
                target_path = Path(out_dir) / Path(member[len(top_level)+1:])
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not member.endswith("/"):
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

    print(f"[INFO] Extracted dataset into {out_dir}")
    return str(out_dir)

def find_jsonl_file(data_dir: str) -> str:
    base = Path(data_dir)
    matches = list(base.rglob("*.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No .jsonl file found in {data_dir}")
    jsonl_path = str(matches[0].resolve())
    print(f"[INFO] Found dataset JSONL: {jsonl_path}")
    return jsonl_path

DATA_DIR = fetch_and_extract_from_s3(S3_BUCKET, S3_KEY, DATA_DIR)
print("[INFO] data dir : ", DATA_DIR)
DATASET_JSONL = find_jsonl_file(DATA_DIR)
print("[INFO] dataset jsonl : ", DATASET_JSONL)

# ============ Dataset ============
def resolve_img(p_str: str, jsonl_root: Path, data_root: Path) -> Optional[str]:
    p = Path(p_str)
    if p.is_absolute():
        return str(p) if p.exists() else None
    candidates = [
        jsonl_root / p,
        data_root / p,
        data_root.parent / p,
        Path.cwd() / p,
    ]
    for q in candidates:
        if q.exists():
            return str(q.resolve())
    return None

class JsonlImageCaptionDataset(Dataset):
    def __init__(self, jsonl_path: str, image_size: int, data_root: Optional[str] = None):
        self.items: List[Dict] = []
        jsonl_abs = Path(jsonl_path).resolve()
        jsonl_root = jsonl_abs.parent
        data_root = Path(data_root).resolve() if data_root else jsonl_root

        kept, dropped = 0, 0
        with open(jsonl_abs, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                img_path = obj.get("image") or obj.get("image_path")
                cap = obj.get("caption") or obj.get("text") or ""
                if not img_path:
                    dropped += 1; continue
                resolved = resolve_img(img_path, jsonl_root, data_root)
                if resolved is None:
                    dropped += 1; continue
                self.items.append({"image": resolved, "caption": cap})
                kept += 1

        print(f"[INFO] Dataset loader: kept {kept} items, dropped {dropped} (unresolvable paths).")
        self.size = image_size

    def __len__(self): return len(self.items)

    def _pil_to_tensor(self, im: Image.Image) -> torch.Tensor:
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((self.size, self.size), Image.BICUBIC)
        arr = np.array(im.convert("RGB"), dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img = Image.open(rec["image"]).convert("RGB")
        pixel_values = self._pil_to_tensor(img)
        return {"pixel_values": pixel_values, "caption": rec["caption"], "path": rec["image"]}

def collate(batch):
    imgs = torch.stack([b["pixel_values"] for b in batch], dim=0)
    caps = [b["caption"] for b in batch]
    paths = [b["path"] for b in batch]
    return {"pixel_values": imgs, "captions": caps, "paths": paths}

# ============ FlowMatch helpers ============
def ensure_flowmatch_schedule(scheduler, device, n_train_steps: Optional[int] = None):
    num = int(n_train_steps or getattr(getattr(scheduler, "config", object()), "num_train_timesteps", 1000))
    try: scheduler.set_timesteps(num, device=device)
    except TypeError:
        try: scheduler.set_timesteps(num)
        except Exception: pass
    sched = getattr(scheduler, "timesteps", None)
    if not (isinstance(sched, torch.Tensor) and sched.numel() > 0):
        sched = torch.linspace(0.0, 1.0, steps=num, device=device, dtype=torch.float32)
        scheduler.timesteps = sched
    if not hasattr(scheduler, "schedule_timesteps"):
        try: scheduler.schedule_timesteps = scheduler.timesteps
        except Exception: pass

def _get_schedule_tensor(scheduler) -> torch.Tensor:
    sched = getattr(scheduler, "schedule_timesteps", None)
    if isinstance(sched, torch.Tensor) and sched.numel() > 0: return sched
    sched = getattr(scheduler, "timesteps", None)
    if isinstance(sched, torch.Tensor) and sched.numel() > 0: return sched
    raise RuntimeError("Scheduler has no valid timesteps after ensure_flowmatch_schedule().")

def _map_t_to_scheduler_timesteps(scheduler, t: torch.Tensor) -> torch.Tensor:
    schedule = _get_schedule_tensor(scheduler)
    if t.ndim == 0: t = t.unsqueeze(0)
    t = t.to(schedule.device, dtype=torch.float32).clamp_(0, 1 - 1e-6)
    idx = (t * (schedule.shape[0] - 1)).long()
    return schedule[idx]

def scale_noise_compat(scheduler, latents, ts_discrete, noise):
    try:
        return scheduler.scale_noise(latents, ts_discrete, noise=noise)
    except Exception:
        return latents + noise

# ============ SD3 attention safety ============
def _is_sd3_transformer(obj) -> bool:
    try:
        from diffusers.models.transformers.transformer_sd3 import TransformerSD3Model
        return isinstance(obj, TransformerSD3Model)
    except Exception:
        return obj.__class__.__name__.lower().startswith("transformersd3")

def _force_dual_return_on_sd3_attention(transformer):
    """
    Wrap attention processors so if a processor returns a single tensor we
    coerce it into (out, None). Native SD3 processors that already return a
    tuple pass through unchanged.
    """
    try:
        from diffusers.models.attention import Attention
    except Exception:
        return

    class _DualReturnWrapper:
        def __init__(self, proc): self.proc = proc
        def __call__(self, *args, **kwargs):
            out = self.proc(*args, **kwargs)
            if isinstance(out, tuple):
                return out
            return out, None

    for m in transformer.modules():
        if m.__class__.__name__ == "Attention" or isinstance(m, Attention):
            proc = getattr(m, "processor", None)
            if proc is not None and not isinstance(proc, _DualReturnWrapper):
                try:
                    m.processor = _DualReturnWrapper(proc)
                except Exception:
                    pass

# ============ SD3.5 prompt encoding ============
def encode_prompts_compat(pipe: StableDiffusion3Pipeline, prompts: List[str], device: torch.device, do_cfg: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        enc = pipe.encode_prompt(prompt=prompts, prompt_2=prompts, prompt_3=prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=do_cfg)
    except TypeError:
        enc = pipe.encode_prompt(prompt=prompts, device=device, num_images_per_prompt=1, do_classifier_free_guidance=do_cfg)
    prompt_embeds = enc[0]
    pooled_proj   = enc[2] if len(enc) > 2 else None
    if pooled_proj is None:
        dcap = getattr(pipe.transformer.config, "caption_projection_dim", prompt_embeds.shape[-1])
        pooled_proj = torch.zeros((prompt_embeds.shape[0], dcap), device=device, dtype=prompt_embeds.dtype)
    return prompt_embeds, pooled_proj

#=========== LoRA via parametrizations ============
class LoRAWeight(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: int):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / float(rank) if rank > 0 else 1.0
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        if rank > 0:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if self.rank == 0: return W
        delta = (self.B @ self.A)
        delta = (self.scale * delta).to(dtype=W.dtype, device=W.device)
        return W + delta

def is_target_linear(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear): return False
    # SD3-friendly, memory-lean: only Q/K/V (drop to_out.0)
    return any(name.endswith(e) for e in ("to_q", "to_k", "to_v"))

def add_lora_parametrizations(root: nn.Module, rank: int, alpha: int):
    trainable = []
    manifest = {}
    for name, module in root.named_modules():
        if is_target_linear(name, module):
            in_f = module.in_features; out_f = module.out_features
            module.weight.requires_grad_(False)
            lora = LoRAWeight(in_f, out_f, rank, alpha)
            parametrize.register_parametrization(module, "weight", lora, unsafe=True)
            trainable.extend([lora.A, lora.B])
            manifest[name] = {"in": in_f, "out": out_f, "rank": rank, "alpha": alpha}
    return trainable, manifest

def collect_lora_state(root: nn.Module) -> Dict[str, torch.Tensor]:
    state = {}
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and parametrize.is_parametrized(module):
            try:
                for p in parametrize.get_parametrizations(module, "weight"):
                    if isinstance(p, LoRAWeight):
                        state[f"{name}.lora_A"] = p.A.detach().cpu()
                        state[f"{name}.lora_B"] = p.B.detach().cpu()
            except Exception:
                continue
    return state

def save_lora_adapters(transformer: nn.Module, out_dir: str, manifest: Dict[str, Dict]):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tensor_dict = collect_lora_state(transformer)
    try:
        import safetensors.torch as st
        st.save_file(tensor_dict, str(Path(out_dir) / "lora_adapters.safetensors"))
    except Exception:
        torch.save(tensor_dict, str(Path(out_dir) / "lora_adapters.pt"))
    with open(Path(out_dir) / "lora_index.json", "w") as f:
        json.dump({"format": "parametrize.lora.v1", "layers": manifest}, f, indent=2)

def load_lora_adapters_into(transformer: nn.Module, path: str):
    p = Path(path)
    if p.is_dir():
        cand = list(p.glob("lora_adapters.safetensors")) + list(p.glob("lora_adapters.pt"))
        if not cand: return
        p = cand[0]
    if p.suffix == ".safetensors":
        import safetensors.torch as st
        state = st.load_file(str(p))
    elif p.suffix == ".pt":
        state = torch.load(str(p), map_location="cpu")
    else:
        return
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear) and parametrize.is_parametrized(module):
            try:
                for paramz in parametrize.get_parametrizations(module, "weight"):
                    if isinstance(paramz, LoRAWeight):
                        keyA = f"{name}.lora_A"; keyB = f"{name}.lora_B"
                        if keyA in state and keyB in state:
                            with torch.no_grad():
                                paramz.A.copy_(state[keyA]); paramz.B.copy_(state[keyB])
            except Exception:
                continue

# ============ Load pipeline ============
if not Path(MODEL_ID).exists():
    raise FileNotFoundError(f"MODEL_ID not found: {MODEL_ID}")
if not Path(DATASET_JSONL).exists():
    raise FileNotFoundError(f"DATASET_JSONL not found: {DATASET_JSONL}")
if not (0.0 < VAL_SPLIT < 1.0):
    raise ValueError("VAL_SPLIT must be in (0,1)")

dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(MIXED_PRECISION, torch.bfloat16)

print(f"[INFO] Loading base model from: {MODEL_ID}")
pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

transformer = pipe.transformer
vae         = pipe.vae
scheduler   = pipe.scheduler

# Gradient checkpointing is OK for SD3/3.5; prefer model's method if present
if GRADIENT_CHECKPOINTING:
    try:
        transformer.enable_gradient_checkpointing()
    except Exception:
        try:
            pipe.transformer.gradient_checkpointing_enable()
        except Exception:
            pass

# IMPORTANT: never force xFormers/vanilla processors onto SD3/3.5
if USE_XFORMERS and not _is_sd3_transformer(transformer):
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

# Extra guard: even if something swapped a single-output processor, wrap it
if _is_sd3_transformer(transformer):
    _force_dual_return_on_sd3_attention(transformer)

# Apply LoRA
trainable_params, lora_manifest = add_lora_parametrizations(transformer, rank=RANK, alpha=LORA_ALPHA)
print(f"[INFO] Attached LoRA to {len(lora_manifest)} attention linears (rank={RANK}, alpha={LORA_ALPHA}).")

# Freeze non-LoRA
for p in vae.parameters(): p.requires_grad = False
for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
    enc = getattr(pipe, name, None)
    if enc is not None:
        for p in enc.parameters(): p.requires_grad = False

# ============ Data ============
class _MyDataset(JsonlImageCaptionDataset): pass
full_ds = _MyDataset(DATASET_JSONL, IMAGE_SIZE)
if len(full_ds) == 0:
    raise RuntimeError("Dataset is empty.")

val_size   = max(1, int(len(full_ds) * VAL_SPLIT))
train_size = len(full_ds) - val_size
g = torch.Generator().manual_seed(SPLIT_SEED)
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=g)

def _collate(batch):
    imgs = torch.stack([b["pixel_values"] for b in batch], dim=0)
    caps = [b["caption"] for b in batch]
    paths = [b["path"] for b in batch]
    return {"pixel_values": imgs, "captions": caps, "paths": paths}

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=_collate)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                          num_workers=max(1, NUM_WORKERS//2), pin_memory=True, drop_last=False, collate_fn=_collate)

print(f"[INFO] Train size: {train_size}, Val size: {val_size}")

# ============ Optimizer / Accelerator ============
optimizer   = torch.optim.AdamW([p for p in trainable_params if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
accelerator = Accelerator(mixed_precision=MIXED_PRECISION if MIXED_PRECISION in ("fp16","bf16") else "no",
                          gradient_accumulation_steps=GRAD_ACCUM)
device = accelerator.device

class _ParamWrap(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = nn.ParameterList([p for p in params if p.requires_grad])

param_wrap = _ParamWrap(trainable_params)
param_wrap, optimizer, train_loader = accelerator.prepare(param_wrap, optimizer, train_loader)
vae.to(device)
pipe.to(device)

def _maybe_num_train_steps():
    return getattr(getattr(scheduler, "config", object()), "num_train_timesteps", 1000)
ensure_flowmatch_schedule(scheduler, device, n_train_steps=_maybe_num_train_steps())
vae_scale = getattr(vae.config, "scaling_factor", 0.18215)

# ===== LR Scheduler =====
total_steps_by_epochs = (len(train_loader) * EPOCHS)
if MAX_TRAIN_STEPS and MAX_TRAIN_STEPS > 0:
    total_training_steps = MAX_TRAIN_STEPS
else:
    total_training_steps = total_steps_by_epochs

if LR_SCHEDULER != "none":
    if WARMUP_STEPS == 0 and WARMUP_RATIO > 0:
        WARMUP_STEPS = int(total_training_steps * WARMUP_RATIO)
    def lr_lambda_linear(step):
        if step < WARMUP_STEPS: return float(step) / max(1, WARMUP_STEPS)
        remain = total_training_steps - WARMUP_STEPS
        if remain <= 0: return 1.0
        prog = float(step - WARMUP_STEPS) / float(remain)
        return max(0.0, 1.0 - prog)
    def lr_lambda_cosine(step):
        if step < WARMUP_STEPS: return float(step) / max(1, WARMUP_STEPS)
        remain = total_training_steps - WARMUP_STEPS
        if remain <= 0: return 1.0
        prog = float(step - WARMUP_STEPS) / float(remain)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))
    if LR_SCHEDULER == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_linear)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine)
else:
    lr_scheduler = None

# ===== Validation helpers =====
def get_validation_prompts(caps: List[str]) -> List[str]:
    if VAL_PROMPTS:
        try:
            if VAL_PROMPTS.strip().startswith("["):
                arr = json.loads(VAL_PROMPTS)
                return [str(x) for x in arr][:max(1, VAL_NUM_IMAGES)]
            else:
                arr = [s for s in VAL_PROMPTS.split(";") if s.strip()]
                return arr[:max(1, VAL_NUM_IMAGES)]
        except Exception:
            pass
    if not caps:
        return ["a high quality photo"]
    return random.sample(caps, k=min(max(1, VAL_NUM_IMAGES), len(caps)))

def should_validate(step: int, epoch_done: bool) -> bool:
    if VAL_EVERY_STEPS and step > 0 and (step % VAL_EVERY_STEPS == 0): return True
    if VAL_EVERY_STEPS == 0 and epoch_done: return True
    return False

def should_metrics(step: int, epoch_done: bool) -> bool:
    if M_METRICS_EVERY_STEPS and step > 0 and (step % M_METRICS_EVERY_STEPS == 0): return True
    if M_METRICS_EVERY_STEPS == 0: return should_validate(step, epoch_done)
    return False

def periodic_save_dir(base: str, step: Optional[int] = None, epoch: Optional[int] = None) -> str:
    tag = f"step_{step}" if step is not None else f"epoch_{epoch}"
    return str(Path(base) / f"ckpt_{tag}")

def enforce_save_limit(base: str, limit: int):
    if limit <= 0: return
    ckpts = sorted(Path(base).glob("ckpt_*"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > limit:
        victim = ckpts.pop(0)
        shutil.rmtree(victim, ignore_errors=True)

# ============ (Optional) Resume ============
if RESUME_FROM:
    try:
        load_lora_adapters_into(transformer, RESUME_FROM)
        print(f"[INFO] Loaded LoRA adapters from: {RESUME_FROM}")
    except Exception as e:
        print(f"[WARN] Failed to load RESUME_FROM='{RESUME_FROM}': {e}")

# ============ W&B Init ============
wandb_run = None
if WANDB_ENABLE and (wandb is not None) and Accelerator().is_main_process:
    os.environ["WANDB_SILENT"] = "true"
    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        entity=(WANDB_ENTITY or None),
        name=WANDB_RUN_NAME or None,
        mode=WANDB_MODE,
        config=dict(
            model_id=MODEL_ID, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS,
            lr=LR, weight_decay=WEIGHT_DECAY, rank=RANK, alpha=LORA_ALPHA,
            grad_accum=GRAD_ACCUM, mixed_precision=MIXED_PRECISION,
            scheduler=LR_SCHEDULER, warmup_steps=WARMUP_STEPS, warmup_ratio=WARMUP_RATIO,
            clip_grad_norm=CLIP_GRAD_NORM, val_guidance=VAL_GUIDANCE, val_steps=VAL_STEPS,
            val_every_steps=VAL_EVERY_STEPS, metrics_every_steps=M_METRICS_EVERY_STEPS,
            val_loss_enable=M_VAL_LOSS_ENABLE, clip_enable=M_CLIP_ENABLE,
            fid_enable=M_FID_ENABLE, kid_enable=M_KID_ENABLE,
            fid_kid_samples=M_FID_KID_SAMPLES, world_size=_world_size(),
        ),
        reinit=True
    )
    if wandb_run:
        print(f"[W&B] Tracking to {wandb_run.url}")

# ============ Lazy loaders for metrics ============
_clip_loaded = False
clip_model = None
clip_processor = None

def _load_clip(device):
    global _clip_loaded, clip_model, clip_processor
    if _clip_loaded: return
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained(M_CLIP_MODEL_ID).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(M_CLIP_MODEL_ID)
    _clip_loaded = True

def _clip_score(imgs: List[Image.Image], texts: List[str], device) -> float:
    if not M_CLIP_ENABLE: return float("nan")
    _load_clip(device)
    with torch.no_grad():
        inputs = clip_processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(device)
        out = clip_model(**inputs)
        logits = out.logits_per_image
        return logits.mean().item()

def _tensor_from_pil_uint8(img: Image.Image, size: int = 299) -> torch.Tensor:
    im = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = torch.from_numpy(np.array(im, dtype=np.uint8))
    return arr.permute(2, 0, 1).unsqueeze(0)

def _collect_real_from_subset(subset: Subset, max_n: int) -> List[Image.Image]:
    ds = subset.dataset
    idxs = subset.indices[:max_n]
    out = []
    for i in idxs:
        p = ds.items[i]["image"]
        try: out.append(Image.open(p).convert("RGB"))
        except Exception: pass
    return out[:max_n]

def _compute_fid_kid(gen_imgs: List[Image.Image], real_imgs: List[Image.Image], device) -> Tuple[Optional[float], Optional[Tuple[float,float]]]:
    if not tm_available or (not (M_FID_ENABLE or M_KID_ENABLE)):
        return None, None
    fid = FrechetInceptionDistance(normalize=False).to(device) if M_FID_ENABLE else None
    kid = KernelInceptionDistance(subset_size=50, normalize=False).to(device) if M_KID_ENABLE else None
    for im in real_imgs:
        t = _tensor_from_pil_uint8(im).to(device)
        if fid is not None: fid.update(t, real=True)
        if kid is not None: kid.update(t, real=True)
    for im in gen_imgs:
        t = _tensor_from_pil_uint8(im).to(device)
        if fid is not None: fid.update(t, real=False)
        if kid is not None: kid.update(t, real=False)
    fid_val = float(fid.compute().item()) if fid is not None else None
    kid_mean, kid_std = (None, None)
    if kid is not None:
        kid_val = kid.compute()
        kid_mean = float(kid_val[0].item()); kid_std = float(kid_val[1].item())
    return fid_val, (kid_mean, kid_std)

def _validation_loss(val_loader, max_samples: int, device, dtype) -> float:
    ensure_flowmatch_schedule(scheduler, device, n_train_steps=_maybe_num_train_steps())
    transformer.eval()
    n = 0
    total = 0.0
    with torch.inference_mode():
        for batch in val_loader:
            imgs = batch["pixel_values"].to(device=device, dtype=dtype)
            posterior = vae.encode(imgs).latent_dist
            latents = posterior.sample() * vae_scale
            noise = torch.randn_like(latents, dtype=dtype)
            t_cont = torch.rand((latents.shape[0],), device=device, dtype=torch.float32)
            t_discrete = _map_t_to_scheduler_timesteps(scheduler, t_cont)
            noisy_latents = scale_noise_compat(scheduler, latents, t_discrete, noise)
            prompt_embeds, pooled_proj = encode_prompts_compat(pipe, batch["captions"], device, do_cfg=False)
            out = transformer(
                hidden_states=noisy_latents,
                timestep=t_discrete,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_proj,
            ).sample
            loss = F.mse_loss(out, noise, reduction="mean")
            bs = imgs.shape[0]
            total += loss.item() * bs
            n += bs
            if n >= max_samples:
                break
    transformer.train()
    return total / max(1, n)

# ---- High-level validation + metrics runners ----
def run_validation_and_log(epoch: int, global_step: int):
    gen_images, gen_texts = [], []
    with torch.inference_mode():
        caps = []
        for vb in val_loader:
            caps.extend(vb["captions"])
            if len(caps) >= 8:
                break
        val_prompts = get_validation_prompts(caps)
        out_dir = Path(OUTPUT_DIR) / "val_samples"; out_dir.mkdir(parents=True, exist_ok=True)
        wb_imgs = []
        for i, ptxt in enumerate(val_prompts):
            images = pipe(ptxt, prompt_2=ptxt, prompt_3=ptxt, num_inference_steps=VAL_STEPS, guidance_scale=VAL_GUIDANCE).images
            for j, img in enumerate(images):
                fpath = out_dir / f"epoch{epoch}_sample{i}_{j}.png"
                img.save(fpath)
                print(f"[VAL] Saved {fpath}")
                gen_images.append(img)
                gen_texts.append(ptxt)
                if WANDB_ENABLE and wandb_run and Accelerator().is_main_process:
                    try:
                        wb_imgs.append(wandb.Image(str(fpath), caption=f"ep{epoch}/p{i}_{j}: {ptxt[:120]}"))
                    except Exception:
                        pass
        if WANDB_ENABLE and wandb_run and Accelerator().is_main_process and wb_imgs:
            wandb.log({"val/samples": wb_imgs, "epoch": epoch, "step": global_step}, step=global_step)
    ensure_flowmatch_schedule(scheduler, device, n_train_steps=_maybe_num_train_steps())
    return gen_images, gen_texts

def run_metrics_and_log(epoch: int, global_step: int, gen_images: List[Image.Image], gen_texts: List[str]):
    metrics = {}
    if M_VAL_LOSS_ENABLE:
        vloss = _validation_loss(val_loader, M_VAL_LOSS_MAX_SAMP, device, dtype)
        metrics["val/loss"] = float(vloss)
        print(f"[METRICS] Val loss (<= {M_VAL_LOSS_MAX_SAMP} samples): {vloss:.4f}")
    if M_CLIP_ENABLE and len(gen_images) > 0:
        try:
            cs = _clip_score(gen_images, gen_texts, device)
            metrics["val/clip_score"] = float(cs)
            print(f"[METRICS] CLIP score (avg): {cs:.4f}")
        except Exception as e:
            print(f"[METRICS][WARN] CLIP failed: {e}")
    if (M_FID_ENABLE or M_KID_ENABLE) and tm_available:
        try:
            real_imgs = _collect_real_from_subset(val_ds, M_FID_KID_SAMPLES)
            need = M_FID_KID_SAMPLES
            gen_for_fid = gen_images[:need] if len(gen_images) >= need else []
            if len(gen_for_fid) < need:
                caps2 = []
                for vb in val_loader:
                    caps2.extend(vb["captions"])
                    if len(caps2) >= need:
                        break
                caps2 = caps2[:need] if len(caps2) else ["a high quality photo"] * need
                with torch.inference_mode():
                    for k, ptxt in enumerate(caps2):
                        im = pipe(ptxt, prompt_2=ptxt, prompt_3=ptxt, num_inference_steps=max(12, VAL_STEPS//2), guidance_scale=VAL_GUIDANCE).images[0]
                        gen_for_fid.append(im)
                        if len(gen_for_fid) >= need: break
                ensure_flowmatch_schedule(scheduler, device, n_train_steps=_maybe_num_train_steps())
            fid_val, kid_pair = _compute_fid_kid(gen_for_fid, real_imgs, device)
            if fid_val is not None:
                metrics["val/fid"] = float(fid_val)
                print(f"[METRICS] FID: {fid_val:.3f}")
            if kid_pair is not None and kid_pair[0] is not None:
                metrics["val/kid_mean"] = float(kid_pair[0]); metrics["val/kid_std"] = float(kid_pair[1])
                print(f"[METRICS] KID: mean {kid_pair[0]:.5f} ± {kid_pair[1]:.5f}")
        except Exception as e:
            print(f"[METRICS][WARN] FID/KID failed: {e}")
    elif (M_FID_ENABLE or M_KID_ENABLE) and not tm_available:
        print("[METRICS][WARN] torchmetrics[image] not available; skip FID/KID.")
    if WANDB_ENABLE and wandb_run:
        wandb.log({**metrics, "epoch": epoch, "step": global_step}, step=global_step)

# ============ Training ============
print(f"[INFO] Starting training for {EPOCHS} epochs (max_steps={MAX_TRAIN_STEPS or 'epoch-based'})")
post_status("TRAINING_STARTED")
transformer.train()
global_step = 0
os.makedirs(os.path.join(OUTPUT_DIR, "val_samples"), exist_ok=True)

stop_now = False
for epoch in range(EPOCHS):
    for step, batch in enumerate(train_loader):
        if stop_now: break
        step_t0 = time.perf_counter()
        with accelerator.accumulate(param_wrap):
            imgs = batch["pixel_values"].to(device=device, dtype=dtype)
            with torch.no_grad():
                posterior = vae.encode(imgs).latent_dist
                latents = posterior.sample() * vae_scale
            noise = torch.randn_like(latents, dtype=dtype)
            t_cont = torch.rand((latents.shape[0],), device=device, dtype=torch.float32)
            t_discrete = _map_t_to_scheduler_timesteps(scheduler, t_cont)
            noisy_latents = scale_noise_compat(scheduler, latents, t_discrete, noise)
            prompt_embeds, pooled_proj = encode_prompts_compat(pipe, batch["captions"], device, do_cfg=False)
            out = transformer(
                hidden_states=noisy_latents,
                timestep=t_discrete,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_proj,
            ).sample
            loss = F.mse_loss(out, noise)
            accelerator.backward(loss)
            if CLIP_GRAD_NORM > 0.0:
                accelerator.clip_grad_norm_(param_wrap.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None:
                lr_scheduler.step()

        iter_time = time.perf_counter() - step_t0
        cur_lr = _get_lr(optimizer)
        gnorm = _grad_global_norm(param_wrap.parameters()) if LOG_GRAD_NORM else None
        gmem  = _gpu_mem_mb() if LOG_GPU_MEMORY else None
        thr   = ((BATCH_SIZE * accelerator.gradient_accumulation_steps * _world_size()) / iter_time) if LOG_THROUGHPUT and iter_time > 0 else None

        if (global_step % LOG_EVERY == 0) or (step == 0):
            msg = f"Epoch {epoch} Step {global_step} Loss {loss.item():.4f}"
            if VERBOSE_LOGS:
                if cur_lr is not None: msg += f" LR {cur_lr:.6g}"
                if gnorm is not None:  msg += f" |GradNorm {gnorm:.3f}"
                if thr is not None:    msg += f" |Throughput {thr:.1f} samples/s"
                if gmem is not None:   msg += f" |GPU MaxMem {gmem:.0f}MB"
            print(msg)

        if WANDB_ENABLE and wandb_run and Accelerator().is_main_process:
            log_dict = {"loss": float(loss.item()), "step": int(global_step), "epoch": int(epoch)}
            if cur_lr is not None: log_dict["lr"] = float(cur_lr)
            if gnorm is not None:  log_dict["grad_norm"] = float(gnorm)
            if thr is not None:    log_dict["throughput_sps"] = float(thr)
            if gmem is not None:   log_dict["gpu_max_mem_mb"] = float(gmem)
            wandb.log(log_dict, step=global_step)

        if VAL_EVERY_STEPS and global_step > 0 and (global_step % VAL_EVERY_STEPS == 0) and Accelerator().is_main_process:
            gen_imgs, gen_txts = run_validation_and_log(epoch, global_step)
            if should_metrics(global_step, epoch_done=False):
                run_metrics_and_log(epoch, global_step, gen_imgs, gen_txts)

        if Accelerator().is_main_process and SAVE_EVERY_STEPS and global_step > 0 and (global_step % SAVE_EVERY_STEPS == 0):
            out_dir = periodic_save_dir(OUTPUT_DIR, step=global_step)
            save_lora_adapters(transformer, out_dir, lora_manifest)
            enforce_save_limit(OUTPUT_DIR, SAVE_LIMIT)
            print(f"[SAVE] Saved checkpoint: {out_dir}")
            if WANDB_ENABLE and wandb_run:
                try:
                    art = wandb.Artifact(f"lora_ckpt_step_{global_step}", type="model")
                    ad = os.path.join(out_dir, "lora_adapters.safetensors")
                    ix = os.path.join(out_dir, "lora_index.json")
                    if os.path.exists(ad): art.add_file(ad)
                    if os.path.exists(ix): art.add_file(ix)
                    wandb.log_artifact(art)
                except Exception:
                    pass

        global_step += 1
        if MAX_TRAIN_STEPS and global_step >= MAX_TRAIN_STEPS:
            stop_now = True
            break

    epoch_done = True
    if Accelerator().is_main_process and SAVE_EVERY_EPOCHS and ((epoch + 1) % SAVE_EVERY_EPOCHS == 0):
        out_dir = periodic_save_dir(OUTPUT_DIR, epoch=epoch + 1)
        save_lora_adapters(transformer, out_dir, lora_manifest)
        enforce_save_limit(OUTPUT_DIR, SAVE_LIMIT)
        print(f"[SAVE] Saved checkpoint: {out_dir}")
        if WANDB_ENABLE and wandb_run:
            try:
                art = wandb.Artifact(f"lora_ckpt_epoch_{epoch+1}", type="model")
                ad = os.path.join(out_dir, "lora_adapters.safetensors")
                ix = os.path.join(out_dir, "lora_index.json")
                if os.path.exists(ad): art.add_file(ad)
                if os.path.exists(ix): art.add_file(ix)
                wandb.log_artifact(art)
            except Exception:
                pass

    if should_validate(global_step, epoch_done=True) and Accelerator().is_main_process:
        gen_imgs, gen_txts = run_validation_and_log(epoch, global_step)
        if should_metrics(global_step, epoch_done=True):
            run_metrics_and_log(epoch, global_step, gen_imgs, gen_txts)

    transformer.train()
    if stop_now:
        break

# ============ Save final ============
accelerator = Accelerator()  # ensure we can check rank for final ops
accelerator.wait_for_everyone()
if Accelerator().is_main_process:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    post_status("SAVING_WEIGHTS")
    save_lora_adapters(transformer, OUTPUT_DIR, lora_manifest)
    enforce_save_limit(OUTPUT_DIR, SAVE_LIMIT)
print(f"[INFO] LoRA adapters saved at {OUTPUT_DIR} (lora_adapters.safetensors + lora_index.json)")

if WANDB_ENABLE and wandb_run and Accelerator().is_main_process:
    try:
        final_art = wandb.Artifact(f"lora_final_{WANDB_RUN_NAME or 'run'}", type="model")
        final_adapters = os.path.join(OUTPUT_DIR, "lora_adapters.safetensors")
        final_index    = os.path.join(OUTPUT_DIR, "lora_index.json")
        if os.path.exists(final_adapters): final_art.add_file(final_adapters)
        if os.path.exists(final_index):    final_art.add_file(final_index)
        wandb.log_artifact(final_art)
    except Exception:
        pass
    wandb.finish()

# ==========================
# Presigned Upload (MinIO/AWS safe)
# ==========================
_ADDR_STYLE       = "path" if S3_ENDPOINT else "virtual"
_PAYLOAD_SIGNING  = False
_PART_MB          = 64
_MPU_THRESHOLD_MB = 128
_PRESIGN_TTL      = 3600
_REQ_TIMEOUT      = 120
_MAX_RETRIES_FILE = 3
_MAX_RETRIES_PART = 3
_PART_SIZE        = _PART_MB * 1024 * 1024
_MPU_THRESHOLD    = _MPU_THRESHOLD_MB * 1024 * 1024

def _strip_proxies():
    for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy","NO_PROXY","no_proxy"):
        os.environ.pop(k, None)

def _dst_s3_presign():
    cfg = BotoConfig(
        signature_version="s3v4",
        s3={"addressing_style": _ADDR_STYLE, "payload_signing": _PAYLOAD_SIGNING},
        retries={"max_attempts": 3, "mode": "standard"},
    )
    kwargs = dict(service_name="s3", config=cfg, region_name="us-east-1")
    if S3_ENDPOINT:
        kwargs.update(endpoint_url=S3_ENDPOINT,
                      aws_access_key_id=S3_ACCESS_KEY or None,
                      aws_secret_access_key=S3_SECRET_KEY or None)
    return boto3.client(**kwargs)

def _guess_type(path: Path):
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype or "application/octet-stream"

def _sha256_file(path: Path, block=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()

def _upload_single_put(dst, local_path: Path, bucket: str, key: str):
    _strip_proxies()
    size    = local_path.stat().st_size
    ctype   = _guess_type(local_path)
    sha_hex = _sha256_file(local_path)
    headers = {"Content-Type": ctype, "Content-Length": str(size), "Connection": "close"}
    last_err = None
    for attempt in range(1, _MAX_RETRIES_FILE + 1):
        try:
            url = dst.generate_presigned_url(
                ClientMethod="put_object",
                Params={"Bucket": bucket, "Key": key, "ContentType": ctype},
                ExpiresIn=_PRESIGN_TTL, HttpMethod="PUT",
            )
            with open(local_path, "rb") as f:
                resp = requests.put(url, data=f, headers=headers, timeout=_REQ_TIMEOUT)
            if 200 <= resp.status_code < 300:
                print(f"✅ {local_path} -> s3://{bucket}/{key} (sha256={sha_hex[:12]}..., {size/1e6:.1f} MB)")
                return size
            last_err = f"HTTP {resp.status_code} {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(min(10, 2 ** attempt))
    print(f"❌ PUT failed for {local_path}: {last_err}")
    return 0

def _upload_multipart(dst, local_path: Path, bucket: str, key: str):
    _strip_proxies()
    size = local_path.stat().st_size
    ctype = _guess_type(local_path)
    total_parts = max(1, math.ceil(size / _PART_SIZE))
    try:
        mpu = dst.create_multipart_upload(Bucket=bucket, Key=key, ContentType=ctype)
        upload_id = mpu["UploadId"]
    except ClientError as e:
        print(f"❌ create_multipart_upload failed: {e}")
        return 0

    etags = []
    try:
        with open(local_path, "rb") as f:
            for part_num in range(1, total_parts + 1):
                to_read = min(_PART_SIZE, size - f.tell())
                if to_read <= 0: break
                data = f.read(to_read)
                if not data: break

                last_err = None
                for attempt in range(1, _MAX_RETRIES_PART + 1):
                    try:
                        url = dst.generate_presigned_url(
                            ClientMethod="upload_part",
                            Params={"Bucket": bucket, "Key": key, "UploadId": upload_id, "PartNumber": part_num},
                            ExpiresIn=_PRESIGN_TTL, HttpMethod="PUT",
                        )
                        headers = {"Content-Length": str(len(data)), "Connection": "close"}
                        resp = requests.put(url, data=data, headers=headers, timeout=_REQ_TIMEOUT)
                        if 200 <= resp.status_code < 300:
                            etag = resp.headers.get("ETag") or resp.headers.get("Etag") or resp.headers.get("etag")
                            if not etag:
                                md5_hex = hashlib.md5(data).hexdigest()
                                etag = f"\"{md5_hex}\""
                            etags.append({"ETag": etag, "PartNumber": part_num})
                            print(f"   part {part_num:03d}/{total_parts:03d} OK ({len(data)/1e6:.1f} MB)")
                            break
                        last_err = f"HTTP {resp.status_code} {resp.text[:200]}"
                    except Exception as e:
                        last_err = str(e)
                    time.sleep(min(10, 2 ** attempt))
                else:
                    raise RuntimeError(f"Exhausted retries for part {part_num}: {last_err}")

        dst.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": etags}
        )
        print(f"✅ (MPU) {local_path} -> s3://{bucket}/{key} ({size/1e6:.1f} MB, {total_parts} parts)")
        return size
    except Exception as e:
        print(f"❌ MPU failed: {e} — aborting…")
        try:
            dst.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        except Exception:
            pass
        return 0

def upload_file_presigned(local_path: Path, bucket: str, key: str):
    dst = _dst_s3_presign()
    size = local_path.stat().st_size
    if size >= _MPU_THRESHOLD:
        return _upload_multipart(dst, local_path, bucket, key)
    else:
        return _upload_single_put(dst, local_path, bucket, key)

def upload_directory_presigned(local_dir: Path, s3_url: str, folder_name: Optional[str] = None) -> bool:
    if not s3_url.startswith("s3://"):
        raise ValueError(f"❌ Invalid S3 path: {s3_url}")
    bucket = s3_url.split("/")[2]
    prefix = "/".join(s3_url.split("/")[3:]).rstrip("/")
    if folder_name:
        prefix = f"{prefix}/{folder_name}".rstrip("/")

    ok = True
    for root, _, files in os.walk(local_dir):
        for fname in files:
            lp = Path(root) / fname
            rel = lp.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}"
            post_status("UPLOADING_WEIGHTS")
            sent = upload_file_presigned(lp, bucket, key)
            if sent <= 0:
                ok = False
                print(f"❌ Failed upload: {lp} -> s3://{bucket}/{key}")
    return ok

# Upload final artifacts
if Accelerator().is_main_process:
    try:
        post_status("UPLOADING_WEIGHTS")
        success = upload_directory_presigned(Path(OUTPUT_DIR), WEIGHTS_PREFIX, folder_name=("SD35_LoRA" if RUN_NAME == "" else RUN_NAME))
        if not success:
            post_status("FAILED")
            raise RuntimeError("One or more artifact uploads failed.")
        post_status("COMPLETED")
        print("✅ Upload complete")
    except Exception as e:
        post_status("FAILED")
        raise
