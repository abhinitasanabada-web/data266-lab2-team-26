"""Reusable Part 2 data-preparation, LoRA-training, and generation utilities.

The training function follows the Stable Diffusion 1.5 latent-training recipe:
the VAE and text encoder are frozen, only UNet attention LoRA processors are
optimized, and the same data/step budget is used for both ranks.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

SEED = 266
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def inspect_images(root: str | Path) -> dict[str, Any]:
    """Validate images and remove duplicate candidates from the returned list."""

    paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    valid, corrupt, duplicates = [], [], 0; hashes: set[str] = set()
    for path in paths:
        try:
            with Image.open(path) as image: image.verify()
            with Image.open(path) as image: digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            if digest in hashes: duplicates += 1; continue
            hashes.add(digest); valid.append(str(path))
        except Exception: corrupt.append(str(path))
    return {"valid": len(valid), "corrupt": len(corrupt), "duplicates": duplicates, "files": valid, "corrupt_files": corrupt}


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in columns}
    return next((lowered[candidate] for candidate in candidates if candidate in lowered), None)


def load_caption_dataset(dataset_name: str = "lambdalabs/naruto-blip-captions", split: str = "train"):
    """Load a Hugging Face image/caption dataset with tolerant column detection."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before loading the Part 2 dataset.") from exc
    dataset = load_dataset(dataset_name, split=split)
    image_column = _find_column(dataset.column_names, ["image", "img", "pixel_values"])
    caption_column = _find_column(dataset.column_names, ["text", "caption", "captions", "prompt", "description"])
    if not image_column or not caption_column: raise ValueError(f"Could not identify image/caption columns in {dataset.column_names}")
    return dataset, image_column, caption_column


def prepare_dataset(dataset_name: str, out_dir: str | Path, seed: int = SEED, test_fraction: float = 0.1) -> dict[str, Any]:
    """Remove invalid/duplicate images, split 90/10, and save metadata."""

    set_seed(seed); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dataset, image_column, caption_column = load_caption_dataset(dataset_name)
    seen: set[str] = set(); valid_rows: list[dict[str, str]] = []; removed_corrupt = removed_duplicate = 0
    for index, row in enumerate(dataset):
        try:
            image = row[image_column].convert("RGB") if hasattr(row[image_column], "convert") else Image.open(row[image_column]).convert("RGB")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in seen: removed_duplicate += 1; continue
            seen.add(digest); valid_rows.append({"source_index": str(index), "caption": str(row[caption_column]), "image_hash": digest})
        except Exception: removed_corrupt += 1
    random.Random(seed).shuffle(valid_rows); cutoff = max(1, int(len(valid_rows) * (1.0 - test_fraction)))
    train_rows, test_rows = valid_rows[:cutoff], valid_rows[cutoff:]
    metadata = {"dataset_name": dataset_name, "seed": seed, "image_column": image_column, "caption_column": caption_column, "original_count": len(dataset), "valid_count": len(valid_rows), "removed_corrupt": removed_corrupt, "removed_duplicates": removed_duplicate, "train_count": len(train_rows), "test_count": len(test_rows), "train": train_rows, "test": test_rows}
    (out_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8"); return metadata


@dataclass
class TrainingRecord:
    run: str; rank: int; learning_rate: float; steps: int; batch_size: int; trainable_parameters: int; training_seconds: float; gpu_hours: float; peak_gpu_memory_gb: float; adapter_path: str


def _lora_processors(unet, rank: int):
    """Build legacy attention processors for older Diffusers releases.

    Current Diffusers releases use ``unet.add_adapter`` with PEFT instead;
    this helper is retained only for installations that do not expose that
    method.
    """
    try:
        from diffusers.models.attention_processor import LoRAAttnProcessor
    except ImportError:  # pragma: no cover - older Diffusers releases
        from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LoRAAttnProcessor

    processors = {}; block_out_channels = unet.config.block_out_channels
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"): hidden_size = block_out_channels[-1]
        elif name.startswith("up_blocks"): hidden_size = list(reversed(block_out_channels))[int(name.split(".")[1])]
        else: hidden_size = block_out_channels[int(name.split(".")[1])]
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        try:
            processors[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
            )
        except TypeError as keyword_error:
            try:
                processors[name] = LoRAAttnProcessor(hidden_size, cross_attention_dim, rank)
            except TypeError:
                raise keyword_error
    return processors


def train_lora(base_model: str, dataset_name: str, out_dir: str | Path, run_name: str, rank: int, steps: int = 1000, batch_size: int = 1, learning_rate: float = 1e-4, seed: int = SEED) -> TrainingRecord:
    """Train one LoRA adapter and save adapter weights plus run metadata."""

    set_seed(seed)
    try:
        import torch
        from accelerate import Accelerator
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        from datasets import load_dataset
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Install the pinned GPU requirements before Part 2 training.") from exc
    if not torch.cuda.is_available(): raise RuntimeError("Part 2 training requires a CUDA GPU.")
    out_dir = Path(out_dir); adapter_dir = out_dir / run_name; adapter_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    dataset, image_column, caption_column = load_caption_dataset(dataset_name)
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def collate(rows):
        images, captions = [], []
        for row in rows:
            image = row[image_column].convert("RGB") if hasattr(row[image_column], "convert") else Image.open(row[image_column]).convert("RGB")
            images.append(transform(image)); captions.append(str(row[caption_column]))
        tokens = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {"pixel_values": torch.stack(images), "input_ids": tokens.input_ids}

    from torch.utils.data import DataLoader
    # Native Windows is more reliable with single-process DataLoaders.
    loader = DataLoader(dataset, shuffle=True, collate_fn=collate, batch_size=batch_size, num_workers=0, pin_memory=True, persistent_workers=False)
    text_encoder.requires_grad_(False); vae.requires_grad_(False); unet.requires_grad_(False)

    # Diffusers 0.35+ exposes the PEFT adapter API. The older
    # ``set_attn_processor`` route now fails because LoRA processor classes no
    # longer accept hidden_size/cross_attention_dim constructor arguments.
    use_peft = hasattr(unet, "add_adapter")
    if use_peft:
        from peft import LoraConfig

        unet_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)
        lora_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
        if not lora_parameters:
            raise RuntimeError("PEFT added no trainable LoRA parameters to the UNet.")
        optimizer = torch.optim.AdamW(lora_parameters, lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
        text_encoder, vae, unet, optimizer, loader = accelerator.prepare(text_encoder, vae, unet, optimizer, loader)
        accumulate_target = unet
    else:
        from diffusers.loaders import AttnProcsLayers

        unet.set_attn_processor(_lora_processors(unet, rank)); lora_layers = AttnProcsLayers(unet.attn_processors)
        optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)
        text_encoder, vae, unet, lora_layers, optimizer, loader = accelerator.prepare(text_encoder, vae, unet, lora_layers, optimizer, loader)
        accumulate_target = lora_layers
    text_encoder.to(accelerator.device); vae.to(accelerator.device); unet.train()
    if not use_peft:
        lora_layers.train()
    start = time.perf_counter(); completed = 0; iterator = iter(loader)
    while completed < steps:
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        with accelerator.accumulate(accumulate_target):
            latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents); timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps); hidden = text_encoder(batch["input_ids"])[0]
            prediction = unet(noisy_latents, timesteps, hidden).sample
            target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(latents, noise, timesteps)
            loss = torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="mean")
            accelerator.backward(loss); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        completed += 1
    unwrapped_unet = accelerator.unwrap_model(unet)
    if use_peft:
        from diffusers import StableDiffusionPipeline
        from peft.utils import get_peft_model_state_dict

        try:
            from diffusers.utils import convert_state_dict_to_diffusers
            lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        except ImportError:
            lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(adapter_dir),
            unet_lora_layers=lora_state_dict,
            safe_serialization=True,
        )
    else:
        unwrapped_unet.save_attn_procs(str(adapter_dir))
    elapsed = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    trainable_parameters = sum(parameter.numel() for parameter in unwrapped_unet.parameters() if parameter.requires_grad)
    record = TrainingRecord(run_name, rank, learning_rate, steps, batch_size, trainable_parameters, elapsed, elapsed / 3600.0, peak_memory, str(adapter_dir))
    (adapter_dir / "training_record.json").write_text(json.dumps(asdict(record), indent=2), encoding="utf-8"); return record


def generate_images(base_model: str, adapter_paths: dict[str, str | Path], prompts: list[str], out_dir: str | Path, seeds: tuple[int, ...] = (0, 1), steps: int = 30) -> list[dict[str, Any]]:
    """Generate base/run-1/run-2 images with identical sampler settings."""

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc: raise RuntimeError("Install diffusers and torch before generation.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"; dtype = torch.float16 if device == "cuda" else torch.float32
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); results = []
    for model_name, adapter in {"base": None, **{name: str(path) for name, path in adapter_paths.items()}}.items():
        # Keep the diffusion safety checker enabled for the required NSFW safety analysis.
        pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype).to(device)
        if adapter: pipe.load_lora_weights(adapter)
        pipe.set_progress_bar_config(disable=True)
        for prompt_index, prompt in enumerate(prompts):
            for seed in seeds:
                generator = torch.Generator(device=device).manual_seed(seed)
                output = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5, generator=generator)
                image = output.images[0]
                path = out_dir / f"{model_name}_prompt{prompt_index:02d}_seed{seed}.png"; image.save(path)
                results.append({"model": model_name, "prompt": prompt, "prompt_index": prompt_index, "seed": seed, "steps": steps, "path": str(path), "nsfw_flag": bool(getattr(output, "nsfw_content_detected", [False])[0])})
        del pipe
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    (out_dir / "generation_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8"); return results


def save_run_metrics(out: str | Path, run_name: str, **values: Any) -> Path:
    path = Path(out); path.mkdir(parents=True, exist_ok=True); target = path / f"{run_name}_metrics.json"; target.write_text(json.dumps(values, indent=2), encoding="utf-8"); return target


def training_record(run_name: str, learning_rate: float, steps: int, batch_size: int, trainable_parameters: int, started: float, peak_memory: float = 0.0) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    return {"run": run_name, "learning_rate": learning_rate, "steps": steps, "batch_size": batch_size, "trainable_parameters": trainable_parameters, "training_seconds": elapsed, "gpu_hours": elapsed / 3600.0, "peak_gpu_memory_gb": peak_memory, "status": "computed_during_training"}
