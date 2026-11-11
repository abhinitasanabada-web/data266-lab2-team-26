"""Compute the Part 2 notebook metrics from generated files.

The function intentionally raises on missing real images instead of returning
placeholder values. Human ratings remain a separate, auditable CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _tensor_images(paths, device):
    import torch
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    return torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).to(device)


def evaluate_manifest(manifest_path: str | Path, real_paths: list[str | Path], out_path: str | Path) -> dict[str, Any]:
    """Return KID, CLIP alignment, and two-seed LPIPS by model."""

    try:
        import torch
        from torchmetrics.image.kid import KernelInceptionDistance
        import lpips
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("Install torchmetrics, lpips, transformers, and torchvision before evaluation.") from exc
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest or not real_paths: raise ValueError("Generation manifest and real test images are required.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    results: dict[str, Any] = {}
    for model_name in sorted({row["model"] for row in manifest}):
        rows = [row for row in manifest if row["model"] == model_name]
        in_domain = [row for row in rows if int(row["prompt_index"]) < 10]
        fake_paths = [row["path"] for row in in_domain]
        kid = KernelInceptionDistance(subset_size=min(10, len(real_paths), len(fake_paths)), normalize=True).to(device)
        real = (_tensor_images(real_paths[: len(fake_paths)], device) * 255).byte(); fake = (_tensor_images(fake_paths, device) * 255).byte()
        kid.update(real, real=True); kid.update(fake, real=False); kid_mean, kid_std = kid.compute()
        similarities = []
        for row in rows:
            image = Image.open(row["path"]).convert("RGB")
            inputs = clip_processor(text=[row["prompt"]], images=image, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                image_features = clip_model.get_image_features(pixel_values=inputs["pixel_values"]); text_features = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            similarities.append(float(torch.nn.functional.cosine_similarity(image_features, text_features).item()))
        by_prompt = {}
        for row in rows: by_prompt.setdefault(int(row["prompt_index"]), {})[int(row["seed"])] = row["path"]
        diversity = []
        for pair in by_prompt.values():
            if 0 in pair and 1 in pair:
                a = _tensor_images([pair[0]], device) * 2 - 1; b = _tensor_images([pair[1]], device) * 2 - 1
                diversity.append(float(lpips_model(a, b).item()))
        results[model_name] = {"kid_mean": float(kid_mean), "kid_std": float(kid_std), "clip_score": float(np.mean(similarities)), "lpips_diversity": float(np.mean(diversity)) if diversity else None, "n_images": len(rows)}
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8"); return results
