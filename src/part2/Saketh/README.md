# Saketh Part 2 — Stable Diffusion LoRA Baseline

## Run configuration

- Base model: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Dataset: `lambdalabs/naruto-blip-captions`
- Seed: `266`
- Resolution: `512 x 512`
- Learning rate: `1e-4`
- Training budget: `1,000` steps per run
- Batch size: `1`
- Runs: LoRA rank 4 and LoRA rank 16
- Generation: 20 prompts, seeds 0 and 1, 30 inference steps, guidance scale 7.5

Run `notebooks/p2_finetune.ipynb` with **Restart & Run All** on a CUDA GPU. The notebook expects the Part 2 code under `part2/src` and writes adapters, dataset metadata, prompts, and generated-image metadata under `part2/outputs`.

The committed local records show 1,221 valid images, 0 corrupt images, 0 exact duplicates, and a deterministic 1,098/123 train/test split. The rank-4 and rank-16 adapter records are retained under `outputs/lora_rank4/` and `outputs/lora_rank16/`.

## Important completion items

The individual notebook does not yet save KID, CLIP, LPIPS, NSFW-count, memorization, or human-rating outputs. Compute those from the real `generation_manifest.json` before using this baseline as a final Part 2 result. Upload adapters and generated images to Hugging Face and place the links in the root `links.txt`; do not commit the large image set to GitHub.
