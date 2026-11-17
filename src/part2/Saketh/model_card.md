# Part 2 model card — Saketh LoRA baseline

## Summary

This model card documents the independent DATA 266 Lab 2 baseline. It fine-tunes Stable Diffusion 1.5 with UNet attention LoRA adapters on the Naruto BLIP Captions dataset. The artifacts are intended for non-commercial academic research and teaching only.

## Base model

`stable-diffusion-v1-5/stable-diffusion-v1-5`, generated at `512 x 512` resolution.

## Data

- Dataset: `lambdalabs/naruto-blip-captions`
- Original images: `1,221`
- Valid images: `1,221`
- Corrupt images removed: `0`
- Exact duplicates removed: `0`
- Split seed: `266`
- Train/test split in the saved run: `1,098 / 123`

The data is narrow, franchise-specific, and caption quality is inherited from BLIP-generated descriptions. The characters are copyrighted; this adapter is not presented as a commercial asset.

## Training settings and measured cost

| Run | Rank | Trainable parameters | Steps | Batch | Learning rate | Time | GPU-hours | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA rank 4 | 4 | 797,184 | 1,000 | 1 | 1e-4 | 194.62 s | 0.0541 | 6.64 GB |
| LoRA rank 16 | 16 | 3,188,736 | 1,000 | 1 | 1e-4 | 191.60 s | 0.0532 | 10.68 GB |

All values above come from the saved `training_record.json` files. The generation run used 20 fixed prompts, two seeds per prompt, and 30 inference steps for the base model and both adapters, producing 120 manifest entries.

## Results status

The individual baseline repository copy does not contain a completed KID, CLIP, LPIPS, NSFW, memorization, or human-evaluation output. Those measurements must be computed from the saved generation manifest and included before this card is used as the final model card. Do not copy the separate team-final LoRA/DoRA metrics into this baseline card.

## Safety and limitations

The diffusion safety checker must remain enabled for every generation used in the final evaluation. Flagged outputs must be counted and excluded according to the evaluation protocol. A ten-image nearest-neighbor CLIP memorization check and three closest pairs are also required. The adapter may over-specialize to the franchise, reduce prompt fidelity on out-of-domain prompts, and reproduce training-style composition. Use is limited to non-commercial academic work subject to the base-model and dataset terms.
