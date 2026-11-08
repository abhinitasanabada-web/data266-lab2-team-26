# Part 2 — Fine-Tuning Stable Diffusion for Domain-Specific Image Generation

**DATA 266 Lab 2 · Pair_Programming_Team_26 · Notebook: [`notebooks/p2_finetune.ipynb`](notebooks/p2_finetune.ipynb) · Git tag: `p2-v1.0`**

We fine-tuned Stable Diffusion 1.5 to generate images in the Naruto anime style, compared two parameter-efficient fine-tuning
runs (LoRA and DoRA) with the base model, and evaluated quality, prompt alignment, diversity, cost and safety. Everything runs
in one notebook on a single 16 GB GPU (Colab T4); we use no hosted APIs.

| | |
|---|---|
| Base model | [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), 512×512 |
| Dataset | [`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) (1,221 captioned images) |
| Run 1 / Run 2 | LoRA rank 16 / DoRA rank 16 on the UNet attention layers — same data, seed and step budget |
| Training | 3,000 steps × batch 4, lr 1e-4, 10% caption dropout, adapters saved at 1,000 / 2,000 / 3,000 steps |
| Checkpoint choice | Best checkpoint × adapter strength per run, chosen on separate validation prompts (never on test data) |
| Seed | 266 everywhere (split, prompts, training, sampling, human-evaluation pairs) |

## Contents
1. [What we did](#1-what-we-did) · 2. [Failure analysis of our first run and what we changed](#2-failure-analysis-of-our-first-run-and-what-we-changed) ·
3. [Results](#3-results) · 4. [How to run it](#4-how-to-run-it) · 5. [Where things are stored](#5-where-things-are-stored) ·
6. [Requirement checklist](#6-requirement-checklist) · 7. [Software and references](#7-software-and-references)

## 1. What we did

**Data preparation (Task 2.1).** We loaded the images as raw bytes (so one broken file cannot crash the run), removed corrupted
images (undecodable, truncated, < 64 px, blank) and empty captions, and removed exact duplicates (MD5 of the pixels) and
near-duplicates (64-bit perceptual hash, Hamming distance ≤ 4) *before* splitting, so no image is in both train and test. We
split 90/10 with seed 266 (1,099 train / 122 test), center-cropped to 512×512, and printed the statistics for the dataset
description and its two limitations: generic, sometimes wrong BLIP captions, and a small one-franchise dataset skewed toward
male characters.

**Fixed prompts (Task 2.3).** Before training, the notebook writes and **commits** 20 evaluation prompts
(`prompts/p2_prompts_naruto.json`: 10 in-domain test captions, 5 out-of-domain, 5 compositional) and 12 separate validation
prompts (`prompts/p2_val_prompts_naruto.json`: 8 training captions + 4 of our own). The Git timestamp shows they were fixed in
advance.

**Fine-tuning (Task 2.2).** We pre-compute the VAE latents (each image + its flip) and text embeddings once, then train with PEFT
following the diffusers `train_text_to_image_lora.py` recipe: noise-prediction MSE with Min-SNR weighting (γ = 5), AdamW, cosine
schedule with 100 warm-up steps, fp16 with fp32 adapter weights, gradient checkpointing and 10% caption dropout. The loop is
resumable (batch order, noise and timesteps come from the seed and step; checkpoints every 250 steps, backed up to Hugging Face).
The notebook prints a table per run (trainable parameters, learning rate, steps, batch size, peak GPU memory, training time) and
pushes both adapters with a model card.

**Choosing each run's checkpoint.** For every saved checkpoint (1,000 / 2,000 / 3,000 steps) at adapter strength 0.75 and 1.0 we
generate the 12 validation prompts and pick, with a rule fixed in advance, the setting with the **highest style score** (CLIP
similarity to the average training image) among those with ≤ 25% blocked images and at most 1.5 CLIP points below the base
model. The test set and the 20 evaluation prompts are never used for this.

**Image generation (Task 2.3).** Base model, Run 1 and Run 2 use identical settings (DPM-Solver++, 30 steps, guidance 7.5,
512×512, the same negative prompt "nsfw, nudity, naked") and seeds 0 and 1 → 40 images per model. We show side-by-side grids for
8 prompts and upload all images to a Hugging Face dataset whose `metadata.csv` lists prompt, seed and model.

**Evaluation (Task 2.4).**

| Measurement | What we computed |
|---|---|
| Primary | **KID** (in-domain generations vs. held-out test images, torchmetrics) and **CLIP score** between each prompt and its image (`openai/clip-vit-large-patch14`) |
| Secondary | **Diversity:** average LPIPS (AlexNet) between the seed-0 and seed-1 image of each prompt |
| Human evaluation | 25 blinded pairs (final run vs. base, left/right randomized); 4 raters — both of us + two people outside the team — choose the better image for domain style and for prompt match; we report the win rate and agreement |
| Efficiency | GPU-hours per training run, with peak GPU memory |

The final run is chosen with a rule fixed in advance (lower in-domain KID wins unless the gap is within the KID std; then higher
CLIP). The notebook also shows automatically selected **failure modes** (an out-of-domain prompt taken over by the anime style,
low diversity between seeds, a failed compositional prompt, over-blocking by the safety filter).

**Safety (C4).** *License:* the images show copyrighted Naruto characters and the dataset states no license, so our adapters and
images are for **non-commercial academic research only** (SD 1.5 is CreativeML OpenRAIL-M). *NSFW filter:* the SD safety checker
stays on for every image; blocked (black) images are counted and left out of the quality metrics. *Memorization:* for 10
in-domain images of the final run we find the most similar training image by CLIP and show the 3 closest pairs, calibrated
against how close the never-trained test images are to the training set.

## 2. Failure analysis of our first run and what we changed
Our first full run (Colab T4; LoRA and DoRA r16, **2,000 steps**, no caption dropout; 39.8 / 48.4 min, 0.66 / 0.81 GPU-hours,
peak 3.2 / 3.3 GB) is kept with its outputs as `notebooks/archive/p2_finetune_T4_run1.ipynb`. Its numbers are quoted here only
to explain our changes; **the numbers in our report come from the final notebook.**

| # | What went wrong in the first run | What we changed | Why |
|---|---|---|---|
| F1 | The safety checker blacked out **12/40 (LoRA)** and **11/40 (DoRA)** images, but **0/40** base images | Same negative prompt `"nsfw, nudity, naked"` for **all three** models | Steers away from flagged content; identical for every model, so the comparison stays fair |
| F2 | The metrics counted those black images as real images (e.g. the "low diversity" example ID07 was two black images, LPIPS 0.000; the memorization check matched black images) | Blocked images are left out of CLIP / KID / LPIPS, the human-rating pairs and the memorization check; the counts are reported | A black image says nothing about style or prompt match; the filter itself stays on |
| F3 | The extended KID standard deviation was exactly **0.0000** | KID subset size = min(#images, 50) | Every subset contained all images, so the std carried no information |
| F4 | Outputs looked anime-like but **not clearly Naruto**, and compositional prompts lost prompt alignment | **3,000 steps** with adapters saved at 1,000 / 2,000 / 3,000; **10% caption dropout** | More training for the under-learned style; dropout trains the unconditional branch that classifier-free guidance uses (as in SD's own training) |
| F5 | The loss was flat (≈ 0.06) for both runs, so it could not tell us which checkpoint is best | Checkpoint × strength chosen on validation prompts with a fixed rule (Section 1) | Picks the best style/prompt trade-off without touching the test data |
| F6 | The prompts were committed only after training, and no Git commit was printed in cell 1 | The notebook clones the repo in cell 1 and **commits and pushes the prompt files before training** | Lab rules C1 and C2.3 |
| — | Risk of reusing an outdated adapter from Hugging Face | A **recipe hash** of all training settings is stored with every run | A saved run is reused only if the settings match |

Everything else stayed the same: data, split, base model, step budget per run (identical for both runs), sampler, the 20
evaluation prompts and the final-run rule. The cost is about 1.5× the training time per run plus ~15 minutes for the
validation step. **Limitations:** 12 validation prompts × 1 seed is a small sample, the style score is only a proxy, and we did
not run a separate ablation for each change, so we cannot say how much each one contributes on its own.

## 3. Results
All numbers come from the executed notebook (Section 6.3) and are saved as small text files:

| File | Contents |
|---|---|
| `results/p2/full/p2_results_table.md` | Baseline (base SD 1.5) vs. final run on every measurement |
| `results/p2/full/p2_training_table.csv` | Per-run training table |
| `results/p2/full/p2_run_comparison.csv` | Run 1 vs. Run 2 on every measurement |
| `results/p2/full/p2_validation_grid.csv`, `p2_adapter_selection.json` | Validation scores of every checkpoint × strength and the chosen setting |
| `results/p2/full/p2_report_numbers.json` | Every number used in the report |
| `src/part2/abhinita/human_eval/` | Rating key and the raters' returned CSV files |
| `results/p2/split_naruto.json` | Indices of the train/test split |

## 4. How to run it
1. Open `notebooks/p2_finetune.ipynb` in Colab with a **T4 GPU** (it also runs on an A100 / RTX 4090, where it caps memory at
   14.5 GiB, but the lab names the T4/P100).
2. Add the secrets `HF_TOKEN` (Hugging Face, write access) and `GH_TOKEN` (GitHub, read/write to this repo) — key icon →
   *Notebook access* on.
3. Set `GIT_REPO` in the first cell to `abhinitasanabada-web/data266-lab2-team-26`.
4. *Runtime → Restart session and run all* and allow the Google Drive prompt. The prompt files are committed right after
   Section 2. A full run takes about 3.5 hours on a T4. (`SMOKE = True` runs a 20-step test of everything first.)
5. Send `rating_sheet.html` (Section 5.3) to the raters (we had 4: both of us + 2 outside), commit their CSV files to `src/part2/abhinita/human_eval/responses_full/`
   and run all again — finished runs are restored from Hugging Face, so this is quick.
6. Save the notebook with all outputs, commit it and tag the commit `p2-v1.0`.

## 5. Where things are stored
| What | Where |
|---|---|
| LoRA adapter + model card | [huggingface.co/sanabada/data266-team26-p2-naruto-run1-lora-r16](https://huggingface.co/sanabada/data266-team26-p2-naruto-run1-lora-r16) |
| DoRA adapter + model card | [huggingface.co/sanabada/data266-team26-p2-naruto-run2-dora-r16](https://huggingface.co/sanabada/data266-team26-p2-naruto-run2-dora-r16) |
| Generated images + `metadata.csv` | [huggingface.co/datasets/sanabada/data266-team26-p2-naruto-generations](https://huggingface.co/datasets/sanabada/data266-team26-p2-naruto-generations) |
| Prompts, results tables, ratings | `prompts/`, `results/p2/` in this repository |
| Data, latents, checkpoints, images | the notebook's work folder on Google Drive — never committed |

Model weights, images and secrets are never committed to GitHub (see `.gitignore`).

## 6. Requirement checklist
| Lab requirement | Where |
|---|---|
| 2.1 Corrupted/duplicate removal with counts, 90/10 split (seed 266), 512×512, description + 2 limitations | Notebook Section 1 |
| 2.2 Two runs with the same data and step budget; per-run table; both models on Hugging Face + model card | Sections 3.3–3.6, 6.2 |
| 2.3 20 prompts committed before training (10/5/5); 2 seeds × 3 models; grids for 8 prompts; images on a Hugging Face dataset | Sections 2, 4 |
| 2.4 KID + CLIP, LPIPS, 25 blinded pairs with 4 raters (≥ 3 required), GPU-hours + peak memory, ≥ 2 failure modes, which run and why | Sections 5.1–5.5, 6.1 |
| C4 License, NSFW filter, memorization check (10 images, 3 closest pairs) | Section 5.6 |
| C1 Run-info first cell, pinned requirements, how-to-run README, tag `p2-v1.0`, weights/images on Hugging Face | Cell 1, `requirements-p2.txt`, this file |
| C2 Every number from a notebook output | Section 6.3 (`p2_report_numbers.json`) |

## 7. Software and references
Python 3.13 with the pinned versions in [`requirements-p2.txt`](requirements-p2.txt) (Parts 1 and 3 have their own files):
`diffusers==0.42.1`, `peft==0.21.2`, `transformers==5.21.0`, `accelerate==1.15.0`, `torchmetrics==1.9.0`, `torch-fidelity==0.4.0`,
`lpips==0.1.4` (PyTorch 2.11 from the Colab image). The exact versions that ran are in `results/p2/full/p2_env.txt`.
**Hosted APIs:** none — SD 1.5, `openai/clip-vit-large-patch14`, Inception-v3 (torch-fidelity) and AlexNet-LPIPS are open weights.

- Rombach et al., 2022. *High-Resolution Image Synthesis with Latent Diffusion Models.* [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Liu et al., 2024. *DoRA: Weight-Decomposed Low-Rank Adaptation.* [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- Carlini et al., 2023. *Extracting Training Data from Diffusion Models.* [arXiv:2301.13188](https://arxiv.org/abs/2301.13188)

Who did what and our meeting recordings are in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md); the AI coding assistants we used are in
[`AI_USAGE.md`](AI_USAGE.md).
