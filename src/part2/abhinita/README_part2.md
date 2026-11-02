# Part 2 — Fine-Tuning Stable Diffusion for Domain-Specific Image Generation

**DATA 266 Lab 2 · Pair_Programming_Team_26 · Notebook: [`notebooks/p2_finetune.ipynb`](notebooks/p2_finetune.ipynb) · Git tag: `p2-v1.0`**

In Part 2 we fine-tuned Stable Diffusion 1.5 to generate images in the Naruto anime style. We trained two
parameter-efficient adapters (LoRA and DoRA) with the same data, seed and step budget, compared them with the base
model, and evaluated image quality, prompt alignment, diversity, cost and safety. Everything runs in one notebook on a
single 16 GB GPU, and no hosted APIs are used.

| | |
|---|---|
| Base model | [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), 512×512 |
| Dataset | [`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) (1,221 captioned images) |
| Run 1 | LoRA, rank 16, on the UNet attention layers |
| Run 2 | DoRA, rank 16, same layers, same data, seed and steps |
| Training | 3,000 steps × batch 4, 10% caption dropout, adapters saved at 1,000 / 2,000 / 3,000 steps |
| Checkpoint choice | Best checkpoint × adapter strength per run, chosen on separate validation prompts (never on test data) |
| Hardware | One NVIDIA T4 (16 GB) on Google Colab |
| Seed | 266 everywhere (split, prompts, training, sampling, human-evaluation pairs) |

## What we did

### 1. Data preparation (Task 2.1)
- Loaded the dataset with images as raw bytes, so one broken file could not crash the run.
- Removed **corrupted** images (undecodable, truncated, smaller than 64 px, blank) and empty captions.
- Removed **duplicates**: exact duplicates by MD5 hash of the decoded pixels, and near-duplicates by a 64-bit perceptual hash
  (Hamming distance ≤ 4). We did this *before* splitting, so the same image cannot be in both train and test.
- Split 90/10 with seed 266 (1,099 train / 122 test). The test images are used only for evaluation.
- Center-cropped and resized every image to 512×512.
- Printed dataset statistics (caption length, vocabulary, original sizes, gender terms) for the description and the two
  limitations in the report: generic, sometimes wrong BLIP captions, and a small dataset from one franchise that is skewed
  toward male characters.

### 2. Fixed evaluation prompts (Task 2.3)
Before training we wrote 20 prompts and committed them to Git ([`prompts/p2_prompts_naruto.json`](prompts/)); the commit
timestamp shows they were fixed in advance:
- 10 **in-domain** prompts: test-set captions chosen with seed 266, copied exactly as BLIP wrote them
- 5 **out-of-domain** prompts (e.g. "a lighthouse on a rocky coast during a storm")
- 5 **compositional** prompts (e.g. "two characters, one in red and one in blue")

In the same step we also committed 12 separate **validation prompts** ([`prompts/p2_val_prompts_naruto.json`](prompts/)):
8 training captions and 4 prompts we wrote ourselves. We use them only to choose each run's checkpoint (Section 3 below).

### 3. Fine-tuning (Task 2.2)
- Pre-computed the VAE latents (each image and its horizontal flip) and the CLIP text embeddings once, so only the UNet is on
  the GPU during training and both runs see exactly the same inputs.
- Trained with PEFT, following the recipe of the diffusers `train_text_to_image_lora.py` example: noise-prediction MSE with
  Min-SNR weighting (γ = 5), AdamW (lr 1e-4), cosine schedule with 100 warm-up steps, batch size 4, 3,000 steps, fp16 with the
  adapter weights in fp32, and gradient checkpointing.
- Added **caption dropout**: 10% of the captions are replaced by the empty prompt during training, so the adapter also learns
  the style for the unconditional branch that classifier-free guidance uses.
- Saved the adapter after 1,000, 2,000 and 3,000 steps, then **chose the checkpoint and adapter strength (0.75 or 1.0) of each
  run on the validation prompts** with a rule fixed in advance: skip settings where the NSFW filter blocks more than 25% of the
  images, keep settings that lose at most 1.5 CLIP points against the base model, and take the one closest to the Naruto style
  (CLIP similarity to the average training image). The scatter plot `figures/validation_selection.png` shows every setting.
  [`README_p2_improvements.md`](README_p2_improvements.md) explains why we made these changes after our first run.
- Made the training loop **resumable**: batch order, flips, noise and timesteps all come from the seed and the step number, and
  a checkpoint is saved every 250 steps (also backed up to Hugging Face), so a Colab disconnect does not lose progress.
- The notebook prints one table per run with the trainable parameters, learning rate, steps, batch size, peak GPU memory and
  training time, and checks that peak memory stays below 16 GB.
- Pushed both adapters to Hugging Face with a model card (base model, data, training settings, results, limitations, license).

### 4. Image generation (Task 2.3)
- Generated 2 images per prompt (seeds 0 and 1) for the base model and both runs with identical sampler settings
  (DPM-Solver++, 30 steps, guidance scale 7.5, 512×512, the same negative prompt "nsfw, nudity, naked"), i.e. 40 images
  per model. Each run uses its selected checkpoint and strength. The starting noise comes from a CPU
  generator, so the same seed gives the same image on any GPU.
- Made side-by-side grids (base vs. Run 1 vs. Run 2) for 8 prompts: 4 in-domain, 2 out-of-domain, 2 compositional.
- Uploaded every image to a Hugging Face dataset whose `metadata.csv` lists the prompt, seed and model of each image.

### 5. Evaluation (Task 2.4)
| Measurement | What we computed |
|---|---|
| Primary | **KID** between the in-domain generations and the held-out test images (torchmetrics, Inception-v3 features, 100 subsets), and **CLIP score** between each prompt and its image (`openai/clip-vit-large-patch14`) |
| Secondary | **Diversity**: average LPIPS (AlexNet) distance between the seed-0 and seed-1 images of each prompt |
| Human evaluation | A blinded HTML rating sheet with 25 pairs (final run vs. base, left/right randomized). Three raters — both of us and one person outside the team — chose the better image for domain style and for prompt match. We report the win rate, per-rater results and agreement (Fleiss' kappa). |
| Efficiency | GPU-hours per training run, together with peak GPU memory |

We chose the final run with a rule fixed before looking at the images: lower in-domain KID wins unless the gap is within
the KID standard deviation; then the higher mean CLIP score wins. The notebook also shows automatically selected **failure
modes** (an out-of-domain prompt taken over by the anime style, low diversity between seeds, a compositional prompt that
fails, and over-blocking by the NSFW filter).

### 6. Safety (C4)
- **License.** The images show copyrighted Naruto characters and the dataset states no license, so our adapters and generated
  images are for **non-commercial academic research only**. SD 1.5 is under the CreativeML OpenRAIL-M license.
- **NSFW filter.** The Stable Diffusion safety checker stays on for every generated image. It blocks noticeably more outputs
  of the fine-tuned models than of the base model; we report these counts and leave the blocked (black) images out of the
  quality metrics, because a black image says nothing about style or prompt match.
- **Memorization check.** For 10 in-domain images of the final run we found the most similar training image by CLIP
  similarity and show the 3 closest pairs. As a reference we also measured how similar each *test* image is to its nearest
  training image, since test images were never trained on.

## Results
All numbers come from the executed notebook (Section 6.3) and are also saved as small text files:

| File | Contents |
|---|---|
| [`results/p2/full/p2_results_table.md`](results/p2/full/) | Results table for the baseline (base SD 1.5) and the final run |
| `results/p2/full/p2_training_table.csv` | Per-run training table |
| `results/p2/full/p2_run_comparison.csv` | Run 1 vs. Run 2 on every measurement |
| `results/p2/full/p2_validation_grid.csv`, `p2_adapter_selection.json` | Validation scores of every checkpoint × strength and the chosen setting per run |
| `results/p2/full/p2_report_numbers.json` | Every number used in the report |
| `results/p2/human_eval/` | Rating key and the raters' returned CSV files |
| `results/p2/split_naruto.json` | Indices of the train/test split |

## How to run it
1. Open `notebooks/p2_finetune.ipynb` in Colab (or Kaggle) and choose a **T4 GPU** runtime. The notebook also runs on larger
   GPUs (A100, RTX 4090); there it caps usable memory at 14.5 GiB so training still fits a 16 GB GPU.
2. Add two secrets (Colab: key icon → *Notebook access* on):
   - `HF_TOKEN` — a Hugging Face token with write access
   - `GH_TOKEN` — a GitHub token with read/write access to this repository
3. Set `GIT_REPO` in the first cell to `<github-owner>/data266-lab2-team26`.
4. *Runtime → Restart session and run all* and allow the Google Drive prompt (the work folder lives on Drive so it survives
   disconnects). The prompt files are committed automatically right after Section 2. A full run takes about 3.5 hours on a T4.
5. Send `rating_sheet.html` (Section 5.3) to the three raters, commit their CSV files to `results/p2/human_eval/responses_full/`
   and run all cells again. Finished training runs are restored from Hugging Face instead of being trained again (only if
   they were trained with the current settings; the notebook checks a hash of the training recipe).
6. Save the notebook with all outputs, commit it and tag the commit `p2-v1.0`.

`SMOKE = True` in the configuration cell runs a 20-step test of the whole pipeline in a separate folder.

## Where things are stored
| What | Where |
|---|---|
| LoRA adapter + model card | [huggingface.co/sanabada/data266-team26-p2-naruto-run1-lora-r16](https://huggingface.co/sanabada/data266-team26-p2-naruto-run1-lora-r16) |
| DoRA adapter + model card | [huggingface.co/sanabada/data266-team26-p2-naruto-run2-dora-r16](https://huggingface.co/sanabada/data266-team26-p2-naruto-run2-dora-r16) |
| Generated images + `metadata.csv` | [huggingface.co/datasets/sanabada/data266-team26-p2-naruto-generations](https://huggingface.co/datasets/sanabada/data266-team26-p2-naruto-generations) |
| Prompts, results tables, ratings | `prompts/`, `results/p2/` in this repository |
| Data, latents, checkpoints, images | the notebook's work folder (Google Drive) — never committed |

Model weights, images and secrets are never committed to GitHub (see `.gitignore`).

## Software
Python 3.13 with the pinned versions in [`requirements-p2.txt`](requirements-p2.txt) (Parts 1 and 3 use `requirements.txt`): `diffusers==0.42.1`, `peft==0.21.2`,
`transformers==5.21.0`, `accelerate==1.15.0`, `torchmetrics==1.9.0`, `torch-fidelity==0.4.0`, `lpips==0.1.4`
(PyTorch 2.11 from the Colab image). The notebook writes the exact versions that ran to `results/p2/full/p2_env.txt`.

**Hosted APIs:** none. All models are open weights from Hugging Face: SD 1.5, `openai/clip-vit-large-patch14` (CLIP score and
memorization check), Inception-v3 via torch-fidelity (KID) and AlexNet-LPIPS.

## References
- Rombach et al., 2022. *High-Resolution Image Synthesis with Latent Diffusion Models.* [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Hu et al., 2021. *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Liu et al., 2024. *DoRA: Weight-Decomposed Low-Rank Adaptation.* [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- Carlini et al., 2023. *Extracting Training Data from Diffusion Models.* [arXiv:2301.13188](https://arxiv.org/abs/2301.13188)

Who did what, and our meeting recordings, are in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md); the AI coding assistants we used are
listed in [`AI_USAGE.md`](AI_USAGE.md).
