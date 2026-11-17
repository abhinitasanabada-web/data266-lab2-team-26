# DATA 266 Lab 2 — Pair_Programming_Team_26

**Team:** Abhinita Sanabada · Saketh Prudhvi Marru
**Course:** DATA 266 (Generative AI), San José State University

This repository has our work for all three parts of Lab 2:

| Part | Topic | What we do |
|---|---|---|
| 1 | Multimodal RAG | We answer questions over a set of PDF documents, including their tables and figures, with grounded answers and citations |
| 2 | Fine-tuning a diffusion model | We fine-tune Stable Diffusion 1.5 on Naruto-style images and evaluate it against the base model |
| 3 | Multimodal multi-agent system | We build five agents that understand an image, write a prompt, generate a new image and check their own result |

Each part has its own README in its folder under `src/`, with the details of what we did and how to run it.

## Final version of each part

We each built our own version of every part and compared them. For every part, the final (submitted) version is Abhinita's:

| Part | Final notebook | Model | Run | Key results |
|---|---|---|---|---|
| 1 | [`notebooks/p1_rag.ipynb`](notebooks/p1_rag.ipynb) | Qwen3.5-9B (4-bit) reads charts and writes answers; bge-small-en-v1.5 + BM25 retrieval with bge-reranker-base | v7, Kaggle T4, commit `0cd3c08`, tag `p1-v1.0` | Public leaderboard 0.910 (submitted T4 run; 0.911 on the A100 development run); groundedness 3.74 / 5 |
| 2 | [`notebooks/p2_finetune.ipynb`](notebooks/p2_finetune.ipynb) | Stable Diffusion 1.5 + LoRA rank 16 (Run 1), checkpoint `step_1000`, strength 1.0 | commit `a4a4ae6`, tag `p2-v1.0` | In-domain KID 0.1117 → 0.0087; raters preferred its style in 84% of judgements |
| 3 | [`notebooks/p3_agents.ipynb`](notebooks/p3_agents.ipynb) | Qwen3.5-9B (same as Part 1), SD 1.5 img2img + our Part 2 LoRA, InstructPix2Pix, CLIP ViT-L/14 | v5, Colab T4, commit `4dc1418`, tag `p3-v1.0` | 11/15 scenarios passed, 0/10 attacks succeeded (both human-rated) |

Saketh's independent versions are in `src/partN/saketh/`; the report compares them with the final versions.

## Where models, checkpoints and outputs are stored

Model weights and data are never committed to GitHub.

| Part | What | Where |
|---|---|---|
| 1 | No trained model (every model is downloaded from Hugging Face). Final graded run: notebook, logs, `submission.csv`, `human_eval.csv`, figure-reading cache `kb_cache/` | Kaggle notebook [p1-rag-v7](https://www.kaggle.com/code/abhinitasanabada/p1-rag-v7) → Output tab |
| 1 | Submission, golden set, groundedness ratings | `src/part1/abhinita/output/`, `src/part1/abhinita/golden_set.csv`, `src/part1/abhinita/human_eval/` |
| 2 | **Final adapter** (Run 1, LoRA r16): checkpoints `step_1000/` (selected), `step_2000/`, `step_3000/`, model card, resumable state `checkpoint/ckpt.pt` | [sanabada/data266-team26-p2-naruto-run1-lora-r16](https://huggingface.co/sanabada/data266-team26-p2-naruto-run1-lora-r16) |
| 2 | Run 2 (DoRA r16): checkpoints `step_1000/`, `step_2000/`, `step_3000/`, model card | [sanabada/data266-team26-p2-naruto-run2-dora-r16](https://huggingface.co/sanabada/data266-team26-p2-naruto-run2-dora-r16) |
| 2 | 120 generated images + `metadata.csv` | [sanabada/data266-team26-p2-naruto-generations](https://huggingface.co/datasets/sanabada/data266-team26-p2-naruto-generations) |
| 2 | Prompts, results tables, human ratings | `prompts/`, `src/part2/abhinita/results/output/`, `src/part2/abhinita/human_eval/` |
| 3 | No new weights; uses the Part 2 adapter `run1-lora-r16`, subfolder `step_1000`, strength 1.0 | (Hugging Face link above) |
| 3 | Final run outputs: generated images, per-request traces, blinded rating sheet | [Google Drive folder](https://drive.google.com/drive/folders/1fDg98VaLxSVqzOWTRfaUpnIwKjMPv7bA) (anyone with the link can view) |
| 3 | Results tables, rating key, our ratings and adjudication | `results/p3/full/`, `results/p3/human_eval_qwen35/`, `src/part3/abhinita/human_eval/` |

Load the final Part 2 adapter with:

```python
pipe.load_lora_weights("sanabada/data266-team26-p2-naruto-run1-lora-r16", subfolder="step_1000")
```

## Folder structure

```
data266-lab2-team-26/
├── README.md          ← this overview
├── links.txt          ← repository, tag commits, Kaggle notebook, Hugging Face and Drive links
├── notebooks/         ← our final notebook for each part (the best version, with all outputs)
│   ├── p1_rag.ipynb
│   ├── p2_finetune.ipynb
│   └── p3_agents.ipynb
├── prompts/           ← Part 2 test and validation prompts
├── results/p3/        ← Part 3 result tables and rating key
└── src/
    ├── part1/
    │   ├── abhinita/  ← Abhinita's independent work and notebook versions
    │   └── saketh/    ← Saketh's independent work and notebook versions
    ├── part2/
    │   ├── abhinita/
    │   └── saketh/
    └── part3/
        ├── abhinita/
        └── saketh/
```

- **`src/`:** each of us works independently in our own folder for every part and keeps all our notebook versions there.
- **`notebooks/`:** when we agree on the best version for a part, we put it here. These are the notebooks we submit.
