# AI assistance disclosure — draft for team review

On 2026-10-09, OpenAI Codex assisted with the initial project implementation: notebook explanations, PDF ingestion, retrieval and validation logic, diffusion experiment wrappers, image-agent orchestration, evaluation utilities, draft test questions/scenarios and automated control-flow tests.

The Part 1 golden set was drafted from the supplied source PDFs without reading Questions.csv. It is labelled AI-assisted and requires both students' independent review and revision before commitment or use. Test doubles used by unit tests do not represent model or human-evaluation results.

No trained Lab 2 weights, Kaggle scores, completed human judgements, meeting recordings, or authorship contributions were invented. Model quality, GPU performance and final report claims require actual student runs and review.

Team members must update this file with subsequent tools, dates, purposes, reviewed code and any corrections. This is a disclosure template, not a claim that review has been completed.

## Part 2 — Claude (Anthropic), 2026-10-29

We used Claude (Anthropic, in the Claude app) while finishing `notebooks/p2_finetune.ipynb`:

- **Requirements review:** compared the notebook and its outputs with the Lab 2 Part 2 requirements and listed the gaps.
- **Text cleanup:** rewrote the notebook's markdown and code comments for clarity and grammar, and drafted `README_part2.md`.
- **Code changes we reviewed and kept:**
  - moved the `torchao` removal into the install cell;
  - added the Run 1 vs. Run 2 comparison table;
  - safe Git handling: the notebook works without `git`, and the prompt file and small results files are committed and pushed automatically;
  - Google Drive mount on Colab, and restoring finished runs from Hugging Face instead of retraining;
  - GPU checks so the notebook runs on a T4, A100 or RTX 4090 within a 16 GB memory cap;
  - images blocked by the NSFW filter are excluded from CLIP/KID/LPIPS, the human-evaluation pairs and the memorization check, and the blocked counts are reported;
  - fixed the KID subset size (the extended KID std was 0);
  - display fixes ("LoRA"/"DoRA" names, rounding of CLIP scores).
- **2026-11-01, fine-tuning improvements** (proposed by Claude after looking at our first T4 run; we reviewed and kept them):
  caption dropout (10%), 3,000 training steps with adapters saved at 1,000/2,000/3,000, choosing each run's checkpoint and
  adapter strength on separate validation prompts with a fixed rule, a shared safety negative prompt, and a training-recipe
  hash so outdated runs are not reused. Claude also drafted `README_p2_improvements.md`.
- **2026-11-01, Part 3** (Claude): built the self-contained `notebooks/p3_agents.ipynb` — the five agents, the shared typed
  state and orchestrator with JSON traces and failure handling, the CPU self-test, the Gradio app, the draft of the 15 scenarios
  and 10 red-team cases with rubrics, the blinded rating sheet and the metric calculations — and drafted `README_part3.md`.
  It reused ideas from the earlier Codex scaffold (strict JSON contracts, request screening, safe image loading).
- **Not done by AI:** all training runs, image generation and metric numbers come from our own executed notebook runs on Colab. The human ratings come from our own raters (three in Part 2, both of us in Part 3). The report analysis is written by us.

*TODO before submission: confirm here that both team members reviewed these changes and re-ran the notebook, with the date.*
