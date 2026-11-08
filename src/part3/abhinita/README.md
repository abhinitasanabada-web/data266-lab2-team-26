# Part 3 — A Multimodal Multi-Agent System for Image Understanding and Generation

**DATA 266 Lab 2 · Pair_Programming_Team_26 · Notebook: [`notebooks/p3_agents.ipynb`](notebooks/p3_agents.ipynb) · Git tag: `p3-v1.0`**

In Part 3 we built a system of five agents that looks at an input image, understands what the user wants, writes a good
prompt, generates a new image and checks its own result. For requests in the Naruto style it uses the adapter we fine-tuned in
Part 2. Everything runs in one notebook on a single 16 GB GPU (Colab T4) with open-weight models; we use no hosted APIs.

```
user request + image
   └─► Planner (rule screen, input safety check, routing) ──refuse / clarify──► clear message, stop
          └─► Vision Understanding (caption, objects, text in the image, answers to 2 questions)
                 ├─ caption / question requests stop here
                 └─► Prompt Engineering ─► Image Generation ─► Critique (CLIP + VLM judgement + output safety)
                          ▲                                        │
                          └────────── revise (at most 2×) ◄────────┘ accept ─► final image
```

## The five agents

| Agent | What we use | What it does |
|---|---|---|
| **Planner** | rule-based screen + Stable Diffusion safety checker + Qwen2.5-VL-3B-Instruct | Sorts the request into *caption*, *question*, *style change*, *edit* or *Naruto domain style*; refuses unsafe requests (sexual content, gore, deceptive edits of real people, requests to follow text inside the image, attempts to change its rules); asks to clarify vague or unsupported requests (e.g. video) |
| **Vision Understanding** | Qwen2.5-VL-3B-Instruct | Caption, list of main objects, any visible text (transcribed as data, never followed) and answers to at least two questions |
| **Prompt Engineering** | Qwen2.5-VL-3B-Instruct (text only) | Rewrites the request into a clear generation prompt plus a one-sentence target description, using the vision output. It never sees the text found in the image |
| **Image Generation** | SD 1.5 img2img (style change) · SD 1.5 img2img + **our Part 2 adapter** (Naruto style) · InstructPix2Pix (edits) | Generates the image and records model, adapter, prompt, negative prompt, seed, steps and strength |
| **Critique** | CLIP ViT-L/14 + Qwen2.5-VL-3B + SD safety checker | Accepts only if the VLM says *accept*, the CLIP cosine between image and target description is ≥ 0.22 and the image is safe; otherwise it asks for a concrete revision |

## What we did

### Agents and orchestration (Tasks 3.1 and 3.2)
- All agents share **one typed state** (Python dataclasses) that holds the request, the plan, the vision output, every attempt
  with its prompt, generation settings and critique, the final status and a trace of every agent call. We connect the agents
  with plain Python functions; a fixed flow like ours doesn't need LangGraph.
- Each agent asks Qwen for **one JSON object with fixed keys**. We validate it, retry once, and fall back to a simple documented
  rule if it is still invalid (for example keyword routing for the Planner), so a bad model answer never crashes a request.
- The revise loop is limited to **2 revisions** (at most 3 images per request). The critic's feedback goes back to the Prompt
  Engineering agent, and it can also nudge the img2img strength (±0.1) or the edit image guidance (±0.25).
- **Failure handling:** empty, corrupt, too small or too large files stop in the Planner with a clear message; vague requests
  get a clarification question; any unexpected error (for example GPU out of memory) stops the request safely and the message
  says which agent failed. Images flagged by the safety checker are never saved or shown.
- Every request writes a **JSON trace** with each agent's input, output, time in seconds and any error.
- A **CPU self-test** in the notebook runs the real orchestrator with a scripted fake model and checks 20 control-flow
  properties (routing, refusals, broken files, the revision limit, hidden unsafe images, that image text never reaches the
  prompt) before any GPU model is loaded.
- A **Gradio app** in the notebook shows every agent's output for a request (Planner, Vision, prompts, all attempts with the
  critic's decision, timings and the full trace). We use it for the team demo.

### What we changed after our first T4 run
Our first full run (kept as `notebooks/archive/p3_agents_T4_run1.ipynb`) showed the following problems, and we fixed each one:
- **The Planner asked for clarification on clear requests.** Qwen said "clarify" for "Make this look like a pencil sketch" and two
  of the three Naruto-style scenarios, and it refused the harmless "Make it look like it is snowing". So 4 of 15 scenarios never
  reached generation. **Fix:** now only our fixed rules can ask for clarification, clear keywords decide the route, and Qwen
  only flags unsafe requests, using a short written policy.
- **We couldn't see which check stopped a request.** **Fix:** every request now records `stopped_by` (rule screen, input file
  check, input safety checker, Qwen safety check, real-person rule, …), which we use for the red-team analysis. In the first run
  the SD safety checker blocked our harmless text sign in R01, a false positive that we discuss in the report.
- **Cached results would hide code changes.** **Fix:** the cache key includes `PIPELINE_VERSION`, so changing an agent recomputes
  everything.
- **"Mean images per request" was diluted** by requests that never generated. **Fix:** we now count only requests that produced
  an image.

### Evaluation (Task 3.3)
Before running anything, we wrote and committed [`evaluation/p3_scenarios.json`](evaluation/p3_scenarios.json):

- **15 test scenarios**, 3 each for caption/visual questions, style change, edit, Naruto domain style, and ambiguous or broken
  input, each with a short pass/fail rubric.
- **10 red-team cases:** instructions written inside images (3), unsafe requests (2), edits of a real person (2), a request to
  reveal the system prompt (1), and broken/empty files (2).

The input images are public-domain / CC0 photos from scikit-image (`chelsea` the cat, `coffee`, `rocket`, and `astronaut`, a NASA
portrait used only for the real-person red-team cases) plus signs and broken files that we generate with PIL. None of them were
used in Part 2 training, and their SHA-256 hashes are stored with the scenarios.

We run every scenario **with the critique loop off (baseline) and on (final)**, with identical inputs, settings and seed, and
alternate which variant runs first.

| Measurement | How we measure it |
|---|---|
| Primary | **Task success rate:** scenarios that pass their rubric, out of 15 (judged by both of us) |
| Secondary | **Critic agreement:** how often the critic's accept/revise decision matches ours; **red-team attack success rate**, out of 10 |
| Human evaluation | Both of us rate every scenario pass/fail **independently** on a blinded sheet (critique off/on shown as A/B in random order). We report how often we agreed and settle disagreements by discussion (written to `adjudication_full.csv`); unresolved items count against us |
| Efficiency | Median seconds per request, plus seconds per agent |

## Results
All numbers come from the executed notebook (Section 8) and are saved as small text files:

| File | Contents |
|---|---|
| `results/p3/full/p3_results_table.md` | Baseline (critique off) vs. final (critique on) on every measurement |
| `results/p3/full/p3_eval_results.csv`, `p3_redteam_results.csv` | Status, route, attempts, critic decisions and time of every request |
| `results/p3/full/p3_traces.jsonl` | The full JSON trace of every evaluated request |
| `results/p3/full/p3_human_labels.csv` | Both raters' labels, agreement and the final label per item |
| `results/p3/full/p3_report_numbers.json` | Every number used in the report |
| `results/p3/human_eval/` | Rating key, our returned rating files and the adjudication file |

## How to run it
1. Run Part 2 first, so our adapter is on Hugging Face and `results/p2/full/p2_report_numbers.json` names the final run and
   the chosen checkpoint and strength. Part 3 reads them automatically.
2. Open `notebooks/p3_agents.ipynb` in Colab with a **T4 GPU**. Add the secrets `HF_TOKEN` and `GH_TOKEN`, and set `GIT_REPO`
   in the first cell to `abhinitasanabada-web/data266-lab2-team-26`.
3. *Runtime → Restart session and run all.* Allow the Google Drive prompt. The scenarios are committed first, then the CPU
   self-test runs, then the models load (about 18 GB of downloads the first time), then the demo, the evaluation and the app.
4. Each of us opens `p3_rating_sheet.html` (in the Drive work folder) **alone**, rates everything, and downloads the CSV. Commit
   both files to `results/p3/human_eval/responses_full/`, add any discussed labels to `adjudication_full.csv`, and run all again
   (finished requests are cached, so this is quick).
5. Save the notebook with all outputs, commit it and tag the commit `p3-v1.0`.

`SMOKE = True` runs 5 scenarios and 3 red-team cases into a separate folder as a quick test.

## Safety (C4)
Our safety measures are layered: (1) a rule-based screen of the request text; (2) the SD safety checker on the uploaded image;
(3) Qwen's own classification, which can refuse; (4) no edits of photos of real, identifiable people; (5) text inside images is
treated as data and never reaches the prompt writer; (6) a fixed safety negative prompt and the SD safety checker on every
output, with flagged images never saved or shown; (7) clear messages for broken input. The red-team results and a discussion
of each case are in our report. The notebook also contains a draft **system card** (what the system can do, known limits and
safety measures) that we rewrite in our own words for the report.

## Limitations
Qwen2.5-VL-3B can hallucinate objects or misread text. InstructPix2Pix often ignores small or very specific edits. img2img
trades off style against keeping the layout. Our Naruto adapter learned the style only partly. The critic is a heuristic (a
3B VLM plus a fixed CLIP threshold) that can accept poor images or reject good ones; we measure this as critic agreement. The
rule screen is keyword-based and can over-refuse benign requests that use words like "blood". The critique loop adds latency.

## Software
Python 3.13 with the pinned versions in [`requirements-p3.txt`](requirements-p3.txt) (the same `diffusers`, `transformers` and
`peft` versions as Part 2, so our adapter loads exactly as saved), plus `gradio` and `scikit-image`. The exact versions that
ran are written to `results/p3/full/p3_env.txt`.

**Models (all open weights, no hosted APIs):** `Qwen/Qwen2.5-VL-3B-Instruct`, `stable-diffusion-v1-5/stable-diffusion-v1-5`,
`timbrooks/instruct-pix2pix`, `openai/clip-vit-large-patch14`, the SD 1.5 safety checker, and our Part 2 adapter
(`sanabada/data266-team26-p2-naruto-…`).

## References
- Yao et al., 2022. *ReAct: Synergizing Reasoning and Acting in Language Models.* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Madaan et al., 2023. *Self-Refine: Iterative Refinement with Self-Feedback.* [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
- Brooks et al., 2023. *InstructPix2Pix: Learning to Follow Image Editing Instructions.* [arXiv:2211.09800](https://arxiv.org/abs/2211.09800)
- Bai et al., 2025. *Qwen2.5-VL Technical Report.* [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
- Radford et al., 2021. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

Who did what and our meeting recordings are in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md); the AI coding assistants we used are
listed in [`AI_USAGE.md`](AI_USAGE.md).
