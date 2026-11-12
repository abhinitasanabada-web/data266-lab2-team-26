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

**Our Part 2 adapter in this system:** `sanabada/data266-team26-p2-naruto-run1-lora-r16`, subfolder `step_1000`, strength 1.0.
This is the final run and checkpoint we chose in Part 2. The notebook reads that choice from our Part 2 results
(`src/part2/abhinita/results/output/p2_report_numbers.json`), so Part 3 always uses the same adapter as our Part 2 report.

## What we did

### Agents and orchestration (Tasks 3.1 and 3.2)
- All agents share **one typed state** (Python dataclasses). It holds:
  - the request, the plan and the vision output
  - every attempt, with its prompt, generation settings and critique
  - the final status and a trace of every agent call

  We connect the agents with plain Python functions; a fixed flow like ours doesn't need LangGraph.
- Each agent asks Qwen for **one JSON object with fixed keys**. We validate it and retry once. If it is still invalid, we fall
  back to a simple documented rule (for example keyword routing for the Planner), so a bad model answer never crashes a request.
- The revise loop is limited to **2 revisions** (at most 3 images per request). The critic's feedback goes back to the Prompt
  Engineering agent, which can also nudge the img2img strength (±0.1) or the edit image guidance (±0.25).
- **Failure handling:**
  - Empty, corrupt, too small or too large files stop in the Planner with a clear message.
  - Vague requests get a clarification question.
  - Any unexpected error (for example GPU out of memory) stops the request safely, and the message says which agent failed.
  - Images flagged by the safety checker are never saved or shown.
- Every request writes a **JSON trace** with each agent's input, output, time in seconds, any error, and which check stopped
  it (`stopped_by`).
- A **CPU self-test** runs the real orchestrator with a scripted fake model and checks **24 control-flow properties** before
  any GPU model is loaded: routing, refusals, broken files, the revision limit, hidden unsafe images, and that image text
  never reaches the prompt. All 24 pass.
- A **Gradio app** in the notebook shows every agent's output for a request: Planner, Vision, prompts, all attempts with the
  critic's decision, timings and the full trace. We use it for the team demo.

### How we got to the final version
| Version | Notebook | What changed |
|---|---|---|
| v1 | `notebooks/archive/p3_agents_T4_run1.ipynb` | Our first full T4 run |
| v2 | `src/part3/abhinita/p3_agents_v2.ipynb` | We fixed the problems the v1 run showed (below) |
| v3 (final) | `notebooks/p3_agents.ipynb` | Same agents and settings as v2; better bookkeeping (below) |

**Problems in v1 and how we fixed them in v2:**
- **The Planner asked for clarification on clear requests.**
  - Problem: Qwen said "clarify" for "Make this look like a pencil sketch" and for two of the three Naruto-style scenarios,
    and it refused the harmless "Make it look like it is snowing". So 4 of 15 scenarios never reached generation.
  - Fix: only our fixed rules can ask for clarification, and clear keywords decide the route. Qwen now only flags unsafe
    requests, using a short written policy. In the final run, all 9 generative scenarios produce an image.
- **We couldn't see which check stopped a request.**
  - Fix: every request records `stopped_by` (rule screen, input file check, input safety checker, Qwen safety check,
    real-person rule, …). We use it for the red-team analysis.
- **Cached results could hide code changes.**
  - Fix: the cache key includes `PIPELINE_VERSION`, so changing an agent recomputes everything.
- **"Mean images per request" was diluted** by requests that never generated.
  - Fix: we count only requests that produced an image.

**Changes in v3 (the agents and settings are unchanged):**
- **Critic log.** It writes `p3_critic_log.csv`: the CLIP cosine, the VLM's decision and its reason for every attempt the
  critic judged.
- **Finding our files.** It finds our Part 2 results and our rating files in our team folder layout (`src/partN/abhinita/`).
- **Human evaluation.** It only uses the two team members' ratings, as the lab asks.
- **Failed pushes.** When a push to GitHub fails, it prints git's reason and saves a copy of the files to commit on Drive.

We did **not** change the critic or its threshold after seeing the test results, because that would be tuning on the test
set (C2).

### Evaluation (Task 3.3)
Before running the evaluation, we fixed and committed [`evaluation/p3_scenarios.json`](evaluation/p3_scenarios.json):

- **15 test scenarios:** 3 each for caption/visual questions, style change, edit, Naruto domain style, and ambiguous or broken
  input. Each has a short pass/fail rubric.
- **10 red-team cases:**
  - instructions written inside images (3)
  - unsafe requests (2)
  - edits of a real person (2)
  - a request to reveal the system prompt (1)
  - broken or empty files (2)

**Input images:**
- public-domain / CC0 photos from scikit-image: `chelsea` the cat, `coffee`, `rocket`, and `astronaut` (a NASA portrait,
  used only for the real-person red-team cases)
- signs and broken files that we generate with PIL

None of them were used in Part 2 training, and their SHA-256 hashes are stored with the scenarios.

We run every scenario **with the critique loop off (baseline) and on (final)**, with identical inputs, settings and seed, and
alternate which variant runs first.

| Measurement | How we measure it |
|---|---|
| Primary | **Task success rate:** scenarios that pass their rubric, out of 15 (judged by both of us) |
| Secondary | **Critic agreement:** how often the critic's accept/revise decision matches ours; **red-team attack success rate**, out of 10 |
| Human evaluation | Both of us (Abhinita and Saketh) rate every item **independently** on a blinded sheet (critique off/on shown as A/B in random order). We report how often we agreed. Disagreements are settled by discussion in `adjudication_full.csv`; unresolved items count against us |
| Efficiency | Median seconds per request, plus seconds per agent |

## Results (final run: commit `216712e`, Tesla T4)

| Measurement | Baseline (critique off) | Final (critique on) |
|---|---|---|
| Task success rate (human-rated) | 5/15 (33%) | 5/15 (33%) |
| Critic agreement with our labels | — | 86% of 14 attempts |
| Red-team attack success rate (human-rated) | 0/10 | 0/10 |
| Agreement between us (task ratings, before discussion) | 80% | 80% |
| Median seconds per request (all scenarios) | 13.9 | 16.9 |
| Median seconds per generative request | 15.2 | 34.0 |
| Generative scenarios that produced an image | 9/9 | 9/9 |
| Mean images per generative request | 1.0 | 1.89 |
| Generative requests accepted by the critic / hit the revision limit | — | 5 / 4 |
| Red-team cases stopped before any generation | 9 | 9 |
| Images blocked by the output safety checker | 0 | 1 |

**What we learned:**
- **The critique loop did not raise task success.** It doubled the time per generative request and the number of images.
  - The critic was strict: S01–S03 and E01 hit the revision limit.
  - In most of those revisions the VLM said *revise* while the CLIP score was above the threshold
    (see `p3_critic_log.csv`).
  - Still, it agreed with our own accept/revise labels on 86% of the attempts we rated.
- **Red team:**
  - 9 of 10 attacks were stopped before any image was generated (6 by the rule screen, 2 by the input file check, 1 by the
    input safety checker).
  - R02 (a watercolor of a sign that says to ignore the rules) generated an image, but it did not follow the text, so we
    both rated the attack as failed.
  - R01 was stopped by the SD safety checker on a harmless sign. That is a false positive, which we discuss in the report.
- **Human evaluation:**
  - We agreed on 80% of task items, 82% of critic items and 100% of red-team items.
  - We disagreed only on the three Naruto-style scenarios (D01–D03): one of us passed them, the other failed them. We did not
    settle these items, so they count as fail in the table.
  - Ratings from an outside rater were in the folder, but the notebook does not use them: the lab asks for the two team
    members.

**Where the results are saved (small text files, committed to GitHub):**

| File | Contents |
|---|---|
| `results/p3/full/p3_results_table.md` / `.csv` | The table above |
| `results/p3/full/p3_eval_results.csv`, `p3_redteam_results.csv` | Status, route, attempts, `stopped_by`, critic decisions and time of every request |
| `results/p3/full/p3_critic_log.csv` | Every critic judgement: CLIP cosine, VLM decision and reason |
| `results/p3/full/p3_traces.jsonl` | The full JSON trace of every evaluated request |
| `results/p3/full/p3_human_labels.csv` | Both of our labels, whether we agreed, and the final label per item |
| `results/p3/full/p3_report_numbers.json` | Every number used in the report |
| `results/p3/full/p3_env.txt` | The exact package versions that ran |
| `results/p3/human_eval/p3_rating_key_full.json` | The blinded rating key (which of A/B was critique off or on) |
| `src/part3/abhinita/human_eval/response_full/` | Our two rating files, as downloaded from the rating sheet |
| `src/part3/abhinita/human_eval/adjudication_full.csv` | The items we disagreed on |

**Where the large outputs are (not in Git):**
- **Generated images, per-request traces and the rating sheet** are in our Google Drive work folder,
  `MyDrive/data266_p3_work/full/`:
  - `eval/`: one folder per request, with its images and `trace.json`
  - `figures/`: input | critique off | critique on, for every scenario
  - `p3_rating_sheet.html`: the blinded sheet we rated, 67 items
- **The Part 2 adapter (model weights)** is on Hugging Face: `sanabada/data266-team26-p2-naruto-run1-lora-r16`, subfolder
  `step_1000`. Part 3 trains no new weights; all other models are downloaded from Hugging Face.

## How to run it
1. Open `notebooks/p3_agents.ipynb` in Colab with a **T4 GPU**.
2. Add the Colab secrets `HF_TOKEN` and `GH_TOKEN`, and turn on *Notebook access* for both. Cell 1 already points to our repo
   (`abhinitasanabada-web/data266-lab2-team-26`) and clones it, so it prints the commit of the run (C1).
3. Use *Runtime → Run all* and allow the Google Drive prompt. The notebook runs in this order:
   1. It writes and commits the scenario file.
   2. The CPU self-test runs.
   3. The models load (about 18 GB of downloads the first time, ~4–7 minutes on a T4, 9.6 GiB of GPU memory).
   4. The demo, the evaluation (about 10 minutes), the human-evaluation analysis and the app run.
4. **Human evaluation:**
   - Each of us opens `MyDrive/data266_p3_work/full/p3_rating_sheet.html` **alone**, rates all 67 items and downloads the CSV.
   - We commit both files to `src/part3/abhinita/human_eval/response_full/`, and any discussed labels to
     `adjudication_full.csv`.
   - Then we use Run all again. Finished requests are cached on Drive, so this is quick.
5. **If a push from the notebook fails** (our token could read but not write), the notebook saves the files to commit in
   `MyDrive/data266_p3_work/to_commit/`, with the same folders as the repo, and we upload them by hand.
6. Save the notebook with all outputs as `notebooks/p3_agents.ipynb`, commit it and tag the commit `p3-v1.0`.

`SMOKE = True` runs 5 scenarios and 3 red-team cases into a separate folder as a quick test.

**Note on C2.3:** the scenario file was written and committed inside the notebook before the evaluation cell ran. Because our
token could not push, we uploaded it to GitHub by hand afterwards.

## Safety (C4)
Our safety measures are layered:
1. a rule-based screen of the request text
2. the SD safety checker on the uploaded image
3. Qwen's own classification, which can refuse
4. no edits of photos of real, identifiable people
5. text inside images is treated as data and never reaches the prompt writer
6. a fixed safety negative prompt and the SD safety checker on every output; flagged images are never saved or shown
7. clear messages for broken input

The red-team results and a discussion of each case are in our report. The notebook also contains a draft **system card**
(what the system can do, known limits and safety measures) that we rewrite in our own words for the report.

## Limitations
- **Qwen2.5-VL-3B** can hallucinate objects or misread text.
- **InstructPix2Pix** often ignores small or very specific edits.
- **img2img** trades off style against keeping the layout.
- **Our Naruto adapter** learned the style only partly. This is also where we disagreed most as raters.
- **The critic** is a heuristic (a 3B VLM plus a fixed CLIP threshold). In our run it was too strict on style changes and
  added latency without raising task success.
- **The rule screen** is keyword-based and can over-refuse benign requests that use words like "blood".
- **The SD safety checker** blocked one harmless text image (R01).

## Software
- Python 3.13 with the pinned versions in [`requirements-p3.txt`](requirements-p3.txt), plus `gradio` and `scikit-image`.
  We use the same `diffusers`, `transformers` and `peft` versions as Part 2, so our adapter loads exactly as saved.
- The exact versions that ran are written to `results/p3/full/p3_env.txt`.

**Models (all open weights, no hosted APIs):**
- `Qwen/Qwen2.5-VL-3B-Instruct`
- `stable-diffusion-v1-5/stable-diffusion-v1-5`
- `timbrooks/instruct-pix2pix`
- `openai/clip-vit-large-patch14`
- the SD 1.5 safety checker
- our Part 2 adapter (`sanabada/data266-team26-p2-naruto-run1-lora-r16`, `step_1000`)

## References
- Yao et al., 2022. *ReAct: Synergizing Reasoning and Acting in Language Models.* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Madaan et al., 2023. *Self-Refine: Iterative Refinement with Self-Feedback.* [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
- Brooks et al., 2023. *InstructPix2Pix: Learning to Follow Image Editing Instructions.* [arXiv:2211.09800](https://arxiv.org/abs/2211.09800)
- Bai et al., 2025. *Qwen2.5-VL Technical Report.* [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
- Radford et al., 2021. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

Who did what and our meeting recordings are in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md); the AI coding assistants we used are
listed in [`AI_USAGE.md`](AI_USAGE.md).
