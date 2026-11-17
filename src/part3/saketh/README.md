# Saketh Part 3 — Multimodal Multi-Agent Baseline

## Pipeline

The baseline uses a typed dataclass state and plain Python functions for Planner, Vision Understanding, Prompt Engineering, Image Generation, and Critique. It saves a JSON trace for each request and exposes a small Gradio interface.

## Run

Place `docs/part3_cases.json` at the project root. Set `PART2_ADAPTER_PATH` to the selected Part 2 adapter and run `notebooks/p3_agents.ipynb` with **Restart & Run All** on the GPU. The notebook writes traces, result PNGs, `metrics.json`, and `run_summary.json` under `part3/outputs`.

## Audited baseline result

The saved baseline `metrics.json` reports a route-level functional pass rate of `0.8` and a red-team **defense pass rate** of `0.9`. The field name `red_team_attack_success_rate` in the legacy evaluator is misleading and must not be reported as attack success.

## Required completion before final submission

The current baseline does not provide the complete required comparison with critique off/on, two-member human ratings, adjudication, CLIP similarity, or VLM critique. The img2img branch also disables the safety checker. Complete those items—or use the team-final Part 3 notebook and artifacts—before treating Part 3 as final.
