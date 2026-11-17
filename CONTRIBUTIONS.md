# DATA 266 Lab 2 — Contributions

Team: Abhinita Sanabada and Saketh Prudhvi Marru  
Repository: <https://github.com/abhinitasanabada-web/data266-lab2-team-26>

Both members contributed to all three parts. The division below records the independent implementations and the team integration work visible in the repository.

## Part 1 — Multimodal RAG

| Member | Contribution |
|---|---|
| Saketh Prudhvi Marru | Implemented the independent RAG baseline in `src/part1/saketh/`: PDF/figure ingestion, hybrid retrieval, conversation memory, answer formatting, evaluation utilities, and submission validation. |
| Abhinita Sanabada | Developed the selected v7 multimodal RAG pipeline, chart-reading workflow, reranking/grounding controls, Kaggle notebook, golden-set reporting, and final integration. |
| Both | Reviewed retrieval behavior, safety handling, JSON output requirements, and the final Part 1 selection. |

## Part 2 — Domain-specific diffusion

| Member | Contribution |
|---|---|
| Saketh Prudhvi Marru | Implemented and ran the independent Stable Diffusion 1.5 LoRA training/generation baseline in `src/part2/Saketh/`, including rank-4 and rank-16 adapters, dataset preparation, training records, and generation manifest code. |
| Abhinita Sanabada | Developed the selected team LoRA/DoRA runs, validation-based checkpoint selection, KID/CLIP/LPIPS evaluation, human-evaluation workflow, model cards, and external artifact publishing. |
| Both | Reviewed the fixed prompt design, safety filtering, failure cases, and model-selection trade-offs. |

## Part 3 — Multimodal multi-agent system

| Member | Contribution |
|---|---|
| Saketh Prudhvi Marru | Implemented the independent typed-state agent baseline in `src/part3/saketh/`: planner, vision fallback, prompt engineering, image generation, critique, trace writing, evaluator, and Gradio interface. |
| Abhinita Sanabada | Developed the selected five-agent evaluation pipeline, critique-off/on comparison, red-team protocol, human rating/adjudication workflow, system card, and final report integration. |
| Both | Reviewed agent routing, safety behavior, demo scenarios, limitations, and final system selection. |

## Verification

This file describes repository evidence; it does not replace the required executed notebooks, evaluation outputs, or recordings.
