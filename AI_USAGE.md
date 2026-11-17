# AI_USAGE.md

## Tools we used

We used ChatGPT/OpenAI Codex as a coding and documentation assistant, mainly for:

- **Debugging:** understanding library and API errors, and finding compatible Diffusers/PEFT changes.
- **Reproducibility:** reviewing our file organization, run instructions, and reproducibility checks.
- **Writing support:** drafting README material, evaluation checklists, and the IEEE-format report layout.
- **Submission checks:** confirming that all required fields, safety sections, and links were present.

## What we did ourselves

- **Design decisions:** We chose the models, datasets, prompts, seeds, training budgets, routing rules, and evaluation protocols.
- **Experiments:** We ran all training, generation, and notebook execution, computed the metrics, collected the human ratings, and made the red-team judgements. Everything is recorded in our project artifacts.
- **Reported results:** Every number in our report comes from executed notebook outputs or committed result files. Where a measurement was missing, we left it missing rather than filling it with a generated estimate.
- **Access and publishing:** We never gave the assistant secrets or access tokens. We submitted to Kaggle, pushed Git tags, and published Hugging Face artifacts ourselves.

## Disclosure

We used AI assistance only for implementation support, debugging, organization, and editing. We take full responsibility for the code, experiments, citations, safety analysis, and final submission.
