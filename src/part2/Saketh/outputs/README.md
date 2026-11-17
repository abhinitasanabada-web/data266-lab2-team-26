# Part 2 output manifest

The saved baseline run should contain:

```text
data/dataset_metadata.json
prompts_before_training.json
lora_rank4/pytorch_lora_weights.safetensors
lora_rank4/training_record.json
lora_rank16/pytorch_lora_weights.safetensors
lora_rank16/training_record.json
generated/generation_manifest.json
generated/<base|run1|run2>_prompt##_seed#.png
metrics.json                    # required: KID, CLIP, LPIPS, NSFW counts
memorization_check.json         # required: 10 test images and 3 closest pairs
human_eval/                     # required: blinded 25-pair ratings and summary
```

Large weights and generated images should be uploaded to Hugging Face; keep only small metadata and links in GitHub.
