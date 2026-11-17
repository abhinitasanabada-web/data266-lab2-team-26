# Part 1 output manifest

After a real run, this directory should contain:

```text
submission.csv          # exact Questions.csv row order; validated locally
metrics.json             # run metadata, timing, model and GPU information
records.json             # extracted text/table/figure evidence with provenance
golden_metrics.json      # only after a real 25-item golden_set.json is committed
figures/                 # extracted/annotated figure assets
chroma/                  # persisted vector index when Chroma is available
```

The raw PDFs and competition CSVs belong in the local/Kaggle data directory and should not be committed to GitHub.
