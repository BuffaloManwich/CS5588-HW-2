# CS5588-HW-2: SearchLVLM Baseline Reproduction

This repo contains a reproducibility baseline for the [SearchLVLMs](https://github.com/NeverMoreLCH/SearchLVLMs) system, adapted for our environment.

## What works now
- Environment pinned via `environment.yml` (Python 3.9.20, torch 2.2.0+cu118, numpy 1.26.3, transformers 4.47.0).
- Deterministic seeds and logging scripts under `reproduce/`.
- Baseline eval run with `llava_v1.5_7b` on a toy 2-sample JSONL dataset, predictions saved to `runs/baseline/llava_v1.5_7b.min_pred.json`.
- Provenance captured under `logs/baseline_provenance.json`.

## Next steps
- Expand to full OneDrive datasets (mapped to schema: `image`, `text`, `question_id`, `category`, `answer_text`).
- Package reproducibility report per course Guide.

---
